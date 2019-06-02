import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PPO:
    def __init__(self, env, policy, device, args):
        self.env = env
        self.args = args
        self.obs_shape = self.env.observation_space.shape
        self.nA = self.env.action_space.n
        self.policy = policy(obs_dim=self.obs_shape[-1], act_dim=self.nA)
        self.policy.to(device)
        self.device = device

    def _init_train_ops(self):
        self.n_step = self.args.n_step
        self.learning_rate = self.args.lr
        self.gamma = self.args.gamma
        self.tau = self.args.tau
        self.vf_coef = self.args.vf_coef
        self.ent_coef = self.args.ent_coef
        self.threads = self.args.threads
        self.n_batch = self.env.num_envs * self.threads
        self.train_iters = (self.args.timesteps // n_batch) + 1
        self.max_grad_norm = 0.5
        if not os.path.exists(self.args.outdir):
            os.mkdir(self.args.outdir)
        if hasattr(self.args, 'game_lives'):
            logger = Logger(self.args.outdir, self.env.num_envs, self.args.game_lives)
        else:
            logger = Logger(self.args.outdir, self.env.num_envs)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # PPO specific parameters
        self.surr_batches = self.args.surrogate_batches
        self.surr_epochs = self.args.surrogate_epochs
        self.surr_batch_size = self.threads // self.surr_batches
        self.clip_frac = self.args.clip_frac
        return logger

    def sample(self, logger):
        values, log_probs = [], []
        ep_X, ep_A, ep_R, ep_D = [], [], [], []
        for n in range(self.n_step):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(self.obs).float().to(self.device)
                pi, v = self.policy(obs_tensor)
                act_tensor = pi.sample()
                actions = act_tensor.cpu().numpy()
                new_obs, rews, dones, infos = self.env.step(actions)
                logger.log(rews, dones)

                # storing data needed for updates
                ep_X.append(self.obs)
                ep_A.append(actions)
                ep_R.append(rews)
                ep_D.append(dones)
                values.append(v)
                entropy.append(pi.entropy())
                log_probs.append(pi.log_prob(act_tensor))
                self.obs = new_obs
        with torch.no_grad():
            last_obs_tensor = torch.from_numpy(self.obs).float().to(self.device)
            _, last_v = self.policy(last_obs_tensor)

        # return estimation
        returns = np.zeros_like(values); _ret = 0
        for i in range(self.n_step - 1, -1, -1):
            _ret = ep_R[i] + self.gamma * _ret * (1 - ep_D[i])
            returns[i] = _ret
        return log_probs, values, returns, ep_X, ep_A, ep_D

    def _forward_policy(self, ep_X, ep_A, ep_D, old_log_probs):
        values, entropy, log_probs = [], [], []
        for t in range(self.n_step):
            pi, v = self.policy(ep_X[t])
            values.append(v)
            entropy.append(pi.entropy())
            log_probs.append(pi.log_prob(ep_A[t]) - old_log_probs[t])
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        entropy = torch.stack(entropy)

        # Generalized Advantage Estimation
        # TODO: make sure this is correct
        advantages = torch.zeros_like(values); last_gae_lam = 0
        for i in range(self.n_step - 1, -1, -1):
            if i == self.n_step - 1:
                next_done = ep_D[i]
                next_v = values[i]
            else:
                next_done = ep_D[i + 1]
                next_v = values[i + 1]
            delta = ep_R[i] + self.gamma * next_v * (1. - next_done) - values[i]
            advantages[i] = last_gae_lam = delta + self.gamma * self.tau * (1. - next_done) * last_gae_lam
        returns = advantages + values
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        return log_probs, advantages, values, entropy


    def loss(self, old_log_probs, old_values, returns, ep_X, ep_A, ep_D, idx):
        log_ratios, advantages, values, entropy = self._forward_policy(ep_X, ep_A, ep_D, olg_log_probs)

        # clipped pg loss
        ratio = torch.exp(log_ratios)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, min=1.0 - self.clip_frac, max=1.0 + self.clip_frac)
        pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2)[:, idx])

        # clipped value loss
        values_clipped = old_values + torch.clamp(values.squeeze() - old_values, min=-self.clip_frac, max=self.clip_frac)
        vf_loss1 = (values.squeeze() - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        value_loss = torch.mean(torch.max(vf_loss1, vf_loss2)[:, idx])

        # entropy loss
        entropy_loss = torch.mean(entropy[:, idx])
        return pg_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

    def update(self, old_log_probs, old_values, returns, ep_X, ep_A, ep_D):
        for _ in range(self.surr_epochs):
            for i in range(self.surr_batches):
                sample_idx = np.random.choice(self.threads, self.surr_batch_size, replace=False)
                self.optimizer.zero_grad()
                loss = self.loss(old_log_probs, old_values, returns, ep_X, ep_A, ep_D, idx=sample_idx)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def train(self):
        logger = self._init_train_ops()
        self.obs = self.env.reset()
        log_iters = 50
        for t in range(self.train_iters):
            env_steps = t * self.n_batch
            old_log_probs, old_values, returns, ep_X, ep_A, ep_D = self.sample(logger)
            self.update(old_log_probs, old_values, returns, ep_X, ep_A, ep_D)
            if t % log_iters == 0 and t != 0:
                logger.dump(env_steps, {})
                logger.save_policy(self.polcy, t)


if __name__ == '__main__':
    main()
