import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.utils import Logger, RewardTracker

class A2C:
    def __init__(self, env, policy, device, args):
        self.env = env
        self.args = args
        self.obs_shape = self.env.observation_space.shape
        self.nA = self.env.action_space.n
        self.policy = policy(obs_dim=self.obs_shape[-1], act_dim=self.nA)
        self.policy.to(device)
        self.device = device
        self.recurrent = self.args.recurr
        self.gru_size = 256

    def _gae(self, values, last_v, ep_R, ep_D):
        advantages = np.zeros((values.size(0), values.size(1)), dtype=np.float32)
        gae_lam = 0
        for i in range(self.n_step - 1, -1, -1):
            if i == self.n_step - 1:
                _next_val = last_v.squeeze().cpu().numpy()
            else:
                _next_val = values[i + 1].cpu().numpy()
            delta = ep_R[i] + self.gamma * _next_val * (1. - ep_D[i]) - values[i].cpu().numpy()
            advantages[i] = gae_lam = delta + self.gamma * self.tau * (1. - ep_D[i]) * gae_lam
        returns = np.array([advantages[i] + values[i].cpu().numpy() for i in range(len(advantages))])
        return returns

    def _init_train_ops(self):
        self.n_step = self.args.n_step
        self.learning_rate = self.args.lr
        self.gamma = self.args.gamma
        self.tau = self.args.tau
        self.vf_coef = self.args.vf_coef
        self.ent_coef = self.args.ent_coef
        self.threads = self.args.threads
        self.n_batch = self.n_step * self.threads
        self.train_iters = (self.args.timesteps // self.n_batch) + 1
        self.max_grad_norm = 0.5
        if not os.path.exists(self.args.outdir):
            os.mkdir(self.args.outdir)

        self.rew_tracker = RewardTracker(self.threads, self.args.game_lives)
        self.logger = Logger(self.args.outdir)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # TODO: may not be the best way to do this
        headers = ["timestep", 'mean_rew', "best_mean_rew", "episodes",
                    "policy_loss", "value_loss", "entropy", "time_elapsed"]
        self.logger.set_headers(headers)

    def sample(self):
        log_probs, values = [], []
        ep_X, ep_A, ep_R, ep_D = [], [], [], []
        if self.recurrent:
            self.start_hx = self.hx.clone()
        for n in range(self.n_step):
            with torch.no_grad():
                obs_tensor = (torch.from_numpy(self.obs).float()/ 255.).to(self.device)
                if self.recurrent:
                    pi, v, self.hx = self.policy(obs_tensor, self.hx)
                else:
                    pi, v = self.policy(obs_tensor)
                act_tensor = pi.sample()
                actions = act_tensor.cpu().numpy()

                new_obs, rews, dones, infos = self.env.step(actions)
                self.rew_tracker.log(rews, dones)
                ep_X.append(self.obs)
                ep_A.append(actions)
                ep_R.append(rews)
                ep_D.append(dones.astype(np.float32))
                values.append(v)
                log_probs.append(pi.log_prob(act_tensor))
                self.obs = new_obs
                if self.recurrent:
                    dones_tensor = torch.from_numpy(dones.astype(np.float32)).to(self.device)
                    self.hx[dones_tensor == 1, :] = 0.

        with torch.no_grad():
            last_obs_tensor = (torch.from_numpy(self.obs).float() / 255.).to(self.device)
            if self.recurrent:
                _, last_v, _ = self.policy(last_obs_tensor, self.hx)
            else:
                _, last_v = self.policy(last_obs_tensor)

        values = torch.stack(values).squeeze()
        returns = self._gae(values, last_v, ep_R, ep_D)
        sample_dict = {}
        sample_dict['old_log_probs'] = torch.stack(log_probs)
        sample_dict['old_values'] = values
        sample_dict['ep_X'] = torch.from_numpy(np.array(ep_X, dtype=np.float32) / 255.).to(self.device)
        sample_dict['ep_A'] = torch.from_numpy(np.array(ep_A, dtype=np.int32)).to(self.device)
        sample_dict['returns'] = torch.from_numpy(returns).to(self.device)
        return sample_dict

    def _forward_policy(self, sample_dict, idx=None, ratio=False):
        if idx is None:
            idx = np.arange(self.threads)
        ep_X = sample_dict['ep_X'][:, idx]
        ep_A = sample_dict['ep_A'][:, idx]
        old_log_probs = sample_dict['old_log_probs'][:, idx]
        returns = sample_dict['returns'][:, idx]

        if not self.recurrent:
            # can pass sample_dict in batch thru network
            ep_X = ep_X.view(-1, *self.obs_shape)
            pi, values = self.policy(ep_X)
            entropy = pi.entropy().view(self.n_step, idx.shape[0])
            values = values.view(self.n_step, idx.shape[0])
            log_probs = pi.log_prob(ep_A.view(-1)).view(self.n_step, idx.shape[0])
            if ratio:
                log_probs -= old_log_probs
        else:
            values, entropy, log_probs = [], [], []
            hx = self.start_hx
            for t in range(self.n_step):
                pi, v, hx = self.policy(ep_X[t], hx)
                values.append(v)
                entropy.append(pi.entropy())
                if ratio:
                    log_probs.append(pi.log_prob(ep_A[t]) - old_log_probs[t])
                else:
                    log_probs.append(pi.log_prob(ep_A[t]))
            log_probs = torch.stack(log_probs)
            values = torch.stack(values)
            entropy = torch.stack(entropy)
        advantages = returns - values.squeeze().detach()
        return log_probs, advantages, values, entropy

    def loss(self, sample_dict):
        log_probs, advantages, values, entropy = self._forward_policy(sample_dict)
        pg_loss = -(advantages * log_probs).mean()
        value_loss = (sample_dict['returns'] - values.squeeze()).pow(2).mean()
        entropy_loss = torch.mean(entropy)
        self.logger.set("policy_loss", pg_loss.item())
        self.logger.set("value_loss", value_loss.item())
        self.logger.set("entropy", entropy_loss.item())
        return pg_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

    def update(self, sample_dict):
        self.optimizer.zero_grad()
        loss = self.loss(sample_dict)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def train(self):
        self._init_train_ops()
        self.obs = self.env.reset()
        if self.recurrent:
            self.hx = torch.zeros(self.threads, self.gru).to(self.device)
        start_time = time.time()
        for t in range(self.train_iters):
            # Updating network
            env_steps = (t + 1) * self.n_batch
            sample_dict = self.sample()
            self.update(sample_dict)

            # Logging values
            self.logger.set("timestep", env_steps)
            self.logger.set("time_elapsed", time.time() - start_time)
            self.logger.set("mean_rew", self.rew_tracker.mean_rew)
            self.logger.set("best_mean_rew", self.rew_tracker.best_mean_rew)
            self.logger.set("episodes", self.rew_tracker.total_episodes)
            if env_steps % int(1e5) == 0:
                self.logger.dump()
                self.logger.save_policy(self.policy, env_steps)


if __name__ == '__main__':
    main()
