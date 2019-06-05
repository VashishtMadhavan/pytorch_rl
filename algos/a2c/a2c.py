import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.utils import Logger

class A2C:
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
        self.n_batch = self.n_step * self.threads
        self.train_iters = (self.args.timesteps // self.n_batch) + 1
        self.max_grad_norm = 0.5

        if not os.path.exists(self.args.outdir):
            os.mkdir(self.args.outdir)
        if hasattr(self.args, 'game_lives'):
            logger = Logger(self.args.outdir, self.env.num_envs, self.args.game_lives)
        else:
            logger = Logger(self.args.outdir, self.env.num_envs)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        return logger

    # TODO: add GAE
    def sample(self, logger):
        ep_X, ep_A, ep_R, ep_D = [], [], [], []
        for n in range(self.n_step):
            with torch.no_grad():
                obs_tensor = (torch.from_numpy(self.obs).float()/ 255.).to(self.device)
                pi, v = self.policy(obs_tensor)
                act_tensor = pi.sample()
                actions = act_tensor.cpu().numpy()

                new_obs, rews, dones, infos = self.env.step(actions)
                logger.log(rews, dones)
                ep_X.append(self.obs)
                ep_A.append(actions)
                ep_R.append(rews)
                ep_D.append(dones.astype(np.float32))
                self.obs = new_obs
        with torch.no_grad():
            last_obs_tensor = (torch.from_numpy(self.obs).float() / 255.).to(self.device)
            _, last_v = self.policy(last_obs_tensor)

        # Return Estimation
        returns = np.zeros((len(ep_R), ep_R[0].shape[0]), dtype=np.float32)
        _ret = last_v.squeeze().cpu().numpy()
        for i in range(self.n_step - 1, -1, -1):
            _ret = ep_R[i] + self.gamma * (1.0 - ep_D[i]) * _ret
            returns[i] = _ret

        ep_X = torch.from_numpy(np.array(ep_X, dtype=np.float32) / 255.).to(self.device)
        ep_A = torch.from_numpy(np.array(ep_A, dtype=np.int32)).to(self.device)
        returns = torch.from_numpy(returns).to(self.device)
        return ep_X, ep_A, returns

    def _forward_policy(self, ep_X, ep_A, returns):
        values, entropy, log_probs = [], [], []
        for t in range(self.n_step):
            pi, v = self.policy(ep_X[t])
            values.append(v)
            entropy.append(pi.entropy())
            log_probs.append(pi.log_prob(ep_A[t]))
        log_probs = torch.stack(log_probs)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        values = torch.stack(values)
        entropy = torch.stack(entropy)
        advantages = returns - values.squeeze()
        return log_probs, advantages, values, entropy

    def loss(self, ep_X, ep_A, returns):
        log_probs, advantages, values, entropy = self._forward_policy(ep_X, ep_A, returns)
        pg_loss = -(advantages.detach() * log_probs).mean()
        value_loss = advantages.pow(2).mean()
        entropy_loss = torch.mean(entropy)
        return pg_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

    def update(self, ep_X, ep_A, returns):
        self.optimizer.zero_grad()
        loss = self.loss(ep_X, ep_A, returns)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def train(self):
        logger = self._init_train_ops()
        self.obs = self.env.reset()
        for t in range(self.train_iters):
            env_steps = t * self.n_batch
            ep_X, ep_A, returns = self.sample(logger)
            self.update(ep_X, ep_A, returns)
            if env_steps % int(1e5) == 0 and env_steps != 0:
                logger.dump(env_steps, {})
                logger.save_policy(self.policy, t)


if __name__ == '__main__':
    main()
