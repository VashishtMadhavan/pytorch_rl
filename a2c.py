import argparse
import numpy as np
import os
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        self.n_batch = self.env.num_envs * self.threads
        self.train_iters = (self.args.timesteps // n_batch) + 1
        self.max_grad_norm = 0.5

        if not os.path.exists(self.args.outdir):
            os.mkdir(self.args.outdir)
        if hasattr(self.args, 'game_lives'):
            logger = Logger(self.args.outdir, self.env.num_envs, self.args.game_lives)
        else:
            logger = Logger(self.args.outdir, self.env.num_envs)
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)
        return logger

    def sample(self):
        values, log_probs, entropy, ep_R, ep_D = [], [], [], [], []
        for n in range(self.n_step):
            obs_tensor = torch.from_numpy(self.obs).float().to(self.device)
            pi, v = self.policy(obs_tensor)
            act_tensor = pi.sample()
            actions = act_tensor.cpu().numpy()

            new_obs, rews, dones, infos = self.env.step(actions)
            ep_R.append(rews)
            ep_D.append(dones)
            values.append(v)
            entropy.append(pi.entropy())
            log_probs.append(pi.log_prob(act_tensor))
            self.obs = new_obs
        last_obs_tensor = torch.from_numpy(self.obs).float().to(self.device)
        _, last_v = self.policy(last_obs_tensor)

        values = torch.stack(values)
        log_probs = torch.stack(log_probs) if log_probs.dim() <= 2 else torch.sum(torch.stack(log_probs), dim=2)
        entropy = torhc.stack(entropy)

        # Generalized Advantage Estimation
        advantages = torch.zeros_like(values).float(); last_gae_lam = 0
        for i in range(self.n_step - 1, -1, -1):
            if i == self.n_step - 1:
                next_done = dones
                next_v = last_v
            else:
                next_done = ep_D[i + 1]
                next_v = values[i + 1]
            delta = ep_R[i] + self.gamma * next_v * (1. - next_done) - values[i]
            advantages[i] = last_gae_lam = delta + self.gamma * self.tau * (1. - next_done) * last_gae_lam
        disc_returns = advantages + values
        return log_probs, advantages, values, entropy, disc_returns

    def loss(self):
        log_probs, advantages, values, entropy, returns = self.sample()
        pg_loss = torch.mean(-advantages * torch.exp(log_probs))
        value_loss = torch.mean((values.squeeze() - returns) ** 2)
        entropy_loss = torch.mean(entropy)
        return pg_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

    def update(self):
        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def train(self):
        logger = self._init_train_ops()
        self.obs = self.env.reset()
        log_iters = (self.args.timesteps // 100)
        for t in range(self.train_iters):
            env_steps = t * self.n_batch
            self.update(episodes)
            if t % log_iters == 0 and t != 0:
                logger.dump(env_steps, {})


if __name__ == '__main__':
    main()
