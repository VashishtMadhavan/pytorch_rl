import argparse
import numpy as np
import os
from itertools import count
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb


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

            new_obs, rews. dones, infos = self.env.step(actions)
            ep_R.append(rews)
            ep_D.append(dones)
            values.append(v)
            entropy.append(pi.entropy())
            log_probs.append(pi.log_prob(act_tensor))
            self.obs = new_obs

        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropy = torhc.stack(entropy)
        advantages = gae(values, tau=self.tau) # TODO: implement function, pass in tau as argument
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        return log_probs, advantages, values, entropy, ep_R, ep_D

    def loss(self):
        log_probs, advantages, values, entropy, ep_R, ep_D = self.sample()
        pg_loss = -advantages * torch.exp(log_probs)
        value_loss = (values.squeeze() - returns) ** 2 # TODO: calculate returns
        entropy_loss = torch.mean(torch.sum(entropy, dim=0)) # TODO: confirm u did this right
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
