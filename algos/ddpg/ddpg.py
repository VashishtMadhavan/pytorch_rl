import gym
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.utils import *

class DDPG:
    def __init__(self, env, actor_network, critic_network, device, args):
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.args = args
        self.expl_noise = args.expl_noise

        self.Q = critic_network(self.obs_dim, self.act_dim)
        self.Q_target = critic_network(self.obs_dim, self.act_dim)

        self.pi = actor_network(self.obs_dim, self.act_dim, self.max_action, self.expl_noise)
        self.pi_target = actor_network(self.obs_dim, self.act_dim, self.max_action, self.expl_noise)

        # Freezing target network weights
        for p in self.Q_target.parameters():
            p.requires_grad = False
        for p in self.pi_target.parameters():
            p.requires_grad = False

        self.Q.to(device)
        self.Q_target.to(device)
        self.pi.to(device)
        self.pi_target.to(device)
        self.device = device

    def _select_action(self, obs):
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        act_sample = self.pi(obs_tensor).sample().cpu().numpy()
        return act_sample

    def _sample_replay_data(self):
        # Sample data from buffer
        X_batch, A_batch, R_batch, X_tp1_batch, D_batch = self.pool.sample(self.batch_size)
        X_tensor = torch.from_numpy(X_batch).float().to(self.device)
        X_tp1_tensor = torch.from_numpy(X_tp1_batch).float().to(self.device)
        A_tensor = torch.from_numpy(A_batch).float().to(self.device)
        R_tensor = torch.from_numpy(R_batch).float().to(self.device)
        D_tensor = torch.from_numpy(D_batch).float().to(self.device)
        return X_tensor, A_tensor, R_tensor, X_tp1_tensor, D_tensor

    def _init_train_ops(self):
        self.pool = ReplayMemory(capacity=self.args.replay_size)
        self.timesteps = self.args.timesteps
        self.gamma = self.args.gamma
        self.tau = self.args.tau
        self.updates = self.args.updates
        self.batch_size = self.args.batch_size
        if not os.path.exists(self.args.outdir):
            os.mkdir(self.args.outdir)

        self.rew_tracker = RewardTracker(self.env.num_envs, 1)
        self.logger = Logger(self.args.outdir)
        headers = ["timesteps", "mean_rew", "best_mean_rew", "episodes", "time_elapsed"]
        self.logger.set_headers(headers)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=self.args.critic_lr, weight_decay=1e-2)
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=self.args.actor_lr)
        self.pi_target.load_state_dict(self.pi.state_dict()) # copy init pi_vars to target
        self.Q_target.load_state_dict(self.Q.state_dict()) # copy init Q_vars to target

    def _update_q_network(self):
        for _ in range(self.updates):
            X_t, A_t, R_t, X_tp1, D_t = self._sample_replay_data()

            # Compute critic loss
            self.Q_optimizer.zero_grad()
            Q_t = self.Q(X_t, A_t)
            A_target = self.pi_target(X_tp1)
            y_t = R_t + self.gamma * (1.0 - D_t) * self.Q_target(X_tp1, A_target).detach()
            critic_loss = F.smooth_l1_loss(Q_t, y_t)
            critic_loss.backward()
            self.Q_optimizer.step()

            # Compute actor loss
            self.pi_optimizer.zero_grad()
            actor_loss = - self.Q(X_t, self.pi(X_t)).mean()
            actor_loss.backward()
            self.pi_optimizer.step()

    def train(self):
        self._init_train_ops()
        obs = self.env.reset()
        iters = (self.timesteps // self.env.num_envs) + 1
        start_time = time.time()
        for t in range(iters):
            # Step in Environment
            curr_step = self.env.num_envs * t
            actions = self._select_action(obs)
            new_obs, rews, dones, infos = self.env.step(actions)
            self.rew_tracker.log(rews, dones)
            for i in range(len(obs)):
                self.pool.push(obs[i], actions[i], rews[i], new_obs[i], float(dones[i]))
            obs = new_obs

            if curr_step >= 2500:
                self._update_q_network()
                # Update target network
                for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.pi.parameters(), self.pi_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                # Logging metrics
                self.logger.set("timesteps", (t + 1) * self.env.num_envs)
                self.logger.set("mean_rew", self.rew_tracker.mean_rew)
                self.logger.set("best_mean_rew", self.rew_tracker.best_mean_rew)
                self.logger.set("episodes", self.rew_tracker.total_episodes)
                self.logger.set("time_elapsed", time.time() - start_time)
                if ((t + 1) * self.env.num_envs) % int(1e5) == 0 and t != 0:
                    self.logger.dump()
                    self.logger.save_policy(self.pi, curr_step)

if __name__=="__main__":
    main()