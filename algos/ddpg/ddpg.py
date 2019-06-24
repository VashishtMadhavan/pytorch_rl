import gym
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.utils import *
from tqdm import tqdm

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class TD3:
    def __init__(self, env, eval_env, actor_network, critic_network, device, args):
        self.env = env
        self.eval_env = eval_env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.args = args

        self.Q = critic_network(self.obs_dim, self.act_dim)
        self.Q_target = critic_network(self.obs_dim, self.act_dim)

        self.pi = actor_network(self.obs_dim, self.act_dim, self.max_action)
        self.pi_target = actor_network(self.obs_dim, self.act_dim, self.max_action)

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

    def _select_action(self, obs, noise=False):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            act = self.pi(obs_tensor).cpu().numpy()
        if noise:
            act = (act + np.random.normal(0, self.expl_noise, size=self.env.action_space.shape[0]))
            act = act.clip(self.env.action_space.low, self.env.action_space.high)
        return act

    def _init_train_ops(self):
        self.pool = ReplayMemory(capacity=self.args.replay_size)
        self.timesteps = self.args.timesteps
        self.gamma = self.args.gamma
        self.tau = self.args.tau
        self.log_iters = self.args.log_iters
        self.expl_steps = self.args.expl_steps
        self.batch_size = self.args.batch_size
        self.expl_noise = self.args.expl_noise
        self.policy_freq = self.args.policy_freq
        self.policy_noise = self.args.policy_noise
        self.noise_clip = self.args.noise_clip
        if not os.path.exists(self.args.outdir):
            os.mkdir(self.args.outdir)

        self.rew_tracker = RewardTracker(self.env.num_envs, 1)
        self.logger = Logger(self.args.outdir)
        headers = ["timesteps", "mean_rew", "best_mean_rew", "episodes", "time_elapsed"]
        self.logger.set_headers(headers)
        self.pi_target.load_state_dict(self.pi.state_dict()) # copy init pi_vars to target
        self.Q_target.load_state_dict(self.Q.state_dict()) # copy init Q_vars to target
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=self.args.critic_lr)
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=self.args.actor_lr)

    def _update_q_network(self, t):
        # Sample data from buffer
        X_batch, A_batch, R_batch, X_tp1_batch, D_batch = self.pool.sample(self.batch_size)
        X_t = torch.from_numpy(X_batch).float().to(self.device)
        X_tp1 = torch.from_numpy(X_tp1_batch).float().to(self.device)
        A_t = torch.from_numpy(A_batch).float().to(self.device)
        R_t = torch.from_numpy(R_batch).float().to(self.device)
        D_t = torch.from_numpy(D_batch).float().to(self.device)

        # Compute critic loss
        noise = torch.FloatTensor(A_batch).data.normal_(0, self.policy_noise).to(self.device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_A = (self.pi_target(X_tp1) + noise).clamp(-self.max_action, self.max_action)

        target_Q1, target_Q2 = self.Q_target(X_tp1, next_A)
        target_Q = torch.min(target_Q1, target_Q2)
        y_t = R_t.view(-1, 1) + self.gamma * (1.0 - D_t.view(-1, 1)) * target_Q.detach()
        Q1, Q2 = self.Q(X_t, A_t)

        critic_loss = F.mse_loss(Q1, y_t) + F.mse_loss(Q2, y_t)
        self.Q_optimizer.zero_grad()
        critic_loss.backward()
        self.Q_optimizer.step()

        # Compute actor loss
        if t % self.policy_freq == 0:
            actor_loss = - self.Q.Q1(X_t, self.pi(X_t)).mean()
            self.pi_optimizer.zero_grad()
            actor_loss.backward()
            self.pi_optimizer.step()

            # Updating target network parameters
            for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.pi.parameters(), self.pi_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self):
        self._init_train_ops()
        obs = self.env.reset(); eval_obs = self.eval_env.reset()
        start_time = time.time()
        for t in range(self.timesteps):
            # Step in Environment
            if t < self.expl_steps:
                act = self.env.action_space.sample()[None]
            else:
                act = self._select_action(obs, noise=True)
            eval_act = self._select_action(eval_obs, noise=False)
            new_obs, rew, done, info = self.env.step(act)
            eval_obs, eval_rew, eval_done, _ = self.eval_env.step(eval_act)

            self.rew_tracker.log(eval_rew, eval_done)
            self.pool.push(obs[0], act[0], rew[0], new_obs[0], float(done[0]))
            obs = new_obs

            if t > self.batch_size:
                self._update_q_network(t)
                # Logging metrics
                self.logger.set("timesteps", t + 1)
                self.logger.set("mean_rew", self.rew_tracker.mean_rew)
                self.logger.set("best_mean_rew", self.rew_tracker.best_mean_rew)
                self.logger.set("episodes", self.rew_tracker.total_episodes)
                self.logger.set("time_elapsed", time.time() - start_time)
                if (t + 1) % self.log_iters == 0 and t != 0:
                    self.logger.dump()
                    self.logger.save_policy(self.pi, t)

if __name__=="__main__":
    main()