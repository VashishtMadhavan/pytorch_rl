import gym
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.utils import *

class DQN:
    def __init__(self, env, q_network, device, args):
        self.env = env
        self.obs_shape = self.env.observation_space.shape
        self.args = args
        self.nA = self.env.action_space.n

        self.Q = q_network(obs_dim=self.obs_shape[-1], act_dim=self.nA)
        self.Q_target = q_network(obs_dim=self.obs_shape[-1], act_dim=self.nA)
        # Freezing target network weights
        for p in self.Q_target.parameters():
            p.requires_grad = False

        self.Q.to(device)
        self.Q_target.to(device)
        self.device = device

    def _select_action(self, obs, epsilon):
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            acts = self.Q(obs_tensor).max(1)[0].cpu().numpy()
        rand_idx = np.random.random(acts.shape) < epsilon
        acts[rand_idx] = np.random.randint(0, high=self.nA, size=sum(rand_idx))
        return acts.astype(np.int32)

    def _sample_replay_data(self):
        # Sample data from buffer
        X_batch, A_batch, R_batch, X_tp1_batch, D_batch = self.pool.sample(self.batch_size)
        X_tensor = torch.from_numpy(X_batch).float().to(self.device)
        X_tp1_tensor = torch.from_numpy(X_tp1_batch).float().to(self.device)
        A_tensor = torch.from_numpy(A_batch).long().to(self.device)
        R_tensor = torch.from_numpy(R_batch).float().to(self.device)
        D_tensor = torch.from_numpy(D_batch).float().to(self.device)
        return X_tensor, A_tensor, R_tensor, X_tp1_tensor, D_tensor

    def _init_train_ops(self):
        self.exploration_schedule = LinearSchedule(self.args.timesteps, final_p=0.01)
        self.pool = ReplayMemory(capacity=self.args.replay_size)
        self.timesteps = self.args.timesteps
        self.learning_rate = self.args.lr
        self.gamma = self.args.gamma
        self.updates = self.args.updates
        self.batch_size = self.args.batch_size
        if not os.path.exists(self.args.outdir):
            os.mkdir(self.args.outdir)
        if hasattr(self.args, 'game_lives'):
            logger = Logger(self.args.outdir, self.env.num_envs, self.args.game_lives)
        else:
            logger = Logger(self.args.outdir, self.env.num_envs)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        self.Q_target.load_state_dict(self.Q.state_dict()) # copy init Q_vars to target
        return logger

    def _update_q_network(self):
        for _ in range(self.updates):
            X_t, A_t, R_t, X_tp1, D_t = self._sample_replay_data()
            # Compute loss + udpate network
            self.optimizer.zero_grad()
            Q_t = self.Q(X_t).gather(1, A_t.view(-1, 1)).squeeze()
            Q_tp1_max = self.Q_target(X_tp1).detach().max(1)[0]
            y_t = R_t + self.gamma * (1.0 - D_t) * Q_tp1_max
            loss = F.smooth_l1_loss(Q_t, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(self.Q.parameters(), 10)
            self.optimizer.step()

    # TODO: add logic to save network
    def train(self):
        logger = self._init_train_ops()
        obs = self.env.reset()
        iters = (self.timesteps // self.env.num_envs) + 1
        target_update_freq = 1000 // self.env.num_envs
        log_update_freq = 100
        for t in range(iters):
            curr_step = self.env.num_envs * t
            eps = self.exploration_schedule.value(curr_step)
            actions = self._select_action(obs, eps)
            new_obs, rews, dones, infos = self.env.step(actions)
            logger.log(rews, dones)
            for i in range(len(obs)):
                self.pool.push(obs[i], actions[i], np.sign(rews[i]), new_obs[i], float(dones[i]))
            obs = new_obs

            if curr_step >= 2500:
                self._update_q_network()
                # Update target network
                if t % target_update_freq == 0:
                    self.Q_target.load_state_dict(self.Q.state_dict())
                if t % log_update_freq == 0 and t != 0:
                    logger.dump(curr_step, {})
                    logger.save_policy(self.Q, t)

if __name__=="__main__":
    main()