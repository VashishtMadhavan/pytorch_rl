import gym
import os
import gym.wrappers as wrappers

import numpy as np
from collections import namedtuple
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
from utils import *
import pdb

BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 1000
TRAIN_FREQ = 4
MIN_MEM_SIZE = 2500
GRAD_NORM_CLIP = 10
LOG_ITERS = int(1e5)
REPLAY_SIZE = int(2.5e5)

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="PongNoFrameskip-v4")
parser.add_argument("--seed", type=int, default=534)
parser.add_argument("--threads", type=int, default=20)
parser.add_argument("--updates", type=int, default=10)
parser.add_argument("--num_timesteps", type=int, default=int(1e6))
parser.add_argument("--output_dir", type=str, default='tmp')
args = parser.parse_args()

env, game_lives = get_env(args.env, args.seed, args.threads)
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
if game_lives == 0:
    game_lives += 1
torch.manual_seed(args.seed)

nT = args.threads
nU = args.updates
nA = env.action_space.n
exploration_schedule = LinearSchedule(args.num_timesteps, final_p=0.05) 
memory = ReplayMemory(capacity=REPLAY_SIZE)

class DQN(nn.Module):
    def __init__(self, num_actions=18):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(9 * 9 * 32, 256)
        self.out = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.out(x)

Q = DQN(num_actions=env.action_space.n)
Q_target = DQN(num_actions=env.action_space.n)
Q_target.load_state_dict(Q.state_dict())
Q_target.eval()

def select_action(state, t):
    state = torch.div(torch.from_numpy(state).float(), 255.).permute(0, 3, 1, 2)
    eps = exploration_schedule.value(t)
    with torch.no_grad():
        act = Q(state).max(1)[1].numpy()
    rand_idx = np.random.random(act.shape[0]) <= eps
    act[rand_idx] = np.random.randint(0, high=nA, size=sum(rand_idx))
    return act

def main():
    optimizer = optim.Adam(Q.parameters(), lr=1e-4)
    obs = env.reset()
    iters = (args.num_timesteps // nT) + 1
    logger = Logger(args.output_dir, nT, game_lives)

    for t in range(iters):
        env_steps = t * nT
        action = select_action(obs, env_steps)
        new_obs, rews, dones, _ = env.step(action)

        for i in range(len(obs)):
            memory.push(obs[i], action[i], np.sign(rews[i]), new_obs[i], float(dones[i]))
        obs = new_obs
        logger.log(rews, dones)

        if t >= MIN_MEM_SIZE:
            if t % TRAIN_FREQ == 0:
                for _ in range(nU):
                    X_batch, A_batch, R_batch, X_tp1_batch, D_batch = memory.sample(BATCH_SIZE)

                    X_batch = torch.div(torch.from_numpy(X_batch).float(), 255.).permute(0, 3, 1, 2)
                    X_tp1_batch = torch.div(torch.from_numpy(X_tp1_batch).float(), 255.).permute(0, 3, 1, 2)

                    q = Q(X_batch).gather(1, torch.from_numpy(A_batch).unsqueeze(1))
                    q_next = Q_target(X_tp1_batch).max(1)[0].detach()
                    yt = torch.from_numpy(R_batch).float() + GAMMA * (1. - torch.from_numpy(D_batch).float()) * q_next

                    loss = F.smooth_l1_loss(q.squeeze(), yt)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(Q.parameters(), GRAD_NORM_CLIP)
                    optimizer.step()
    
            if env_steps % TARGET_UPDATE == 0:
                Q_target.load_state_dict(Q.state_dict())

        if env_steps % LOG_ITERS == 0 and env_steps != 0:
            logger.dump(env_steps, {})

if __name__=="__main__":
    main()