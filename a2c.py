import argparse
import numpy as np
import os
from itertools import count
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torchvision.transforms as T
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="PongNoFrameskip-v4")
parser.add_argument('--threads', type=int, default=20)
parser.add_argument('--steps', type=int, default=5)
parser.add_argument('--timesteps', type=int, default=int(10e6))
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', type=int, default=543)
parser.add_argument('--output_dir', type=str, default='tmp')
args = parser.parse_args()

env, game_lives = get_env(args.env, args.seed, args.threads)
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

if game_lives == 0:
    game_lives += 1
torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self, num_actions=18):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(9 * 9 * 32, 256)
        self.pi = nn.Linear(256, num_actions)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        pi = self.pi(x)
        v = self.v(x)
        return pi, v

model = Policy(num_actions=env.action_space.n)
optimizer = optim.RMSprop(model.parameters(), lr=7e-4, eps=1e-5)
T = args.timesteps
nT = args.threads
nS = args.steps
n_batch = nT * nS
VF_COEF = 0.5
ENT_COEF = 0.01
LOG_ITERS = int(1e5)

def select_action(state):
    state = torch.div(torch.from_numpy(state).float(), 255.).permute(0, 3, 1, 2)
    pi, v = model(state)
    m = Categorical(logits=pi)
    action = m.sample()
    return action, m.log_prob(action), v, m.entropy()

def main():
    obs = env.reset()
    train_iters = (T // n_batch) + 1
    nW, nH, nC = env.observation_space.shape
    batch_ob_shape = (n_batch, nW, nH, nC)
    logger = Logger(args.output_dir, nT, game_lives)

    for t in range(train_iters):
        env_steps = t * n_batch
        ep_R, ep_D, ep_V, ep_P, ep_H = [], [], [], [], []

        for step in range(nS):
            act, logp, values, entropy = select_action(obs)
            new_obs, rews, dones, _ = env.step(act.numpy())

            logger.log(rews, dones)
            ep_D.append(dones)
            ep_V.append(values)
            ep_P.append(logp)
            ep_H.append(entropy)
            ep_R.append(rews)
            obs = new_obs

        ep_R = np.array(ep_R).astype(np.float)
        ep_D = np.array(ep_R).astype(np.float)

        _, _, R, _ = select_action(obs)
        R = (1. - torch.from_numpy(ep_D[-1]).float()) * R.squeeze()

        for i in reversed(range(len(ep_R))):
            R = args.gamma * R + torch.from_numpy(ep_R[i]).float()
            adv = R - ep_V[i]

            if i == len(ep_R) - 1:
                policy_loss = ep_P[i] * adv
                value_loss = 0.5 * adv.pow(2)
                entropy_loss = ep_H[i]
            else:
                policy_loss = torch.cat((policy_loss, ep_P[i] * adv), 0)
                value_loss = torch.cat((value_loss, 0.5 * adv.pow(2)), 0)
                entropy_loss = torch.cat((entropy_loss, ep_H[i]))


        optimizer.zero_grad()
        loss = policy_loss.mean() - ENT_COEF * entropy_loss.mean() + VF_COEF * value_loss.mean()
        loss.backward()
        optimizer.step()

        if env_steps % LOG_ITERS == 0 and env_steps != 0:
            logger.dump(env_steps, {})


if __name__ == '__main__':
    main()