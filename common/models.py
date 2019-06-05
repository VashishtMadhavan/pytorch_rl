import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

def xavier_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

class NatureCnn(nn.Module):
    def __init__(self, obs_dim):
        super(NatureCnn, self).__init__()
        self.obs_dim = obs_dim
        self.conv1 = nn.Conv2d(self.obs_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)

    def forward(self, x):
        out = x.permute(0, 3, 1, 2)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc(out))
        return out

class AtariQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(AtariQNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.encoder = NatureCnn(self.obs_dim)
        self.final = nn.Linear(self.encoder.fc.out_features, self.act_dim)
        self.apply(xavier_init)
        
    def forward(self, x):
        out = self.encoder(x)
        return self.final(out)

class AtariPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(AtariPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.encoder = NatureCnn(self.obs_dim)
        self.pi = nn.Linear(self.encoder.fc.out_features, self.act_dim)
        self.v = nn.Linear(self.encoder.fc.out_features, 1)
        self.apply(xavier_init)

    def forward(self, x):
        out = self.encoder(x)
        return Categorical(logits=self.pi(out)), self.v(out)