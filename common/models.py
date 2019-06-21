import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal

def xavier_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super(MLPActor, self).__init__()
        self.l1 = nn.Linear(obs_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, act_dim)
        self.max_action = max_action

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        act_mean = self.max_action * torch.tanh(self.l3(out))
        return act_mean

class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(MLPCritic, self).__init__()
        # Q1 Architecture
        self.l1 = nn.Linear(obs_dim + act_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 Architecture
        self.l4 = nn.Linear(obs_dim + act_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, a):
        concat = torch.cat([x, a], 1)
        x1 = F.relu(self.l1(concat))
        x1 = F.relu(self.l2(x1))

        x2 = F.relu(self.l4(concat))
        x2 = F.relu(self.l5(x2))
        return self.l3(x1), self.l6(x2)

    def Q1(self, x, a):
        concat = torch.cat([x, a], 1)
        x1 = F.relu(self.l1(concat))
        x1 = F.relu(self.l2(x1))
        return self.l3(x1)

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, expl_noise=0.1):
        super(MLPPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.expl_noise = expl_noise
        self.policy = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.act_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_noise = torch.ones(self.act_dim) * self.expl_noise
        self.action_noise.to(device)
        self.apply(xavier_init)

    def forward(self, x):
        act_mean = self.policy(x)
        pi = MultivariateNormal(act_mean, torch.diag(self.action_noise))
        return pi, self.value(x)

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

class AtariGRUPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, gru_size=256):
        super(AtariGRUPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gru_size = gru_size
        self.encoder = NatureCnn(self.obs_dim)
        self.gru = nn.GRUCell(self.encoder.fc.out_features, hidden_size=self.gru_size)
        self.pi = nn.Linear(self.gru_size, self.act_dim)
        self.v = nn.Linear(self.gru_size, 1)
        self.apply(xavier_init)

    def forward(self, x, hx):
        out = self.encoder(x)
        h = self.gru(out, hx)
        return Categorical(logits=self.pi(h)), self.v(h), h