"""

Script to evaluate and visualize a Atari policies

"""
import gym
import argparse
import torch
import numpy as np

import common.vec_env.wrappers as wrappers
from common.models import *

def load_policy(env, args):
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.n
    if args.algo == 'dqn':
        policy_fn = AtariQNetwork
    else:
        policy_fn = AtariGRUPolicy if args.recurr else AtariPolicy
    policy = policy_fn(obs_dim, act_dim)
    policy.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    return policy

def select_action(policy, obs, args, hx=None):
    x = torch.from_numpy(obs[None]).float()
    if args.algo == "dqn":
        act = policy(x).max(1)[1]
    else:
        if hx is None:
            pi, v = policy(x)
        else:
            pi, v, hx = policy(x, hx)
        act = pi.probs.detach().max(1)[1] if args.greedy else pi.sample()
    return act.cpu().numpy()[0], hx

def main(args):
    env = wrappers.make_atari(args.env)
    fstack_val = 2 if args.recurr else 4
    env = wrappers.wrap_deepmind(env, episode_life=False, frame_stack=True, frame_stack_val=fstack_val, scale=True)
    policy = load_policy(env, args)

    # Running Policy
    returns = []
    for _ in range(args.test_eps):
        rew_t = 0; obs = env.reset(); done = False
        hx = torch.zeros(1, 256) if args.recurr else None
        while not done:
            if args.render: env.render()
            action, hx = select_action(policy, obs, args, hx)
            obs, rew, done, info = env.step(action)
            rew_t += rew
        returns.append(rew_t)
    env.close()

    # Logging output
    print("MeanRew: ", np.mean(returns))
    print("Std. Dev of MeanRew: ", np.std(returns) / np.sqrt(len(returns)))
    print("Episodes: ", args.test_eps)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='environment name')
    parser.add_argument('--algo', type=str, default='ppo', choices=['a2c', 'dqn', 'ppo'], help='algo for checkpoint')
    parser.add_argument('--checkpoint', type=str, help='checkpoint to evaluate/visualize')
    parser.add_argument('--test_eps', type=int, default=10, help='number of episodes to evaluate')
    parser.add_argument('--recurr', action='store_true', help='whether to use a recurrent policy or not')
    parser.add_argument('--render', action='store_true', help='whether to render the policy')
    parser.add_argument('--greedy', action='store_true', help='whether to greedily select action')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)