"""

Script to evaluate and visualize a Mujoco policies

"""
import gym
import argparse
import torch
import numpy as np
from common.models import MLPPolicy
from common.utils import make_mujoco_env

def load_policy(env, args):
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    policy_fn = MLPPolicy
    policy = policy_fn(obs_dim, act_dim)
    policy.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    return policy

def select_action(policy, obs, args):
    x = torch.from_numpy(obs[None]).float()
    pi, v = policy(x)
    return pi.sample().cpu().numpy()[0]

def main(args):
    env = make_mujoco_env(args.env)
    policy = load_policy(env, args)

    # Running Policy
    returns = []
    for _ in range(args.test_eps):
        rew_t = 0; obs = env.reset(); done = False
        while not done:
            if args.render: env.render()
            action, hx = select_action(policy, obs, args)
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
    parser.add_argument('--env', type=str, default='HalfCheetah-v2', help='environment name')
    parser.add_argument('--algo', type=str, default='ppo', choices=['a2c', 'ppo'], help='algo for checkpoint')
    parser.add_argument('--checkpoint', type=str, help='checkpoint to evaluate/visualize')
    parser.add_argument('--test_eps', type=int, default=10, help='number of episodes to evaluate')
    parser.add_argument('--render', action='store_true', help='whether to render the policy')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)