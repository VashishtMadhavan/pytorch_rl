import torch
import os
import argparse
import gym
from algos.ddpg.ddpg import TD3
from common.utils import make_mujoco_env
from common.models import MLPActor, MLPCritic

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2', help='environment name')
    parser.add_argument('--timesteps', type=int, default=int(1e6), help='number of timesteps to train for')
    parser.add_argument('--seed', type=int, default=0, help='seed for gym, numpy, and torch')
    parser.add_argument('--outdir', type=str, default='ddpg_mujoco_debug/', help='location of saved output from training')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--actor_lr', type=float, default=1e-3, help='learning rate for training the policy')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='learning rate for training the Q function')
    parser.add_argument('--log_iters', type=int, default=int(1e4), help='log at this frequency of timesteps')
    parser.add_argument('--gpu', type=str, default='0', help='which GPU to use. If no GPU use -1')

    # TD3 specific parameters
    parser.add_argument('--replay_size', type=int, default=int(1e6), help='number of transitions stored in the replay buffer')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training the Q and pi network')
    parser.add_argument('--tau', type=float, default=0.005, help='target network update rate')
    parser.add_argument('--expl_steps', type=int, default=int(1e4), help='number of exploratory steps before training')
    parser.add_argument('--expl_noise', type=float, default=0.1, help='action noise for exploration')
    parser.add_argument('--policy_noise', type=float, default=0.2, help='noise for policy training')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='range to clip target policy noise')
    parser.add_argument('--policy_freq', type=int, default=2, help='frequency of delayed policy updates')
    return parser.parse_args()

def main(args):
    # Load Mujoco Env
    env = make_mujoco_env(args.env, args.seed, normalize=False)
    eval_env = make_mujoco_env(args.env, args.seed, normalize=False)

    # Setting GPU + device information
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train PPO
    ddpg = TD3(env, eval_env, MLPActor, MLPCritic, device, args)
    ddpg.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)