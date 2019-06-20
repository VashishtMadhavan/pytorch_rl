import torch
import os
import argparse
from algos.ddpg.ddpg import DDPG
from common.utils import make_mujoco_env
from common.models import MLPActor, MLPCritic

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2', help='environment name')
    parser.add_argument('--timesteps', type=int, default=int(1e6), help='number of timesteps to train for')
    parser.add_argument('--outdir', type=str, default='ddpg_mujoco_debug/', help='location of saved output from training')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='learning rate for training the policy')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='learning rate for training the Q function')
    parser.add_argument('--log_iters', type=int, default=int(1e4), help='log at this frequency of timesteps')
    parser.add_argument('--gpu', type=str, default='0', help='which GPU to use. If no GPU use -1')

    # DDPG specific parameters
    parser.add_argument('--replay_size', type=int, default=int(1e6), help='number of transitions stored in the replay buffer')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training the Q and pi network')
    parser.add_argument('--expl_noise', type=float, default=0.2, help='action noise for exploration')
    return parser.parse_args()

def main(args):
    # Load Atari Env
    env = make_mujoco_env(args.env)

    # Setting GPU + device information
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train PPO
    ddpg = DDPG(env, MLPActor, MLPCritic, device, args)
    ddpg.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)