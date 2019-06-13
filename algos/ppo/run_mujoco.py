import torch
import os
import argparse
from algos.ppo.ppo import PPO
from common.utils import make_mujoco_env
from common.models import MLPPolicy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2', help='environment name')
    parser.add_argument('--timesteps', type=int, default=int(1e6), help='number of timesteps to train for')
    parser.add_argument('--outdir', type=str, default='ppo_mujoco_debug/', help='location of saved output from training')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate for training the policy')
    parser.add_argument('--log_iters', type=int, default=int(1e4), help='log at this frequency of timesteps')
    parser.add_argument('--gpu', type=str, default='0', help='which GPU to use. If no GPU use -1')

    # PPO specific params
    parser.add_argument('--n_step', type=int, default=2048, help='number of steps to collect before update')
    parser.add_argument('--tau', type=float, default=0.95, help='discount factor for GAE')
    parser.add_argument('--vf_coef', type=float, default=0.25, help='coef for value function loss')
    parser.add_argument('--ent_coef', type=float, default=0.0, help='coef for entropy loss')
    parser.add_argument('--surr_batches', type=int, default=32, help='number of surrogate batch updates')
    parser.add_argument('--surr_epochs', type=int, default=10, help='number of surrogate epochs')
    parser.add_argument('--clip_frac', type=float, default=0.2, help='clip fraction for advantage ratio')
    return parser.parse_args()

def main(args):
    # Load Atari Env
    env = make_mujoco_env(args.env)

    # Setting GPU + device information
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train PPO
    ppo = PPO(env, MLPPolicy, device, args)
    ppo.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)