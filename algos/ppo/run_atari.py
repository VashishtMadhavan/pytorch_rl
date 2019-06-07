import torch
import os
import argparse
from algos.ppo.ppo import PPO
from common.utils import make_atari_env
from common.models import AtariPolicy, AtariGRUPolicy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='environment name')
    parser.add_argument('--threads', type=int, default=20, help='number of parallel threads to collect data')
    parser.add_argument('--seed', type=int, default=0, help='seed for experiment')
    parser.add_argument('--timesteps', type=int, default=int(10e6), help='number of timesteps to train for')
    parser.add_argument('--outdir', type=str, default='ppo_debug/', help='location of saved output from training')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate for training the policy')
    parser.add_argument('--recurr', action='store_true', help='whether to use a recurrent policy or not')
    parser.add_argument('--gpu', type=str, default='0', help='which GPU to use. If no GPU use -1')

    # PPO specific params
    parser.add_argument('--n_step', type=int, default=125, help='number of steps to collect before update')
    parser.add_argument('--tau', type=float, default=0.95, help='discount factor for GAE')
    parser.add_argument('--vf_coef', type=float, default=0.25, help='coef for value function loss')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='coef for entropy loss')
    parser.add_argument('--surr_batches', type=int, default=4, help='number of surrogate batch updates')
    parser.add_argument('--surr_epochs', type=int, default=3, help='number of surrogate epochs')
    parser.add_argument('--clip_frac', type=float, default=0.2, help='clip fraction for advantage ratio')
    return parser.parse_args()

def main(args):
    # Load Atari Env
    # Generally, with recurrent policies, we change the frame_stack to 1 or 2, instead of 4
    # this is because of the partial observability problem
    frame_stack = 2 if args.recurr else 4
    env, game_lives = make_atari_env(args.env, args.threads, args.seed, frame_stack)
    args.game_lives = game_lives

    # Setting GPU + device information
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train PPO
    policy = AtariGRUPolicy if args.recurr else AtariPolicy
    ppo = PPO(env, policy, device, args)
    ppo.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)