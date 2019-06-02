import torch
import os
import argparse
from algos.ppo.ppo import PPO
from common.utils import make_atari_env
from common.models import AtariPolicy

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='environment name')
	parser.add_argument('--threads', type=int, default=8, help='number of parallel threads to collect data')
	parser.add_argument('--seed', type=int, default=1234, help='seed for experiment')
	parser.add_argument('--timesteps', type=int, default=int(1e6), help='number of timesteps to train for')
	parser.add_argument('--outdir', type=str, default='a2c_debug/', help='location of saved output from training')
	parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
	parser.add_argument('--lr', type=float, default=7e-4, help='learning rate for training the policy')
	parser.add_argument('--gpu', type=str, default='0', help='which GPU to use. If no GPU use -1')

	# PPO specific params
	parser.add_argument('--n_step', type=int, default=32, help='number of steps to collect before update')
	parser.add_argument('--tau', type=float, default=0.95, help='discount factor for GAE')
	parser.add_argument('--vf_coef', type=float, default=0.5, help='coef for value function loss')
	parser.add_argument('--ent_coef', type=float, default=0.01, help='coef for entropy loss')
	parser.add_argument('--surr_batches', type=int, default=4, help='number of surrogate batch updates')
	parser.add_argument('--surr_epochs', type=int, default=4, help='number of surrogate epochs')
	parser.add_argument('--clip_frac', type=float, default=0.2, help='clip fraction for advantage ratio')
	return parser.parse_args()

def main(args):
	# Load Atari Env
	env, game_lives = make_atari_env(args.env, args.threads, args.seed)
	args.game_lives = game_lives

	# Setting GPU + device information
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Train PPO
	ppo = PPO(env, AtariQNetwork, device, args)
	ppo.train()

if __name__ == "__main__":
	args = parse_args()
	main(args)