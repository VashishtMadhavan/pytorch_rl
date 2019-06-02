import torch
import os
import argparse
from algos.dqn.dqn import DQN
from common.utils import make_atari_env
from common.models import AtariQNetwork

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='environment name')
	parser.add_argument('--threads', type=int, default=100, help='number of parallel threads to collect data')
	parser.add_argument('--updates', type=int, default=25, help='number of Q updates per step')
	parser.add_argument('--seed', type=int, default=1234, help='seed for experiment')
	parser.add_argument('--timesteps', type=int, default=int(1e6), help='number of timesteps to train for')
	parser.add_argument('--outdir', type=str, default='dqn_debug/', help='location of saved output from training')
	parser.add_argument('--replay_size', type=int, default=int(1e6), help='number of transitions stored in the replay buffer')
	parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
	parser.add_argument('--batch_size', type=int, default=128, help='batch size for training the Q network')
	parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training the Q network')
	parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
	return parser.parse_args()

def main(args):
	# Load Atari Env
	env, game_lives = make_atari_env(args.env, args.threads, args.seed)
	args.game_lives = game_lives

	# set GPU + device information
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Train DQN
	dqn = DQN(env, AtariQNetwork, device, args)
	dqn.train()

if __name__ == "__main__":
	args = parse_args()
	main(args)