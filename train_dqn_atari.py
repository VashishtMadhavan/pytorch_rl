import gym
import os
import numpy as np
import argparse
from utils import make_atari_env
from dqn import DQN
from models import AtariQNetwork

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='environment name')
	parser.add_argument('--num_threads', type=int, default=8, help='number of parallel threads to collect data')
	parser.add_argument('--seed', type=int, default=1234, help='seed for experiment')
	parser.add_argument('--timesteps', type=int, default=int(1e6), help='number of timesteps to train for')
	parser.add_argument('--outdir', type=str, default='atari_pong_debug/', help='location of saved output from training')
	parser.add_argument('--replay_size', type=int, default=int(1e6), help='number of transitions stored in the replay buffer')
	parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
	parser.add_argument('--batch_size', type=int, default=128, help='batch size for training the Q network')
	parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training the Q network')
	return parser.parse_args()

def main(args):
	# Load Atari Env
	env, game_lives = make_atari_env(args.env, args.num_threads, args.seed)
	args.game_lives = game_lives

	# Train DQN
	dqn = DQN(env, AtariQNetwork, args)
	dqn.train()

if __name__ == "__main__":
	args = parse_args()
	main(args)