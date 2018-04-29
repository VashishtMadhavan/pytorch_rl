from vec_env.vec_frame_stack import VecFrameStack
from vec_env.subproc_vec_env import SubprocVecEnv

import os
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import h5py

def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank):
        def _thunk():
            env = wrappers.make_atari(env_id)
            env.seed(seed + rank)
            return wrappers.wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    np.random.seed(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def get_env(env_name, seed, num_threads):
    tenv = gym.make(env_name)
    game_lives = tenv.unwrapped.ale.lives()
    if game_lives == 0:
        game_lives += 1
    wrapper_dict = {'episode_life': True, 'clip_rewards': False, 'frame_stack': False, 'scale': False}
    env = VecFrameStack(make_atari_env(env_name, num_threads, seed, wrapper_kwargs=wrapper_dict), 4)
    return env, game_lives

class LinearSchedule(object):
    def __init__(self, tsteps, final_p, initial_p=1.0):
        self.tsteps = tsteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction  = min(float(t) / self.tsteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, obs, act, rew, next_obs, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (obs, act, rew, next_obs, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        rand_idx = np.random.randint(0, len(self.memory), size=batch_size)
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in rand_idx:
            data = self.memory[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def __len__(self):
        return len(self.memory)


class Logger:
    def __init__(self, expt_dir, num_threads, game_lives):
        self.log_file = expt_dir + '/log.txt'
        self.num_threads = num_threads
        self.game_lives = game_lives
        self.tot_rews = []
        self.ep_rews = np.zeros(num_threads)
        self.ep_life_counter = np.zeros(num_threads)

    def set_headers(self, headers):
        with open(self.log_file, 'a') as f:
            print(*headers, file=f)

    def log(self, rewards, dones):
        self.ep_rews += rewards
        for i, done in enumerate(dones):
            if done:
                self.ep_life_counter[i] += 1
            if self.ep_life_counter[i] == self.game_lives:
                self.tot_rews.append(float(self.ep_rews[i]))
                self.ep_life_counter[i] = 0
                self.ep_rews[i] = 0.

    def dump(self, itr, info):
        if len(self.tot_rews) > 0:
            rew_mean = np.mean(self.tot_rews[-100:])
            best_rew_mean = np.max(self.tot_rews)
        
        if itr > 0:
            print_values = [itr, rew_mean, best_rew_mean]
            print("Timestep %d" % (itr,))
            print("mean reward (100 episodes) %f" % rew_mean)
            print("best mean reward %f" % best_rew_mean)
            print("episodes %d" % len(self.tot_rews))
            for k in info:
                print(k + ' ' + str(info[k]))
                print_values.append(info[k])
            print()
            sys.stdout.flush()

            with open(self.log_file, 'a') as f:
                print(*print_values, file=f)