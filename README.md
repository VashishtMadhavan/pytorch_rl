# Pytorch Implementation of Basic Deep RL Algorithms

### Algorithms Implemented

- DQN - {Discrete Action}
- A2C - {Discrete Action}
- PPO - {Discrete Action}

### Instructions to Train Each Algorithm

To train a DQN agent
```
python -m algos.dqn.run_atari --env 'PongNoFrameskip-v4' # to train a DQN on Pong
```

To train an A2C agent
```
python -m algos.a2c.run_atari --env 'PongNoFrameskip-v4' # to train an A2C agent on Pong
```

To train a PPO agent
```
python -m algos.ppo.run_atari --env 'PongNoFrameskip-v4' # to train a PPO agent on Pong
```

### TODO
- validate PPO
- setup for continuous e