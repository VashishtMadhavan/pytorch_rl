# Lightweight PyTorch Implementations of Deep RL Algorithms

### Algorithms

[TODO: add more descriptions to this]
- DQN - Discrete Action
- DDPG - Continuous Action
- A2C - Discrete Action
- PPO - Discrete/Continuous Action

### Instructions to Train Each Algorithm

To train a DQN agent
```
python -m algos.dqn.run_atari --env 'PongNoFrameskip-v4' #  train a DQN on Pong
```
To train a DDPG agent
```
python -m algos.ddpg.run_mujoco --env 'Ant-v2' # train a DDPG agent on Ant
```
To train an A2C agent
```
python -m algos.a2c.run_atari --env 'PongNoFrameskip-v4' # train an A2C agent on Pong
python -m algos.a2c.run_atari --env 'PongNoFrameskip-v4' --recurr # train a recurrent A2C agent on Pong
```
To train a PPO agent
```
python -m algos.ppo.run_atari --env 'PongNoFrameskip-v4' # train a FF PPO agent on Pong
python -m algos.ppo.run_atari --env 'PongNoFrameskip-v4' --recurr # train a recurrent PPO agent on Pong
python -m algos.ppo.run_mujoco --env 'HalfCheetah-v2' # train an MLP agent on Ant
```
### Evaluating Trained Agents

To evaluate/visualize a trained Atari agent
```
python -m scripts.eval_atari --env {env_name} --algo {a2c,ppo,dqn} --checkpoint {policy_checkpoint}
```
To evaluate/visualize a trained Mujoco agent
```
python -m scripts.eval_mujoco --env {env_name} --algo {a2c,ppo,ddpg} --checkpoint {policy_checkpoint}
```

### TODO
- Validate PPO for continuous actions
- Validate DDPG on Mujoco
- Add video saving to eval script