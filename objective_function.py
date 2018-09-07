import torch
import math


def gym_objective_func(network, env, max_steps_in_episodes, gamma):
    """run one episode in given gym environment and return the collected reward"""
    episode_return = 0.0
    state = env.reset()
    for t in range(max_steps_in_episodes):
        state = torch.from_numpy(state).float()
        action = network.forward(state)
        state, reward, done, _ = env.step(action)
        episode_return += reward * math.pow(gamma, t)
        if done:
            break
    return episode_return
