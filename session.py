import os
import numpy as np
from collections import deque
from gym import wrappers
import helper
import torch


def train(agent, config):
    """run max_number_of_episodes learning epochs"""
    scores_deque = deque(maxlen=100)
    scores = []
    for epoch in range(1, config["max_number_of_episodes"] + 1):
        reward = agent.learn()
        scores.append(reward)
        scores_deque.append(reward)
        if np.mean(scores_deque) >= config["average_score_for_solving"] and len(scores_deque) >= 100:
            print("\nEnvironment solved after episode: {}".format(epoch - 100))
            print("\nMean Reward over 100 episodes: {}".format(np.mean(scores_deque)))
            break
        print("Episode: {} - Mean Reward: {}".format(epoch, np.mean(scores_deque)))
    return scores


def _set_up_monitoring(env, config):
    """wrap the environment to allow rendering and set up a save directory"""
    helper.mkdir(os.path.join(".",
                              *config["monitor_dir"],
                              config["env_name"]))
    current_date_time = helper.get_current_date_time()
    current_date_time = current_date_time.replace(" ", "__").replace("/", "_").replace(":", "_")

    monitor_path = os.path.join(".", *config["monitor_dir"], config["env_name"], current_date_time)
    env = wrappers.Monitor(env, monitor_path)
    return env


def evaluate(agent, env, config, num_test_runs=3):
    """run objective function num_test_runs times and monitor performance"""
    env = _set_up_monitoring(env, config)
    for eval_step in range(num_test_runs):
        episode_return = 0.0
        state = env.reset()
        while True:
            state = torch.from_numpy(state).float()
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            episode_return += reward
            if done:
                break
        print("Evaluation Episode: {} - Reward: {}".format(eval_step, episode_return))
