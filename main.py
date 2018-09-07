import json
import os
import gym
import argparse
import session
import helper
import numpy as np
from agent import CrossEntropyAgent


def main():

    parser = argparse.ArgumentParser(description="Run Extended Q-Learning with given config")
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        metavar="",
                        required=True,
                        help="Config file name - file must be available as .json in ./configs")

    args = parser.parse_args()

    with open(os.path.join(".", "configs", args.config), "r") as read_file:
        config = json.load(read_file)

    np.random.seed(101)
    agent = CrossEntropyAgent(config)  
    
    if config["is_training"]:
        total_reward = session.train(agent, config)
        agent.save()
        helper.plot_scores(total_reward)
    else:
        env = gym.make(config["env_name"])
        agent.load()
        session.evaluate(agent, env, config)
        env.env.close()
        env.close()


if __name__ == "__main__":
    main()
