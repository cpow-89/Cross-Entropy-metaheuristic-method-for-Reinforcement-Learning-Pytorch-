import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def mkdir(path):
    """if needed create a folder at given path"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_current_date_time():
    """get current datetime as string in the form of %Y/%m/%d %H:%M:%S"""
    return datetime.now().strftime('%Y/%m/%d %H:%M:%S')


def plot_scores(scores):
    """plot the current list of scores"""
    fig = plt.figure()
    _ = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()
