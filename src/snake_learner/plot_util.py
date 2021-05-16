import numpy as np
from matplotlib import pyplot as plt


def plot_rewards_history(history, output_path):
    rewards = [history_point["rewards"] for history_point in history]
    x = np.arange(len(rewards))
    fig, ax = plt.subplots()

    ax.plot(x, rewards)
    fig.savefig(output_path)
