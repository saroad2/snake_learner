import numpy as np
from matplotlib import pyplot as plt


def plot_field_history(history, output_dir, field, window=None):
    rewards = [history_point[field] for history_point in history]
    if window is not None:
        rewards = [
            np.mean(rewards[i: i + window]) for i in range(0, len(rewards), window)
        ]
    x = np.arange(len(rewards))
    fig, ax = plt.subplots()

    ax.plot(x, rewards)
    field_title = field.replace("_", " ").title()
    ax.set_title(f"{field_title} History")
    ax.set_xlabel("Time")
    ax.set_ylabel(field_title)
    fig.savefig(output_dir / f"{field}_history.png")
