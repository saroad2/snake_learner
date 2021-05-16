import numpy as np
from matplotlib import pyplot as plt


def plot_field_history(history, output_dir, field):
    rewards = [history_point[field] for history_point in history]
    rewards = moving_max(rewards)
    x = np.arange(len(rewards))
    fig, ax = plt.subplots()

    ax.plot(x, rewards)
    field_title = field.replace("_", " ").title()
    ax.set_title(f"{field_title} History")
    ax.set_xlabel("Time")
    ax.set_ylabel(field_title)
    fig.savefig(output_dir / f"{field}_history.png")


def moving_max(values):
    new_values = []
    max_val = None
    for val in values:
        if max_val is None or val > max_val:
            max_val = val
        new_values.append(max_val)
    return new_values
