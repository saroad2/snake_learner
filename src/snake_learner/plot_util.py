import numpy as np
from matplotlib import pyplot as plt


def plot_field_history(history, output_dir, field):
    values = [history_point[field] for history_point in history]
    x = np.arange(len(values))
    fig, ax = plt.subplots()

    ax.plot(x, values)
    field_title = field.replace("_", " ").title()
    ax.set_title(f"{field_title} History")
    ax.set_xlabel("Time")
    ax.set_ylabel(field_title)
    fig.savefig(output_dir / f"{field}_history.png")


def plot_int_field_histogram(history, output_dir, field):
    values = [history_point[field] for history_point in history]
    mean_value, std_value = np.mean(values), np.std(values)
    unique, counts = np.unique(values, return_counts=True)
    fig, ax = plt.subplots()

    ax.bar(unique, counts)
    field_title = field.replace("_", " ").title()
    ax.set_title(
        fr"{field_title} Histogram (Mean=${mean_value:.3f}\pm{std_value:.3f}$)"
    )
    ax.set_xlabel(field_title)
    ax.set_ylabel("Count")
    fig.savefig(output_dir / f"{field}_histogram.png")


def plot_float_field_histogram(history, output_dir, field, bins):
    values = [history_point[field] for history_point in history]
    fig, ax = plt.subplots()

    ax.hist(values, bins=bins)
    field_title = field.replace("_", " ").title()
    ax.set_title(f"{field_title} Histogram")
    ax.set_xlabel(field_title)
    ax.set_ylabel("Count")
    fig.savefig(output_dir / f"{field}_histogram.png")
