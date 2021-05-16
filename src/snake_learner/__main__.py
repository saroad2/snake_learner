import json
from pathlib import Path

import click
from snake_learner.learner import SnakeLearner
from snake_learner.plot_util import plot_field_history


@click.command()
@click.argument("output-dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--rows", type=int, default=8, help="How many rows in board")
@click.option("--columns", type=int, default=8, help="How many columns in board")
@click.option(
    "--iterations", type=int, default=1_000, help="How many iterations to run."
)
@click.option("--plot-window", type=int, help="Windowing for plotting.")
@click.option("--loss-penalty", type=int, default=0, help="Penalty for losing.")
@click.option("--eat-reward", type=int, default=1, help="Base reward for eating")
@click.option(
    "--reward-decay", type=float, default=0.15, help="Decay for moving"
)
def learn_snake(
    output_dir,
    rows,
    columns,
    iterations,
    plot_window,
    loss_penalty,
    eat_reward,
    reward_decay,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    learner = SnakeLearner(
        rows=rows,
        columns=columns,
        sight_distance=3,
        loss_penalty=loss_penalty,
        eat_reward=eat_reward,
        reward_decay=reward_decay,
    )
    click.echo("Start learning...")
    with click.progressbar(length=iterations, show_pos=True) as bar:
        for _ in bar:
            learner.run_iteration()
            bar.label = (
                f"Max score = {learner.max_score}, "
                f"Max rewards sum - {learner.max_rewards_sum:.4f}, "
                f"Longest duration - {learner.longest_duration}"
            )
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="rewards_sum",
        window=plot_window,
    )
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="rewards_max",
        window=plot_window,
    )
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="score",
        window=plot_window,
    )
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="duration",
        window=plot_window,
    )
    with open(output_dir / "q_probabilities.json", mode="w") as fd:
        q_as_dict = {key: val.tolist() for key, val in learner.q.items()}
        json.dump(q_as_dict, fd, indent=1)


if __name__ == '__main__':
    learn_snake()
