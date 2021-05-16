import json
from pathlib import Path

import click
from snake_learner.learner import SnakeLearner
from snake_learner.plot_util import plot_field_history
from snake_learner.view_getter import DistancesViewGetter


@click.command()
@click.argument("output-dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--rows", type=int, default=8, help="How many rows in board")
@click.option("--columns", type=int, default=8, help="How many columns in board")
@click.option(
    "--iterations", type=int, default=1_000, help="How many iterations to run."
)
@click.option("--loss-penalty", type=int, default=0, help="Penalty for losing.")
@click.option("--eat-reward", type=int, default=1, help="Base reward for eating")
@click.option(
    "--reward-decay", type=float, default=0.15, help="Decay for moving"
)
@click.option(
    "--epsilon", type=float, default=0.1, help="Policy epsilon"
)
def learn_snake(
    output_dir,
    rows,
    columns,
    iterations,
    loss_penalty,
    eat_reward,
    reward_decay,
    epsilon,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    view_getter = DistancesViewGetter()
    learner = SnakeLearner(
        rows=rows,
        columns=columns,
        view_getter=view_getter,
        loss_penalty=loss_penalty,
        eat_reward=eat_reward,
        reward_decay=reward_decay,
        epsilon=epsilon,
    )
    click.echo("Start learning...")
    with click.progressbar(length=iterations, show_pos=True) as bar:
        for _ in bar:
            learner.run_iteration()
            bar.label = (
                f"Max score - {learner.max_score}, "
                f"Max rewards sum - {learner.max_rewards_sum:.4f}, "
                f"Longest duration - {learner.longest_duration}"
            )
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="rewards_sum",
    )
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="rewards_max",
    )
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="score",
    )
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="duration",
    )
    with open(output_dir / "q_probabilities.json", mode="w") as fd:
        q_as_dict = {key: val.tolist() for key, val in learner.q.items()}
        json.dump(q_as_dict, fd, indent=1)


if __name__ == '__main__':
    learn_snake()
