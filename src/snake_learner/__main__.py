import json
from pathlib import Path

import click
from snake_learner.learner import SnakeLearner
from snake_learner.plot_util import plot_field_history, plot_int_field_histogram, \
    plot_float_field_histogram
from snake_learner.view_getter import DistancesViewGetter


@click.command()
@click.argument("output-dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "-q", "--q-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Existing Q file to update",
)
@click.option("--rows", type=int, default=8, help="How many rows in board")
@click.option("--columns", type=int, default=8, help="How many columns in board")
@click.option(
    "--iterations", type=int, default=1_000, help="How many iterations to run."
)
@click.option("--loss-penalty", type=int, default=0, help="Penalty for losing.")
@click.option("--eat-reward", type=int, default=1, help="Base reward for eating")
@click.option(
    "--reward-change", type=float, default=0.15, help="Reward change after n foods."
)
@click.option(
    "--epsilon", type=float, default=0.1, help="Policy epsilon"
)
@click.option(
    "--sight-distance", type=int, help="How far can the snake see"
)
def learn_snake(
    output_dir,
    q_file,
    rows,
    columns,
    iterations,
    loss_penalty,
    eat_reward,
    reward_change,
    epsilon,
    sight_distance,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    view_getter = DistancesViewGetter(sight_distance=sight_distance)
    learner = SnakeLearner(
        rows=rows,
        columns=columns,
        view_getter=view_getter,
        loss_penalty=loss_penalty,
        eat_reward=eat_reward,
        reward_change=reward_change,
        epsilon=epsilon,
    )
    if q_file is not None:
        learner.load_q_from_file(q_file)
    click.echo("Start learning...")
    with click.progressbar(length=iterations, show_pos=True) as bar:
        try:
            for _ in bar:
                learner.run_iteration()
                bar.label = (
                    f"Max score - {learner.max_score}, "
                    f"Max rewards sum - {learner.max_rewards_sum:.4f}, "
                    f"Longest duration - {learner.longest_duration}"
                )
        except KeyboardInterrupt:
            click.echo()
            if not click.confirm(
                "Training interrupted. Would you like to save the results?",
                default=False
            ):
                return
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="rewards_sum",
    )
    plot_float_field_histogram(
        history=learner.history,
        output_dir=output_dir,
        field="rewards_sum",
        bins=50,
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
    plot_int_field_histogram(
        history=learner.history,
        output_dir=output_dir,
        field="score",
    )
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="duration",
    )
    plot_int_field_histogram(
        history=learner.history,
        output_dir=output_dir,
        field="duration",
    )
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="states",
        max_val=False,
    )
    with open(output_dir / "q_values.json", mode="w") as fd:
        q_as_dict = {key: val.tolist() for key, val in learner.q.items()}
        json.dump(q_as_dict, fd, indent=1)


if __name__ == '__main__':
    learn_snake()
