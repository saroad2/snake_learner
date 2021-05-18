import json
from pathlib import Path

import click

from snake_learner.board import SnakeBoard
from snake_learner.learner import SnakeLearner
from snake_learner.plot_util import plot_field_history, plot_int_field_histogram, \
    plot_float_field_histogram
from snake_learner.snake_animation import SnakeAnimation
from snake_learner.view_getter import DistancesViewGetter


@click.group()
def snake():
    pass


@snake.command("train")
@click.option(
    "-o", "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Output directory"
)
@click.option(
    "-c", "--configuration-file",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Configuration file",
)
@click.option(
    "--iterations", type=int, default=1_000, help="How many iterations to run."
)
@click.option(
    "-q", "--q-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Existing Q file to update",
)
def train_snake(
    output_dir,
    q_file,
    configuration_file,
    iterations,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(configuration_file, mode="r") as fd:
        configuration = json.load(fd)
    view_getter = DistancesViewGetter(
        sight_distance=configuration.pop("sight_distance")
    )
    learner = SnakeLearner(view_getter=view_getter, **configuration)
    if q_file is not None:
        learner.load_q_from_file(q_file)
    click.echo("Start learning...")
    with click.progressbar(length=iterations, show_pos=True) as bar:
        try:
            for _ in bar:
                learner.run_train_iteration()
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


@snake.command("play")
@click.option(
    "-q", "--q-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Existing Q file to update",
)
@click.option(
    "-c", "--configuration-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Configuration file",
    required=True,
)
@click.option("--rows", type=int, help="Override number of rows.")
@click.option("--columns", type=int, help="Override number of columns.")
@click.option(
    "-m", "--max-games", type=int, default=1, help="How many games to play."
)
def play_snake(q_file, configuration_file, rows, columns, max_games):
    with open(configuration_file, mode="r") as fd:
        configuration = json.load(fd)
    view_getter = DistancesViewGetter(
        sight_distance=configuration.pop("sight_distance")
    )
    if rows is not None:
        configuration["rows"] = rows
    if columns is not None:
        configuration["columns"] = columns
    learner = SnakeLearner(view_getter=view_getter, **configuration)
    learner.load_q_from_file(q_file)
    SnakeAnimation(learner=learner, max_games=max_games).play()


if __name__ == '__main__':
    snake()
