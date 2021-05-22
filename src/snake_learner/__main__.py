import json
import shutil
from pathlib import Path

import click

from snake_learner.learner import SnakeLearner
from snake_learner.plot_util import plot_field_history, plot_int_field_histogram, \
    plot_float_field_histogram, plot_recent_mean_field_history, plot_max_field_history, \
    plot_values_histogram
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
@click.option(
    "-p", "--plot-window",
    type=int,
    default=1000,
    help="Window size for mean calculation in plots",
)
@click.option(
    "--animate/--no-animate",
    is_flag=True,
    default=True,
    help="Should animate best game"
)
@click.option(
    "--animation-output-type",
    type=click.Choice(["gif", "mp4"], case_sensitive=False),
    default="gif"
)
@click.option(
    "--min-state-strength",
    type=float,
    help="Minimum strength for state to be saved"
)
def train_snake(
    output_dir,
    q_file,
    configuration_file,
    iterations,
    plot_window,
    animate,
    animation_output_type,
    min_state_strength,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(configuration_file, mode="r") as fd:
        configuration = json.load(fd)
    view_getter = DistancesViewGetter(
        sight_distance=configuration.pop("sight_distance", None)
    )
    learner = SnakeLearner(view_getter=view_getter, **configuration)
    if q_file is not None:
        learner.load_q_from_file(q_file)
    click.echo("Start learning...")
    with click.progressbar(length=iterations, show_pos=True) as bar:
        try:
            for _ in bar:
                learner.run_train_iteration()
                score_mean = learner.recent_scores_mean(plot_window)
                rewards_mean = learner.recent_rewards_mean(plot_window)
                duration_mean = learner.recent_duration_mean(plot_window)
                bar.label = (
                    f"Recent scores mean - {score_mean :.2f}, "
                    f"Recent rewards mean - {rewards_mean :.2f}, "
                    f"Recent durations mean - {duration_mean :.2f}, "
                    f"States - {learner.states_number}"
                )
        except KeyboardInterrupt:
            click.echo()
            if not click.confirm(
                "Training interrupted. Would you like to save the results?",
                default=False
            ):
                return
    if min_state_strength is not None:
        learner.clear_states_by_strength(min_state_strength)
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="rewards",
    )
    plot_recent_mean_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="rewards",
        n=plot_window,
    )
    plot_max_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="rewards",
    )
    plot_float_field_histogram(
        history=learner.history,
        output_dir=output_dir,
        field="rewards",
        bins=50,
    )
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="score",
    )
    plot_recent_mean_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="score",
        n=plot_window,
    )
    plot_max_field_history(
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
    plot_recent_mean_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="duration",
        n=plot_window,
    )
    plot_max_field_history(
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
        field="velocity",
    )
    plot_recent_mean_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="velocity",
        n=plot_window,
    )
    plot_max_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="velocity",
    )
    plot_float_field_histogram(
        history=learner.history,
        output_dir=output_dir,
        field="velocity",
        bins=50,
    )
    plot_field_history(
        history=learner.history,
        output_dir=output_dir,
        field="states",
    )
    plot_values_histogram(
        values=learner.state_strengths,
        title="Strength histogram",
        xlabel="Strength",
        bins=50,
        output_path=output_dir / "q_strength_histogram.png"
    )
    if animate:
        SnakeAnimation(history=learner.best_game.history).save(
            Path(output_dir) / f"snake_game.{animation_output_type}"
        )
    shutil.copyfile(configuration_file, output_dir / "configuration.json")
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
@click.option(
    "-o", "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory"
)
@click.option("--rows", type=int, help="Override number of rows.")
@click.option("--columns", type=int, help="Override number of columns.")
@click.option("-b", "--best-of", type=int, help="Show best of n games")
@click.option(
    "--output-type",
    type=click.Choice(["gif", "mp4"], case_sensitive=False),
    default="gif"
)
@click.option("-e", "--epsilon", type=float, help="Override epsilon value.")
def play_snake(
    q_file,
    configuration_file,
    output_dir,
    rows,
    columns,
    best_of,
    epsilon,
    output_type
):
    with open(configuration_file, mode="r") as fd:
        configuration = json.load(fd)
    view_getter = DistancesViewGetter(
        sight_distance=configuration.pop("sight_distance", None)
    )
    extra_config = dict(rows=rows, columns=columns, epsilon=epsilon)
    extra_config = {
        key: value for key, value in extra_config.items() if value is not None
    }
    configuration.update(extra_config)
    learner = SnakeLearner(view_getter=view_getter, **configuration)
    learner.load_q_from_file(q_file)
    if best_of is None:
        board = learner.play()
    else:
        click.echo(f"Look for best game of {best_of} games")
        boards = [learner.play() for _ in range(best_of)]
        board = max(boards, key=lambda b: b.score)
        click.echo(f"Found game with score {board.score}")
    animation = SnakeAnimation(history=board.history)
    if output_dir is None:
        animation.play()
    else:
        animation.save(Path(output_dir) / f"snake_game.{output_type}")


if __name__ == '__main__':
    snake()
