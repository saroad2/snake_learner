import json
from pathlib import Path

import click
from snake_learner.learner import SnakeLearner
from snake_learner.plot_util import plot_field_history


@click.command()
@click.argument("output-dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "--iterations", type=int, default=1_000, help="How many iterations to run."
)
@click.option("--plot-window", type=int, help="Windowing for plotting.")
def learn_snake(output_dir, iterations, plot_window):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    learner = SnakeLearner(rows=8, columns=8, sight_distance=3)
    click.echo("Start learning...")
    with click.progressbar(length=iterations, show_pos=True) as bar:
        for _ in bar:
            learner.run_iteration()
            bar.label = f"Max score - {learner.max_score}"
    click.echo(f"Max score = {learner.max_score}")
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
