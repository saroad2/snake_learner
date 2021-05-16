from pathlib import Path

import click
from snake_learner.learner import SnakeLearner
from snake_learner.plot_util import plot_rewards_history


@click.command()
@click.argument("output-dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "--iterations", type=int, default=1_000, help="How many iterations to run."
)
def learn_snake(output_dir, iterations):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    learner = SnakeLearner(rows=8, columns=8, sight_distance=3)
    click.echo("Start learning...")
    with click.progressbar(length=iterations, show_pos=True) as bar:
        for _ in bar:
            learner.run_iteration()
            bar.label = f"Max score - {learner.max_score}"
    click.echo(f"Max score = {learner.max_score}")
    plot_rewards_history(
        history=learner.history, output_path=output_dir / "rewards.png"
    )


if __name__ == '__main__':
    learn_snake()
