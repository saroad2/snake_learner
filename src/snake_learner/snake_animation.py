from typing import List

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from snake_learner.history import HistoryPoint

FOOD_BASE_SIZE = 7
CREATURE_BASE_SIZE = 10


class SnakeAnimation:

    def __init__(self, history: List[HistoryPoint]):
        self.history = history
        self.fig, self.ax = plt.subplots()
        self.anim = FuncAnimation(
            self.fig, self.update, frames=len(self.history), interval=200
        )

    def save(self, output):
        self.anim.save(str(output))

    def play(self):
        plt.show()

    def update(self, i):
        if i >= len(self.history):
            return
        history_point = self.history[i]

        self.ax.cla()

        rows, columns = history_point.shape
        self.ax.set_xlim(-1, columns)
        self.ax.set_ylim(-1, rows)
        self.ax.hlines(rows, -1, columns, color="black")
        self.ax.hlines(-1, -1, columns, color="black")
        self.ax.vlines(rows, -1, rows, color="black")
        self.ax.vlines(-1, -1, rows, color="black")
        self.ax.set_title(
            f"Move {history_point.moves}/{len(self.history)}, "
            f"Score {history_point.score}"
        )

        self.snake_scatter(history_point)
        self.food_scatter(history_point)

    def food_scatter(self, history_point):
        self.ax.scatter(
            [history_point.food[1]], [history_point.food[0]], c="orange"
        )

    def snake_scatter(self, history_point: HistoryPoint):
        snake_cells_without_head = list(history_point.snake)[:-1]
        snake_cell_x = [cell[1] for cell in snake_cells_without_head]
        snake_cell_y = [cell[0] for cell in snake_cells_without_head]
        self.ax.scatter(snake_cell_x, snake_cell_y, c="green")
        self.ax.scatter(
            [history_point.head[1]], [history_point.head[0]], c="blue"
        )
