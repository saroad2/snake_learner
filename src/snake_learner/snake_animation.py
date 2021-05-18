from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from snake_learner.board import SnakeBoard
from snake_learner.learner import SnakeLearner

FOOD_BASE_SIZE = 7
CREATURE_BASE_SIZE = 10


class SnakeAnimation:

    def __init__(self, learner: SnakeLearner, max_games=1):
        self.learner = learner
        self.snake_board = SnakeBoard(rows=learner.rows, columns=learner.columns)
        self.max_games = max_games
        self.game = 1
        self.fig, self.ax = plt.subplots()
        self.anim = FuncAnimation(self.fig, self.update, interval=150)

    def save(self, output):
        self.anim.save(output)

    def play(self):
        plt.show()

    def update(self, i):
        if self.snake_board.done:
            if self.game >= self.max_games:
                return
            self.game += 1
            self.snake_board.restart()
        self.learner.make_move(self.snake_board, update_q=False)

        self.ax.cla()

        self.ax.set_xlim(-2, self.snake_board.shape[1] + 2)
        self.ax.set_ylim(-2, self.snake_board.shape[0] + 2)
        self.ax.set_title(
            f"Game {self.game}, "
            f"Move {self.snake_board.moves}, "
            f"Score {self.snake_board.score}"
        )

        self.snake_scatter()
        self.food_scatter()

    def food_scatter(self):
        self.ax.scatter(
            [self.snake_board.food[1]], [self.snake_board.food[0]], c="orange"
        )

    def snake_scatter(self):
        snake_cells_without_head = list(self.snake_board.snake)[:-1]
        snake_cell_x = [cell[1] for cell in snake_cells_without_head]
        snake_cell_y = [cell[0] for cell in snake_cells_without_head]
        self.ax.scatter(snake_cell_x, snake_cell_y, c="green")
        self.ax.scatter(
            [self.snake_board.head[1]], [self.snake_board.head[0]], c="blue"
        )
