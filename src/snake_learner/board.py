import numpy as np
from collections import deque

from snake_learner.direction import Direction


class SnakeBoard:

    def __init__(self, rows, columns):
        self.shape = (rows, columns)
        self.snake = [
            np.array([rows // 2 - 2, columns // 2]),
            np.array([rows // 2 - 1, columns // 2]),
            np.array([rows // 2, columns // 2]),
        ]
        self.done = False
        self.food = None
        self.put_random_food()

    @property
    def snake(self):
        return self._snake

    @snake.setter
    def snake(self, snake):
        self._snake = deque([np.array(cell) for cell in snake])

    @property
    def head(self):
        return self.snake[-1]

    @property
    def score(self):
        return len(self.snake)

    def move(self, direction: Direction):
        new_head = self.head + direction.to_array()
        self.snake.append(new_head)
        if np.array_equal(new_head, self.food):
            self.put_random_food()
        else:
            self.snake.popleft()
        if not self.is_valid_location(location=self.head, include_head=False):
            self.done = True
            return

    def is_valid_location(self, location, include_head=True):
        if location[0] < 0 or location[0] >= self.shape[0]:
            return False
        if location[1] < 0 or location[1] >= self.shape[1]:
            return False
        if self.location_in_snake(location, include_head=include_head):
            return False
        return True

    def put_random_food(self):
        self.food = (
            np.random.randint(self.shape[0]),
            np.random.randint(self.shape[1]),
        )
        while self.location_in_snake(self.food):
            self.food = (
                np.random.randint(self.shape[0]),
                np.random.randint(self.shape[1]),
            )

    def location_in_snake(self, location, include_head=True):
        snake_cells = list(self.snake)
        if not include_head:
            snake_cells = snake_cells[:-1]
        for snake_cell in snake_cells:
            if np.array_equal(snake_cell, location):
                return True
        return False
