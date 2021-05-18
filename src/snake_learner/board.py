import numpy as np
from collections import deque

from snake_learner.direction import Direction


class SnakeBoard:

    def __init__(self, rows, columns, initial_size=3):
        self.shape = (rows, columns)
        self.initial_size = initial_size
        self.snake = []
        self.moves = 0
        self.food = None
        self.initialize_snake()
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

    @property
    def done(self):
        return not self.is_valid_location(self.head, include_head=False)

    def restart(self):
        self.snake = []
        self.moves = 0
        self.food = None
        self.initialize_snake()
        self.put_random_food()

    def move(self, direction: Direction):
        self.moves += 1
        new_head = self.head + direction.to_array()
        self.snake.append(new_head)
        if np.array_equal(new_head, self.food):
            self.put_random_food()
        else:
            self.snake.popleft()

    def is_valid_location(self, location, include_head=True):
        if location[0] < 0 or location[0] >= self.shape[0]:
            return False
        if location[1] < 0 or location[1] >= self.shape[1]:
            return False
        if self.location_in_snake(location, include_head=include_head):
            return False
        return True

    def initialize_snake(self):
        self.snake.append(self.random_location())
        for _ in range(1, self.initial_size):
            new_head = self.head + np.random.choice(Direction).to_array()
            while not self.is_valid_location(new_head):
                new_head = self.head + np.random.choice(Direction).to_array()
            self.snake.append(new_head)

    def put_random_food(self):
        self.food = self.random_location()
        while self.location_in_snake(self.food):
            self.food = self.random_location()

    def random_location(self):
        return np.array(
            [
                np.random.randint(self.shape[0]),
                np.random.randint(self.shape[1]),
            ]
        )

    def location_in_snake(self, location, include_head=True):
        snake_cells = list(self.snake)
        if not include_head:
            snake_cells = snake_cells[:-1]
        for snake_cell in snake_cells:
            if np.array_equal(snake_cell, location):
                return True
        return False
