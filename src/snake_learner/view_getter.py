import numpy as np

from snake_learner.board import SnakeBoard
from snake_learner.direction import Direction


class ViewGetter:

    def get_view(self, board: SnakeBoard):
        raise NotImplementedError("ViewGetter.get_view() is not implemented")


class GridViewGetter(ViewGetter):

    EMPTY = " "
    BLOCK = "X"
    FOOD = "F"

    def __init__(self, sight_distance):
        self.sight_distance = sight_distance

    def get_view(self, board: SnakeBoard):
        sight = []
        for i in np.arange(-self.sight_distance, self.sight_distance + 1):
            for j in np.arange(-self.sight_distance, self.sight_distance + 1):
                loc = board.head + np.array([i, j])
                sight.append(self.get_cell_type(board, loc))
        return "".join(sight)

    @classmethod
    def get_cell_type(cls, board, location):
        if np.all(np.equal(location, board.food)):
            return cls.FOOD
        if board.is_valid_location(location):
            return cls.EMPTY
        return cls.BLOCK


class DistancesViewGetter(ViewGetter):
    DIRECTIONS = [
        Direction.UP.to_array(),
        Direction.UP.to_array() + Direction.RIGHT.to_array(),
        Direction.RIGHT.to_array(),
        Direction.DOWN.to_array() + Direction.RIGHT.to_array(),
        Direction.DOWN.to_array(),
        Direction.DOWN.to_array() + Direction.LEFT.to_array(),
        Direction.LEFT.to_array(),
        Direction.UP.to_array() + Direction.LEFT.to_array(),
    ]

    def get_view(self, board: SnakeBoard):
        distances = [
            self.get_distance(board, direction) for direction in self.DIRECTIONS
        ]
        food_vector = board.food - board.head
        food_direction = self.get_closest_direction(food_vector)
        food_distance = int(np.sum(np.fabs(food_vector)))
        return (
            f"{'_'.join([str(dist) for dist in distances])}"
            f":{food_direction}_{food_distance}"
        )

    @classmethod
    def get_closest_direction(cls, vector):
        multiplications = [
            np.dot(vector, direction) / (np.linalg.norm(vector) * np.linalg.norm(direction))
            for direction in cls.DIRECTIONS
        ]
        return np.argmax(multiplications)

    @classmethod
    def get_distance(cls, board, direction):
        i = 0
        while True:
            loc = board.head + (i + 1) * direction
            if not board.is_valid_location(loc):
                return i
            i += 1
