import numpy as np

from snake_learner.board import SnakeBoard
from snake_learner.direction import Direction
from snake_learner.linalg_util import closest_direction, project_to_direction
from snake_learner.snake_action import SnakeAction


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

    def __init__(self, sight_distance=None):
        self.sight_distance = sight_distance

    def get_view(self, board: SnakeBoard):
        distances = [
            self.get_distance(board, direction)
            for direction in self.get_cross_directions(board.direction)
        ]
        food_vector = board.food - board.head
        food_vector = project_to_direction(
            sight_vector=food_vector, direction=board.direction
        )
        return (
            f"{'_'.join([str(dist) for dist in distances])}:"
            f"{self.normalize_coordinate(food_vector[0])}_"
            f"{self.normalize_coordinate(food_vector[1])}"
        )

    @classmethod
    def get_cross_directions(cls, direction: Direction):
        return [
            direction.to_array(),
            direction.to_array() + SnakeAction.TURN_RIGHT.turn(direction).to_array(),
            SnakeAction.TURN_RIGHT.turn(direction).to_array(),
            -direction.to_array() + SnakeAction.TURN_RIGHT.turn(direction).to_array(),
            -direction.to_array() + SnakeAction.TURN_LEFT.turn(direction).to_array(),
            SnakeAction.TURN_LEFT.turn(direction).to_array(),
            direction.to_array() + SnakeAction.TURN_LEFT.turn(direction).to_array(),
        ]

    def get_distance(self, board, direction):
        i = 0
        while True:
            loc = board.head + (i + 1) * direction
            if not board.is_valid_location(loc):
                break
            i += 1
        return self.normalize_coordinate(i)

    def normalize_coordinate(self, coord):
        if self.sight_distance is None:
            return coord
        if coord > self.sight_distance:
            return self.sight_distance
        if coord < -self.sight_distance:
            return -self.sight_distance
        return coord
