from enum import Enum

from snake_learner.direction import Direction


ACTION_TO_INDEX_DELTA = {
    "TURN_LEFT": -1,
    "TURN_RIGHT": 1,
    "FORWARD": 0,
}


class SnakeAction(Enum):

    TURN_LEFT = 0
    FORWARD = 1
    TURN_RIGHT = 2

    def turn(self, direction: Direction):
        direction_index = (
            direction.value + ACTION_TO_INDEX_DELTA[self.name]
        ) % len(Direction)
        return Direction(direction_index)
