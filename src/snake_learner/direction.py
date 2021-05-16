from enum import Enum, auto
import numpy as np


TO_ARRAY = {
    "UP": np.array([-1, 0]),
    "DOWN": np.array([1, 0]),
    "RIGHT": np.array([0, 1]),
    "LEFT": np.array([0, -1]),
}


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    RIGHT = auto()
    LEFT = auto()

    def to_array(self):
        return TO_ARRAY[self.name]
