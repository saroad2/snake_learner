from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class HistoryPoint:

    shape: np.ndarray
    moves: int
    snake: List[np.ndarray]
    food: np.ndarray

    @property
    def score(self):
        return len(self.snake)

    @property
    def head(self):
        return self.snake[-1]
