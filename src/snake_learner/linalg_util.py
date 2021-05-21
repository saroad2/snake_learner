import numpy as np

from snake_learner.direction import Direction
from snake_learner.snake_action import SnakeAction


def block_distance(sight_vector):
    return int(np.sum(np.fabs(sight_vector)))


def closest_direction(sight_vector):
    multiplications = [
        cosine_multiplication(sight_vector, direction.to_array())
        for direction in Direction
    ]
    return Direction(np.argmax(multiplications))


def project_to_direction(sight_vector: np.ndarray, direction: Direction):
    a1 = np.dot(sight_vector, direction.to_array())
    a2 = np.dot(sight_vector, SnakeAction.TURN_RIGHT.turn(direction).to_array())
    return np.array([a1, a2])


def cosine_multiplication(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
