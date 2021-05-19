import numpy as np

from snake_learner.direction import Direction


def block_distance(sight_vector):
    return int(np.sum(np.fabs(sight_vector)))


def closest_direction(sight_vector):
    multiplications = [
        cosine_multiplication(sight_vector, direction.to_array())
        for direction in Direction
    ]
    return Direction(np.argmax(multiplications))


def cosine_multiplication(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
