from pytest_cases import parametrize_with_cases

from snake_learner.direction import Direction
from snake_learner.linalg_util import block_distance, closest_direction


def case_up_direction():
    vec = Direction.UP.to_array()
    distance = 1
    direction = Direction.UP
    return vec, distance, direction


def case_down_direction():
    vec = Direction.DOWN.to_array()
    distance = 1
    direction = Direction.DOWN
    return vec, distance, direction


def case_left_direction():
    vec = Direction.LEFT.to_array()
    distance = 1
    direction = Direction.LEFT
    return vec, distance, direction


def case_right_direction():
    vec = Direction.RIGHT.to_array()
    distance = 1
    direction = Direction.RIGHT
    return vec, distance, direction


def case_long_up_direction():
    n = 8
    vec = n * Direction.UP.to_array()
    distance = n
    direction = Direction.UP
    return vec, distance, direction


def case_long_down_direction():
    n = 8
    vec = n * Direction.DOWN.to_array()
    distance = n
    direction = Direction.DOWN
    return vec, distance, direction


def case_long_right_direction():
    n = 8
    vec = n * Direction.RIGHT.to_array()
    distance = n
    direction = Direction.RIGHT
    return vec, distance, direction


def case_long_left_direction():
    n = 8
    vec = n * Direction.LEFT.to_array()
    distance = n
    direction = Direction.LEFT
    return vec, distance, direction


def case_up_right_direction():
    n, m = 3, 2
    vec = n * Direction.UP.to_array() + m * Direction.RIGHT.to_array()
    distance = n + m
    direction = Direction.UP
    return vec, distance, direction


def case_right_up_direction():
    n, m = 3, 2
    vec = m * Direction.UP.to_array() + n * Direction.RIGHT.to_array()
    distance = n + m
    direction = Direction.RIGHT
    return vec, distance, direction


@parametrize_with_cases(argnames=["vec", "distance", "direction"], cases=".")
def test_linalg_utils(vec, distance, direction):
    assert block_distance(vec) == distance
    assert closest_direction(vec) == direction
