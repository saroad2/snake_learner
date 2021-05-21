import numpy as np
from pytest_cases import parametrize_with_cases, case

from snake_learner.direction import Direction
from snake_learner.linalg_util import block_distance, closest_direction, \
    project_to_direction

CLOSEST_DIRECTION = "closest_direction"
PROJECT_TO_DIRECTION = "project_to_direction"


@case(tags=[CLOSEST_DIRECTION])
def case_closest_direction_up_direction():
    vec = Direction.UP.to_array()
    distance = 1
    direction = Direction.UP
    return vec, distance, direction


@case(tags=[CLOSEST_DIRECTION])
def case_closest_direction_down_direction():
    vec = Direction.DOWN.to_array()
    distance = 1
    direction = Direction.DOWN
    return vec, distance, direction


@case(tags=[CLOSEST_DIRECTION])
def case_closest_direction_left_direction():
    vec = Direction.LEFT.to_array()
    distance = 1
    direction = Direction.LEFT
    return vec, distance, direction


@case(tags=[CLOSEST_DIRECTION])
def case_closest_direction_right_direction():
    vec = Direction.RIGHT.to_array()
    distance = 1
    direction = Direction.RIGHT
    return vec, distance, direction


@case(tags=[CLOSEST_DIRECTION])
def case_closest_direction_long_up_direction():
    n = 8
    vec = n * Direction.UP.to_array()
    distance = n
    direction = Direction.UP
    return vec, distance, direction


@case(tags=[CLOSEST_DIRECTION])
def case_closest_direction_long_down_direction():
    n = 8
    vec = n * Direction.DOWN.to_array()
    distance = n
    direction = Direction.DOWN
    return vec, distance, direction


@case(tags=[CLOSEST_DIRECTION])
def case_closest_direction_long_right_direction():
    n = 8
    vec = n * Direction.RIGHT.to_array()
    distance = n
    direction = Direction.RIGHT
    return vec, distance, direction


@case(tags=[CLOSEST_DIRECTION])
def case_closest_direction_long_left_direction():
    n = 8
    vec = n * Direction.LEFT.to_array()
    distance = n
    direction = Direction.LEFT
    return vec, distance, direction


@case(tags=[CLOSEST_DIRECTION])
def case_closest_direction_up_right_direction():
    n, m = 3, 2
    vec = n * Direction.UP.to_array() + m * Direction.RIGHT.to_array()
    distance = n + m
    direction = Direction.UP
    return vec, distance, direction


@case(tags=[CLOSEST_DIRECTION])
def case_closest_direction_right_up_direction():
    n, m = 3, 2
    vec = m * Direction.UP.to_array() + n * Direction.RIGHT.to_array()
    distance = n + m
    direction = Direction.RIGHT
    return vec, distance, direction


@case(tags=[PROJECT_TO_DIRECTION])
def case_project_to_direction_up():
    vec = np.array([5, 2])
    direction = Direction.UP
    result = np.array([5, 2])
    return vec, direction, result


@case(tags=[PROJECT_TO_DIRECTION])
def case_project_to_direction_right():
    vec = np.array([5, 2])
    direction = Direction.RIGHT
    result = np.array([2, -5])
    return vec, direction, result


@case(tags=[PROJECT_TO_DIRECTION])
def case_project_to_direction_down():
    vec = np.array([5, 2])
    direction = Direction.DOWN
    result = np.array([-5, -2])
    return vec, direction, result


@case(tags=[PROJECT_TO_DIRECTION])
def case_project_to_direction_left():
    vec = np.array([5, 2])
    direction = Direction.LEFT
    result = np.array([-2, 5])
    return vec, direction, result


@parametrize_with_cases(
    argnames=["vec", "distance", "direction"], cases=".", has_tag=CLOSEST_DIRECTION
)
def test_closest_direction(vec, distance, direction):
    assert block_distance(vec) == distance
    assert closest_direction(vec) == direction



@parametrize_with_cases(
    argnames=["vec", "direction", "result"], cases=".", has_tag=PROJECT_TO_DIRECTION
)
def test_project_to_direction(vec, direction, result):
    np.testing.assert_array_equal(
        project_to_direction(sight_vector=vec, direction=direction), result
    )