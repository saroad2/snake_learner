from pytest_cases import parametrize_with_cases

from snake_learner.direction import Direction
from snake_learner.snake_action import SnakeAction


def case_snake_action_up():
    direction = Direction.UP
    right_result = Direction.RIGHT
    left_result = Direction.LEFT

    return direction, right_result, left_result


def case_snake_action_right():
    direction = Direction.RIGHT
    right_result = Direction.DOWN
    left_result = Direction.UP

    return direction, right_result, left_result


def case_snake_action_down():
    direction = Direction.DOWN
    right_result = Direction.LEFT
    left_result = Direction.RIGHT

    return direction, right_result, left_result


def case_snake_action_left():
    direction = Direction.LEFT
    right_result = Direction.UP
    left_result = Direction.DOWN

    return direction, right_result, left_result


@parametrize_with_cases(
    argnames=["direction", "right_result", "left_result"], cases="."
)
def test_snake_action(direction, right_result, left_result):
    assert right_result == SnakeAction.TURN_RIGHT.turn(direction)
    assert left_result == SnakeAction.TURN_LEFT.turn(direction)
    assert direction == SnakeAction.FORWARD.turn(direction)
