import numpy as np
from unittest import mock

from snake_learner.board import SnakeBoard
from snake_learner.direction import Direction


def test_board_constructor():
    board = SnakeBoard(rows=8, columns=8)

    np.testing.assert_array_equal(
        board.snake,
        np.array(
            [
                [3, 4],
                [4, 4],
            ]
        )
    )
    np.testing.assert_array_equal(board.head, np.array([4, 4]))
    assert not board.done
    assert board.score == 2


def test_move_down():
    board = SnakeBoard(rows=8, columns=8)
    board.snake = [
        [3, 4],
        [4, 4],
        [5, 4],
        [5, 5],
        [5, 6]
    ]
    board.food = [1, 1]
    board.move(Direction.DOWN)

    assert not board.done
    np.testing.assert_array_equal(
        board.snake,
        [
            [4, 4],
            [5, 4],
            [5, 5],
            [5, 6],
            [6, 6],
        ]
    )
    assert board.score == 5


def test_move_up():
    board = SnakeBoard(rows=8, columns=8)
    board.snake = [
        [3, 4],
        [4, 4],
        [4, 6],
        [4, 7]
    ]
    board.food = [1, 1]
    board.move(Direction.UP)

    assert not board.done
    np.testing.assert_array_equal(
        board.snake,
        [
            [4, 4],
            [4, 6],
            [4, 7],
            [3, 7],
        ]
    )
    assert board.score == 4


def test_move_left():
    board = SnakeBoard(rows=8, columns=8)
    board.snake = [
        [3, 4],
        [2, 4],
        [2, 5],
        [2, 6],
        [1, 6]
    ]
    board.food = [1, 1]
    board.move(Direction.LEFT)

    assert not board.done
    np.testing.assert_array_equal(
        board.snake,
        [
            [2, 4],
            [2, 5],
            [2, 6],
            [1, 6],
            [1, 5],
        ]
    )
    assert board.score == 5


def test_move_right():
    board = SnakeBoard(rows=8, columns=8)
    board.snake = [
        [3, 4],
        [2, 4],
        [2, 5],
        [3, 5]
    ]
    board.food = [1, 1]
    board.move(Direction.RIGHT)

    assert not board.done
    np.testing.assert_array_equal(
        board.snake,
        [
            [2, 4],
            [2, 5],
            [3, 5],
            [3, 6],
        ]
    )
    assert board.score == 4


def test_move_to_food():
    board = SnakeBoard(rows=8, columns=8)
    board.snake = [
        [3, 4],
        [4, 4],
        [5, 4],
        [5, 5],
    ]
    board.food = [5, 6]
    assert board.score == 4

    board.move(Direction.RIGHT)

    assert not board.done
    np.testing.assert_array_equal(
        board.snake,
        [
            [3, 4],
            [4, 4],
            [5, 4],
            [5, 5],
            [5, 6],
        ]
    )
    assert board.score == 5
    assert np.any(np.not_equal(board.food, [5, 6]))


def test_snake_eat_itself():
    board = SnakeBoard(rows=8, columns=8)
    board.snake = [
        [3, 4],
        [4, 4],
        [5, 4],
        [6, 4],
        [6, 5],
        [5, 5],
    ]
    board.food = [1, 1]
    assert board.score == 6

    board.move(Direction.LEFT)

    assert board.done
    assert board.score == 6


def test_move_too_down():
    board = SnakeBoard(rows=8, columns=8)
    board.snake = [
        [3, 4],
        [4, 4],
        [5, 4],
        [6, 4],
        [7, 4],
    ]
    board.food = [1, 1]
    assert board.score == 5
    assert not board.done

    board.move(Direction.DOWN)

    assert board.done
    assert board.score == 5


def test_move_too_up():
    board = SnakeBoard(rows=8, columns=8)
    board.snake = [
        [4, 4],
        [3, 4],
        [2, 4],
        [1, 4],
        [0, 4],
    ]
    board.food = [1, 1]
    assert board.score == 5
    assert not board.done

    board.move(Direction.UP)

    assert board.done
    assert board.score == 5


def test_move_too_right():
    board = SnakeBoard(rows=8, columns=8)
    board.snake = [
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],

    ]
    board.food = [1, 1]
    assert board.score == 4
    assert not board.done

    board.move(Direction.RIGHT)

    assert board.done
    assert board.score == 4


def test_move_too_left():
    board = SnakeBoard(rows=8, columns=8)
    board.snake = [
        [4, 5],
        [4, 4],
        [4, 3],
        [4, 2],
        [4, 1],
        [4, 0],
    ]
    board.food = [1, 1]
    assert board.score == 6
    assert not board.done

    board.move(Direction.LEFT)

    assert board.done
    assert board.score == 6


def test_random_food_in_snake():
    with mock.patch("numpy.random.randint") as randint:
        randint.side_effect = [3, 4, 5, 6]
        board = SnakeBoard(rows=8, columns=8)
        np.testing.assert_array_equal(board.food, [5, 6])
        assert randint.call_count == 4
