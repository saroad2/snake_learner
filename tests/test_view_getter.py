import numpy as np

from snake_learner.board import SnakeBoard
from snake_learner.view_getter import ViewGetter


def test_default_view():
    board = SnakeBoard(rows=8, columns=8)
    board.food = [0, 0]
    view_getter = ViewGetter(sight_distance=2)

    assert view_getter.get_view(board) == (
        "     "
        "  X  "
        "  X  "
        "     "
        "     "
    )


def test_default_view_with_food():
    board = SnakeBoard(rows=8, columns=8)
    board.food = board.head + np.array([1, 2])
    view_getter = ViewGetter(sight_distance=2)

    assert view_getter.get_view(board) == (
        "     "
        "  X  "
        "  X  "
        "    F"
        "     "
    )


def test_view_in_front_of_wall():
    board = SnakeBoard(rows=8, columns=8)
    board.snake = [
        [4, 5],
        [4, 6],
        [5, 6],
    ]
    board.food = board.head + np.array([2, 1])
    view_getter = ViewGetter(sight_distance=2)

    assert view_getter.get_view(board) == (
        "    X"
        " XX X"
        "  X X"
        "    X"
        "   FX"
    )

def test_view_in_corner():
    board = SnakeBoard(rows=8, columns=8)
    board.snake = [
        [1, 5],
        [0, 5],
        [0, 6],
        [0, 7],
    ]
    board.food = [7, 0]
    view_getter = ViewGetter(sight_distance=2)

    assert view_getter.get_view(board) == (
        "XXXXX"
        "XXXXX"
        "XXXXX"
        "X  XX"
        "   XX"
    )
