from snake_learner.direction import Direction


def test_all_directions():
    assert len(Direction) == 4
    assert list(Direction) == [
        Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT
    ]


def test_up_index():
    assert Direction(0) == Direction.UP


def test_right_index():
    assert Direction(1) == Direction.RIGHT


def test_down_index():
    assert Direction(2) == Direction.DOWN


def test_left_index():
    assert Direction(3) == Direction.LEFT
