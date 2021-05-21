import numpy as np

from snake_learner.stat_util import moving_max, moving_mean


def test_moving_max():
    values = np.array([1, 5, 2, 3, 7, 2, 10])
    np.testing.assert_array_equal(
        np.array([1, 5, 5, 5, 7, 7, 10]),
        moving_max(values),
    )


def test_moving_mean():
    values = np.array([1, 5, 2, 3, 7, 2, 10])
    n = 2
    np.testing.assert_array_equal(
        np.array([3, 3.5, 2.5, 5, 4.5, 6]),
        moving_mean(values, n=n),
    )
