import numpy as np
from blink_analysis.categorize import *

def test_smooth():
    data     = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0], dtype='bool')
    expected = np.array([1, 0, 0, 1, 1, 1, 1, 0, 0], dtype='bool')
    np.testing.assert_equal(smooth(data, 2), expected)

def test_smooth_small_peak():
    data     = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0], dtype='bool')
    expected = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0], dtype='bool')
    np.testing.assert_equal(smooth(data, 2), expected)

def test_smooth_passthrough():
    data = np.array([1, 0, 0, 1, 1, 0, 1], dtype='bool')
    np.testing.assert_equal(smooth(data, 1), data)

def test_categorize():
    data = np.zeros((4, 9, 9), dtype='float')
    data[1, 2:-2, 2:-2] = 10
    data[2, :, :] = 1
    data[3, :, :] = 1
    data[3, 2:-2, 2:-2] = 0

    expected = np.array([0, 1, 0, 0], dtype='bool')
    np.testing.assert_equal(categorize(data), expected)

def test_masks():
    expected = (
        np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ], dtype='bool'),
        np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ], dtype='bool'),
    )

    for test, e in zip(masks((5, 5)), expected):
        np.testing.assert_equal(test, e)
