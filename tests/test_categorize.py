import numpy as np
from categorize import *

def test_smooth():
    data = np.array([1, 0, 0, 1, 1, 0, 1], dtype='bool')
    expected = np.array([1, 0, 0, 1, 1, 1, 1], dtype='bool')
    np.testing.assert_equal(smooth(data, 2), expected)

def test_smooth_passthrough():
    data = np.array([1, 0, 0, 1, 1, 0, 1], dtype='bool')
    np.testing.assert_equal(smooth(data, 1), data)
