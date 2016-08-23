from project import *

def test_multireduce():
    assert multiReduce([min, max], range(20)) == (0, 19)

def test_multireduce_np():
    import numpy as np
    from numpy.testing import assert_equal

    data = np.array([[1, 2, 3], [1, 4, -10], [5, 2, 0]])
    reduced = multiReduce([np.fmax, np.fmin], data)
    expected = map(np.asarray, ([5, 4, 3], [1, 2, -10]))
    for r, e in zip(reduced, expected):
        assert_equal(r, e)
