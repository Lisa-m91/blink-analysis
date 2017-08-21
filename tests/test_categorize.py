import numpy as np
from blink_analysis.categorize import *

import pytest
from click.testing import CliRunner

@pytest.fixture
def runner():
    return CliRunner()

def test_smooth():
    data     = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0], dtype='bool')
    expected = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0], dtype='bool')
    np.testing.assert_equal(smooth(data, (2, 2)), expected)

def test_smooth_small_peak():
    data     = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0], dtype='bool')
    expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0], dtype='bool')
    np.testing.assert_equal(smooth(data, (2, 2)), expected)

def test_smooth_passthrough():
    data = np.array([1, 0, 0, 1, 1, 0, 1], dtype='bool')
    np.testing.assert_equal(smooth(data, (1, 1)), data)

def test_smooth_different():
    data     = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0], dtype='bool')
    expected = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype='bool')
    np.testing.assert_equal(smooth(data, (2, 3)), expected)

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

def test_image_grid():
    data = np.arange(16).reshape((4, 2, 2))
    expected = np.array([[ 0,  1,  4,  5],
                         [ 2,  3,  6,  7],
                         [ 8,  9, 12, 13],
                         [10, 11, 14, 15]])
    np.testing.assert_equal(image_grid(data, 2), expected)

    expected = np.array([[ 0,  1,  4,  5],
                         [ 2,  3,  6,  7],
                         [ 8,  9,  0,  0],
                         [10, 11,  0,  0]])
    np.testing.assert_equal(image_grid(data[:3], 2), expected)

def test_plot_grid(runner, tmpdir):
    rois = np.random.randint(1, 200, size=(10, 20, 4, 4)).astype('uint8')
    roi_f = tmpdir.join("rois.pickle")
    with roi_f.open("wb") as f:
        for roi in rois:
            dump(roi, f)

    ons = np.random.randint(0, 2, size=rois.shape[:2]).astype('uint8')
    on_f = tmpdir.join("ons.pickle")
    with on_f.open("wb") as f:
        for on in ons:
            dump(on, f)

    result = runner.invoke(
        plot, ["--output", str(tmpdir.join("test.pdf")), "grid", str(roi_f), str(on_f)]
    )
    print(result.output)
    assert result.exit_code == 0

def test_plot_traces(runner, tmpdir):
    rois = np.random.randint(0, 200, size=(10, 20, 4, 4)).astype('uint8')
    roi_f = tmpdir.join("rois.pickle")
    with roi_f.open("wb") as f:
        for roi in rois:
            dump(roi, f)

    ons = np.random.randint(0, 2, size=rois.shape[:2]).astype('uint8')
    on_f = tmpdir.join("ons.pickle")
    with on_f.open("wb") as f:
        for on in ons:
            dump(on, f)

    result = runner.invoke(
        plot, ["--output", str(tmpdir.join("test.pdf")), "traces", str(roi_f), str(on_f)]
    )
    print(result.output)
    assert result.exit_code == 0
