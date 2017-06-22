import numpy as np
from functools import partial
from pickle import dump, HIGHEST_PROTOCOL
dump = partial(dump, protocol=HIGHEST_PROTOCOL)
from blink_analysis.analyse import *

def test_main(tmpdir, capsys):
    np.random.seed(4)

    rois_f = tmpdir.join('rois.pickle')
    rois = np.random.normal(size=(2, 10, 9, 9))
    with rois_f.open("wb") as f:
        for roi in rois:
            dump(roi, f)

    ons_f = tmpdir.join('on.pickle')
    ons = np.random.normal(size=(2, 10)) > 0
    with ons_f.open("wb") as f:
        for on in ons:
            dump(on, f)

    main([str(rois_f), str(ons_f)])


def test_calculate_stats():
    signal = np.array([
        [[1, 1], [1, 2]],
        [[1, 2], [2, 2]],
        [[1, 5], [5, 2]], # 13
        [[1, 5], [8, 2]], # 16
        [[1, 1], [1, 2]],
        [[1, 1], [1, 2]],
        [[5, 9], [9, 2]],
        [[1, 2], [0, 6]], # 9
        [[1, 2], [1, 0]],
        [[1, 2], [1, 2]],
    ])

    on = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0], dtype='bool')
    expected = {
        'frame_photons': 38 / 3,
        'blink_photons': 19,
        'total_photons': 38,
        'blink_times': 1.5,
        'total_times': 3,
        'total_blinks': 2,
        'on_rate': 1/4,
        'off_rate': 2/3,
    }

    assert calculateStats(signal, on) == expected
    assert calculateStats(signal[:-2], on[:-2]) == expected

def test_calculate_blank_stats():
    signal = np.array([
        [[1, 1], [1, 2]],
        [[1, 2], [2, 2]],
        [[1, 5], [5, 2]],
        [[1, 5], [8, 2]],
        [[1, 1], [1, 2]],
        [[1, 1], [1, 2]],
        [[5, 9], [9, 2]],
        [[1, 2], [0, 6]],
        [[1, 2], [1, 0]],
        [[1, 2], [1, 2]],
    ])

    on = np.zeros(len(signal), dtype='bool')
    expected = {}

    assert calculateStats(signal, on) == expected
