import numpy as np
from functools import partial
from pickle import dump, HIGHEST_PROTOCOL
dump = partial(dump, protocol=HIGHEST_PROTOCOL)
from blink_analysis.analyse import *
from math import isnan

import pytest

on_states = np.asarray([1], dtype='uint8')

@pytest.fixture()
def runner():
    from click.testing import CliRunner
    return CliRunner()

def test_main(tmpdir, capsys, runner):
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

    result = runner.invoke(main, [str(rois_f), str(ons_f)])
    assert result.exit_code == 0


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
        'on_rate': 2/5,
        'off_rate': 2/3,
    }

    assert calculateStats(signal, on, on_states) == expected
    assert calculateStats(signal[:-2], on[:-2], on_states) == expected

    photons = 2.5
    exposure = 3.8
    expected = {
        'frame_photons': expected['frame_photons'] * photons,
        'blink_photons': expected['blink_photons'] * photons,
        'total_photons': expected['total_photons'] * photons,
        'blink_times': expected['blink_times'] * exposure,
        'total_times': expected['total_times'] * exposure,
        'total_blinks': expected['total_blinks'],
        'on_rate': expected['on_rate'] / exposure,
        'off_rate': expected['off_rate'] / exposure,
    }
    for k, v in calculateStats(signal, on, on_states, photons, exposure).items():
        np.testing.assert_almost_equal(v, expected[k])
    for k, v in calculateStats(signal[:-2], on[:-2], on_states, photons, exposure).items():
        np.testing.assert_almost_equal(v, expected[k])

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

    assert calculateStats(signal, on, on_states) == expected

def test_calculate_single_blink_stats():
    signal = np.array([
        [[1, 1], [1, 2]], # 5
        [[1, 2], [2, 2]], # 7
        [[1, 5], [5, 2]], # 13
        [[1, 5], [8, 2]],
        [[1, 1], [1, 2]],
        [[1, 1], [1, 2]],
        [[5, 9], [9, 2]],
        [[1, 2], [0, 6]],
        [[1, 2], [1, 0]],
        [[1, 2], [1, 2]],
    ])

    on = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype='bool')
    expected = {
        'frame_photons': 25 / 3,
        'blink_photons': 25,
        'total_photons': 25,
        'blink_times': 3,
        'total_times': 3,
        'total_blinks': 1,
        'off_rate': 1/3,
    }

    stats = calculateStats(signal, on, on_states)
    assert isnan(stats.pop('on_rate'))
    assert stats == expected
