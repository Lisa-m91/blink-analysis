from blink_analysis.hmm import *

import numpy as np
from pickle import loads
import pytest
from click.testing import CliRunner

@pytest.fixture
def runner():
    return CliRunner()

def test_gen_trans_size():
    expected = normalize(np.array([[1.0, 1.0, 0.1],
                                   [1.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]]))
    np.testing.assert_equal(gen_trans([1, 1]), expected)
    expected = normalize(np.array([[1.0, 1.0, 1.0, 1.0, 0.1],
                                   [1.0, 1.0, 1.0, 1.0, 0.1],
                                   [1.0, 1.0, 1.0, 1.0, 0.0],
                                   [1.0, 1.0, 1.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0]]))
    np.testing.assert_equal(gen_trans([2, 2]), expected)
    expected = normalize(np.array([[1.0, 1.0, 1.0, 0.1],
                                   [1.0, 1.0, 1.0, 0.1],
                                   [1.0, 1.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0]]))
    np.testing.assert_equal(gen_trans([2, 1]), expected)

def test_gen_trans_bleachprob():
    expected = normalize(np.array([[1.0, 1.0, 0.0],
                                   [1.0, 1.0, 0.2],
                                   [0.0, 0.0, 1.0]]))
    np.testing.assert_equal(gen_trans([1, 1], [0.0, 0.2]), expected)

def untrace(values):
    shape = (5, 5)
    fg_mask, bg_mask = masks(shape)
    signal = np.random.normal(size=(len(values),) + shape)
    signal[:, fg_mask] += values
    return signal

# TODO: Do saving as resultcallback, for easier testing
def test_train(runner, tmpdir):
    rng = np.random.RandomState(4)

    model = hmm.GaussianHMM(n_components=3)
    model.means_ = np.array([100, 0, 0])[:, None]
    model.covars_ = np.array([10, 1, 1])[:, None]
    model.startprob_ = normalize(np.array([[1, 1, 1]]))[0]
    model.transmat_ = normalize(
        np.array([[1.0, 1.0, 0.1], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )

    signals = []
    for path in map(tmpdir.join, map("{}.pickle".format, range(20))):
        sample, states = model.sample(100, rng)
        signals.append(str(path))
        with path.open("wb") as f:
            dump(untrace(sample), f)

    cmd = signals + [
            str(tmpdir.join("model.pickle")), "--noise", "1",
            "--signal", "100", "--variance", "10"
        ]
    result = runner.invoke(train, cmd)
    assert result.exit_code == 0

    with tmpdir.join("model.pickle").open("rb") as f:
        new_model = load(f)
    np.testing.assert_allclose(model.means_, new_model.means_, rtol=0.01, atol=1)
    np.testing.assert_allclose(model.transmat_, new_model.transmat_, rtol=0.15)

def test_categorize(runner, tmpdir):
    rng = np.random.RandomState(4)

    model = hmm.GaussianHMM(n_components=3)
    model.means_ = np.array([100, 0, 0])[:, None]
    model.covars_ = np.array([10, 1, 1])[:, None]
    model.startprob_ = normalize(np.array([[1, 1, 1]]))[0]
    model.transmat_ = normalize(
        np.array([[1.0, 1.0, 0.1], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    model_path = tmpdir.join("model.pickle")
    with model_path.open("wb") as f:
        dump(model, f)

    for sample_path in map(tmpdir.join, map("{}.pickle".format, range(20))):
        sample, states = model.sample(100, rng)
        with sample_path.open("wb") as f:
            dump(untrace(sample), f)

        state_path = tmpdir.join("{}.states.pickle")
        cmd = [str(model_path), str(sample_path), str(state_path)]
        result = runner.invoke(categorize, cmd)
        assert result.exit_code == 0

        with state_path.open("rb") as f:
            categories = load(f)
        np.testing.assert_equal(categories, states)
