from pathlib import Path
from functools import partial
from itertools import accumulate
from pickle import load, dump, HIGHEST_PROTOCOL
dump = partial(dump, protocol=HIGHEST_PROTOCOL)

import click
from hmmlearn import hmm
from sklearn import cluster
import numpy as np

from .categorize import masks

@click.group("hmm")
def main():
    pass

def loadAll(f):
    while True:
        try:
            yield load(f)
        except EOFError:
            return

def normalize(matrix):
    return matrix / np.sum(matrix, axis=1, keepdims=True)

def trace(roi):
    fg_mask, bg_mask = masks(roi.shape[1:])
    signal = roi[:, fg_mask]
    background = roi[:, bg_mask]
    return (signal - np.mean(background, axis=-1, keepdims=True)).sum(axis=-1)

state_names = ['on', 'off', 'bleached']
state_colors = list(map("C{}".format, range(len(state_names))))
on_states = np.array([0])

@main.command()
@click.argument("signals", type=Path, nargs=-1)
@click.argument("output", type=Path)
@click.option("--bias", type=float)
@click.option("--noise", type=float)
@click.option("--signal", type=float)
@click.option("--variance", type=float)
def train(signals, output, bias=0.0, noise=0.0, signal=1.0, variance=0.0):
    means = np.array([signal, bias, bias]).reshape(-1, 1)
    covars = np.array([variance, noise, noise]).reshape(-1, 1)
    startprob = normalize(np.asarray([[1, 1, 1]])).reshape(-1)
    trans = normalize(np.array([[0.5, 0.5, 0.1], # on
                                [0.1, 0.9, 0.0], # off
                                [0.0, 0.0, 1.0]])) # bleached

    model = hmm.GaussianHMM(
        n_components=len(trans), #transmat_prior=trans,
        #means_prior=means, means_weight=1.0,
        #covars_prior=covars, covars_weight=1.0,
        params='stmc', init_params='', algorithm='viterbi'
    )
    model.means_ = means
    model.covars_ = covars
    model.transmat_ = trans
    model.startprob_ = startprob

    traces = []
    for signal in signals:
        with signal.open("rb") as f:
            traces.extend(map(trace, loadAll(f)))

    model = model.fit(np.concatenate(traces).reshape(-1, 1), list(map(len, traces)))

    with output.open("wb") as f:
        dump(model, f)

@main.command()
@click.argument("model", type=Path)
@click.argument("signal", type=Path)
@click.argument("output", type=Path)
def categorize(model, signal, output):
    with model.open("rb") as f:
        model = load(f)
    with signal.open("rb") as f, output.open("wb") as out_f:
        traces = map(partial(np.reshape, newshape=(-1, 1)), map(trace, loadAll(f)))
        for states in map(model.predict, traces):
            dump(states.astype('uint8'), out_f)
