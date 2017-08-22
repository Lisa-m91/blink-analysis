from pathlib import Path
from functools import partial
from itertools import accumulate
from pickle import load, dump, HIGHEST_PROTOCOL
import operator as op
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
    return (signal - np.mean(background, axis=-1, keepdims=True)).mean(axis=-1)

state_names = ['on', 'off', 'bleached']
state_colors = list(map("C{}".format, range(len(state_names))))
on_states = np.array([0])

def gen_trans(nstates, bleach_prob=[0.1, 0.0]):
    # Fully connected on/off transitions
    trans = np.ones((sum(nstates),) * 2, dtype='double')
    bleach_prob = np.repeat(bleach_prob, nstates)
    trans = np.concatenate([trans, bleach_prob[:, None]], axis=1)
    stay_bleached = np.repeat([0.0, 1.0], [sum(nstates), 1])
    trans = np.concatenate([trans, stay_bleached[None, :]], axis=0)
    return trans

def jitter(probs, amplitude=0.1, rng=None):
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    return probs * rng.uniform(1-amplitude, 1+amplitude, probs.shape)

@main.command()
@click.argument("signals", type=Path, nargs=-1)
@click.argument("output", type=Path)
@click.option("--bias", type=float, default=0.0)
@click.option("--noise", type=float, default=0.0)
@click.option("--signal", type=float, default=1.0)
@click.option("--variance", type=float, default=0.0)
@click.option("--nstates", type=(int, int), default=(1, 1))
@click.option("--seed", type=int, default=None)
@click.option("--mean-weight", type=float, default=0.0)
def train(signals, output, bias=0.0, noise=0.0, signal=1.0, variance=0.0,
          nstates=(1, 1), seed=None, mean_weight=0.0):
    rng = np.random.RandomState(seed)

    means = np.repeat([signal, bias], [nstates[0], nstates[1] + 1])[:, None]
    means = jitter(means, rng=rng)
    covars = np.repeat([variance, noise], [nstates[0], nstates[1] + 1])[:, None]
    startprob = normalize(jitter(np.ones((1, sum(nstates) + 1)), rng=rng))[0]
    trans = normalize(jitter(gen_trans(nstates), rng=rng))

    model = hmm.GaussianHMM(
        n_components=len(trans), #transmat_prior=trans,
        means_prior=means, means_weight=mean_weight,
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

    model = model.fit(np.concatenate(traces)[:, None], list(map(len, traces)))

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
        traces = map(op.itemgetter((slice(None), None)), map(trace, loadAll(f)))
        for states in map(model.predict, traces):
            dump(states.astype('uint8'), out_f)
