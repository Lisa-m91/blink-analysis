#!/usr/bin/env python3
from itertools import chain
from functools import partial
import numpy as np
from scipy.stats import ttest_ind
from pickle import load
from pathlib import Path

import click

from .categorize import masks

def loadAll(f):
    while True:
        try:
            yield load(f)
        except EOFError:
            break

def mean(iterable):
    total = next(iterable)
    ctr = 1
    for i in iterable:
        total += i
        ctr += 1
    return total / ctr

stat_names = ["frame_photons", "blink_photons", "total_photons", "blink_times",
              "total_times", "total_blinks", "on_rate", "off_rate"]

def calculateStats(signal, states, on_states, photons=1.0, exposure=1.0):
    stats = {}

    on = (states == on_states.reshape(-1, 1)).any(axis=0)
    signal = signal.reshape(len(signal), -1)
    blinks = np.split(signal, np.flatnonzero(np.diff(on)) + 1)
    if on[0]:
        on_blinks, off_blinks = blinks[::2], blinks[1::2]
    else:
        off_blinks, on_blinks = blinks[::2], blinks[1::2]
    if not on[-1]:
        off_blinks = off_blinks[:-1]
    try:
        last = np.flatnonzero(on)[-1]
    except IndexError:
        return {} # Will cause nothing to be appended to stats

    stats["frame_photons"] = signal[on].sum(axis=1).mean() * photons
    stats["blink_photons"] = mean(map(np.sum, on_blinks)) * photons
    stats["total_photons"] = signal[on].sum() * photons
    stats["blink_times"] = mean(map(len, on_blinks)) * exposure
    stats["total_times"] = on.sum() * exposure
    stats["total_blinks"] = len(on_blinks)
    # Add 2 to get time of last off-event
    try:
        stats["on_rate"] = stats["total_blinks"] / (sum(map(len, off_blinks)) * exposure)
    except ZeroDivisionError:
        stats["on_rate"] = float('nan')
    stats["off_rate"] = stats["total_blinks"] / stats["total_times"]

    return stats

def analyze(rois, statess, *args, **kwargs):
    stats = {k: [] for k in stat_names}
    for roi, states in zip(rois, statess):
        fg_mask, bg_mask = masks(roi.shape[1:])
        signal = roi[:, fg_mask]
        background = roi[:, bg_mask]

        signal = (signal - background.mean(axis=1, keepdims=True)).clip(min=0)
        for stat, vs in calculateStats(signal, states, *args, **kwargs).items():
            stats[stat].append(vs)
    return stats

@click.command("analyse")
@click.argument("rois", type=Path)
@click.argument("statefile", type=Path)
@click.option("--photons", type=float, default=1.0,
              help="The conversion of ADU to photons")
@click.option("--exposure", type=float, default=1.0,
              help="The conversion of frames to seconds")
@click.option("--on-state", type=int, multiple=True,
              help="The labels of the on-states")
def main(rois, statefile, photons=1.0, exposure=1.0, on_state=()):
    from sys import stdout
    import csv

    on_state = np.asarray(on_state)

    with rois.open("rb") as roi_f, statefile.open("rb") as state_f:
        stats = analyze(loadAll(roi_f), loadAll(state_f), on_states=on_state,
                        photons=photons, exposure=exposure)

    writer = csv.DictWriter(stdout, sorted(stats.keys()))
    writer.writeheader()
    for row in zip(*stats.values()):
        writer.writerow(dict(zip(stats.keys(), row)))
