#!/usr/bin/env python3
from itertools import chain
from functools import partial
import numpy as np
from scipy.stats import ttest_ind
from pickle import load

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

def bin(roi, width=1):
    end = len(roi) // width * width
    return sum(map(lambda start: roi[start:end:width], range(width)))

stat_names = ["frame_photons", "blink_photons", "total_photons",
              "blink_times", "total_times", "total_blinks", "on_rate"]

def calculateStats(signal, on):
    stats = {}

    signal = signal.reshape(len(signal), -1)
    blinks = np.split(signal, np.flatnonzero(np.diff(on)) + 1)
    if on[0]:
        on_blinks, off_blinks = blinks[::2], blinks[1::2]
    else:
        off_blinks, on_blinks = blinks[::2], blinks[1::2]

    last = np.flatnonzero(on)[-1] # index of last on event

    stats["frame_photons"] = signal[on].sum(axis=1).mean()
    stats["blink_photons"] = mean(map(np.sum, on_blinks))
    stats["total_photons"] = signal[on].sum()
    stats["blink_times"] = mean(map(len, on_blinks))
    stats["total_times"] = on.sum()
    stats["total_blinks"] = len(on_blinks)
    # Add 2 to get time of last off-event
    stats["on_rate"] = stats["total_blinks"] / (last + 2)

    return dict(stats)

def analyze(rois, ons):
    stats = {k: [] for k in stat_names}
    for roi, on in zip(rois, ons):
        fg_mask, bg_mask = masks(roi.shape[1:])
        signal = roi[:, fg_mask]
        background = roi[:, bg_mask]

        signal = (signal - background.mean(axis=1, keepdims=True)).clip(min=0)
        for stat, vs in calculateStats(signal, on).items():
            stats[stat].append(vs)
    return stats

def main(args=None):
    from sys import argv, stdout
    from argparse import ArgumentParser
    from pathlib import Path
    import csv

    parser = ArgumentParser(description="Analyze single-particle traces.")
    parser.add_argument("ROIs", type=Path, help="The pickled ROIs to process")
    parser.add_argument("onfile", type=Path, help="The pickled ROIs to process")
    parser.add_argument("--bin", type=int, default=1, help="Number of frames to bin.")
    args = parser.parse_args(argv[1:] if args is None else args)

    with args.ROIs.open("rb") as roi_f, args.onfile.open("rb") as on_f:
        stats = analyze(loadAll(roi_f), loadAll(on_f))

    writer = csv.DictWriter(stdout, sorted(stats.keys()))
    writer.writeheader()
    for row in zip(*stats.values()):
        writer.writerow(dict(zip(stats.keys(), row)))

if __name__ == "__main__":
    main()
