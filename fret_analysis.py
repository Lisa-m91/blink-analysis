#!/usr/bin/env python3
from itertools import chain
from functools import partial
from numpy import (sum as asum, max as amax, min as amin, mean, clip, zeros)
from math import inf
from scipy.stats import ttest_ind
from collections import defaultdict
from pickle import load, dump, HIGHEST_PROTOCOL
dump = partial(dump, protocol=HIGHEST_PROTOCOL)
import yaml

def loadAll(f):
    while True:
        try:
            yield load(f)
        except EOFError:
            break

def groupWith(a, b):
    from itertools import groupby

    for key, group in groupby(zip(b, a), lambda x: x[0]):
        yield key, map(lambda x: x[1], group)

def bin(roi, width=1):
    end = len(roi) // width * width
    return sum(map(lambda start: roi[start:end:width], range(width)))

stat_names = ["frame_photons", "blink_photons", "total_photons",
              "blink_times", "total_times", "total_blinks"]

def calculateStats(roi, on):
    stats = {k: [] for k in stat_names}

    background = mean(roi[~on])
    signal = clip(roi - background, a_min=0, a_max=inf)
    blinks = groupWith(signal, on)
    on_blinks = map(lambda x: x[1], filter(lambda x: x[0], blinks))

    photons_by_blink = list(map(list, map(partial(map, asum), on_blinks)))
    stats["frame_photons"] = list(chain.from_iterable(photons_by_blink))
    stats["blink_photons"] = list(map(sum, photons_by_blink))
    stats["total_photons"] = sum(map(sum, photons_by_blink))
    stats["blink_times"] = list((map(len, photons_by_blink)))
    stats["total_times"] = sum(map(len, photons_by_blink))
    stats["total_blinks"] = len(photons_by_blink)
    return dict(stats)

def calculateThreshold(trace):
    return (amin(trace) + (amax(trace) - amin(trace)) / 2)

def analyze(rois):
    bin_trace = partial(bin, width=args.bin)
    stats = defaultdict(list)
    for roi in map(bin_trace, loadAll(f)):
        trace = mean(roi, axis=(1, 2))
        threshold = calculateThreshold(trace)
        on = trace > threshold

        for stat, vs in calculateStats(roi, on).items():
            stats[stat].append(vs)
    return dict(stats)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description="Analyze single-particle traces.")
    parser.add_argument("ROIs", type=Path, help="The pickled ROIs to process")
    parser.add_argument("outfile", type=Path, help="The file to write stats to")
    parser.add_argument("--bin", type=int, default=1, help="Number of frames to bin.")

    args = parser.parse_args()
    with args.ROIs.open("rb") as f:
        stats = analyze(loadAll(f))
    with args.outfile.open("wb") as f:
        dump(stats, f)
