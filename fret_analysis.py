#!/usr/bin/env python3
from itertools import chain
from functools import partial
from numpy import (amax, amin, sum as asum, mean, std, percentile, clip,
                   linspace, array, arange, reshape)
from collections import defaultdict, namedtuple
from math import inf
from pickle import load, dump, HIGHEST_PROTOCOL
dump = partial(dump, protocol=HIGHEST_PROTOCOL)
import yaml

Summary = namedtuple("Summary", ["mean", "std_dev"])

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

def roundMean(mean, sigma):
    from math import log10, floor

    if sigma == 0:
        return mean, sigma
    digits = int(floor(log10(abs(sigma))))

    mean = round(mean, -digits)
    sigma = round(sigma, -digits)
    return mean, sigma

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

def analyze(args):
    bin_trace = partial(bin, width=args.bin)
    stats = defaultdict(list)
    with args.ROIs.open("rb") as f:
        for roi in map(bin_trace, loadAll(f)):
            trace = mean(roi, axis=(1, 2))
            threshold = calculateThreshold(trace)
            on = trace > threshold

            for stat, vs in calculateStats(roi, on).items():
                stats[stat].append(vs)
    with args.outfile.open("wb") as f:
        dump(dict(stats), f)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description="Analyze single-particle traces.")
    subparsers = parser.add_subparsers()
    parser_analyze = subparsers.add_parser('analyze', help="Generate statistics from a ROI")
    parser_analyze.add_argument("ROIs", type=Path,
                                help="The pickled ROIs to process")
    parser_analyze.add_argument("outfile", type=Path,
                                help="The file to write stats to")
    parser_analyze.add_argument("--bin", type=int, default=1,
                                help="Number of frames to bin.")
    parser_analyze.set_defaults(func=analyze)

    args = parser.parse_args()
    args.func(args)
