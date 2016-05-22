#!/usr/bin/env python3
from itertools import chain
from functools import partial
from numpy import amax, amin, sum as asum, mean, std, percentile, clip, linspace
from collections import OrderedDict

def loadAll(f):
    from pickle import load

    while True:
        try:
            yield load(f)
        except EOFError:
            break

def groupWith(a, b):
    from itertools import groupby

    for key, group in groupby(zip(b, a), lambda x: x[0]):
        yield key, map(lambda x: x[1], group)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    import matplotlib
    import csv

    parser = ArgumentParser(description="Analyze single-particle traces.")
    parser.add_argument("rois", nargs='+', type=Path,
                        help="The pickled ROIs to process")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="The fold-increase over background necessary for a spot to be on.")
    parser.add_argument("--output", type=str, required=False,
                        help="The base name for saving data.")

    args = parser.parse_args()

    if args.output is not None:
        matplotlib.use('agg')
    # Must be imported after backend is set
    import matplotlib.pyplot as plt

    rois = []
    for roi_path in args.rois:
        with roi_path.open("rb") as f:
            rois.append(list(loadAll(f)))
    traces = list(map(list, map(partial(map, partial(amax, axis=(1, 2))), rois)))
    nrois = sum(map(len, rois))

    stats = OrderedDict([("on times", []), ("blink times", []), ("# of blinks", []),
                         ("# of photons", []), ("photons/blink", []), ("photons/frame", [])])
    for ds_rois, ds_traces in zip(rois, traces):
        ds_stats = {k: [] for k in stats.keys()}
        for roi, trace in zip(ds_rois, ds_traces):
            on = (trace > args.threshold)

            # FIXME: Use raw intensity or intensity/background?
            background = mean(roi[~on])
            signal = clip(roi - background, a_min=0, a_max=float('inf'))
            blinks = groupWith(signal, on)
            on_blinks = map(lambda x: x[1], filter(lambda x: x[0], blinks))

            photons_by_blink = list(map(list, map(partial(map, asum), on_blinks)))
            ds_stats["photons/frame"].extend(chain.from_iterable(photons_by_blink))
            ds_stats["photons/blink"].extend(map(sum, photons_by_blink))
            ds_stats["# of photons"].append(sum(map(sum, photons_by_blink)))
            ds_stats["blink times"].extend(map(len, photons_by_blink))
            ds_stats["on times"].append(sum(map(len, photons_by_blink)))
            ds_stats["# of blinks"].append(len(photons_by_blink))
        for k, v in ds_stats.items():
            stats[k].append(v)

    fig = plt.figure(figsize=(8, 12))

    if args.output is not None:
        with open("{}_stats_summary.csv".format(args.output), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=("name", "mean", "standard deviation"))
            writer.writeheader()
            for title, stat in stats.items():
                grand_mean = mean(list(chain.from_iterable(stat)))
                variation = std(list(map(mean, stat)))
                writer.writerow({"name": title, "mean": grand_mean,
                                 "standard deviation": variation})
    else:
        for title, stat in stats.items():
            grand_mean = mean(list(chain.from_iterable(stat)))
            variation = std(list(map(mean, stat)))
            print("{}: μ = {}, σ = {}".format(title, grand_mean, variation))


    axes = map(partial(fig.add_subplot, len(stats) // 2, 2),
               range(1, len(stats) + 1))
    for ax, (title, data_sets) in zip(axes, stats.items()):
        data = list(chain.from_iterable(data_sets))
        ax.set_title(title)
        bound = percentile(data, 95)
        bins = linspace(0, bound, min(bound, 20))
        ax.hist(data, bins)

    if args.output is not None:
        fig.tight_layout()
        fig.savefig("{}_stats.png".format(args.output))

    if args.output is None:
        plt.show()
