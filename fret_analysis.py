#!/usr/bin/env python3
from itertools import chain
from functools import partial
from numpy import amax, amin, sum as asum, mean, std, percentile, clip, linspace
from collections import defaultdict

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

def roundMean(mean, sigma):
    from math import log10, floor

    if sigma == 0:
        return mean, sigma
    digits = int(floor(log10(abs(sigma))))

    mean = round(mean, -digits)
    sigma = round(sigma, -digits)
    return mean, sigma

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    import matplotlib
    import csv

    parser = ArgumentParser(description="Analyze single-particle traces.")
    parser.add_argument("--experiment", nargs='+', type=str, action="append",
                        help="The pickled ROIs to process")
    parser.add_argument("--output", type=str, required=False,
                        help="The base name for saving data.")
    parser.add_argument("--bin", type=int, default=1,
                        help="Number of frames to bin.")

    args = parser.parse_args()

    if args.output is not None:
        matplotlib.use('agg')
    # Must be imported after backend is set
    import matplotlib.pyplot as plt

    stats = defaultdict(lambda: defaultdict(list))
    for experiment in args.experiment:
        name = experiment[0]
        for datafile in map(Path, experiment[1:]):
            ds_stats = defaultdict(list)
            with datafile.open("rb") as f:
                for roi in loadAll(f):
                    end = len(roi) // args.bin * args.bin
                    roi = sum(map(lambda start: roi[start:end:args.bin], range(args.bin)))
                    trace = mean(roi, axis=(1, 2))
                    threshold = (amin(trace) + (amax(trace) - amin(trace)) / 2)
                    on = trace > threshold

                    background = mean(roi[~on])
                    # FIXME: Use raw intensity or intensity/background?
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

            for stat, v in ds_stats.items():
                stats[name][stat].append(v)

    fig = plt.figure(figsize=(8, 12))

    if args.output is not None:
        for name, exp_stats in stats.items():
            with open("{}_{}.csv".format(args.output, name), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=("name", "mean", "standard deviation"))
                writer.writeheader()
                for title, stat in sorted(exp_stats.items()):
                    grand_mean = mean(list(chain.from_iterable(stat)))
                    variation = std(list(map(mean, stat)))
                    writer.writerow({"name": title, "mean": grand_mean,
                                     "standard deviation": variation})
    else:
        for name, exp_stats in stats.items():
            print(name)
            for title, stat in sorted(exp_stats.items()):
                grand_mean = mean(list(chain.from_iterable(stat)))
                variation = std(list(map(mean, stat)))
                grand_mean, variation = roundMean(grand_mean, variation)
                print("{}: μ = {}, σ = {}".format(title, grand_mean, variation))
            print()

    titles = ["photons/frame", "photons/blink", "# of photons", "blink times",
              "on times", "# of blinks",]
    axes = map(partial(fig.add_subplot, len(titles) // 2, 2),
               range(1, len(titles) + 1))
    for ax, title in zip(axes, titles):
        ax.set_title(title)

        grand_means, variations, data = [], [], []
        for name, exp_stats in stats.items():
            grand_means.append(mean(list(chain.from_iterable(exp_stats[title]))))
            variations.append(std(list(map(mean, exp_stats[title]))))
            data.append(list(chain.from_iterable(exp_stats[title])))

        bound = percentile(list(chain.from_iterable(data)), 95)
        bins = linspace(0, bound, min(bound, 20))
        _, _, patches = ax.hist(data, bins, normed=True, label=stats)
        for grand_mean, variation, patch in zip(grand_means, variations, patches):
            ax.axvline(grand_mean, color=patch[0].get_facecolor())
            ax.axvline(grand_mean - variation, color=patch[0].get_facecolor(), linestyle='dashed')
            ax.axvline(grand_mean + variation, color=patch[0].get_facecolor(), linestyle='dashed')
        ax.legend()

    if args.output is not None:
        fig.tight_layout()
        fig.savefig("{}.png".format(args.output))

    if args.output is None:
        plt.show()
