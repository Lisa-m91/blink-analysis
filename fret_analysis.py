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
    stats["frame_photons"].extend(chain.from_iterable(photons_by_blink))
    stats["blink_photons"].extend(map(sum, photons_by_blink))
    stats["total_photons"].append(sum(map(sum, photons_by_blink)))
    stats["blink_times"].extend(map(len, photons_by_blink))
    stats["total_times"].append(sum(map(len, photons_by_blink)))
    stats["total_blinks"].append(len(photons_by_blink))
    return stats

def analyze(args):
    bin_trace = partial(bin, width=args.bin)
    stats = defaultdict(list)
    for roi_file in args.ROIs:
        ds_stats = defaultdict(list)
        with roi_file.open("rb") as f:
            for roi in map(bin_trace, loadAll(f)):
                trace = mean(roi, axis=(1, 2))
                threshold = (amin(trace) + (amax(trace) - amin(trace)) / 2)
                on = trace > threshold

                for stat, vs in calculateStats(roi, on).items():
                    ds_stats[stat].extend(vs)
        for stat, vs in ds_stats.items():
            stats[stat].append(vs)
    with args.outfile.open("wb") as f:
        with args.metadata.open("r") as mf:
            dump(yaml.load(mf), f)
        dump(stats, f)

def plot(args):
    if args.outdir is not None:
        matplotlib.use('agg')
    # Must be imported after backend is set
    import matplotlib.pyplot as plt

    stats = defaultdict(partial(defaultdict, list))
    for stat_file in args.statfile:
        with stat_file.open("rb") as f:
            name = load(f)
            dataset_stats = load(f)
            for stat_name, vs in dataset_stats.items():
                stats[name][stat_name].append(vs)

    summaries = defaultdict(dict)
    for exp_name, exp_stats in stats.items():
        for stat_name, ds_stats in exp_stats.items():
            means = list(map(mean, stats[exp_name][stat_name]))
            summaries[exp_name][stat_name] = Summary(mean(means), std(means))
            stats[exp_name][stat_name] = list(chain.from_iterable(ds_stats))

    figs = {}
    fig, axs = plt.subplots(2, 3, figsize=(8, 12))
    figs["histograms"] = fig
    for stat_name, ax in zip(stat_names, chain.from_iterable(axs)):
        ax.set_title(stat_name)
        exp_names = list(stats.keys())
        data = [stats[exp_name][stat_name] for exp_name in exp_names]
        bound = percentile(list(chain.from_iterable(data)), 95)
        bins = linspace(0, bound, min(bound, 20))
        ax.hist(data, bins, normed=True, label=exp_names)
        ax.legend()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    figs["survival"] = fig
    ax.set_title("Survival Curve")
    ax.set_ylabel("% Remaining")
    ax.set_yscale("log")
    ax.set_xlabel("Lifetime")
    for experiment_name, stat in stats.items():
        on_times = array(stat["total_times"]).reshape(-1)
        xs = arange(0, on_times.max() + 1)
        surviving = asum(reshape(on_times, (-1, 1)) >= reshape(xs, (1, -1)), axis=0)
        ax.plot(surviving / len(on_times) * 100, label=experiment_name)
    ax.legend()

    if args.outdir is not None:
        for name, stat_summaries in summaries.items():
            with (args.outdir / "{}.csv".format(name)).open("w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=("name", "mean", "standard deviation"))
                writer.writeheader()
                for stat, summary in stat_summaries.items():
                    writer.writerow({"name": stat, "mean": summary.mean,
                                     "standard_deviation": summary.std_dev})
        for fig_name, fig in figs.items():
            fig.tight_layout()
            fig.savefig("{}.pdf".format(fig_name))
    else:
        for name, stat_summaries in summaries.items():
            print("{} (N = {})".format(name, len(stats[name]['total_times'])))
            for stat, summary in stat_summaries.items():
                print("{}: μ = {}, σ = {}"
                      .format(stat, *roundMean(summary.mean, summary.std_dev)))
            print()
        plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    import matplotlib
    import csv

    parser = ArgumentParser(description="Analyze single-particle traces.")
    subparsers = parser.add_subparsers()
    parser_analyze = subparsers.add_parser('analyze', help="Generate statistics from a ROI")
    parser_analyze.add_argument("metadata", type=Path,
                                help="The metadata .yaml file")
    parser_analyze.add_argument("ROIs", type=Path, nargs='+',
                                help="The pickled ROIs to process")
    parser_analyze.add_argument("outfile", type=Path,
                                help="The file to write stats to")
    parser_analyze.add_argument("--bin", type=int, default=1,
                                help="Number of frames to bin.")
    parser_analyze.set_defaults(func=analyze)

    parser_plot = subparsers.add_parser('plot', help="Plot analysis of generated statistics")
    parser_plot.add_argument("statfile", type=Path, nargs='+',
                             help="The directory where stats are stored")
    parser_plot.add_argument("--outdir", type=Path,
                             help="The directory to save the output files in")
    parser_plot.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)
