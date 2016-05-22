#!/usr/bin/env python3
from itertools import chain
from functools import partial
from numpy import amax, amin, sum as asum, mean, std, percentile, clip, concatenate, linspace
from random import sample, seed

def loadAll(f):
    from pickle import load

    while True:
        try:
            yield load(f)
        except EOFError:
            break

def makeSegments(y, x=None):
    from numpy import arange, concatenate, array

    if x is None:
        x = arange(len(y))
    points = array([x, y]).T.reshape(-1, 1, 2)
    return concatenate([points[:-1], points[1:]], axis=1)

def groupWith(a, b):
    from itertools import groupby

    for key, group in groupby(zip(b, a), lambda x: x[0]):
        yield key, map(lambda x: x[1], group)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    import matplotlib
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.collections import LineCollection
    from matplotlib.cm import get_cmap
    import csv

    parser = ArgumentParser(description="Analyze single-particle traces.")
    parser.add_argument("rois", nargs='+', type=Path,
                        help="The pickled ROIs to process")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="The fold-increase over background necessary for a spot to be on.")
    parser.add_argument("--ntraces", type=int, default=5,
                        help="The number of (randomly chosen) traces to display.")
    parser.add_argument("--output", type=str, required=False,
                        help="The base name for saving data.")
    parser.add_argument("--seed", type=int, default=4,
                        help="The seed for random processed (e.g. selecting sample traces)")

    args = parser.parse_args()

    seed(args.seed)

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

    fig = plt.figure(figsize=(8, 12))
    sample_idxs = list(sample(range(sum(map(len, rois))), args.ntraces))
    sample_rois  = map(list(chain.from_iterable(rois)).__getitem__, sample_idxs)
    sample_traces = list(map(list(chain.from_iterable(traces)).__getitem__, sample_idxs))
    vmin = min(map(amin, sample_traces))
    vmax = max(map(amax, sample_traces))
    plt_indices = range(1, len(sample_idxs) * 2, 2)
    for i, roi, trace in zip(plt_indices, sample_rois, sample_traces):
        on = trace > args.threshold

        cmap = ListedColormap(['r', 'b'])
        norm = BoundaryNorm([-float('inf'), 0.5, float('inf')], cmap.N)

        lc = LineCollection(makeSegments(trace), cmap=cmap, norm=norm)
        lc.set_array(on)

        ax = fig.add_subplot(plt_indices.stop, 1, i)
        ax.set_ylabel("max. intensity")
        ax.set_xlabel("frame")
        ax.add_collection(lc)
        ax.set_xlim(0, len(trace))
        ax.set_ylim(vmin, vmax)
        ax.axhline(y=args.threshold)

        ax = fig.add_subplot(plt_indices.stop, 1, i+1)
        rowsize = 409 # Factors 8998
        show = concatenate([concatenate(roi[i:i+rowsize], axis=-1)
                            for i in range(0, len(roi), rowsize)], axis=-2)
        ax.imshow(show, vmax=vmax, vmin=vmin,
                  cmap=get_cmap('gray'), interpolation="nearest")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    if args.output is not None:
        fig.tight_layout()
        fig.savefig("{}_traces.png".format(args.output))

    on_times = []
    blink_times = []
    blink_counts = []
    photon_counts = []
    blink_photons = []
    frame_photons = []
    for ds_rois, ds_traces in zip(rois, traces):
        ds_on_times = []
        ds_blink_times = []
        ds_blink_counts = []
        ds_photon_counts = []
        ds_blink_photons = []
        ds_frame_photons = []
        for roi, trace in zip(ds_rois, ds_traces):
            on = (trace > args.threshold)

            # FIXME: Use raw intensity or intensity/background?
            background = mean(roi[~on])
            signal = clip(roi - background, a_min=0, a_max=float('inf'))
            blinks = groupWith(signal, on)
            on_blinks = map(lambda x: x[1], filter(lambda x: x[0], blinks))

            photons_by_blink = list(map(list, map(partial(map, asum), on_blinks)))
            ds_frame_photons.extend(chain.from_iterable(photons_by_blink))
            ds_blink_photons.extend(map(sum, photons_by_blink))
            ds_photon_counts.append(sum(map(sum, photons_by_blink)))
            ds_blink_times.extend(map(len, photons_by_blink))
            ds_on_times.append(sum(map(len, photons_by_blink)))
            ds_blink_counts.append(len(photons_by_blink))
        on_times.append(ds_on_times)
        blink_times.append(ds_blink_times)
        blink_counts.append(ds_blink_counts)
        photon_counts.append(ds_photon_counts)
        blink_photons.append(ds_blink_photons)
        frame_photons.append(ds_frame_photons)

    fig = plt.figure(figsize=(8, 12))
    stats = [on_times, blink_times, blink_counts, photon_counts,
             blink_photons, frame_photons]
    titles = ["on times", "blink times", "# of blinks", "# of photons",
              "photons/blink", "photons/frame (AU)"]

    if args.output is not None:
        with open("{}_stats_summary.csv".format(args.output), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=("name", "mean", "standard deviation"))
            writer.writeheader()
            for title, stat in zip(titles, stats):
                grand_mean = mean(list(chain.from_iterable(stat)))
                variation = std(list(map(mean, stat)))
                writer.writerow({"name": title, "mean": grand_mean,
                                 "standard deviation": variation})
    else:
        for title, stat in zip(titles, stats):
            grand_mean = mean(list(chain.from_iterable(stat)))
            variation = std(list(map(mean, stat)))
            print("{}: μ = {}, σ = {}".format(title, grand_mean, variation))


    axes = map(partial(fig.add_subplot, len(titles) // 2, 2),
               range(1, len(titles) + 1))
    for ax, data_sets, title in zip(axes, stats, titles):
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
