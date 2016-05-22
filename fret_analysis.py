#!/usr/bin/env python3
from itertools import chain
from functools import partial
from numpy import amax, amin, sum as asum, mean, std, percentile, clip, linspace

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
