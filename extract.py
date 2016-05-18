#!/usr/bin/env python3
from functools import partial, reduce
from itertools import chain
from tifffile import imread

from blob import findBlobs

from numpy import (std, mean, amax, amin, sum as asum, median, array, empty, clip, linspace)

def extract(peak, image, expansion=1):
    scale, *pos = peak
    scale = round(int(scale) * expansion)
    roi = (slice(None),) + tuple(slice(p - scale, p + scale) for p in pos)
    return image[roi]

def peakEnclosed(peak, shape, expansion=1):
    scale, *pos = peak
    scale = scale * expansion
    return (all(scale <= p for p in pos) and
            all(scale < (s - p) for s, p in zip(shape, pos)))

def rollingMedian(data, width, pool=None):
    slices = (data[start:start+width] for start in range(0, len(data) - (width - 1)))
    if pool is None:
        return map(partial(median, axis=0), slices)
    else:
        return pool.imap(partial(median, axis=0), slices)

def makeSegments(y, x=None):
    from numpy import arange, concatenate

    if x is None:
        x = arange(len(y))
    points = array([x, y]).T.reshape(-1, 1, 2)
    return concatenate([points[:-1], points[1:]], axis=1)

def groupWith(a, b):
    from itertools import groupby

    for key, group in groupby(zip(b, a), lambda x: x[0]):
        yield key, map(lambda x: x[1], group)

def blinkTimes(iterable):
    count = 0
    for i in iterable:
        if not i and count > 0:
            yield count
        count = (count + 1) if i else 0
    if count != 0:
        yield count

if __name__ == '__main__':
    from argparse import ArgumentParser
    import csv

    import matplotlib
    from matplotlib.cm import get_cmap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.collections import LineCollection

    from numpy import percentile, concatenate, asarray
    from random import seed, sample
    from multiprocessing import Pool

    parser = ArgumentParser(description="Extract points from a video.")
    parser.add_argument("image", type=str, help="The video to process.")
    parser.add_argument("plot", type=str, nargs='?',
                        help="The base name for saving plots.")
    parser.add_argument("--background", type=str, help="A background image to correct for.")
    parser.add_argument("--spot-size", nargs=2, type=int, default=(2, 5),
                        help="The range of spot sizes to search for.")
    parser.add_argument("--max-overlap", type=float, default=0.05,
                        help="The maximum amount of overlap before spots are merged.")
    parser.add_argument("--blob-threshold", type=float, default=0.1,
                        help="The threshold value for the LOG blob detection.")
    parser.add_argument("--count-photons", type=float, default=1.0,
                        help="The number of photons per image count.")
    parser.add_argument("--expansion", type=float, default=1,
                        help="The amount to expand detected points by.")
    parser.add_argument("--ntraces", type=int, default=5,
                        help="The number of (randomly chosen) traces to display.")
    parser.add_argument("--on-threshold", type=float, default=2.0,
                        help="The fraction of the maximum value a spot has to rise above to be 'on'")
    parser.add_argument("--seed", type=int, default=4,
                        help="The seed to use for random processes (e.g. selecting sample traces)")
    args = parser.parse_args()

    if args.plot is not None:
        matplotlib.use('agg')
    # Must be imported after backend is set
    import matplotlib.pyplot as plt

    seed(args.seed)
    p = Pool()

    raw = imread(args.image, memmap=True)
    background = imread(args.background)

    image = empty((len(raw) - 2,) + raw.shape[1:], dtype='float32')
    for i, frame in enumerate(rollingMedian(raw, 3, pool=p)):
        image[i] = frame / background
    del raw

    proj = amax(image, axis=0)
    peaks = findBlobs(proj, scales=range(*args.spot_size),
                      threshold=args.blob_threshold, max_overlap=args.max_overlap)

    # Exclude partial ROIs on edge
    peaks = filter(partial(peakEnclosed, shape=proj.shape, expansion=args.expansion), peaks)
    peaks = asarray(list(peaks))
    rois = list(map(partial(extract, image=image, expansion=args.expansion), peaks))
    traces = list(map(partial(amax, axis=(1, 2)), rois))
    sample_idxs = list(sample(range(len(rois)), args.ntraces))

    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(proj, cmap=get_cmap('gray'), vmax=percentile(proj, 99.5))
    ax.scatter(peaks[:, 2], peaks[:, 1], s=peaks[:, 0] * 20,
               facecolors='None', edgecolors=['b' if idx not in sample_idxs else 'r'
                                              for idx in range(len(rois))])
    ax.set_ylim(0, proj.shape[1])
    ax.set_yticks([])
    ax.set_xlim(0, proj.shape[0])
    ax.set_xticks([])
    ax.set_title("max intensity")
    if args.plot is not None:
        fig.tight_layout()
        fig.savefig("{}_proj.png".format(args.plot))

    fig = plt.figure(figsize=(8, 12))
    samples = list(zip(map(rois.__getitem__, sample_idxs),
                       map(traces.__getitem__, sample_idxs)))
    vmin = min(map(lambda s: amin(s[1]), samples))
    vmax = max(map(lambda s: amax(s[1]), samples))
    plt_indices = range(1, len(samples) * 2, 2)
    for i, (roi, trace) in zip(plt_indices, samples):
        on = trace > args.on_threshold

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
        ax.axhline(y=args.on_threshold)

        ax = fig.add_subplot(plt_indices.stop, 1, i+1)
        rowsize = 409 # Factors 8998
        show = concatenate([concatenate(roi[i:i+rowsize], axis=-1)
                            for i in range(0, len(roi), rowsize)], axis=-2)
        ax.imshow(show, vmax=vmax, vmin=vmin,
                  cmap=get_cmap('gray'), interpolation="nearest")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    if args.plot is not None:
        fig.savefig("{}_traces.png".format(args.plot))

    on_times = []
    blink_times = []
    blink_counts = []
    photon_counts = []
    blink_photons = []
    frame_photons = []
    for roi, trace in zip(rois, traces):
        on = (trace > args.on_threshold).astype('int8')

        on_times.append(asum(on))
        blink_times.extend(blinkTimes(on))
        blink_counts.append(asum((on[1:] - on[:-1]) == -1) + on[-1])

        background = mean(roi[~on])
        # FIXME: Use raw intensity or intensity/background?
        signal = clip(roi - background, a_min=0, a_max=float('inf'))
        frame_photons.extend(map(asum, signal[on]))

        blinks = groupWith(signal, on)
        on_blinks = map(lambda x: list(x[1]), filter(lambda x: x[0], blinks))
        photons = list(map(asum, on_blinks))
        blink_photons.extend(photons)
        photon_counts.append(sum(photons))

    fig = plt.figure(figsize=(8, 12))
    stats = [on_times, blink_times, blink_counts, photon_counts,
             blink_photons, frame_photons]
    titles = ["on times", "blink times", "# of blinks", "# of photons",
              "photons/blink", "photons/frame (AU)"]

    if args.plot is not None:
        with open("{}_stats_summary.csv".format(args.plot), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=("name", "mean", "standard deviation"))
            writer.writeheader()
            for title, stat in zip(titles, stats):
                writer.writerow({"name": title, "mean": mean(stat),
                                 "standard deviation": std(stat)})
    else:
        for title, stat in zip(titles, stats):
            print("{}: μ = {}, σ = {}".format(title, mean(stat, std(stat))))


    for i, (data, title) in enumerate(zip(stats, titles), start=1):
        ax = fig.add_subplot(len(titles) // 2, 2, i)
        ax.set_title(title)
        bound = percentile(data, 95)
        bins = linspace(0, bound, min(bound, 20))
        ax.hist(data, bins)

    if args.plot is not None:
        fig.tight_layout()
        fig.savefig("{}_stats.png".format(args.plot))

    if args.plot is None:
        plt.show()
