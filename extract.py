#!/usr/bin/env python3
from functools import partial, reduce
from itertools import chain
from tifffile import imread

from blob import findBlobs

from numpy import (std, mean, max as np_max, min as np_min, sum as np_sum,
                   median, array, fromiter as np_fromiter)

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

def rollingMedian(data, width):
    slices = (data[start:start+width] for start in range(0, len(data) - (width - 1)))
    return map(partial(median, axis=0), slices)

def tupleReduce(iterable, *functions):
    def reduction(acc, val):
        return tuple(f(a, v) for f, a, v in zip(functions, acc, val))
    return reduce(reduction, iterable)

def getThresholds(data, thresholds):
    from itertools import tee

    data_range = tupleReduce(zip(*tee(data, 2)), min, max)
    return (data_range[0] + (data_range[1] - data_range[0]) * thresholds[0],
            data_range[0] + (data_range[1] - data_range[0]) * thresholds[1])

def makeSegments(y, x=None):
    from numpy import arange, concatenate

    if x is None:
        x = arange(len(y))
    points = array([x, y]).T.reshape(-1, 1, 2)
    return concatenate([points[:-1], points[1:]], axis=1)

def categorize(iterable, on_threshold, off_threshold):
    on = False
    for i in iterable:
        if on and i < off_threshold:
            on = False
        elif not on and i > on_threshold:
            on = True
        yield on

def blinkTimes(iterable):
    count = 0
    for i in iterable:
        if not i and count > 0:
            yield count
        count = (count + 1) if i else 0

if __name__ == '__main__':
    from argparse import ArgumentParser

    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.collections import LineCollection

    from numpy import percentile, concatenate, asarray
    from random import sample

    parser = ArgumentParser(description="Extract points from a video.")
    parser.add_argument("image", type=str, help="The video to process.")
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
    parser.add_argument("--on-threshold", type=float, nargs=2, default=(0.5, 0.3),
                        help="The fraction of the maximum value a spot has to rise above to be 'on'")

    args = parser.parse_args()
    image = imread(args.image).astype('float32')

    #thresholds = getThresholds(image, args.on_threshold)

    proj = np_max(image, axis=0) / mean(image, axis=0)
    peaks = findBlobs(proj, scales=range(*args.spot_size),
                      threshold=args.blob_threshold, max_overlap=args.max_overlap)

    # Exclude partial ROIs on edge
    peaks = filter(partial(peakEnclosed, shape=proj.shape, expansion=args.expansion), peaks)
    peaks = asarray(list(peaks))
    rois = list(map(partial(extract, image=image, expansion=args.expansion), peaks))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(proj, cmap=get_cmap('gray'), vmax=percentile(proj, 99.5))
    ax.scatter(peaks[:, 2], peaks[:, 1], s=peaks[:, 0] * 20,
               facecolors='None', edgecolors='g')
    ax.set_ylim(0, proj.shape[1])
    ax.set_yticks([])
    ax.set_xlim(0, proj.shape[0])
    ax.set_xticks([])
    ax.set_title("max/mean intensity")

    fig = plt.figure()
    ntraces = 5
    samples = list(sample(rois, ntraces))
    vmin = min(map(np_min, samples))
    vmax = max(map(np_max, samples))
    plt_indices = range(1, len(samples) * 2, 2)
    for i, roi in zip(plt_indices, samples):
        trace = np_fromiter(map(np_max, rollingMedian(roi, 3)),
                            dtype='float', count=len(image) - 2)
        thresholds = getThresholds(trace, args.on_threshold)
        on = np_fromiter(map(int, categorize(trace, *thresholds)),
                         dtype='uint8', count=len(trace))

        cmap = ListedColormap(['r', 'b'])
        norm = BoundaryNorm([-float('inf'), 0.5, float('inf')], cmap.N)

        lc = LineCollection(makeSegments(trace), cmap=cmap, norm=norm)
        lc.set_array(on)

        ax = fig.add_subplot(ntraces*2, 1, i)
        ax.set_ylabel("max. intensity")
        ax.set_xlabel("frame")
        ax.add_collection(lc)
        ax.set_xlim(0, len(trace))
        ax.set_ylim(vmin, vmax)
        ax.axhline(y=thresholds[0], color='green')
        ax.axhline(y=thresholds[1], color='red')

        ax = fig.add_subplot(ntraces*2, 1, i+1)
        rowsize = 100
        show = concatenate([concatenate(roi[i:i+rowsize], axis=-1)
                            for i in range(0, len(roi), rowsize)], axis=-2)
        ax.imshow(show, vmax=vmax, vmin=vmin,
                  cmap=get_cmap('gray'), interpolation="nearest")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig = plt.figure()

    on_times = []
    blink_times = []
    for roi in rois:
        trace = np_fromiter(map(np_max, rollingMedian(roi, 3)),
                            dtype='float', count=len(image) - 2)
        thresholds = getThresholds(trace, args.on_threshold)
        on = np_fromiter(map(int, categorize(trace, *thresholds)),
                         dtype='uint8', count=len(trace))
        on_times.append(np_sum(on))
        blink_times.extend(blinkTimes(on))

    ax = fig.add_subplot(2, 1, 1)
    ax.set_title("On times")
    ax.hist(on_times, 30)
    ax = fig.add_subplot(2, 1, 2)
    ax.set_title("Blink times")
    ax.hist(blink_times, 30)

    plt.show()
