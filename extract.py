#!/usr/bin/env python3
from functools import partial, reduce
from itertools import chain
from tifffile import imread

from blob import findBlobs

from numpy import std, mean, max as np_max, min as np_min, median, array

def extract(peak, image, expansion=1):
    scale, *pos = peak
    scale = scale * expansion
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

def getThresholds(image, thresholds):
    medians = rollingMedian(image, 3)
    image_range = tupleReduce(map(lambda x: (np_max(x), np_max(x)), medians), min, max)
    return (image_range[0] + (image_range[1] - image_range[0]) * thresholds[0],
            image_range[0] + (image_range[1] - image_range[0]) * thresholds[1])

def categorize(iterable, on_threshold, off_threshold):
    on = False
    for i in iterable:
        if on and i < off_threshold:
            on = False
        elif not on and i > on_threshold:
            on = True
        yield on


if __name__ == '__main__':
    from argparse import ArgumentParser

    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
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

        trace = np_max(roi, axis=tuple(range(1, roi.ndim)))
        thresholds = getThresholds(roi, args.on_threshold)
        colors = array([0.0, 1.0])
        colors = [colors[i] for i in map(int, categorize(trace, *thresholds))]

        # http://matplotlib.org/examples/pylab_examples/multicolored_line.html
        ax.scatter(trace)
        ax.set_ylabel("max. intensity")
        ax.set_xlabel("frame")
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

    plt.show()
