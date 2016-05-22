#!/usr/bin/env python3
from functools import partial
from pickle import dump, HIGHEST_PROTOCOL

from tifffile import TiffFile

from blob import findBlobs

from numpy import amax, median, percentile, empty, clip

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

def arrangeSubplots(n):
    from math import ceil, sqrt

    nrows = ceil(sqrt(n))
    ncols = ceil(n / nrows)

    return nrows, ncols

def tiffConcat(*series):
    from itertools import accumulate, tee, islice, starmap, chain
    from numpy import empty
    from tifffile.tifffile import TiffPageSeries

    lengths = list(map(lambda img: img.shape[0], series))
    offsets = accumulate(chain((0,), lengths))

    shape = (sum(lengths),) + series[0].shape[1:]
    result = empty(shape, dtype=series[0].dtype)

    offsets, ends = tee(offsets, 2)
    ends = islice(ends, 1, None)
    slices = starmap(slice, zip(offsets, ends))

    for img, s in zip(map(TiffPageSeries.asarray, series), slices):
        result[s] = img
    return result

if __name__ == '__main__':
    from argparse import ArgumentParser
    from sys import stdout

    from numpy import percentile
    from multiprocessing import Pool

    parser = ArgumentParser(description="Extract points from a video.")
    parser.add_argument("images", nargs='+', type=TiffFile, help="The video to process.")
    parser.add_argument("--spot-size", nargs=2, type=int, default=(2, 5),
                        help="The range of spot sizes to search for.")
    parser.add_argument("--max-overlap", type=float, default=0.05,
                        help="The maximum amount of overlap before spots are merged.")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="The threshold value for the LOG blob detection.")
    parser.add_argument("--expansion", type=float, default=1,
                        help="The amount to expand detected points by.")

    args = parser.parse_args()
    p = Pool()

    raw = tiffConcat(*(tif.series[0] for tif in args.images))
    background = percentile(raw, 15.0, axis=0)

    image = empty((len(raw) - 2,) + raw.shape[1:], dtype='float32')
    for i, frame in enumerate(rollingMedian(raw, 3, pool=p)):
        image[i] = frame / background
    del raw

    proj = amax(image, axis=0)
    peaks = findBlobs(proj, scales=range(*args.spot_size),
                        threshold=args.threshold, max_overlap=args.max_overlap)

    peaks = filter(partial(peakEnclosed, shape=proj.shape, expansion=args.expansion), peaks)
    rois = map(partial(extract, image=image, expansion=args.expansion), peaks)

    for roi in rois:
        dump(roi, stdout.buffer, protocol=HIGHEST_PROTOCOL)
