#!/usr/bin/env python3
from functools import partial, reduce
from pickle import dump, HIGHEST_PROTOCOL

from tifffile import TiffFile

from blob import findBlobs

from numpy import amax, median, percentile, empty, clip, fmax

def extract(peak, image, expansion=1, excludes=()):
    scale, *pos = peak
    scale = round(int(scale) * expansion)
    roi = (slice(None),) + tuple(slice(p - scale, p + scale + 1) for p in pos)
    return image[roi]

def excludeFrames(image, exclude=()):
    from numpy import arange, zeros
    import operator as op

    idxs = arange(len(image))
    excluded = map(lambda ex: (idxs >= ex.start) & (idxs < ex.end), exclude)
    excluded = reduce(op.or_, excluded, zeros(len(image), dtype='bool'))
    return image[~excluded]

def peakEnclosed(peak, shape, expansion=1):
    scale, *pos = peak
    scale = scale * expansion
    return (all(scale <= p for p in pos) and
            all(scale < (s - p) for s, p in zip(shape, pos)))

def rollingMedian(data, width, pool=None):
    try:
        # Array
        slices = (data[start:start+width] for start in range(0, len(data) - (width - 1)))
    except TypeError:
        # Iterable
        from itertools import tee, islice, count
        from numpy import stack

        slices = tee(data, width)
        slices = map(lambda data, start: islice(data, start, None), slices, count())
        slices = map(stack, zip(*slices))

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

def tiffChain(*series):
    from itertools import chain
    from tifffile.tifffile import TiffPageSeries

    return chain.from_iterable(map(TiffPageSeries.asarray, series))

class Range:
    def __init__(self, start, end=None):
        if end is None:
            end = value + 1
        self.start = int(start)
        self.end = int(end)

    @classmethod
    def fromString(cls, string):
        return cls(*string.split('-'))

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
    parser.add_argument("--exclude", type=Range.fromString, nargs='*', default=(),
                        help="Any frames to exclude from the extracted sequence")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize for per-pixel percentile (for hot spots)")

    args = parser.parse_args()
    p = Pool()

    series = (tif.series[0] for tif in args.images)

    if args.normalize:
        raw = tiffConcat(*series)
        raw = excludeFrames(raw, exclude=args.exclude)
        background = percentile(raw, 15.0, axis=0)
        proj = reduce(fmax, rollingMedian(raw, 3, pool=p)) / background
    else:
        proj = reduce(fmax, rollingMedian(tiffChain(*series), 3, pool=p))

    peaks = findBlobs(proj, scales=range(*args.spot_size),
                      threshold=args.threshold, max_overlap=args.max_overlap)

    peaks = filter(partial(peakEnclosed, shape=proj.shape, expansion=args.expansion), peaks)
    rois = map(partial(extract, image=raw, expansion=args.expansion), peaks)

    for roi in rois:
        dump(roi, stdout.buffer, protocol=HIGHEST_PROTOCOL)
