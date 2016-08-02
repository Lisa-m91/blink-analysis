#!/usr/bin/env python3
from functools import partial, reduce
from pickle import dump, HIGHEST_PROTOCOL

from tifffile import TiffFile

from blob import findBlobs

from numpy import amax, median, percentile, empty, clip, fmax

def extract(peak, image, expansion=1):
    scale, *pos = peak
    scale = round(int(scale) * expansion)
    roi = (slice(None),) + tuple(slice(p - scale, p + scale + 1) for p in pos)
    return image[roi]

def extractAll(peaks, series, expansion=1):
    from numpy import empty, around, array

    dtype = series[0].dtype
    nframes = sum(s.shape[0] for s in series)

    peaks = array(peaks)
    scales, poss = peaks[:, 0], peaks[:, 1:]
    ndim = poss.shape[1]
    scales = around(scales * expansion).astype('int')

    shapes = map(lambda s: (nframes,) + (s * 2 + 1,) * ndim, scales)
    rois = list(map(partial(empty, dtype=dtype), shapes))
    for i, frame in enumerate(tiffChain(*series)):
        for roi, region in zip(rois, map(partial(extract, image=frame[None, ...]), peaks)):
            roi[i] = region
    return rois

def excludeFrames(image, exclude=()):
    from numpy import arange, zeros
    import operator as op

    try:
        nframes = len(image)
    except TypeError:
        # Iterable
        return (frame for i, frame in enumerate(image)
                if not any(i in ex for ex in exclude))
    else:
        # Array
        idxs = arange(nframes)
        excluded = map(lambda ex: (idxs >= ex.start) & (idxs < ex.end), exclude)
        excluded = reduce(op.or_, excluded, zeros(len(image), dtype='bool'))
        return image[~excluded]

def peakEnclosed(peaks, shape, expansion=1):
    from numpy import asarray

    shape = asarray(shape)

    scales, poss = peaks[:, 0:1], peaks[:, 1:]
    scales = scales * expansion
    return ((scales <= poss).all(axis=-1) &
            (scales < (shape - poss)).all(axis=-1))

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

    def __contains__(self, i):
        return self.start <= i < self.end

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
    parser.add_argument("--plot", action="store_true",
                        help="Plot the picked spots on the projection.")
    parser.add_argument("--filter-length", type=int, default=3,
                        help="The number of frames to median-filter before projection")

    args = parser.parse_args()
    p = Pool()

    series = [tif.series[0] for tif in args.images]

    raw = tiffChain(*series)
    raw = excludeFrames(raw, exclude=args.exclude)
    proj = reduce(fmax, rollingMedian(raw, args.filter_length, pool=p))

    peaks = findBlobs(proj, scales=range(*args.spot_size),
                      threshold=args.threshold, max_overlap=args.max_overlap)
    peaks = peaks[peakEnclosed(peaks, shape=proj.shape, expansion=args.expansion)]

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        ax.imshow(proj, cmap="gray", interpolation="nearest", vmax=percentile(proj, 99.99))
        ax.scatter(*peaks.T[1:][::-1], marker="+", color="red")
        ax.set_xticks([])
        ax.set_yticks([])

    rois = extractAll(list(peaks), series, expansion=args.expansion)

    for roi in rois:
        dump(roi, stdout.buffer, protocol=HIGHEST_PROTOCOL)

    if args.plot:
        fig.tight_layout()
        plt.show()
