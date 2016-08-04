#!/usr/bin/env python3
from functools import partial

def makeRegion(peak, size):
    starts = peak - size
    ends = peak + size + 1
    return tuple(map(slice, starts, ends))

def makeConsecutiveSlices(lengths):
    from itertools import islice, chain, tee, accumulate

    ends = accumulate(lengths)
    starts, ends = tee(ends, 2)
    starts = chain((0,), islice(starts, 1, None))
    ends = chain(ends, (None,))
    return map(slice, starts, ends)

def extractAll(peaks, series, size=1):
    from numpy import empty, around, array
    from tifffile.tifffile import TiffPageSeries

    dtype = series[0].dtype
    nframes = sum(s.shape[0] for s in series)

    peaks = array(peaks)
    ndim = peaks.shape[1]

    shape = (nframes,) + (size * 2 + 1,) * ndim
    rois = empty((len(peaks),) + shape, dtype=dtype)
    regions = map(partial(makeRegion, size=size), peaks)

    data = map(TiffPageSeries.asarray, series)
    slices = makeConsecutiveSlices(s.shape[0] for s in series)
    for i, (s, frames) in enumerate(zip(slices, data)):
        for roi, region in zip(rois, regions):
            roi[s] = frames[(slice(None),) + region]
    return rois

def peakEnclosed(peaks, shape, size=1):
    from numpy import asarray

    shape = asarray(shape)
    return ((size <= peaks).all(axis=-1) & (size < (shape - peaks)).all(axis=-1))

if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    from itertools import chain
    from pickle import load, dump, HIGHEST_PROTOCOL
    dump = partial(dump, protocol=HIGHEST_PROTOCOL)
    from sys import stdout

    from tifffile import TiffFile

    parser = ArgumentParser(description="Extract points from a video.")

    parser.add_argument("peaks", type=Path, help="The locations of peaks to pick")
    parser.add_argument("images", nargs='+', type=TiffFile, help="The video to process.")
    parser.add_argument("--size", type=int, default=2,
                        help="The radius of the spot to extract.")

    args = parser.parse_args()
    series = chain.from_iterable(tif.series for tif in args.images)
    image_shape = args.images[0].series[0].asarray().shape[1:]

    with args.peaks.open("rb") as f:
        peaks = load(f)
    peaks = peaks[peakEnclosed(peaks, shape=image_shape, size=2)]
    rois = extractAll(peaks, list(series), size=args.size)

    for roi in rois:
        dump(roi, stdout.buffer)
