#!/usr/bin/env python3
from functools import partial
from itertools import tee

def makeRegion(peak, size):
    starts = peak - size
    ends = peak + size + 1
    return tuple(map(slice, starts, ends))

def makeConsecutiveSlices(lengths):
    from itertools import chain, tee, accumulate

    ends = accumulate(lengths)
    starts, ends = tee(ends, 2)
    starts = chain((0,), starts)
    return map(slice, starts, ends)

def sliceSeries(seriess, start=0, end=None):
    acc = 0
    for series in seriess:
        l = series.shape[0]
        if end is not None and end < acc:
            return
        stop = min(l, end-acc) if end is not None else None
        if start > (acc + l):
            continue
        begin = max(0, start-acc)
        acc += l
        yield series.asarray()[begin:stop]

def extractAll(peaks, series, size=1, start=0, end=None):
    from numpy import empty, around, array
    from tifffile.tifffile import TiffPageSeries

    dtype = series[0].dtype
    start, end, _ = slice(start, end).indices(sum(s.shape[0] for s in series))
    nframes = end - start

    peaks = array(peaks)
    ndim = peaks.shape[1]

    shape = (nframes,) + (size * 2 + 1,) * ndim
    rois = empty((len(peaks),) + shape, dtype=dtype)
    regions = list(map(partial(makeRegion, size=size), peaks))

    data = sliceSeries(series, start, end)
    data, lens = tee(data, 2)
    lens = map(len, lens)
    slices = makeConsecutiveSlices(lens)
    for s, frames in zip(slices, data):
        for roi, region in zip(rois, regions):
            roi[s] = frames[(slice(None),) + region]
    return rois

if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    from itertools import chain
    from pickle import load, dump, HIGHEST_PROTOCOL
    dump = partial(dump, protocol=HIGHEST_PROTOCOL)
    from sys import stdout
    from functools import partial

    from tifffile import TiffFile

    parser = ArgumentParser(description="Extract points from a video.")

    parser.add_argument("peaks", type=Path, help="The locations of peaks to pick")
    parser.add_argument("images", nargs='+', type=partial(TiffFile, multifile=False),
                        help="The video to process.")
    parser.add_argument("--size", type=int, default=2,
                        help="The radius of the spot to extract.")
    parser.add_argument("--range", type=str, nargs=2, default=("start", "end"),
                        help="The range of frames to extract.")

    args = parser.parse_args()
    series = chain.from_iterable(tif.series for tif in args.images)
    image_shape = args.images[0].series[0].asarray().shape[1:]

    with args.peaks.open("rb") as f:
        peaks = load(f)

    start, end = args.range
    start = 0 if start == "start" else int(start)
    end = None if end == "end" else int(end)
    rois = extractAll(peaks, list(series), size=args.size, start=start, end=end)

    for roi in rois:
        dump(roi, stdout.buffer)
