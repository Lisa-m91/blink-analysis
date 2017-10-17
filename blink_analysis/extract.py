#!/usr/bin/env python3
from functools import partial
from itertools import tee
from pathlib import Path

import click
from tiffutil.util import SingleTiffFile

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

    dtype = series[0].dtype
    start, end, _ = slice(start, end).indices(sum(s.shape[0] for s in series))
    nframes = end - start

    peaks = array(peaks)
    try:
        ndim = peaks.shape[1]
    except IndexError:
        return array([], dtype=dtype)

    shape = (nframes,) + (size * 2 + 1,) * ndim
    rois = empty((len(peaks),) + shape, dtype=dtype)
    regions = list(map(partial(makeRegion, size=size), peaks[:, ::-1]))

    data = sliceSeries(series, start, end)
    data, lens = tee(data, 2)
    lens = map(len, lens)
    slices = makeConsecutiveSlices(lens)
    for s, frames in zip(slices, data):
        for roi, region in zip(rois, regions):
            roi[s] = frames[(slice(None),) + region]
    return rois

@click.command("extract")
@click.argument("peaks", type=Path)
@click.argument("images", nargs=-1, type=SingleTiffFile)
@click.argument("output", type=Path)
@click.option("--size", type=int, default=2, help="The radius of the spot to extract")
@click.option("--start", type=int, default=0, help="The first frame to extract")
@click.option("--end", type=int, default=None, help="The last frame to extract")
def main(peaks, images, output, size=2, start=0, end=None):
    from itertools import chain
    from pickle import load, dump, HIGHEST_PROTOCOL
    dump = partial(dump, protocol=HIGHEST_PROTOCOL)
    import csv

    series = chain.from_iterable(tif.series for tif in images)
    image_shape = images[0].series[0].asarray().shape[1:]

    with peaks.open("r") as f:
        reader = csv.reader(f)
        peaks = list(map(list, map(partial(map, int), reader)))

    rois = extractAll(peaks, list(series), size=size, start=start, end=end)

    with output.open("wb") as f:
        for roi in rois:
            dump(roi, f)
