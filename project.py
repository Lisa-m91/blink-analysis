#!/usr/bin/env python3

def rollingMedian(data, width, pool=None):
    from functools import partial
    from itertools import tee, islice, count
    from numpy import stack, median

    slices = tee(data, width)
    slices = map(lambda data, start: islice(data, start, None), slices, count())
    slices = map(stack, zip(*slices))

    if pool is None:
        return map(partial(median, axis=0), slices)
    else:
        return pool.imap(partial(median, axis=0), slices)

def tiffChain(series):
    from tifffile.tifffile import TiffPageSeries
    from itertools import chain

    return chain.from_iterable(map(TiffPageSeries.asarray, series))

def multiReduce(functions, iterable):
    def reducer(accs, value):
        return tuple(map(lambda f, a: f(a, value), functions, accs))
    return reduce(reducer, iterable)

class Counter:
    def __init__(self, value=0):
        self.value = value

    def __call__(self, acc, v):
        self.value += 1
        return self.value

if __name__ == "__main__":
    from argparse import ArgumentParser
    from tifffile import TiffFile, TiffWriter
    from pathlib import Path
    from contextlib import ExitStack
    from numpy import fmax
    from operator import add
    from multiprocessing import Pool
    from itertools import chain, islice
    from functools import reduce
    from enum import Enum

    class Projection(Enum):
        mean = add
        maximum = fmax
        max = fmax

    parser = ArgumentParser()
    parser.add_argument("tifs", nargs='+', type=TiffFile)
    parser.add_argument("--range", type=str, nargs=2, default=("start", "end"),
                        help="The frame to start the projection")
    parser.add_argument("--method", nargs=2, action='append', type=str, required=True)
    parser.add_argument("--filter-size", type=int, default=1,
                        help="The number of frames to running-median filter")

    args = parser.parse_args()
    start, end = args.range
    start = None if start == "start" else int(start)
    end = None if end == "end" else int(end)

    methods = tuple(Projection[m[0]] for m in args.method)
    functions = (Counter(),) + tuple(m.value for m in methods)
    outfiles = tuple(TiffWriter(m[1]) for m in args.method)

    pool = Pool()
    with ExitStack() as stack:
        for tif in args.tifs: stack.enter_context(tif)
        frames = islice(tiffChain(chain.from_iterable(tif.series for tif in args.tifs)), start, end)
        count, *projections = multiReduce(functions, rollingMedian(frames, args.filter_size, pool=pool))

    for method, projection, outfile in zip(methods, projections, outfiles):
        with outfile:
            if method is Projection.mean:
                outfile.save((projection / count).astype('float32'))
            else:
                outfile.save(projection.astype('float32'))
