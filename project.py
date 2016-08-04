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

if __name__ == "__main__":
    from argparse import ArgumentParser
    from tifffile import TiffFile, TiffWriter
    from pathlib import Path
    from contextlib import ExitStack
    from numpy import fmax
    from multiprocessing import Pool
    from itertools import chain
    from functools import reduce

    parser = ArgumentParser()
    parser.add_argument("tifs", nargs='+', type=TiffFile, help="The files to project")
    parser.add_argument("outfile", type=TiffWriter, help="The output filename")
    parser.add_argument("--filter-size", type=int, default=1,
                        help="The number of frames to running-median filter")
    args = parser.parse_args()

    pool = Pool()

    with ExitStack() as stack, args.outfile as outfile:
        for tif in args.tifs: stack.enter_context(tif)
        frames = tiffChain(chain.from_iterable(tif.series for tif in args.tifs))
        projection = reduce(fmax, rollingMedian(frames, args.filter_size, pool=pool))
        outfile.save(projection)
