#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology

def ball(r, ndim):
    return ellipse((r,) * (ndim + 1))

def ellipse(rs):
    from functools import reduce
    from operator import sub

    z, *rs = rs
    rs = np.asarray(rs)
    size = np.ceil(2 * rs)
    xs = np.ogrid[tuple(map(slice, -size/2, size/2, size * 1j))]
    xs = map(lambda x, r: np.power(x, 2) / r, xs, rs ** 2)
    ys = np.sqrt((reduce(sub, xs, 1) * z ** 2).clip(0))
    return ys - ys.max(), ys > 0

def smooth(a, radius, invert=False):
    function = morphology.grey_closing if invert else morphology.grey_opening
    structure, footprint = ball(radius, a.ndim)
    return function(a, structure=structure, footprint=footprint)

def main(args=None):
    from sys import argv
    from argparse import ArgumentParser
    from tifffile import TiffFile, TiffWriter
    from functools import partial

    parser = ArgumentParser(description="Background correct an image using the rolling-ball method")
    parser.add_argument("tif", type=partial(TiffFile, multifile=False))
    parser.add_argument("outfile", type=TiffWriter)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args(argv[1:] if args is None else args)

    with args.tif:
        data = args.tif.asarray().astype('float32')

    with args.outfile:
        args.outfile.save(smooth(data, args.radius, args.invert).astype('float32'))

if __name__ == "__main__":
    main()
