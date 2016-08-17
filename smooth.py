#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology

def ball(r, ndim):
    from math import ceil, pi, inf
    from operator import sub
    from functools import reduce, partial

    size = ceil(2*r)
    xs = np.ogrid[(slice(-size/2, size/2, (size * 1j)),) * ndim]
    xs = map(lambda x: np.power(x, 2), xs)
    ys = np.sqrt(reduce(sub, xs, r ** 2).clip(0))
    return ys - ys.max(), ys > 0

def smooth(a, radius, invert=False):
    function = morphology.grey_closing if invert else morphology.grey_opening
    structure, footprint = ball(radius, a.ndim)
    return function(a, structure=structure, footprint=footprint)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from tifffile import TiffFile, TiffWriter

    parser = ArgumentParser(description="Background correct an image using the rolling-ball method")
    parser.add_argument("tif", type=TiffFile)
    parser.add_argument("outfile", type=TiffWriter)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--invert", action="store_true")

    args = parser.parse_args()
    with args.tif:
        data = args.tif.asarray().astype('float32')

    with args.outfile:
        args.outfile.save(smooth(data, args.radius, args.invert).astype('float32'))
