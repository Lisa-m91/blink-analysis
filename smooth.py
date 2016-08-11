#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology

def ball(r, ndim):
    from math import ceil, pi
    from operator import sub
    from functools import reduce, partial

    size = ceil(2*r)
    xs = np.ogrid[(slice(-size/2, size/2, (size * 1j)),) * ndim]
    xs = map(lambda x: np.power(x, 2), xs)
    return np.sqrt(reduce(sub, xs, r ** 2).clip(0)) - r

if __name__ == "__main__":
    from argparse import ArgumentParser
    from tifffile import TiffFile, TiffWriter

    parser = ArgumentParser(description="Background correct an image using the rolling-ball method")
    parser.add_argument("tif", type=TiffFile)
    parser.add_argument("outfile", type=TiffWriter)
    parser.add_argument("radius", type=float)
    parser.add_argument("--invert", action="store_true", type=bool)

    args = parser.parse_args()
    with args.tif:
        data = args.tif.asarray()

    structure = ball(args.radius, data.ndim)
    function = morphology.grey_dilation if args.invert else morphology.grey_erosion
    data = function(data, structure=structure) + args.radius

    with args.outfile:
        args.outfile.save(data.astype('float32'))
