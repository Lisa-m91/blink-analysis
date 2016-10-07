#!/usr/bin/env python3
from itertools import chain
from functools import partial
import numpy as np
from scipy.stats import ttest_ind
from scipy.ndimage.morphology import binary_closing
from pickle import load, dump, HIGHEST_PROTOCOL
dump = partial(dump, protocol=HIGHEST_PROTOCOL)

def loadAll(f):
    while True:
        try:
            yield load(f)
        except EOFError:
            break

bg_mask = np.ones((9, 9), dtype='bool')
bg_mask[(slice(2, -2),) * bg_mask.ndim] = False
fg_mask = np.zeros((9, 9), dtype='bool')
fg_mask[(slice(2, -2),) * fg_mask.ndim] = True

def smooth(on, smoothing=1):
    return on | binary_closing(on, structure=np.ones(smoothing, dtype="bool"))

def categorize(roi):
    signal = roi[:, fg_mask]
    background = roi[:, bg_mask]
    cutoff = 1 / len(roi)

    different = ttest_ind(signal, background, axis=1, equal_var=False).pvalue < cutoff
    higher = np.mean(signal, axis=1) > np.mean(background, axis=1)

    return different & higher

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description="Analyze single-particle traces.")
    parser.add_argument("ROIs", type=Path,
                        help="The pickled ROIs to process")
    parser.add_argument("outfile", type=Path,
                        help="The file to write on/off data to")
    parser.add_argument("--smoothing", type=int, default=1,
                        help="The number of 'off' frames required to end a blink")
    args = parser.parse_args()

    with args.ROIs.open("rb") as roi_f, args.outfile.open("wb") as on_f:
        for on in map(partial(smooth, smoothing=args.smoothing),
                      map(categorize, loadAll(roi_f))):
            dump(on, on_f)
