#!/usr/bin/env python3
from itertools import chain
from functools import partial
import numpy as np
from scipy.stats import ttest_ind
from pickle import load, dump, HIGHEST_PROTOCOL
dump = partial(dump, protocol=HIGHEST_PROTOCOL)

def loadAll(f):
    while True:
        try:
            yield load(f)
        except EOFError:
            break

mask = np.zeros((9, 9), dtype='bool')
mask[(slice(2, -2),) * mask.ndim] = True

def categorize(roi):
    signal = roi[:, mask]
    background = roi[:, ~mask]
    return (ttest_ind(signal, background, axis=1, equal_var=False).pvalue
            < (1 / len(roi)))

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description="Analyze single-particle traces.")
    parser.add_argument("ROIs", type=Path,
                        help="The pickled ROIs to process")
    parser.add_argument("outfile", type=Path,
                        help="The file to write on/off data to")
    args = parser.parse_args()

    with args.ROIs.open("rb") as roi_f, args.outfile.open("wb") as on_f:
        for on in map(categorize, loadAll(roi_f)):
            dump(on, on_f)
