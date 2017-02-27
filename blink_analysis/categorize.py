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

def masks(shape):
    slices = tuple(slice(s//4, -(s//4)) for s in shape)

    bg_mask = np.ones(shape, dtype='bool')
    bg_mask[slices] = False
    fg_mask = np.zeros(shape, dtype='bool')
    fg_mask[slices] = True

    return fg_mask, bg_mask

def smooth(on, smoothing=1):
    return on | binary_closing(on, structure=np.ones(smoothing, dtype="bool"))

def categorize(roi):
    fg_mask, bg_mask = masks(roi.shape[1:])
    signal = roi[:, fg_mask]
    background = roi[:, bg_mask]
    cutoff = 1 / len(roi)

    different = ttest_ind(signal, background, axis=1, equal_var=False).pvalue < cutoff
    higher = np.mean(signal, axis=1) > np.mean(background, axis=1)

    return different & higher

def run(args):
    with args.ROIs.open("rb") as roi_f, args.outfile.open("wb") as on_f:
        for on in map(partial(smooth, smoothing=args.smoothing),
                      map(categorize, loadAll(roi_f))):
            dump(on, on_f)

def image_grid(frames, ncols, fill=0):
    nframes, *shape = frames.shape
    nrows = nframes // ncols + (nframes % ncols != 0)
    grid = np.full([nrows * ncols] + shape, fill, dtype=frames.dtype)
    grid[:nframes] = frames
    grid = grid.reshape((nrows, ncols) + tuple(shape))

    height, width, *shape = shape
    return grid.swapaxes(1, 2).reshape((height * nrows, width * ncols) + tuple(shape))

def plot(args):
    import matplotlib
    if args.outfile is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    np.random.seed(args.seed)

    with args.ROIs.open("rb") as roi_f, args.categories.open("rb") as on_f:
        data = list(zip(loadAll(roi_f), loadAll(on_f)))
    idxs = np.random.choice(len(data), size=args.n, replace=False)

    fig, axs = plt.subplots(args.n, 1)
    for ax, (roi, on) in zip(axs, map(data.__getitem__, idxs)):
        roi = np.pad(roi, [(0, 0), (1, 1), (1, 1)], mode='constant')

        border_mask = np.ones(roi.shape[1:], dtype='bool')
        border_mask[(slice(1, -1),) * border_mask.ndim] = 0
        border_mask = border_mask * on.reshape((-1,) + (1,) * border_mask.ndim)
        roi[border_mask] = roi.max()

        ax.imshow(image_grid(roi, args.ncols), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    if args.outfile is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(str(args.outfile))

def main(args=None):
    from sys import argv
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description="Analyze single-particle traces.")
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("ROIs", type=Path, help="The pickled ROIs to process")
    run_parser.add_argument("outfile", type=Path, help="The file to write on/off data to")
    run_parser.add_argument("--smoothing", type=int, default=1,
                            help="The number of 'off' frames required to end a blink")
    run_parser.set_defaults(func=run)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("ROIs", type=Path, help="The ROIs to plot")
    plot_parser.add_argument("categories", type=Path,
                             help="The saved categories to visualize")
    plot_parser.add_argument("outfile", nargs='?', type=Path, default=None,
                             help="Where to save the plot (omit to display)")
    plot_parser.add_argument("-n", type=int, default=5,
                             help="The number of traces to plot")
    plot_parser.add_argument("--ncols", type=int, default=80,
                             help="The number of columns to stack traces into")
    plot_parser.add_argument("--seed", type=int, default=None,
                             help="The random seed for selecting traces")
    plot_parser.set_defaults(func=plot)

    args = parser.parse_args(argv[1:] if args is None else args)
    args.func(args)

if __name__ == "__main__":
    main()
