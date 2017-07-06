#!/usr/bin/env python3
from itertools import chain
from functools import partial
import numpy as np
from scipy.stats import ttest_ind
from scipy.ndimage.morphology import binary_closing, binary_opening
from pickle import load, dump, HIGHEST_PROTOCOL
dump = partial(dump, protocol=HIGHEST_PROTOCOL)
from pathlib import Path

import click

@click.group("categorize")
def main():
    pass

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

def smooth(on, smoothing=(1, 1)):
    on = on | binary_closing(on, structure=np.ones(smoothing[0], dtype="bool"))
    on = on & binary_opening(on, structure=np.ones(smoothing[1], dtype="bool"))
    return on

def categorize(roi, factor=4.0):
    fg_mask, bg_mask = masks(roi.shape[1:])
    signal = roi[:, fg_mask]
    background = roi[:, bg_mask]

    axes = tuple(range(1, signal.ndim))
    peaks = np.amax(signal, axis=axes)
    return peaks > (np.mean(background, axis=axes)
                    + np.std(background, axis=axes) * factor)

@main.command()
@click.argument("rois", type=Path)
@click.argument("outfile", type=Path)
@click.option("--smoothing", nargs=2, type=int, default=(1, 1),
              help="The number of 'off'/'on' frames required to end/begin a blink")
@click.option("--factor", type=float, default=4.0,
              help="The minimum SNR to categorize an on-frame")
def run(rois, outfile, smoothing=(1, 1), factor=4.0):
    with rois.open("rb") as roi_f, outfile.open("wb") as on_f:
        for on in map(partial(smooth, smoothing=smoothing),
                      map(partial(categorize, factor=factor), loadAll(roi_f))):
            dump(on, on_f)

def image_grid(frames, ncols, fill=0):
    nframes, *shape = frames.shape
    nrows = nframes // ncols + (nframes % ncols != 0)
    grid = np.full([nrows * ncols] + shape, fill, dtype=frames.dtype)
    grid[:nframes] = frames
    grid = grid.reshape((nrows, ncols) + tuple(shape))

    height, width, *shape = shape
    return grid.swapaxes(1, 2).reshape((height * nrows, width * ncols) + tuple(shape))

@main.command()
@click.argument("rois", type=Path)
@click.argument("categories", type=Path)
@click.option("--outfile", type=Path, default=None,
              help="Where to save the plot (omit to display)")
@click.option("-n", type=int, default=5, help="The number of traces to plot")
@click.option("--ncols", type=int, default=80,
              help="The number of columns to stack traces into")
@click.option("--seed", type=int, default=None, help="The random seed for selecting traces")
def plot(rois, categories, outfile=None, n=5, ncols=80, seed=None):
    import matplotlib
    if outfile is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    np.random.seed(seed)

    with rois.open("rb") as roi_f, categories.open("rb") as on_f:
        data = list(zip(loadAll(roi_f), loadAll(on_f)))
    idxs = np.random.choice(len(data), size=n, replace=False)

    fig, axs = plt.subplots(n, 1)
    for ax, (roi, on) in zip(axs, map(data.__getitem__, idxs)):
        roi = np.pad(roi, [(0, 0), (1, 1), (1, 1)], mode='constant')

        border_mask = np.ones(roi.shape[1:], dtype='bool')
        border_mask[(slice(1, -1),) * border_mask.ndim] = 0
        border_mask = border_mask * on.reshape((-1,) + (1,) * border_mask.ndim)
        roi[border_mask] = roi.max()

        ax.imshow(image_grid(roi, ncols), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    if outfile is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(str(outfile))
