#!/usr/bin/env python3
from itertools import chain
from functools import partial, reduce
import numpy as np
from scipy.stats import ttest_ind
from scipy.ndimage.morphology import binary_closing, binary_opening
from pickle import load, dump, HIGHEST_PROTOCOL
dump = partial(dump, protocol=HIGHEST_PROTOCOL)
from pathlib import Path
import operator as op

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
            dump(on.astype('uint8'), on_f)

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
@click.option("--size", type=(float, float), default=(3, 10),
              help="The size (width height, in inches) of the figure")
@click.option("--length", type=int, default=None,
              help="The number of frames to plot")
@click.option("-n", type=(int, int), default=(5, 1),
              help="The number of traces to plot (rows cols)")
@click.option("--ncols", type=int, default=80,
              help="The number of columns to stack traces into")
@click.option("--seed", type=int, default=None, help="The random seed for selecting traces")
def plot(rois, categories, length=None, outfile=None,
         size=(3, 10), n=(5, 1), ncols=80, seed=None):
    import matplotlib
    if outfile is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    np.random.seed(seed)

    with rois.open("rb") as roi_f, categories.open("rb") as on_f:
        getter = op.itemgetter(slice(None, length))
        data = list(zip(map(getter, loadAll(roi_f)), map(getter, loadAll(on_f))))

    fig, axs = plt.subplots(*n, figsize=size, squeeze=False)
    axs = axs.ravel()

    n = reduce(op.mul, n)
    idxs = np.random.choice(len(data), size=min(n, len(data)), replace=False)
    for ax, (roi, on) in zip(axs, map(data.__getitem__, idxs)):
        roi = np.pad(roi, [(0, 0), (1, 1), (1, 1)], mode='constant')
        ax.imshow(image_grid(roi, ncols), cmap='gray')

        for frame, state in enumerate(on):
            row = frame // ncols
            col = frame % ncols
            pos = (row * roi.shape[-2], col * roi.shape[-1])
            ax.add_patch(patches.Rectangle(
                pos[::-1], *[s-1 for s in roi.shape[-2:]],
                fill=False, edgecolor="C{}".format(state),
                linewidth=0.5
            ))

        ax.set_xticks([])
        ax.set_yticks([])

    if outfile is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(str(outfile))
