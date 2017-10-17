#!/usr/bin/env python3
from itertools import chain
from functools import partial, reduce
import numpy as np
from scipy.stats import ttest_ind
from scipy.ndimage.morphology import binary_closing, binary_opening
from scipy.ndimage import gaussian_laplace
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

def categorize(roi, factor=400, sigma=2.0):
    fg_mask, bg_mask = masks(roi.shape[1:])
    signal = roi[:, fg_mask]
    background = roi[:, bg_mask]

    different = ttest_ind(signal, background, axis=1, equal_var=False).pvalue < factor
    higher = np.mean(signal, axis=1) > np.mean(background, axis=1)
    return different & higher


@main.command()
@click.argument("rois", type=Path)
@click.argument("outfile", type=Path)
@click.option("--smoothing", nargs=2, type=int, default=(1, 1),
              help="The number of 'off'/'on' frames required to end/begin a blink")
@click.option("--factor", type=float, default=400,
              help="The minimum peak intensity to categorize an on-frame")
@click.option("--sigma", type=float, default=2,
              help="The minimum SNR to categorize an on-frame")
def run(rois, outfile, smoothing=(1, 1), factor=400, sigma=2):
    with rois.open("rb") as roi_f, outfile.open("wb") as on_f:
        smooth_ = partial(smooth, smoothing=smoothing)
        categorize_ = partial(categorize, factor=factor, sigma=sigma)
        for on in map(smooth_, map(categorize_, loadAll(roi_f))):
            dump(on.astype('uint8'), on_f)

def image_grid(frames, ncols, fill=0):
    nframes, *shape = frames.shape
    nrows = nframes // ncols + (nframes % ncols != 0)
    grid = np.full([nrows * ncols] + shape, fill, dtype=frames.dtype)
    grid[:nframes] = frames
    grid = grid.reshape((nrows, ncols) + tuple(shape))

    height, width, *shape = shape
    return grid.swapaxes(1, 2).reshape((height * nrows, width * ncols) + tuple(shape))

default_colors = list(map("C{}".format, range(10)))
@main.group()
@click.option("--output", type=Path, default=None,
              help="Where to save the plot (omit to display)")
@click.option("--figsize", type=(float, float), default=(3, 10),
              help="The size (width height, in inches) of the figure")
@click.option("--length", type=int, default=None,
              help="The number of frames to plot")
@click.option("-n", type=(int, int), default=(5, 1),
              help="The number of traces to plot (rows cols)")
@click.option("--filter-state", type=int, multiple=True,
              help="Filter out traces with no matching frames")
@click.option("--color", type=str, default=default_colors, multiple=True,
              help="The colors to plot each state in")
@click.option("--seed", type=int, default=None,
              help="The random seed for selecting traces")
@click.pass_context
def plot(ctx, length=None, output=None, figsize=(3, 10),
         n=(5, 1), filter_state=False, color=default_colors, seed=None):
    ctx.obj = {'LENGTH': length, 'COLORS': np.asarray(color), 'N': reduce(op.mul, n),
               'FILTER': filter_state}

    if output is not None:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(*n, figsize=figsize, squeeze=False, sharex=True, sharey=True)
    ctx.obj['AXS'] = axs.ravel()
    ctx.obj['FIG'] = fig

    rng = np.random.RandomState(seed)
    ctx.obj['RNG'] = rng

@plot.resultcallback()
def save(fig, length=None, output=None, figsize=(3, 10),
         n=(5, 1), filter_state=False, color=default_colors, seed=None):
    import matplotlib.pyplot as plt
    fig.set_size_inches(*figsize)
    fig.tight_layout()

    if output:
        # dpi=600 is workaround for matplotlib#8981
        fig.savefig(str(output), dpi=600)
    else:
        plt.show()

def load_data(files, length=None, filter_states=()):
    rois, categories = [], []
    getter = op.itemgetter(slice(None, length))
    for roi_f, category_f in zip(files[0::2], files[1::2]):
        with roi_f.open("rb") as roi_f, category_f.open("rb") as category_f:
            rois.extend(map(getter, loadAll(roi_f)))
            categories.extend(map(getter, loadAll(category_f)))
    if filter_states:
        keep = list(map(np.any, [[c == s for s in filter_states] for c in categories]))
        return np.asarray(rois)[keep], np.asarray(categories)[keep]
    else:
        return np.asarray(rois), np.asarray(categories)

@plot.command()
@click.argument("files", type=Path, nargs=-1)
@click.option("--ncols", type=int, default=80,
              help="The number of columns to stack traces into")
@click.pass_context
def grid(ctx, files, ncols=80):
    import matplotlib.patches as patches

    rois, categories = load_data(files, ctx.obj['LENGTH'], ctx.obj['FILTER'])
    if not len(rois):
        return ctx.obj['FIG']

    idxs = ctx.obj['RNG'].choice(
        len(rois), size=min(ctx.obj['N'], len(rois)), replace=False
    )
    vmin = rois[idxs].min()
    vmax = rois[idxs].max()
    for ax, roi, on in zip(ctx.obj['AXS'], rois[idxs], categories[idxs]):
        roi = np.pad(roi, [(0, 0), (1, 1), (1, 1)], mode='constant')
        ax.imshow(
            image_grid(roi, ncols), vmin=vmin, vmax=vmax,
            cmap='gray', interpolation='nearest'
        )

        for frame, state in enumerate(on):
            row = frame // ncols
            col = frame % ncols
            pos = (row * roi.shape[-2], col * roi.shape[-1])
            ax.add_patch(patches.Rectangle(
                pos[::-1], *[s-1 for s in roi.shape[-2:]],
                fill=False, edgecolor=ctx.obj['COLORS'][state], linewidth=0.5
            ))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    return ctx.obj['FIG']

@plot.command()
@click.argument("files", type=Path, nargs=-1)
@click.pass_context
def traces(ctx, files):
    rois, categories = load_data(files, ctx.obj['LENGTH'], ctx.obj['FILTER'])
    if not len(rois):
        return ctx.obj['FIG']

    idxs = ctx.obj['RNG'].choice(
        len(rois), size=min(ctx.obj['N'], len(rois)), replace=False
    )
    for ax, roi, on in zip(ctx.obj['AXS'], rois[idxs], categories[idxs]):
        fg_mask, bg_mask = masks(roi.shape[1:])
        trace = np.mean(roi[:, fg_mask], axis=1)
        ax.plot(trace, linewidth=0.5, color='black', alpha=0.5)
        ax.scatter(
            np.arange(len(trace)), trace, facecolor=ctx.obj['COLORS'][on], s=1.5,
            edgecolor='none', marker='o'
        )

        bg_trace = np.mean(roi[:, bg_mask], axis=1)
        ax.plot(bg_trace, linewidth=0.5, color='black', alpha=0.3)
        ax.set_ylabel("intensity")
    ctx.obj['AXS'][-1].set_xlabel("frame")
    return ctx.obj['FIG']
