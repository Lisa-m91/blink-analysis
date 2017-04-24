# FRET Tools

This repository contains tools for the analysis of single-molecule blinking
data.

# Installation

No installation is required, this repository is just a group of self-contained
scripts. If desired, the package can be installed with `./setup.py install`.

## Requirements

[Python 3][python], [Scipy][scipy], [Numpy][numpy] and [tifffile][tifffile]. All
are available from [PyPI][pypi] and can be installed as described in the
[pip documentation][pip-install]. If necessary, a more up-to-date installer for
`tifffile` is maintained [here](https://github.com/kwohlfahrt/tifffile).

# Usage

The scripts can be used directly, or, if installed with `setup.py`, with
`blink_analysis <script> <options>`.

A number of scripts are meant to be used in the following order. Each has it's
own options to fine-tune behaviour, which can be described by passing the
`--help` flag to the appropriate script.

The following examples illustrate picking spots from a video split over 2 TIFF
files (`video_1.tif` and `video_2.tif`).

1. The video(s) are maximum-intensity projected into a 2D image
   (`projection.tif`) using the [tiffutil][tiffutil] package. See that package's
   documentation for available options, such as how to select sub-sequences:

   ```
   tiffutil project --projection max video_1.tif video_2.tif projection.tif
   ```

2. The peaks are picked using the picking script from
   the [blob detection][blob-detection] repository. The most important parameter
   is the `threshold` value.

   The `edge` parameter limits how close to the edge of the frame peaks are
   accepted, and should be set to (at least) the same size as used for the
   extraction in step 3.
   
   ```
   blob find --edge 4 --threshold 40 projection.tif > peaks.csv
   ```
   
3. For further analysis, regions of interest (ROI) with a user-defined `size`
   are extracted from the video. The size should be adjusted to include the
   diffraction limited spot and the neighbouring background. The ROIs will be of
   size `2 * size + 1`, centered on the peak.

   ```
   blink_analysis extract --size 4 peaks.csv video_1.tif video_2.tif > rois.pickle
   ```
   
4. The frames are categorized into 'on-state' and 'off-state'. This simply
   compares the background (outer 1/4 of the ROI) to the signal (remaining
   area), on a frame-by-frame basis. A smoothing can be applied that requires a
   number of consecutive 'off-state' and 'on-state' frames to end and start a
   blink respectively.

   ```
   blink_analysis categorize run --smoothing 4 2 rois.pickle on.pickle
   ```

   a. The categorization can be plotted for inspection.

      ```
      blink_analysis categorize plot rois.pickle on.pickle
      ```

5. The following statistics are extracted for each ROI:

   - location
   - mean signal per frame
   - mean total signal per switching event
   - total signal of ROI
   - mean 'on-state' length
   - number of switching events
   - total 'on-state' time

   Location is given in pixels, signal is measured in analog-to-digital units
   (ADU) and time is measured in frames.

   ```
   blink_analysis analyse rois.pickle on.pickle > stats.csv
   ```

If background correction is necessary, [tiffutil][tiffutil] provides a `smooth`
function that performs rolling-ball background correction. However, the methods
used in the picking and categorization software should be robust to
low-frequency variation.

[python]: https://python.org
[scipy]: https://scipy.org
[numpy]: https://www.numpy.org
[tifffile]: http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html
[matplotlib]: http://matplotlib.org
[skimage]: http://scikit-image.org
[skimage-log]: http://scikit-image.org/docs/dev/auto_examples/plot_blob.html#laplacian-of-gaussian-log
[pypi]: https://pypi.python.org/pypi
[pip-install]: https://pip.pypa.io/en/stable/user_guide/#installing-packages
[blob-detection]: https://github.com/TheLaueLab/blob-detection
[tiffutil]: https://github.com/TheLaueLab/tiffutil
