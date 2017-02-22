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
   (`projection.tif`) using the [tiffutil][tiffutil] package:

   ```
   tiffutil project --method max video_1.tif video_2.tif projection.tif
   ```

2. The peaks are picked using the picking script from
   the [blob detection][blob-detection] repository. The most important parameter
   is the `threshold` value.

   The `edge` parameter should be set to (at least) the same as used for the
   extraction in step 3. This avoids the extracted ROI clipping against the edge
   of the frame.
   
   ```
   blob find --size 2 3 --edge 4 --format pickle --threshold 40 projection.tif > peaks.pickle
   ```
   
3. The ROIs are extracted from the video. The ROIs will be of size
   `2 * size + 1`, centered on the peak.

   ```
   blink_analysis extract --size 4 peaks.pickle video_1.tif video_2.tif > rois.pickle
   ```
   
4. The ROIs are categorized into on- and off-states. This simply compares the
   center of the ROI to the edge, on a frame-by-frame basis.
   
   ```
   blink_analysis categorize rois.pickle on.pickle
   ```

5. The resulting traces are summarized. This extracts the following statistics
   for each ROI:

   - location (in pixels)
   - mean photons per frame
   - mean photons per switching event
   - total photons
   - mean on-state length
   - number of switching events
   - total on-state time

   ```
   blink_analysis analyse rois.pickle on.pickle stats.pickle
   ```
   
   The statistics can be converted to a human-readable CSV file with the
   `csvify.py` script.
   
   ```
   blink_analysis csvify peaks.pickle stats.pickle stats.csv
   ```

`smooth.py` is provided to perform background correction, should it be
necessary. However, the methods used in the picking and categorization software
should be robust to low-frequency variation.


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
