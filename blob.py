from numpy import zeros
from numpy.linalg import norm
from math import pi

try:
    from pyfftw.interfaces.numpy_fft import fftn, ifftn
except ImportError:
    from numpy.fft import fftn, ifftn

def gaussianPsf(shape, sigmas, dtype='float'):
    from math import sqrt
    from numpy import ones, ogrid, exp

    output = ones(shape, dtype=dtype)
    grids = ogrid[tuple(slice(None, s) for s in shape)]
    for grid, sigma in zip(grids, sigmas):
        # Pyramid function with integer operations
        grid = (grid.size - abs(grid * 2 - grid.size)) // 2
        values = 1 / (sigma * sqrt(2 * pi)) * exp(-( grid / sigma ) ** 2 / 2)
        output *= values
    return output

def gaussianPsfHat(shape, sigmas, dtype='float'):
    from numpy import ones, ogrid, exp

    output = ones(shape, dtype=dtype)
    grids = ogrid[tuple(slice(None, s) for s in shape)]
    for grid, sigma in zip(grids, sigmas):
        # Pyramid function with integer operations
        grid = (grid.size - abs(grid * 2 - grid.size)) // 2
        values = exp(-( grid * pi * sigma / grid.size ) ** 2 * 2)
        output *= values
    return output

# Circular gradient. Add more padding methods?
def gradient(data, axis):
    from numpy import roll
    return data - roll(data, 1, axis)

def div(*data):
    return sum(gradient(d, axis) for axis, d in enumerate(data))

def laplacianOperator(data):
    return div(*(gradient(data, a) for a in range(data.ndim)))

def localMinima(data, threshold):
    from numpy import ones, roll, nonzero, transpose

    if threshold is not None:
        peaks = data < threshold
    else:
        peaks = ones(data.shape, dtype=data.dtype)

    for axis in range(data.ndim):
        peaks &= data <= roll(data, -1, axis)
        peaks &= data <= roll(data, 1, axis)
    return transpose(nonzero(peaks))

def blobLOG(data, scales=range(1, 10, 1), threshold=-30):
    """Find blobs. Returns [[scale, x, y, ...], ...]"""
    from numpy import empty, asarray
    from itertools import repeat

    data = asarray(data)
    scales = asarray(scales)
    data_hat = fftn(data)

    log = empty((len(scales),) + data.shape, dtype=data.dtype)
    for slog, scale in zip(log, scales):
        blur = gaussianPsfHat(data.shape, repeat(scale), data.dtype)
        blurred = ifftn(data_hat * blur).real
        slog[...] = scale * laplacianOperator(blurred)

    peaks = localMinima(log, threshold=threshold)
    peaks[:, 0] = scales[peaks[:, 0]]
    return peaks

def sphereIntersection(r1, r2, d):
    # http://mathworld.wolfram.com/Sphere-SphereIntersection.html

    return (pi * (r1 + r2 - d) ** 2
            * (d ** 2 + 6 * r2 * r1
               + 2 * d * r1 - 3 * r1 ** 2
               + 2 * d * r2 - 3 * r2 ** 2)
            / (12 * d))

def circleIntersection(r1, r2, d):
    # http://mathworld.wolfram.com/Circle-CircleIntersection.html
    from numpy import arccos, sqrt

    return (r1 ** 2 * arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
            + r2 ** 2 * arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
            - sqrt((-d + r1 + r2) * (d + r1 - r2)
                   * (d - r1 + r2) * (d + r1 + r2)) / 2)

def findBlobs(img, scales=range(1, 10), threshold=30, max_overlap=0.05):
    from numpy import ones, triu

    peaks = blobLOG(img, scales=scales, threshold=-threshold)
    radii = peaks[:, 0]
    positions = peaks[:, 1:]

    distances = norm(positions[:, None, :] - positions[None, :, :], axis=2)

    if positions.shape[1] == 2:
        intersections = circleIntersection(radii, radii.T, distances)
        volumes = pi * radii ** 2
    if positions.shape[1] == 3:
        intersections = sphereIntersection(radii, radii.T, distances)
        volumes = 4/3 * pi * radii ** 3
    delete = ((intersections > (volumes * max_overlap))
              # Remove the smaller of the blobs
              & ((radii[:, None] < radii[None, :])
                 # Tie-break
                 | ((radii[:, None] == radii[None, :])
                    & triu(ones((len(peaks), len(peaks)), dtype='bool'))))
    ).any(axis=1)
    return peaks[~delete]

if __name__ == '__main__':
    from tifffile import imread
    from pathlib import Path
    from matplotlib import pyplot as plt
    from matplotlib.cm import get_cmap

    image = (imread(str(Path(__file__).parent / "hubble_deep_field.tif"))[:500, :500, :]
             .astype('float32').sum(axis=2))
    peaks = findBlobs(image, range(1, 30, 3), 0.05)

    plt.imshow(image, cmap=get_cmap('gray'))
    plt.scatter(peaks[:, 2], peaks[:, 1], s=peaks[:, 0] * 20,
                facecolors='none', edgecolors='g')
    plt.show()