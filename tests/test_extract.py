from unittest import TestCase
import numpy as np
from itertools import chain, zip_longest

from tifffile import TiffFile
from extract import *

class TiffTest(TestCase):
    offsets = 100 * np.arange(1, 4)[:, None]

    def setUp(self):
        from tempfile import TemporaryFile
        from tifffile import imsave

        data = np.arange(10) + self.offsets
        self.tifffile = TemporaryFile()
        imsave(self.tifffile, data)
        self.tifffile.seek(0)

    def tearDown(self):
        self.tifffile.close()

class TextExtractAll(TiffTest):
    def test_iterable(self):
        series = [TiffFile(self.tifffile).series[0]]

        peaks = np.array([[3], [4], [6]])
        expected = [np.arange(2, 5) + self.offsets,
                    np.arange(3, 6) + self.offsets,
                    np.arange(5, 8) + self.offsets,]

        for trial, expected in zip_longest(extractAll(peaks, series), expected):
            np.testing.assert_equal(trial, expected)

if __name__ == "__main__":
    import unittest
    unittest.main()
