from unittest import TestCase
from numpy import arange, array

class TestRollingMedian(TestCase):
    def test_iterable(self):
        from extract import rollingMedian

        data = iter(range(0, 10))
        expected = list(range(1, 9))
        self.assertEqual(list(rollingMedian(data, 3)), expected)

class TestExcludeFrames(TestCase):
    def test_iterable(self):
        from extract import excludeFrames, Range
        excludes = [Range(1, 4), Range(8, 10)]
        data = iter(range(100, 120))
        expected = [100] + list(range(104, 108)) + list(range(110, 120))
        self.assertEqual(list(excludeFrames(data, excludes)), expected)

    def test_array(self):
        from extract import excludeFrames, Range

        excludes = [Range(1, 4), Range(8, 10)]
        data = array(range(100, 120))
        expected = array([100] + list(range(104, 108)) + list(range(110, 120)))
        self.assertTrue((excludeFrames(data, excludes) == expected).all())

class TextExtractAll(TestCase):
    offsets = 100 * arange(1, 4)[:, None]

    def setUp(self):
        from tempfile import TemporaryFile
        from tifffile import imsave

        data = arange(10) + self.offsets
        self.tifffile = TemporaryFile()
        imsave(self.tifffile, data)
        self.tifffile.seek(0)

    def tearDown(self):
        self.tifffile.close()

    def test_iterable(self):
        from extract import extractAll
        from tifffile import TiffFile

        series = [TiffFile(self.tifffile).series[0]]

        peaks = array([[1, 3], [0, 4], [2, 6]])
        expected = [arange(2, 5) + self.offsets,
                    arange(4, 5) + self.offsets,
                    arange(4, 9) + self.offsets,]

        for trial, expected in zip(extractAll(peaks, series), expected):
            self.assertTrue((trial == expected).all())

class TestPeakEnclosed(TestCase):
    def test_simple(self):
        from extract import peakEnclosed
        shape = (5, 10)
        peaks = array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 1],
                       [0, 4, 9],
                       [0, 5, 10],
                       [1, 4, 9],
                       [1, 3, 8],])
        expected = array([True, False, True, True, False, False, True])
        self.assertTrue((peakEnclosed(peaks, shape) == expected).all())

if __name__ == "__main__":
    import unittest
    unittest.main()
