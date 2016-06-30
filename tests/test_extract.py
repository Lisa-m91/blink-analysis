from unittest import TestCase

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

if __name__ == "__main__":
    import unittest
    unittest.main()
