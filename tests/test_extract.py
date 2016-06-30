from unittest import TestCase

class TestRollingMedian(TestCase):
    def test_iterable(self):
        from extract import rollingMedian

        data = iter(range(0, 10))
        expected = list(range(1, 9))
        self.assertEqual(list(rollingMedian(data, 3)), expected)

if __name__ == "__main__":
    import unittest
    unittest.main()
