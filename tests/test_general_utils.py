import unittest

import numpy as np

from hopes.general_utils import (
    get_action_int,
    log_softmax,
    parse_list,
    softmax_1d,
    to_1d_float32,
    to_1d_int64,
    to_2d_float32,
    to_list,
)


class TestGeneralUtils(unittest.TestCase):
    def test_parse_list_accepts_list(self):
        x = [1, 2, 3]
        self.assertEqual(parse_list(x), [1, 2, 3])

    def test_parse_list_parses_string_list(self):
        x = "[1, 2, 3]"
        self.assertEqual(parse_list(x), [1, 2, 3])

    def test_parse_list_raises_on_bad_type(self):
        with self.assertRaises(TypeError):
            parse_list(123)  # int unsupported

    def test_to_list_parses_string(self):
        self.assertEqual(to_list("[1, 2]"), [1, 2])

    def test_to_list_converts_ndarray(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        self.assertEqual(to_list(arr), [1, 2, 3])

    def test_to_list_converts_iterable(self):
        self.assertEqual(to_list((4, 5)), [4, 5])

    def test_softmax_1d_basic_properties(self):
        x = [0.0, 0.0]
        p = softmax_1d(x)

        self.assertIsInstance(p, np.ndarray)
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(float(p.sum()), 1.0, places=7)
        self.assertTrue(np.all(p >= 0.0))
        self.assertTrue(np.all(p <= 1.0))
        self.assertTrue(np.allclose(p, np.array([0.5, 0.5]), atol=1e-7))

    def test_softmax_1d_stability_large_values(self):
        # should not overflow
        x = [1000.0, 1000.0]
        p = softmax_1d(x)
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(float(p.sum()), 1.0, places=7)
        self.assertTrue(np.allclose(p, np.array([0.5, 0.5]), atol=1e-6))

    def test_log_softmax_matches_log_of_softmax(self):
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        ls = log_softmax(x, axis=1)
        p = np.exp(ls)

        # exp(log_softmax) should sum to 1
        self.assertTrue(np.allclose(p.sum(axis=1), np.ones((1,)), atol=1e-6))

        # Compare with softmax_1d on the row
        p_ref = softmax_1d(x[0])
        self.assertTrue(np.allclose(p[0], p_ref, atol=1e-6))

    def test_get_action_int_scalar(self):
        self.assertEqual(get_action_int(1), 1)
        self.assertEqual(get_action_int(1.0), 1)

    def test_get_action_int_list_like(self):
        self.assertEqual(get_action_int([1]), 1)
        self.assertEqual(get_action_int((0,)), 0)
        self.assertEqual(get_action_int(np.array([1], dtype=np.int64)), 1)

    def test_get_action_int_string_list(self):
        self.assertEqual(get_action_int("[1]"), 1)
        self.assertEqual(get_action_int("[0]"), 0)

    def test_to_2d_float32(self):
        x = [[1, 2], [3, 4]]
        arr = to_2d_float32(x)
        self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(arr.shape, (2, 2))

    def test_to_1d_int64(self):
        x = [[1, 2, 3]]
        arr = to_1d_int64(x)
        self.assertEqual(arr.dtype, np.int64)
        self.assertEqual(arr.shape, (3,))
        self.assertTrue(np.array_equal(arr, np.array([1, 2, 3], dtype=np.int64)))

    def test_to_1d_float32(self):
        x = [[1.0, 2.0]]
        arr = to_1d_float32(x)
        self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(arr.shape, (2,))
        self.assertTrue(np.allclose(arr, np.array([1.0, 2.0], dtype=np.float32)))
