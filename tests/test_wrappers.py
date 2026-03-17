import unittest

import numpy as np

from hopes.ope.wrappers import compute_stepwise_ips_wis


class TestWrappers(unittest.TestCase):
    def test_compute_stepwise_ips_wis(self):
        p_b_taken_flat = np.array([0.5, 0.4, 0.2, 0.8], dtype=np.float32)
        p_e_taken_flat = np.array([0.25, 0.2, 0.4, 0.4], dtype=np.float32)
        rew_flat = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        out = compute_stepwise_ips_wis(
            p_b_taken_flat=p_b_taken_flat,
            p_e_taken_flat=p_e_taken_flat,
            rew_flat=rew_flat,
        )

        expected_weights = p_e_taken_flat / p_b_taken_flat
        expected_ips = float(np.mean(expected_weights * rew_flat))
        expected_wis = float(np.sum(expected_weights * rew_flat) / np.sum(expected_weights))

        self.assertIsInstance(out, dict)
        self.assertIn("weights", out)
        self.assertIn("ips", out)
        self.assertIn("wis", out)
        self.assertIn("w_max", out)
        self.assertIn("w_p99", out)

        self.assertTrue(np.allclose(out["weights"], expected_weights))
        self.assertAlmostEqual(out["ips"], expected_ips, places=6)
        self.assertAlmostEqual(out["wis"], expected_wis, places=6)
        self.assertAlmostEqual(out["w_max"], float(expected_weights.max()), places=6)
        self.assertTrue(np.isfinite(out["w_p99"]))

    def test_compute_stepwise_ips_wis_raises_on_length_mismatch(self):
        p_b_taken_flat = np.array([0.5, 0.4], dtype=np.float32)
        p_e_taken_flat = np.array([0.25, 0.2, 0.4], dtype=np.float32)
        rew_flat = np.array([1.0, 2.0], dtype=np.float32)

        with self.assertRaises(ValueError):
            compute_stepwise_ips_wis(
                p_b_taken_flat=p_b_taken_flat,
                p_e_taken_flat=p_e_taken_flat,
                rew_flat=rew_flat,
            )

    def test_compute_stepwise_ips_wis_raises_on_invalid_probs(self):
        p_b_taken_flat = np.array([0.5, 1.2], dtype=np.float32)
        p_e_taken_flat = np.array([0.25, 0.2], dtype=np.float32)
        rew_flat = np.array([1.0, 2.0], dtype=np.float32)

        with self.assertRaises(ValueError):
            compute_stepwise_ips_wis(
                p_b_taken_flat=p_b_taken_flat,
                p_e_taken_flat=p_e_taken_flat,
                rew_flat=rew_flat,
            )
