import unittest

import numpy as np

from hopes.ope.wrappers import compute_stepwise_ips_wis, dr_step_daily, wpdis_daily


class TestWrappers(unittest.TestCase):
    def setUp(self):
        self.num_days = 2
        self.T = 3
        N = self.num_days * self.T

        # deterministic actions alternating
        self.act_flat = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

        # simple rewards
        self.rew_flat = np.array([1.0, 0.5, 0.0, 1.0, 0.5, 0.0], dtype=np.float32)

        # behavior propensities (taken)
        self.p_b_taken_flat = np.full(N, 0.6, dtype=np.float32)

        # target propensities (taken)
        self.p_e_taken_flat = np.full(N, 0.8, dtype=np.float32)

        # no stickiness (all zeros)
        self.sticky_act_flat = np.zeros(N, dtype=np.int64)

    # compute_stepwise_ips_wis
    def test_compute_stepwise_ips_wis_basic(self):
        result = compute_stepwise_ips_wis(
            p_b_taken_flat=self.p_b_taken_flat,
            p_e_taken_flat=self.p_e_taken_flat,
            rew_flat=self.rew_flat,
        )

        self.assertIn("weights", result)
        self.assertIn("ips", result)
        self.assertIn("wis", result)

        w = result["weights"]

        # weights must be finite and positive
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertTrue(np.all(w > 0))

        # IPS and WIS must be finite
        self.assertTrue(np.isfinite(result["ips"]))
        self.assertTrue(np.isfinite(result["wis"]))

    def test_compute_stepwise_invalid_probs(self):
        bad_p = self.p_e_taken_flat.copy()
        bad_p[0] = 1.5  # invalid prob

        with self.assertRaises(ValueError):
            compute_stepwise_ips_wis(
                p_b_taken_flat=self.p_b_taken_flat,
                p_e_taken_flat=bad_p,
                rew_flat=self.rew_flat,
            )

    # wpdis_daily
    def test_wpdis_daily_basic(self):
        mean, lo, hi = wpdis_daily(
            num_days=self.num_days,
            steps_per_episode=self.T,
            rew_flat=self.rew_flat,
            act_flat=self.act_flat,
            p_b_taken_flat=self.p_b_taken_flat,
            p_e_taken_flat=self.p_e_taken_flat,
            sticky_act_flat=self.sticky_act_flat,
            n_boot=100,  # keep small for test speed
            alpha=0.1,
            seed=42,
        )

        # all outputs must be finite scalars
        self.assertTrue(np.isfinite(mean))
        self.assertTrue(np.isfinite(lo))
        self.assertTrue(np.isfinite(hi))

        # CI bounds should be ordered
        self.assertLessEqual(lo, hi)

    def test_wpdis_daily_dimension_mismatch(self):
        # break N consistency
        bad_rew = self.rew_flat[:-1]

        with self.assertRaises(ValueError):
            wpdis_daily(
                num_days=self.num_days,
                steps_per_episode=self.T,
                rew_flat=bad_rew,
                act_flat=self.act_flat,
                p_b_taken_flat=self.p_b_taken_flat,
                p_e_taken_flat=self.p_e_taken_flat,
                sticky_act_flat=self.sticky_act_flat,
            )

    def test_wpdis_daily_with_stickiness(self):
        # introduce stickiness at first step of first episode
        sticky = self.sticky_act_flat.copy()
        sticky[1] = 1  # force switch in episode 1

        mean, lo, hi = wpdis_daily(
            num_days=self.num_days,
            steps_per_episode=self.T,
            rew_flat=self.rew_flat,
            act_flat=self.act_flat,
            p_b_taken_flat=self.p_b_taken_flat,
            p_e_taken_flat=self.p_e_taken_flat,
            sticky_act_flat=sticky,
            n_boot=100,
            alpha=0.1,
            seed=0,
        )

        self.assertTrue(np.isfinite(mean))
        self.assertLessEqual(lo, hi)

    def test_dr_step_daily_shapes_and_finite(self):
        rng = np.random.default_rng(3)
        steps_per_episode = 5
        num_days = 9
        N = steps_per_episode * num_days

        rtg_flat = rng.normal(size=N).astype(np.float32)
        act_flat = rng.integers(0, 2, size=N, dtype=np.int64)

        p_b_taken_flat = rng.uniform(0.2, 0.9, size=N).astype(np.float32)

        P_new = rng.uniform(0.1, 0.9, size=(N, 2)).astype(np.float32)
        P_new /= P_new.sum(axis=1, keepdims=True)

        Q0 = rng.normal(size=N).astype(np.float32)
        Q1 = rng.normal(size=N).astype(np.float32)

        dr_day, w_logged = dr_step_daily(
            num_days=num_days,
            steps_per_episode=steps_per_episode,
            rtg_flat=rtg_flat,
            P_new=P_new,
            act_flat=act_flat,
            p_b_taken_flat=p_b_taken_flat,
            Q0=Q0,
            Q1=Q1,
            eps=1e-12,
            cap=20,
        )

        self.assertEqual(dr_day.shape, (num_days,))
        self.assertEqual(w_logged.shape, (N,))
        self.assertTrue(np.isfinite(dr_day).all())
        self.assertTrue(np.isfinite(w_logged).all())

    def test_dr_step_daily_weight_clipping(self):
        rng = np.random.default_rng(4)
        steps_per_episode = 4
        num_days = 6
        N = steps_per_episode * num_days

        rtg_flat = rng.normal(size=N).astype(np.float32)
        act_flat = rng.integers(0, 2, size=N, dtype=np.int64)

        # Make pb tiny to create huge ratios, then check clipping
        p_b_taken_flat = np.full(N, 1e-6, dtype=np.float32)

        P_new = np.zeros((N, 2), dtype=np.float32)
        P_new[np.arange(N), act_flat] = 0.9
        P_new[np.arange(N), 1 - act_flat] = 0.1

        Q0 = rng.normal(size=N).astype(np.float32)
        Q1 = rng.normal(size=N).astype(np.float32)

        cap = 10
        dr_day, w_logged = dr_step_daily(
            num_days=num_days,
            steps_per_episode=steps_per_episode,
            rtg_flat=rtg_flat,
            P_new=P_new,
            act_flat=act_flat,
            p_b_taken_flat=p_b_taken_flat,
            Q0=Q0,
            Q1=Q1,
            eps=1e-12,
            cap=cap,
        )

        self.assertTrue(np.all(w_logged <= cap + 1e-6))
        self.assertTrue(np.all(w_logged >= (1.0 / cap) - 1e-6))
