import unittest

import numpy as np

from hopes.ope.wrappers import compute_stepwise_ips_wis, dr_step_daily, wpdis_daily


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

    def test_wpdis_daily_returns_metrics(self):
        num_days = 3
        steps_per_episode = 4
        n_samples = num_days * steps_per_episode

        rew_flat = np.array(
            [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5, 2.0, 2.0, 2.0, 2.0],
            dtype=np.float32,
        )
        act_flat = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1], dtype=np.int64)

        p_b_taken_flat = np.full(n_samples, 0.5, dtype=np.float32)
        p_e_taken_flat = np.full(n_samples, 0.5, dtype=np.float32)
        sticky_act_flat = np.zeros(n_samples, dtype=np.int64)

        mean, lower, upper = wpdis_daily(
            num_days=num_days,
            steps_per_episode=steps_per_episode,
            rew_flat=rew_flat,
            act_flat=act_flat,
            p_b_taken_flat=p_b_taken_flat,
            p_e_taken_flat=p_e_taken_flat,
            sticky_act_flat=sticky_act_flat,
            clip=20.0,
            num_bootstrap_samples=200,
            significance_level=0.05,
        )

        expected_daily_returns = rew_flat.reshape(num_days, steps_per_episode).sum(axis=1)
        expected_mean = float(np.mean(expected_daily_returns) / steps_per_episode)

        self.assertIsInstance(mean, float)
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)

        self.assertAlmostEqual(mean, expected_mean / steps_per_episode, places=1)
        self.assertLessEqual(lower, mean)
        self.assertLessEqual(mean, upper)

    def test_wpdis_daily_identity_behavior_case(self):
        num_days = 5
        steps_per_episode = 3
        n_samples = num_days * steps_per_episode

        rew_flat = np.array(
            [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0],
            dtype=np.float32,
        )
        act_flat = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0], dtype=np.int64)

        p_b_taken_flat = np.full(n_samples, 0.5, dtype=np.float32)
        p_e_taken_flat = p_b_taken_flat.copy()
        sticky_act_flat = np.zeros(n_samples, dtype=np.int64)

        mean, lower, upper = wpdis_daily(
            num_days=num_days,
            steps_per_episode=steps_per_episode,
            rew_flat=rew_flat,
            act_flat=act_flat,
            p_b_taken_flat=p_b_taken_flat,
            p_e_taken_flat=p_e_taken_flat,
            sticky_act_flat=sticky_act_flat,
            clip=20.0,
            num_bootstrap_samples=200,
            significance_level=0.05,
        )

        expected = float(
            np.mean(rew_flat.reshape(num_days, steps_per_episode).sum(axis=1)) / steps_per_episode
        )

        self.assertAlmostEqual(mean, expected / steps_per_episode, places=2)
        self.assertLessEqual(lower, mean)
        self.assertLessEqual(mean, upper)

    def test_wpdis_daily_raises_on_invalid_n(self):
        rew_flat = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        act_flat = np.array([0, 1, 0], dtype=np.int64)
        p_b_taken_flat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        p_e_taken_flat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        sticky_act_flat = np.array([0, 0, 0], dtype=np.int64)

        with self.assertRaises(ValueError):
            wpdis_daily(
                num_days=2,
                steps_per_episode=2,
                rew_flat=rew_flat,
                act_flat=act_flat,
                p_b_taken_flat=p_b_taken_flat,
                p_e_taken_flat=p_e_taken_flat,
                sticky_act_flat=sticky_act_flat,
            )

    def test_wpdis_daily_raises_on_length_mismatch(self):
        rew_flat = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        act_flat = np.array([0, 1, 0], dtype=np.int64)
        p_b_taken_flat = np.full(4, 0.5, dtype=np.float32)
        p_e_taken_flat = np.full(4, 0.5, dtype=np.float32)
        sticky_act_flat = np.zeros(4, dtype=np.int64)

        with self.assertRaises(ValueError):
            wpdis_daily(
                num_days=2,
                steps_per_episode=2,
                rew_flat=rew_flat,
                act_flat=act_flat,
                p_b_taken_flat=p_b_taken_flat,
                p_e_taken_flat=p_e_taken_flat,
                sticky_act_flat=sticky_act_flat,
            )

    def test_dr_step_daily_identity_case(self):
        num_days = 4
        steps_per_episode = 3
        num_actions = 2
        n_samples = num_days * steps_per_episode

        rewards_flat = np.array(
            [1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        logged_actions = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=np.int64)

        target_policy_action_probabilities = np.full(
            (n_samples, num_actions),
            0.5,
            dtype=np.float32,
        )
        behavior_policy_action_probabilities = np.full(
            (n_samples, num_actions),
            0.5,
            dtype=np.float32,
        )

        q_values = np.zeros((n_samples, num_actions), dtype=np.float32)

        dr_day, rho = dr_step_daily(
            num_days=num_days,
            steps_per_episode=steps_per_episode,
            rewards_flat=rewards_flat,
            target_policy_action_probabilities=target_policy_action_probabilities,
            behavior_policy_action_probabilities=behavior_policy_action_probabilities,
            logged_actions=logged_actions,
            q_values=q_values,
            apply_stickiness=False,
        )

        expected = rewards_flat.reshape(num_days, steps_per_episode).sum(axis=1)

        self.assertIsInstance(dr_day, np.ndarray)
        self.assertIsInstance(rho, np.ndarray)
        self.assertEqual(dr_day.shape, (num_days,))
        self.assertEqual(rho.shape, (n_samples,))
        self.assertTrue(np.allclose(rho, np.ones(n_samples, dtype=np.float32)))
        self.assertTrue(np.allclose(dr_day, expected))

    def test_dr_step_daily_with_stickiness_runs(self):
        num_days = 3
        steps_per_episode = 4
        num_actions = 3
        n_samples = num_days * steps_per_episode

        rng = np.random.default_rng(0)

        rewards_flat = rng.random(n_samples, dtype=np.float32)
        logged_actions = rng.integers(0, num_actions, size=n_samples, dtype=np.int64)

        target_policy_action_probabilities = np.full(
            (n_samples, num_actions),
            1.0 / num_actions,
            dtype=np.float32,
        )
        behavior_policy_action_probabilities = np.full(
            (n_samples, num_actions),
            1.0 / num_actions,
            dtype=np.float32,
        )

        q_values = rng.random((n_samples, num_actions), dtype=np.float32)
        sticky_actions = np.zeros(n_samples, dtype=np.int64)
        sticky_actions[1] = 1

        dr_day, rho = dr_step_daily(
            num_days=num_days,
            steps_per_episode=steps_per_episode,
            rewards_flat=rewards_flat,
            target_policy_action_probabilities=target_policy_action_probabilities,
            behavior_policy_action_probabilities=behavior_policy_action_probabilities,
            logged_actions=logged_actions,
            q_values=q_values,
            sticky_actions=sticky_actions,
            apply_stickiness=True,
            value_after_switch=1.0,
        )

        self.assertEqual(dr_day.shape, (num_days,))
        self.assertEqual(rho.shape, (n_samples,))
        self.assertTrue(np.all(np.isfinite(dr_day)))
        self.assertTrue(np.all(np.isfinite(rho)))

    def test_dr_step_daily_raises_on_invalid_q_shape(self):
        num_days = 2
        steps_per_episode = 3
        num_actions = 2
        n_samples = num_days * steps_per_episode

        rewards_flat = np.ones(n_samples, dtype=np.float32)
        logged_actions = np.zeros(n_samples, dtype=np.int64)

        target_policy_action_probabilities = np.full(
            (n_samples, num_actions),
            0.5,
            dtype=np.float32,
        )
        behavior_policy_action_probabilities = np.full(
            (n_samples, num_actions),
            0.5,
            dtype=np.float32,
        )

        q_values = np.ones((n_samples, num_actions + 1), dtype=np.float32)

        with self.assertRaises(ValueError):
            dr_step_daily(
                num_days=num_days,
                steps_per_episode=steps_per_episode,
                rewards_flat=rewards_flat,
                target_policy_action_probabilities=target_policy_action_probabilities,
                behavior_policy_action_probabilities=behavior_policy_action_probabilities,
                logged_actions=logged_actions,
                q_values=q_values,
            )

    def test_dr_step_daily_raises_on_invalid_n(self):
        rewards_flat = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        logged_actions = np.array([0, 1, 0], dtype=np.int64)
        q_values = np.ones((3, 2), dtype=np.float32)

        target_policy_action_probabilities = np.full((3, 2), 0.5, dtype=np.float32)
        behavior_policy_action_probabilities = np.full((3, 2), 0.5, dtype=np.float32)

        with self.assertRaises(ValueError):
            dr_step_daily(
                num_days=2,
                steps_per_episode=2,
                rewards_flat=rewards_flat,
                target_policy_action_probabilities=target_policy_action_probabilities,
                behavior_policy_action_probabilities=behavior_policy_action_probabilities,
                logged_actions=logged_actions,
                q_values=q_values,
            )
