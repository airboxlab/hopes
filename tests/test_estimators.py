import unittest

import numpy as np
from action_probs_utils import generate_action_probs

from hopes.ope.estimators import (
    BaseEstimator,
    DirectMethod,
    InverseProbabilityWeighting,
    PerDecisionImportanceSampling,
    SelfNormalizedInverseProbabilityWeighting,
    SelfNormalizedPerDecisionImportanceSampling,
    SelfNormalizedTrajectoryWiseImportanceSampling,
    StickySequentialDoublyRobust,
    StickyTrajectoryWiseIS,
    TrajectoryWiseImportanceSampling,
    WeightedPerDecisionImportanceSampling,
)
from hopes.rew.rewards import RegressionBasedRewardModel


class TestEstimators(unittest.TestCase):
    def test_check_parameters(self):
        ipw = InverseProbabilityWeighting()
        with self.assertRaises(ValueError):
            ipw.check_parameters()

        with self.assertRaises(ValueError):
            ipw.set_parameters(
                target_policy_action_probabilities=np.random.rand(10, 3),
                behavior_policy_action_probabilities=np.random.rand(10, 4),
                rewards=np.random.rand(10),
            )

        with self.assertRaises(ValueError):
            ipw.set_parameters(
                target_policy_action_probabilities=np.random.rand(4, 3),
                behavior_policy_action_probabilities=np.random.rand(10, 3),
                rewards=np.random.rand(10),
            )

    def test_ipw(self):
        ipw = InverseProbabilityWeighting()

        target_policy_action_probabilities = generate_action_probs(traj_length=10, num_actions=3)
        behavior_policy_action_probabilities = generate_action_probs(traj_length=10, num_actions=3)
        rewards = np.random.rand(10)

        ipw.set_parameters(
            target_policy_action_probabilities=target_policy_action_probabilities,
            behavior_policy_action_probabilities=behavior_policy_action_probabilities,
            rewards=rewards,
        )

        wrew = ipw.estimate_weighted_rewards()
        self.assertIsInstance(wrew, np.ndarray)
        self.assertEqual(wrew.shape, (10, 3))
        policy_value = ipw.estimate_policy_value()
        self.assertIsInstance(policy_value, float)
        self.assertGreaterEqual(policy_value, 0.0)

        # test CI
        self._test_ci(ipw)

        # test with zero rewards
        rewards = np.zeros(10)

        ipw.set_parameters(
            target_policy_action_probabilities=target_policy_action_probabilities,
            behavior_policy_action_probabilities=behavior_policy_action_probabilities,
            rewards=rewards,
        )

        policy_value = ipw.estimate_policy_value()
        self.assertIsInstance(policy_value, float)
        self.assertEqual(policy_value, 0.0)

    def test_snipw(self):
        snipw = SelfNormalizedInverseProbabilityWeighting()

        target_policy_action_probabilities = generate_action_probs(traj_length=10, num_actions=3)
        behavior_policy_action_probabilities = generate_action_probs(traj_length=10, num_actions=3)
        rewards = np.random.rand(10)

        snipw.set_parameters(
            target_policy_action_probabilities=target_policy_action_probabilities,
            behavior_policy_action_probabilities=behavior_policy_action_probabilities,
            rewards=rewards,
        )

        wrew = snipw.estimate_weighted_rewards()
        self.assertIsInstance(wrew, np.ndarray)
        self.assertEqual(wrew.shape, (10, 3))
        policy_value = snipw.estimate_policy_value()
        self.assertIsInstance(policy_value, float)
        self.assertGreaterEqual(policy_value, 0.0)

        # test CI
        self._test_ci(snipw)

    def test_dm(self):
        num_actions = 3
        num_obs = 10
        num_samples = 100
        num_steps_per_episode = 2
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)
        rew = np.random.rand(num_samples)
        target_policy_action_probabilities = generate_action_probs(
            traj_length=num_samples, num_actions=num_actions
        )

        dm = DirectMethod(
            q_model_cls=RegressionBasedRewardModel,
            q_model_type="random_forest",
            q_model_params={"max_depth": 5},
            behavior_policy_obs=obs,
            behavior_policy_act=act,
            behavior_policy_rewards=rew,
            steps_per_episode=num_steps_per_episode,
            discount_factor=0.99,
        )
        fit_stats = dm.fit()
        self.assertIsInstance(fit_stats, dict)
        self.assertIn("rmse", fit_stats)

        dm.set_parameters(
            target_policy_action_probabilities=target_policy_action_probabilities,
            behavior_policy_action_probabilities=None,
            rewards=None,
        )

        wrew = dm.estimate_weighted_rewards()
        self.assertIsInstance(wrew, np.ndarray)
        self.assertEqual(wrew.shape, (num_samples // num_steps_per_episode,))
        policy_value = dm.estimate_policy_value()
        self.assertIsInstance(policy_value, float)
        self.assertGreaterEqual(policy_value, 0.0)

        self._test_ci(dm)

    def test_dm_discount_computation(self):
        """Test that DM correctly computes discounted cumulative returns."""
        num_actions = 2
        num_obs = 2
        num_episodes = 3
        num_steps_per_episode = 4
        num_samples = num_episodes * num_steps_per_episode
        discount_factor = 0.9

        # Create deterministic data
        np.random.seed(42)
        obs = np.random.rand(num_samples, num_obs)
        act = np.array([0, 1, 0, 1] * num_episodes)  # alternating actions

        # Create rewards: each episode has rewards [1, 2, 3, 4]
        rew = np.tile([1.0, 2.0, 3.0, 4.0], num_episodes)

        # Expected cumulative discounted returns from initial states:
        # G_0 = r_0 + γ*r_1 + γ²*r_2 + γ³*r_3
        # G_0 = 1 + 0.9*2 + 0.9²*3 + 0.9³*4
        gamma = discount_factor
        expected_return = 1.0 + gamma * 2.0 + gamma**2 * 3.0 + gamma**3 * 4.0

        dm = DirectMethod(
            q_model_cls=RegressionBasedRewardModel,
            q_model_type="linear",  # Use linear for predictability
            behavior_policy_obs=obs,
            behavior_policy_act=act,
            behavior_policy_rewards=rew,
            steps_per_episode=num_steps_per_episode,
            discount_factor=discount_factor,
        )

        fit_stats = dm.fit()
        self.assertIsInstance(fit_stats, dict)

        # Since we have perfect linear model training data, the model should learn
        # to predict similar values for similar initial states
        # Verify the Q model was trained on initial states only
        self.assertEqual(dm.q_model.obs.shape[0], num_episodes)
        self.assertEqual(dm.q_model.act.shape[0], num_episodes)
        self.assertEqual(dm.q_model.rew.shape[0], num_episodes)

        # Verify cumulative returns are approximately correct
        for i in range(num_episodes):
            self.assertAlmostEqual(dm.q_model.rew[i], expected_return, delta=0.01)

    def test_tis_sntis_pdis(self):
        traj_length = 10
        num_episodes = 500
        num_actions = 3

        (
            target_policy_action_probabilities,
            behavior_policy_action_probabilities,
            rewards,
        ) = self._get_is_data(
            traj_length=traj_length, num_actions=num_actions, num_episodes=num_episodes
        )

        # TIS
        tis = TrajectoryWiseImportanceSampling(
            steps_per_episode=traj_length,
            discount_factor=0.99,
        )

        tis.set_parameters(
            target_policy_action_probabilities=target_policy_action_probabilities,
            behavior_policy_action_probabilities=behavior_policy_action_probabilities,
            rewards=rewards,
        )

        wrew = tis.estimate_weighted_rewards()
        self.assertIsInstance(wrew, np.ndarray)
        self.assertEqual(wrew.shape, (num_episodes, 1))

        policy_value = tis.estimate_policy_value()
        self.assertIsInstance(policy_value, float)
        self.assertGreaterEqual(policy_value, 0.0)
        print("tis", policy_value)

        self._test_ci(tis)

    def test_sntis(self):
        traj_length = 10
        num_episodes = 500
        num_actions = 3

        (
            target_policy_action_probabilities,
            behavior_policy_action_probabilities,
            rewards,
        ) = self._get_is_data(
            traj_length=traj_length, num_actions=num_actions, num_episodes=num_episodes
        )

        sntis = SelfNormalizedTrajectoryWiseImportanceSampling(
            steps_per_episode=traj_length,
            discount_factor=0.99,
        )
        sntis.set_parameters(
            target_policy_action_probabilities=target_policy_action_probabilities,
            behavior_policy_action_probabilities=behavior_policy_action_probabilities,
            rewards=rewards,
        )

        wrew = sntis.estimate_weighted_rewards()
        self.assertIsInstance(wrew, np.ndarray)
        self.assertEqual(wrew.shape, (num_episodes, 1))

        policy_value = sntis.estimate_policy_value()
        self.assertIsInstance(policy_value, float)
        self.assertGreaterEqual(policy_value, 0.0)
        print("sntis", policy_value)

        self._test_ci(sntis)

    def test_pdis(self):
        traj_length = 10
        num_episodes = 500
        num_actions = 3

        (
            target_policy_action_probabilities,
            behavior_policy_action_probabilities,
            rewards,
        ) = self._get_is_data(
            traj_length=traj_length, num_actions=num_actions, num_episodes=num_episodes
        )

        pdis = PerDecisionImportanceSampling(
            steps_per_episode=traj_length,
            discount_factor=0.99,
        )
        pdis.set_parameters(
            target_policy_action_probabilities=target_policy_action_probabilities,
            behavior_policy_action_probabilities=behavior_policy_action_probabilities,
            rewards=rewards,
        )

        wrew = pdis.estimate_weighted_rewards()
        self.assertIsInstance(wrew, np.ndarray)
        self.assertEqual(wrew.shape, (num_episodes, 1))

        policy_value = pdis.estimate_policy_value()
        self.assertIsInstance(policy_value, float)
        self.assertGreaterEqual(policy_value, 0.0)
        print("pdis", policy_value)

        self._test_ci(pdis)

    def test_snpdis(self):
        traj_length = 10
        num_episodes = 500
        num_actions = 3

        (
            target_policy_action_probabilities,
            behavior_policy_action_probabilities,
            rewards,
        ) = self._get_is_data(
            traj_length=traj_length, num_actions=num_actions, num_episodes=num_episodes
        )

        snpdis = SelfNormalizedPerDecisionImportanceSampling(
            steps_per_episode=traj_length,
            discount_factor=0.99,
        )
        snpdis.set_parameters(
            target_policy_action_probabilities=target_policy_action_probabilities,
            behavior_policy_action_probabilities=behavior_policy_action_probabilities,
            rewards=rewards,
        )

        wrew = snpdis.estimate_weighted_rewards()
        self.assertIsInstance(wrew, np.ndarray)
        self.assertEqual(wrew.shape, (num_episodes, 1))

        policy_value = snpdis.estimate_policy_value()
        self.assertIsInstance(policy_value, float)
        self.assertGreaterEqual(policy_value, 0.0)
        print("snpdis", policy_value)

        self._test_ci(snpdis)

    def test_neg_rewards(self):
        ipw = InverseProbabilityWeighting()

        target_policy_action_probabilities = generate_action_probs(traj_length=10, num_actions=3)
        behavior_policy_action_probabilities = generate_action_probs(traj_length=10, num_actions=3)
        rewards = -np.random.rand(10)

        with self.assertRaises(ValueError) as e:
            ipw.set_parameters(
                target_policy_action_probabilities=target_policy_action_probabilities,
                behavior_policy_action_probabilities=behavior_policy_action_probabilities,
                rewards=rewards,
            )
            self.assertTrue("The rewards must be non-negative" in str(e.exception))

    def _test_ci(self, estimator: BaseEstimator):
        # test CI
        metrics = estimator.estimate_policy_value_with_confidence_interval(
            num_samples=1000, significance_level=0.05
        )
        self.assertIsInstance(metrics, dict)
        for m in ["mean", "lower_bound", "upper_bound", "std"]:
            self.assertIn(m, metrics)
            self.assertIsInstance(metrics[m], float)
        self.assertTrue(metrics["lower_bound"] <= metrics["mean"] <= metrics["upper_bound"])

    def _get_is_data(
        self, traj_length: int, num_actions: int, num_episodes: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_policy_action_probabilities = np.concatenate(
            [
                generate_action_probs(traj_length=traj_length, num_actions=num_actions)
                for _ in range(num_episodes)
            ]
        )

        behavior_policy_action_probabilities = np.concatenate(
            [
                generate_action_probs(traj_length=traj_length, num_actions=num_actions)
                for _ in range(num_episodes)
            ]
        )

        rewards = np.random.rand(traj_length * num_episodes)

        return (target_policy_action_probabilities, behavior_policy_action_probabilities, rewards)

    def test_wpdis_identity_and_ci(self):
        traj_length = 6
        num_episodes = 30
        num_actions = 2
        N = traj_length * num_episodes

        target = generate_action_probs(traj_length=N, num_actions=num_actions)
        behavior = target.copy()  # identity case

        rewards = np.random.RandomState(0).rand(N).astype(np.float32)

        actions = np.random.randint(0, num_actions, size=N).astype(np.int64)

        # sticky: make it valid and monotonic per episode
        sticky = np.zeros(N, dtype=np.int64)
        for ep in range(num_episodes):
            start = ep * traj_length
            sticky[start + 3 : start + traj_length] = 1  # switch at t=3 always

        wpdis = WeightedPerDecisionImportanceSampling(
            steps_per_episode=traj_length,
            eps=1e-12,
            clip=20.0,
            apply_stickiness=True,
        )
        wpdis.set_logged_data(actions=actions, sticky_actions=sticky)
        wpdis.set_parameters(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            rewards=rewards,
        )

        v = wpdis.estimate_policy_value()
        self.assertIsInstance(v, float)
        self.assertTrue(np.isfinite(v))

        # With rho=1 everywhere, WPDIS reduces to the mean reward across all steps
        expected = float(np.mean(rewards))
        self.assertAlmostEqual(v, expected, places=5)

        # test CI (episode bootstrap in your implementation)
        ci = wpdis.estimate_policy_value_with_confidence_interval(n_boot=200, alpha=0.1, seed=0)
        self.assertIsInstance(ci, dict)
        for k in ["mean", "lower_bound", "upper_bound"]:
            self.assertIn(k, ci)
            self.assertIsInstance(ci[k], float)
            self.assertTrue(np.isfinite(ci[k]))
        self.assertLessEqual(ci["lower_bound"], ci["mean"])
        self.assertLessEqual(ci["mean"], ci["upper_bound"])

    def test_wpdis_requires_logged_actions(self):
        traj_length = 4
        num_episodes = 3
        num_actions = 2
        N = traj_length * num_episodes

        target = generate_action_probs(traj_length=N, num_actions=num_actions)
        behavior = generate_action_probs(traj_length=N, num_actions=num_actions)
        rewards = np.random.rand(N).astype(np.float32)

        wpdis = WeightedPerDecisionImportanceSampling(steps_per_episode=traj_length)

        # Intentionally do NOT call set_logged_data
        with self.assertRaises(ValueError):
            wpdis.set_parameters(
                target_policy_action_probabilities=target,
                behavior_policy_action_probabilities=behavior,
                rewards=rewards,
            )
            _ = wpdis.estimate_policy_value()

    def test_sticky_trajectory_wise_is_identity(self):
        steps_per_episode = 5
        num_eps = 20
        N = steps_per_episode * num_eps

        rng = np.random.default_rng(0)

        # Rewards (any non-pathological values)
        rew_flat = rng.normal(loc=0.0, scale=1.0, size=N).astype(np.float32)

        # Identity: p_e_taken == p_b_taken => rho = 1
        p_b_taken_flat = rng.uniform(low=0.1, high=1.0, size=N).astype(np.float32)
        p_e_taken_flat = p_b_taken_flat.copy()

        # Sticky can be anything; rho is already 1 so correction doesn't change it
        sticky_act_flat = np.zeros(N, dtype=np.int64)
        for ep in range(num_eps):
            start = ep * steps_per_episode
            # switch on at t=2 for half episodes
            if ep % 2 == 0:
                sticky_act_flat[start + 2 : start + steps_per_episode] = 1

        est = StickyTrajectoryWiseIS(steps_per_episode=steps_per_episode, eps=1e-12)

        est.set_parameters(
            p_e_taken_flat=p_e_taken_flat,
            p_b_taken_flat=p_b_taken_flat,
            rew_flat=rew_flat,
            sticky_act_flat=sticky_act_flat,
        )

        v_is = est.estimate_policy_value()
        v_snis = est.estimate_self_normalized_value()

        self.assertTrue(np.isfinite(v_is))
        self.assertTrue(np.isfinite(v_snis))

        # Expected: mean over episodes of G_i where G_i = sum_t r_{i,t}
        G = rew_flat.reshape(num_eps, steps_per_episode).sum(axis=1)
        expected = float(np.mean(G))

        self.assertAlmostEqual(v_is, expected, places=6)
        self.assertAlmostEqual(v_snis, expected, places=6)

    def test_sticky_trajectory_wise_is_requires_parameters(self):
        est = StickyTrajectoryWiseIS(steps_per_episode=5)
        with self.assertRaises(ValueError):
            _ = est.estimate_policy_value()

    def test_sticky_trajectory_wise_is_requires_divisible_length(self):
        est = StickyTrajectoryWiseIS(steps_per_episode=5)

        N = 12  # not divisible by 5
        p = np.full(N, 0.5, dtype=np.float32)
        r = np.ones(N, dtype=np.float32)
        sticky = np.zeros(N, dtype=np.int64)

        est.set_parameters(
            p_e_taken_flat=p,
            p_b_taken_flat=p,
            rew_flat=r,
            sticky_act_flat=sticky,
        )

        with self.assertRaises(ValueError):
            _ = est.estimate_policy_value()

    def test_sticky_sequential_dr_components_shapes(self):
        rng = np.random.default_rng(0)
        steps_per_episode = 5
        num_eps = 8
        N = steps_per_episode * num_eps

        rew_flat = rng.normal(size=N).astype(np.float32)
        act_flat = rng.integers(0, 2, size=N, dtype=np.int64)

        p_b_taken_flat = rng.uniform(0.2, 0.9, size=N).astype(np.float32)

        P_new = rng.uniform(0.1, 0.9, size=(N, 2)).astype(np.float32)
        P_new /= P_new.sum(axis=1, keepdims=True)

        sticky_act_flat = np.zeros(N, dtype=np.int64)

        Q0 = rng.normal(size=N).astype(np.float32)
        Q1 = rng.normal(size=N).astype(np.float32)

        est = StickySequentialDoublyRobust(
            steps_per_episode=steps_per_episode,
            gamma=1.0,
            cap=20.0,
            eps=1e-12,
        )

        est.set_parameters(
            rew_flat=rew_flat,
            act_flat=act_flat,
            p_b_taken_flat=p_b_taken_flat,
            P_new=P_new,
            sticky_act_flat=sticky_act_flat,
            Q0=Q0,
            Q1=Q1,
        )

        rho, W_t, dr_episode = est.estimate_components()

        self.assertEqual(rho.shape, (num_eps, steps_per_episode))
        self.assertEqual(W_t.shape, (num_eps, steps_per_episode))
        self.assertEqual(dr_episode.shape, (num_eps,))
        self.assertTrue(np.isfinite(rho).all())
        self.assertTrue(np.isfinite(W_t).all())
        self.assertTrue(np.isfinite(dr_episode).all())

    def test_sticky_sequential_dr_policy_value_is_mean_of_dr_episode(self):
        rng = np.random.default_rng(1)
        steps_per_episode = 4
        num_eps = 10
        N = steps_per_episode * num_eps

        rew_flat = rng.normal(size=N).astype(np.float32)
        act_flat = rng.integers(0, 2, size=N, dtype=np.int64)

        p_b_taken_flat = rng.uniform(0.2, 0.9, size=N).astype(np.float32)

        P_new = rng.uniform(0.1, 0.9, size=(N, 2)).astype(np.float32)
        P_new /= P_new.sum(axis=1, keepdims=True)

        sticky_act_flat = np.zeros(N, dtype=np.int64)

        Q0 = rng.normal(size=N).astype(np.float32)
        Q1 = rng.normal(size=N).astype(np.float32)

        est = StickySequentialDoublyRobust(steps_per_episode=steps_per_episode)

        est.set_parameters(
            rew_flat=rew_flat,
            act_flat=act_flat,
            p_b_taken_flat=p_b_taken_flat,
            P_new=P_new,
            sticky_act_flat=sticky_act_flat,
            Q0=Q0,
            Q1=Q1,
        )

        rho, W_t, dr_episode = est.estimate_components()
        v = est.estimate_policy_value()

        self.assertTrue(np.isfinite(v))
        self.assertAlmostEqual(v, float(np.mean(dr_episode)), places=6)

    def test_sticky_sequential_dr_missing_params_raises(self):
        est = StickySequentialDoublyRobust(steps_per_episode=4)
        with self.assertRaises(ValueError):
            est.estimate_policy_value()

    def test_sticky_sequential_dr_identity_policy_has_finite_output(self):
        """
        Sanity: if πe(a|s) matches πb on the logged actions, ratios are well-behaved.
        This does NOT enforce a specific numeric value, but catches NaN/Inf explosions.
        """

        rng = np.random.default_rng(2)
        steps_per_episode = 6
        num_eps = 7
        N = steps_per_episode * num_eps

        rew_flat = rng.normal(size=N).astype(np.float32)
        act_flat = rng.integers(0, 2, size=N, dtype=np.int64)

        # behavior taken prob
        p_b_taken_flat = rng.uniform(0.2, 0.9, size=N).astype(np.float32)

        # Build a P_new such that P_new[idx, act_flat] == p_b_taken_flat
        P_new = np.zeros((N, 2), dtype=np.float32)
        idx = np.arange(N)
        P_new[idx, act_flat] = p_b_taken_flat
        P_new[idx, 1 - act_flat] = 1.0 - p_b_taken_flat

        sticky_act_flat = np.zeros(N, dtype=np.int64)

        Q0 = rng.normal(size=N).astype(np.float32)
        Q1 = rng.normal(size=N).astype(np.float32)

        est = StickySequentialDoublyRobust(steps_per_episode=steps_per_episode)

        est.set_parameters(
            rew_flat=rew_flat,
            act_flat=act_flat,
            p_b_taken_flat=p_b_taken_flat,
            P_new=P_new,
            sticky_act_flat=sticky_act_flat,
            Q0=Q0,
            Q1=Q1,
        )

        v = est.estimate_policy_value()
        self.assertTrue(np.isfinite(v))
