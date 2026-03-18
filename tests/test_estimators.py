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
    TrajectoryWiseImportanceSampling,
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

        self._test_ci(ipw)

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
        num_actions = 2
        num_obs = 2
        num_episodes = 3
        num_steps_per_episode = 4
        num_samples = num_episodes * num_steps_per_episode
        discount_factor = 0.9

        np.random.seed(42)
        obs = np.random.rand(num_samples, num_obs)
        act = np.array([0, 1, 0, 1] * num_episodes)
        rew = np.tile([1.0, 2.0, 3.0, 4.0], num_episodes)

        gamma = discount_factor
        expected_return = 1.0 + gamma * 2.0 + gamma**2 * 3.0 + gamma**3 * 4.0

        dm = DirectMethod(
            q_model_cls=RegressionBasedRewardModel,
            q_model_type="linear",
            behavior_policy_obs=obs,
            behavior_policy_act=act,
            behavior_policy_rewards=rew,
            steps_per_episode=num_steps_per_episode,
            discount_factor=discount_factor,
        )

        fit_stats = dm.fit()
        self.assertIsInstance(fit_stats, dict)

        self.assertEqual(dm.q_model.obs.shape[0], num_episodes)
        self.assertEqual(dm.q_model.act.shape[0], num_episodes)
        self.assertEqual(dm.q_model.rew.shape[0], num_episodes)

        for i in range(num_episodes):
            self.assertAlmostEqual(dm.q_model.rew[i], expected_return, delta=0.01)

    def test_tis(self):
        traj_length = 10
        num_episodes = 500
        num_actions = 3

        target, behavior, rewards = self._get_is_data(
            traj_length=traj_length,
            num_actions=num_actions,
            num_episodes=num_episodes,
        )

        tis = TrajectoryWiseImportanceSampling(
            steps_per_episode=traj_length,
            discount_factor=0.99,
        )
        tis.set_parameters(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            rewards=rewards,
        )

        wrew = tis.estimate_weighted_rewards()
        self.assertIsInstance(wrew, np.ndarray)
        self.assertEqual(wrew.shape, (num_episodes, 1))

        policy_value = tis.estimate_policy_value()
        self.assertIsInstance(policy_value, float)
        self.assertGreaterEqual(policy_value, 0.0)

        self._test_ci(tis)

    def test_sntis(self):
        traj_length = 10
        num_episodes = 500
        num_actions = 3

        target, behavior, rewards = self._get_is_data(
            traj_length=traj_length,
            num_actions=num_actions,
            num_episodes=num_episodes,
        )

        sntis = SelfNormalizedTrajectoryWiseImportanceSampling(
            steps_per_episode=traj_length,
            discount_factor=0.99,
        )
        sntis.set_parameters(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            rewards=rewards,
        )

        wrew = sntis.estimate_weighted_rewards()
        self.assertIsInstance(wrew, np.ndarray)
        self.assertEqual(wrew.shape, (num_episodes, 1))

        policy_value = sntis.estimate_policy_value()
        self.assertIsInstance(policy_value, float)
        self.assertGreaterEqual(policy_value, 0.0)

        self._test_ci(sntis)

    def test_pdis(self):
        traj_length = 10
        num_episodes = 500
        num_actions = 3

        target, behavior, rewards = self._get_is_data(
            traj_length=traj_length,
            num_actions=num_actions,
            num_episodes=num_episodes,
        )

        pdis = PerDecisionImportanceSampling(
            steps_per_episode=traj_length,
            discount_factor=0.99,
        )
        pdis.set_parameters(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            rewards=rewards,
        )

        wrew = pdis.estimate_weighted_rewards()
        self.assertIsInstance(wrew, np.ndarray)
        self.assertEqual(wrew.shape, (num_episodes, 1))

        policy_value = pdis.estimate_policy_value()
        self.assertIsInstance(policy_value, float)
        self.assertGreaterEqual(policy_value, 0.0)

        self._test_ci(pdis)

    def test_snpdis(self):
        traj_length = 10
        num_episodes = 500
        num_actions = 3

        target, behavior, rewards = self._get_is_data(
            traj_length=traj_length,
            num_actions=num_actions,
            num_episodes=num_episodes,
        )

        snpdis = SelfNormalizedPerDecisionImportanceSampling(
            steps_per_episode=traj_length,
            discount_factor=0.99,
        )
        snpdis.set_parameters(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            rewards=rewards,
        )

        wrew = snpdis.estimate_weighted_rewards()
        self.assertIsInstance(wrew, np.ndarray)
        self.assertEqual(wrew.shape, (num_episodes, 1))

        policy_value = snpdis.estimate_policy_value()
        self.assertIsInstance(policy_value, float)
        self.assertGreaterEqual(policy_value, 0.0)

        self._test_ci(snpdis)

    def test_tis_with_precomputed_importance_ratios_identity(self):
        steps_per_episode = 4
        num_episodes = 20
        num_actions = 3
        n_samples = steps_per_episode * num_episodes

        rng = np.random.default_rng(0)

        target = generate_action_probs(traj_length=n_samples, num_actions=num_actions)
        behavior = generate_action_probs(traj_length=n_samples, num_actions=num_actions)
        rewards = rng.random(n_samples, dtype=np.float32)

        tis = TrajectoryWiseImportanceSampling(
            steps_per_episode=steps_per_episode,
            discount_factor=1.0,
        )
        tis.set_parameters(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            rewards=rewards,
        )

        rho = np.ones(n_samples, dtype=np.float32)
        tis.set_importance_ratios(rho)

        value = tis.estimate_policy_value()

        expected = float(np.mean(rewards.reshape(num_episodes, steps_per_episode).sum(axis=1)))
        self.assertAlmostEqual(value, expected, places=6)

    def test_snpdis_with_precomputed_importance_ratios_identity(self):
        steps_per_episode = 5
        num_episodes = 12
        num_actions = 3
        n_samples = steps_per_episode * num_episodes

        rng = np.random.default_rng(1)

        target = generate_action_probs(traj_length=n_samples, num_actions=num_actions)
        behavior = generate_action_probs(traj_length=n_samples, num_actions=num_actions)
        rewards = rng.random(n_samples, dtype=np.float32)

        est = SelfNormalizedPerDecisionImportanceSampling(
            steps_per_episode=steps_per_episode,
            discount_factor=1.0,
        )
        est.set_parameters(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            rewards=rewards,
        )

        rho = np.ones(n_samples, dtype=np.float32)
        est.set_importance_ratios(rho)

        value = est.estimate_policy_value()

        expected = float(np.mean(rewards.reshape(num_episodes, steps_per_episode).sum(axis=1)))
        self.assertAlmostEqual(value, expected, places=6)
        self._test_ci(est)

    def test_snpdis_global_with_precomputed_importance_ratios_identity(self):
        steps_per_episode = 5
        num_episodes = 12
        num_actions = 3
        n_samples = steps_per_episode * num_episodes

        rng = np.random.default_rng(11)

        target = generate_action_probs(traj_length=n_samples, num_actions=num_actions)
        behavior = generate_action_probs(traj_length=n_samples, num_actions=num_actions)
        rewards = rng.random(n_samples, dtype=np.float32)

        est = SelfNormalizedPerDecisionImportanceSampling(
            steps_per_episode=steps_per_episode,
            discount_factor=1.0,
            normalization="global",
        )
        est.set_parameters(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            rewards=rewards,
        )

        rho = np.ones(n_samples, dtype=np.float32)
        est.set_importance_ratios(rho)

        value = est.estimate_policy_value()

        expected = float(np.mean(rewards.reshape(num_episodes, steps_per_episode).sum(axis=1)))
        self.assertAlmostEqual(value, expected / steps_per_episode, places=6)

    def test_snpdis_global_bootstrap_ci(self):
        steps_per_episode = 4
        num_episodes = 10
        num_actions = 2
        n_samples = steps_per_episode * num_episodes

        rng = np.random.default_rng(12)

        target = generate_action_probs(traj_length=n_samples, num_actions=num_actions)
        behavior = generate_action_probs(traj_length=n_samples, num_actions=num_actions)
        rewards = rng.random(n_samples, dtype=np.float32)

        est = SelfNormalizedPerDecisionImportanceSampling(
            steps_per_episode=steps_per_episode,
            discount_factor=1.0,
            normalization="global",
        )
        est.set_parameters(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            rewards=rewards,
        )

        rho = np.ones(n_samples, dtype=np.float32)
        est.set_importance_ratios(rho)

        metrics = est.estimate_policy_value_with_confidence_interval(
            method="bootstrap",
            significance_level=0.05,
            num_samples=200,
            random_state=0,
        )

        self.assertIsInstance(metrics, dict)
        for metric in ["mean", "lower_bound", "upper_bound", "std"]:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)

        self.assertLessEqual(metrics["lower_bound"], metrics["mean"])
        self.assertLessEqual(metrics["mean"], metrics["upper_bound"])

    def test_bootstrap_random_state_reproducibility(self):
        steps_per_episode = 4
        num_episodes = 10
        num_actions = 2
        n_samples = steps_per_episode * num_episodes

        rng = np.random.default_rng(123)

        target = generate_action_probs(traj_length=n_samples, num_actions=num_actions)
        behavior = generate_action_probs(traj_length=n_samples, num_actions=num_actions)
        rewards = rng.random(n_samples, dtype=np.float32)

        est = SelfNormalizedPerDecisionImportanceSampling(
            steps_per_episode=steps_per_episode,
            discount_factor=1.0,
            normalization="global",
        )
        est.set_parameters(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            rewards=rewards,
        )

        rho = np.ones(n_samples, dtype=np.float32)
        est.set_importance_ratios(rho)

        metrics_1 = est.estimate_policy_value_with_confidence_interval(
            method="bootstrap",
            significance_level=0.05,
            num_samples=200,
            random_state=42,
        )
        metrics_2 = est.estimate_policy_value_with_confidence_interval(
            method="bootstrap",
            significance_level=0.05,
            num_samples=200,
            random_state=42,
        )

        self.assertEqual(metrics_1, metrics_2)

    def test_sntis_bootstrap_random_state_reproducibility(self):
        traj_length = 10
        num_episodes = 100
        num_actions = 3

        target, behavior, rewards = self._get_is_data(
            traj_length=traj_length,
            num_actions=num_actions,
            num_episodes=num_episodes,
        )

        est = SelfNormalizedTrajectoryWiseImportanceSampling(
            steps_per_episode=traj_length,
            discount_factor=0.99,
        )
        est.set_parameters(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            rewards=rewards,
        )

        metrics_1 = est.estimate_policy_value_with_confidence_interval(
            method="bootstrap",
            significance_level=0.05,
            num_samples=200,
            random_state=7,
        )
        metrics_2 = est.estimate_policy_value_with_confidence_interval(
            method="bootstrap",
            significance_level=0.05,
            num_samples=200,
            random_state=7,
        )

        self.assertEqual(metrics_1, metrics_2)

    def test_snpdis_invalid_normalization_raises(self):
        steps_per_episode = 4
        num_episodes = 5
        num_actions = 2
        n_samples = steps_per_episode * num_episodes

        target = generate_action_probs(traj_length=n_samples, num_actions=num_actions)
        behavior = generate_action_probs(traj_length=n_samples, num_actions=num_actions)
        rewards = np.random.rand(n_samples).astype(np.float32)

        est = SelfNormalizedPerDecisionImportanceSampling(
            steps_per_episode=steps_per_episode,
            discount_factor=1.0,
            normalization="invalid_mode",
        )

        with self.assertRaises(ValueError):
            est.set_parameters(
                target_policy_action_probabilities=target,
                behavior_policy_action_probabilities=behavior,
                rewards=rewards,
            )

    def test_neg_rewards(self):
        ipw = InverseProbabilityWeighting()

        target_policy_action_probabilities = generate_action_probs(traj_length=10, num_actions=3)
        behavior_policy_action_probabilities = generate_action_probs(traj_length=10, num_actions=3)
        rewards = -np.random.rand(10)

        with self.assertRaises(ValueError):
            ipw.set_parameters(
                target_policy_action_probabilities=target_policy_action_probabilities,
                behavior_policy_action_probabilities=behavior_policy_action_probabilities,
                rewards=rewards,
            )

    def _test_ci(self, estimator: BaseEstimator):
        metrics = estimator.estimate_policy_value_with_confidence_interval(
            num_samples=200,
            significance_level=0.05,
            random_state=0,
        )
        self.assertIsInstance(metrics, dict)

        for metric in ["mean", "lower_bound", "upper_bound", "std"]:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)

        self.assertLessEqual(metrics["lower_bound"], metrics["mean"])
        self.assertLessEqual(metrics["mean"], metrics["upper_bound"])

    def _get_is_data(
        self,
        traj_length: int,
        num_actions: int,
        num_episodes: int,
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

        return (
            target_policy_action_probabilities,
            behavior_policy_action_probabilities,
            rewards,
        )
