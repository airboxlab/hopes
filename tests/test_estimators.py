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
