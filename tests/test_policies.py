import unittest

import numpy as np

from hopes.policy.policies import RandomPolicy, RegressionBasedPolicy


class TestPolicies(unittest.TestCase):
    def test_rnd_policy(self):
        rnd_policy = RandomPolicy(num_actions=3)

        log_probs = rnd_policy.log_likelihoods(obs=np.random.rand(10, 5))
        self.assertIsInstance(log_probs, np.ndarray)
        self.assertEqual(log_probs.shape, (10, 3))
        self.assertTrue(np.all(log_probs <= 0.0))
        self.assertTrue(np.all(log_probs >= -np.inf))

        act_probs = np.exp(log_probs)
        self.assertTrue(np.all(act_probs >= 0.0))
        self.assertTrue(np.all(act_probs <= 1.0))
        self.assertTrue(np.allclose(act_probs.sum(axis=1), 1.0))

    def test_regression_policy(self):
        # generate a random dataset of (obs, act) for target policy
        num_actions = 3
        num_obs = 5
        num_samples = 100
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)

        # create and fit a regression-based policy
        reg_policy = RegressionBasedPolicy(obs=obs, act=act, regression_model="logistic")
        reg_policy.fit()

        # check if the policy returns valid log-likelihoods
        new_obs = np.random.rand(10, num_obs)
        act_probs = reg_policy.compute_action_probs(obs=new_obs)

        self.assertIsInstance(act_probs, np.ndarray)
        self.assertEqual(act_probs.shape, (10, 3))
        self.assertTrue(np.all(act_probs >= 0.0))
        self.assertTrue(np.all(act_probs <= 1.0))
        self.assertTrue(np.allclose(act_probs.sum(axis=1), 1.0))

    def test_compute_action_probs(self):
        rnd_policy = RandomPolicy(num_actions=3)
        act_probs = rnd_policy.compute_action_probs(obs=np.random.rand(10, 5))

        self.assertIsInstance(act_probs, np.ndarray)
        self.assertEqual(act_probs.shape, (10, 3))
        self.assertTrue(np.all(act_probs >= 0.0))
        self.assertTrue(np.all(act_probs <= 1.0))
        self.assertTrue(np.allclose(act_probs.sum(axis=1), 1.0))
