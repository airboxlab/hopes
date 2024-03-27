import unittest

import numpy as np

from hopes.policy.policies import RandomPolicy


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

    def test_compute_action_probs(self):
        rnd_policy = RandomPolicy(num_actions=3)
        act_probs = rnd_policy.compute_action_probs(obs=np.random.rand(10, 5))

        self.assertIsInstance(act_probs, np.ndarray)
        self.assertEqual(act_probs.shape, (10, 3))
        self.assertTrue(np.all(act_probs >= 0.0))
        self.assertTrue(np.all(act_probs <= 1.0))
        self.assertTrue(np.allclose(act_probs.sum(axis=1), 1.0))
