import unittest

import numpy as np

from hopes.policy.spaces import discretize_action_space


class TestSpaces(unittest.TestCase):
    def test_discretize_space(self):
        actions = np.arange(0, 100, 0.1)
        bins, binned_actions = discretize_action_space(actions=actions, bins=10)
        self.assertTrue(np.all(binned_actions >= 0))
        self.assertTrue(np.all(binned_actions <= 100))
        self.assertEqual(len(bins), 10)

        bins, binned_actions = discretize_action_space(
            actions=actions, bins=[0, 20, 40, 60, 80, 100]
        )
        self.assertTrue(np.all(binned_actions >= 0))
        self.assertTrue(np.all(binned_actions <= 100))
        self.assertEqual(len(bins), 6)
        self.assertTrue(np.all(binned_actions % 20 == 0))
