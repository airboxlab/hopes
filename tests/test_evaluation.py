import unittest

import numpy as np

from hopes.ope.estimators import (
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
)
from hopes.ope.evaluation import OffPolicyEvaluation
from hopes.ope.selection import OffPolicySelection
from hopes.policy import ClassificationBasedPolicy, RandomPolicy


class TestEvaluation(unittest.TestCase):
    def test_ope(self):
        num_actions = 3
        num_obs = 5
        num_samples = 1000
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)
        rew = np.random.rand(num_samples)

        # create the behavior policy
        behavior_policy = ClassificationBasedPolicy(
            obs=obs, act=act, classification_model="logistic"
        )
        behavior_policy.fit()

        # create the target policy
        target_policy = RandomPolicy(num_actions=num_actions)

        # initialize the estimators
        estimators = [
            InverseProbabilityWeighting(),
            SelfNormalizedInverseProbabilityWeighting(),
        ]

        # run the off-policy evaluation
        ope = OffPolicyEvaluation(
            obs=obs,
            rewards=rew,
            behavior_policy=behavior_policy,
            estimators=estimators,
            fail_fast=True,
            significance_level=0.1,
        )
        results = ope.evaluate(target_policy)
        self.assertEqual(0.1, results.significance_level)

        results_df = results.as_dataframe()
        self.assertEqual(results_df.shape[0], len(estimators))
        self.assertEqual(results_df.shape[1], 4)
        for metric in ["mean", "lower_bound", "upper_bound", "std"]:
            self.assertIn(metric, results_df.columns)

        print(results)

    def test_ops(self):
        num_actions = 3
        num_obs = 5
        num_samples = 1000
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)
        rew = np.random.rand(num_samples)

        # create the behavior policy
        behavior_policy = ClassificationBasedPolicy(
            obs=obs, act=act, classification_model="logistic"
        )
        behavior_policy.fit()

        # create the target policies
        target_policy_1 = RandomPolicy(num_actions=num_actions).with_name("p1")
        target_policy_2 = RandomPolicy(num_actions=num_actions).with_name("p2")
        target_policy_3 = ClassificationBasedPolicy(
            obs=obs, act=act, classification_model="random_forest"
        ).with_name("p3")
        target_policy_3.fit()

        # initialize the estimators
        estimators = [
            InverseProbabilityWeighting(),
            SelfNormalizedInverseProbabilityWeighting(),
        ]

        # run the off-policy evaluation
        ope = OffPolicyEvaluation(
            obs=obs,
            rewards=rew,
            behavior_policy=behavior_policy,
            estimators=estimators,
            fail_fast=True,
            significance_level=0.1,
        )

        results = []
        for target_policy in [target_policy_1, target_policy_2, target_policy_3]:
            results.append(ope.evaluate(target_policy))

        top_k_results = OffPolicySelection.select_top_k(results)
        self.assertEqual(len(top_k_results), 1)
        print(top_k_results[0])

        top_k_results = OffPolicySelection.select_top_k(results, metric="lower_bound")
        self.assertEqual(len(top_k_results), 1)
        print(top_k_results[0])
