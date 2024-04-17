import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dataclasses_json import dataclass_json
from tabulate import tabulate

from hopes.ope.estimators import BaseEstimator
from hopes.policy import Policy


@dataclass_json
@dataclass
class OffPolicyEvaluationResult:
    """A result of the off-policy evaluation of a target policy."""

    mean: float
    std: float
    lower_bound: float
    upper_bound: float


@dataclass
class OffPolicyEvaluationResults:
    policy_name: str
    results: dict[str, OffPolicyEvaluationResult]
    significance_level: float

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.results, orient="index")

    def __str__(self):
        table = tabulate(self.as_dataframe(), headers="keys", tablefmt="rounded_grid")
        return (
            f"Policy: {self.policy_name}"
            f"\nConfidence interval: ±{100 * (1 - self.significance_level)}%"
            f"\n{table}"
        )


class OffPolicyEvaluation:
    """Off-Policy evaluation of a target policy using a behavior policy and a set of estimators.

    Example usage:

    .. code-block:: python

        # create the behavior policy
        behavior_policy = ClassificationBasedPolicy(obs=collected_obs, act=collected_act, classification_model="logistic")
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
            target_policy=target_policy,
            behavior_policy=behavior_policy,
            estimators=estimators,
            fail_fast=True,
            significance_level=0.1
        )
        results = ope.evaluate()
    """

    def __init__(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        behavior_policy: Policy,
        estimators: list[BaseEstimator],
        fail_fast: bool = True,
        significance_level: float = 0.05,
    ):
        """Initialize the off-policy evaluation.

        :param obs: the observations for which to evaluate the target policy
        :param rewards: the rewards associated with the observations, collected using the
            behavior policy
        :param behavior_policy: the behavior policy used to generate the data
        :param estimators: a list of estimators to use to evaluate the target policy
        :param fail_fast: whether to stop the evaluation if one estimator fails
        :param significance_level: the significance level for the confidence intervals
        """
        assert isinstance(obs, np.ndarray), "obs must be a numpy array"
        assert len(obs.shape) == 2, "obs must be a 2D array"
        assert isinstance(rewards, np.ndarray), "rewards must be a numpy array"
        assert len(rewards.shape) == 1, "rewards must be a 1D array"
        assert isinstance(behavior_policy, Policy), "behavior_policy must be an instance of Policy"
        assert len(estimators) > 0, "estimators must be a non-empty list"
        assert all(
            [isinstance(estimator, BaseEstimator) for estimator in estimators]
        ), "estimators must be a list of BaseEstimator instances"
        assert isinstance(fail_fast, bool), "fail_fast must be a boolean"
        assert isinstance(significance_level, float), "significance_level must be a float"
        assert 0 < significance_level < 1, "significance_level must be in (0, 1)"

        self.obs = obs
        self.rewards = rewards
        self.behavior_policy = behavior_policy
        self.estimators = estimators
        self.fail_fast = fail_fast
        self.significance_level = significance_level

    def evaluate(self, target_policy: Policy) -> OffPolicyEvaluationResults:
        """Run the off-policy evaluation and return the estimated value of the target policy.

        :return: a dict of OffPolicyEvaluationResult instances, one for each estimator
        """
        assert isinstance(target_policy, Policy), "target_policy must be an instance of Policy"

        target_policy_action_probabilities = target_policy.compute_action_probs(self.obs)
        behavior_policy_action_probabilities = self.behavior_policy.compute_action_probs(self.obs)

        results = {}

        for estimator in self.estimators:
            try:
                estimator.set_parameters(
                    target_policy_action_probabilities=target_policy_action_probabilities,
                    behavior_policy_action_probabilities=behavior_policy_action_probabilities,
                    rewards=self.rewards,
                )

                eval_results = estimator.estimate_policy_value_with_confidence_interval(
                    significance_level=self.significance_level
                )
                results[estimator.short_name()] = eval_results

            except Exception as e:
                msg = f"Estimator {estimator} failed with exception: {e}"
                if self.fail_fast:
                    logging.error(msg)
                    raise e
                else:
                    logging.warning(msg)

        return OffPolicyEvaluationResults(
            policy_name=target_policy.name,
            results={e: OffPolicyEvaluationResult.from_dict(r) for e, r in results.items()},
            significance_level=self.significance_level,
        )
