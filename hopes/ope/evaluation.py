import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dataclasses_json import dataclass_json
from tabulate import tabulate

from hopes.ope.estimators import BaseEstimator
from hopes.policy import Policy

"""
TODO:
- workflow to run several estimators
- option to fail fast if one estimator fails, or keep going
- option to run estimators in parallel or sequentially
- compute several metrics for each estimator
    - policy value
    - lower bound
    - upper bound
    - mean and variance

- return policies compared by:
    - mean value
    - lower bound
    - upper bound
"""


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
    results: dict[str, OffPolicyEvaluationResult]
    significance_level: float

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.results, orient="index")

    def __str__(self):
        table = tabulate(self.as_dataframe(), headers="keys", tablefmt="rounded_grid")
        return f"{table}\nSignificance level: {self.significance_level}"


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

        results.as_dataframe()

    Expected output:

    =====  =======  =======  =============  =============
    ..        mean      std    lower_bound    upper_bound
    =====  =======  =======  =============  =============
    IPW    499.368  5.28645        491.011        507.669
    SNIPW  499.538  5.21046        490.714        507.695
    =====  =======  =======  =============  =============
    """

    def __init__(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        target_policy: Policy,
        behavior_policy: Policy,
        estimators: list[BaseEstimator],
        fail_fast: bool = True,
        significance_level: float = 0.05,
    ):
        """Initialize the off-policy evaluation.

        :param obs: the observations for which to evaluate the target policy
        :param rewards: the rewards associated with the observations, collected using the
            behavior policy
        :param target_policy: the target policy to evaluate
        :param behavior_policy: the behavior policy used to generate the data
        :param estimators: a list of estimators to use to evaluate the target policy
        :param fail_fast: whether to stop the evaluation if one estimator fails
        :param significance_level: the significance level for the confidence intervals
        """
        assert isinstance(obs, np.ndarray), "obs must be a numpy array"
        assert len(obs.shape) == 2, "obs must be a 2D array"
        assert isinstance(rewards, np.ndarray), "rewards must be a numpy array"
        assert len(rewards.shape) == 1, "rewards must be a 1D array"
        assert isinstance(target_policy, Policy), "target_policy must be an instance of Policy"
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
        self.target_policy = target_policy
        self.behavior_policy = behavior_policy
        self.estimators = estimators
        self.fail_fast = fail_fast
        self.significance_level = significance_level

    def evaluate(self) -> OffPolicyEvaluationResults:
        """Run the off-policy evaluation and return the estimated value of the target policy.

        :return: a dict of OffPolicyEvaluationResult instances, one for each estimator
        """

        target_policy_action_probabilities = self.target_policy.compute_action_probs(self.obs)
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
                results[estimator.short_name()] = OffPolicyEvaluationResult.from_dict(eval_results)

            except Exception as e:
                msg = f"Estimator {estimator} failed with exception: {e}"
                if self.fail_fast:
                    logging.error(msg)
                    raise e
                else:
                    logging.warning(msg)

        return OffPolicyEvaluationResults(results, significance_level=self.significance_level)
