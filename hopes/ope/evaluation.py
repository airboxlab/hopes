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
    """A result of the off-policy evaluation of a target policy.

    It includes:

    - the mean estimate of the policy value
    - the standard deviation of the estimate
    - the lower bound of the confidence interval
    - the upper bound of the confidence interval
    """

    # mean estimate of the policy value
    mean: float
    # standard deviation of the estimate
    std: float
    # lower bound of the confidence interval
    lower_bound: float
    # upper bound of the confidence interval
    upper_bound: float


@dataclass
class OffPolicyEvaluationResults:
    """The results of the off-policy evaluation of a target policy using multiple estimators."""

    # the name of the target policy that was evaluated
    policy_name: str
    # the results of the evaluation for each estimator
    results: dict[str, OffPolicyEvaluationResult]
    # the significance level used to compute the confidence intervals
    significance_level: float

    def as_dataframe(self) -> pd.DataFrame:
        """Return the results as a pandas DataFrame."""
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
        ci_method: str = "bootstrap",
        ci_significance_level: float = 0.05,
    ):
        """Initialize the off-policy evaluation.

        :param obs: the observations for which to evaluate the target policy
        :param rewards: the rewards associated with the observations, collected using the
            behavior policy
        :param behavior_policy: the behavior policy used to generate the data
        :param estimators: a list of estimators to use to evaluate the target policy
        :param fail_fast: whether to stop the evaluation if one estimator fails
        :param ci_method: the method to use to compute the confidence intervals. Can be
            "bootstrap" or "t-test"
        :param ci_significance_level: the significance level for the confidence intervals
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
        assert ci_method in ["bootstrap", "t-test"], "ci_method must be 'bootstrap' or 't-test'"
        assert isinstance(ci_significance_level, float), "significance_level must be a float"
        assert 0 < ci_significance_level < 1, "significance_level must be in (0, 1)"

        self.obs = obs
        self.rewards = rewards
        self.behavior_policy = behavior_policy
        self.estimators = estimators
        self.fail_fast = fail_fast
        self.ci_method = ci_method
        self.significance_level = ci_significance_level

    def evaluate(self, target_policy: Policy) -> OffPolicyEvaluationResults:
        """Run the off-policy evaluation and return the estimated value of the target policy.

        :return: a dict of OffPolicyEvaluationResult instances, one for each estimator
        """
        assert isinstance(target_policy, Policy), "target_policy must be an instance of Policy"

        target_policy_action_probabilities = target_policy.compute_action_probs(self.obs)
        behavior_policy_action_probabilities = self.behavior_policy.compute_action_probs(self.obs)

        results = {}

        # run the evaluation for each estimator
        for estimator in self.estimators:
            try:
                estimator.set_parameters(
                    target_policy_action_probabilities=target_policy_action_probabilities,
                    behavior_policy_action_probabilities=behavior_policy_action_probabilities,
                    rewards=self.rewards,
                )

                eval_results = estimator.estimate_policy_value_with_confidence_interval(
                    method=self.ci_method, significance_level=self.significance_level
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
            results={e: OffPolicyEvaluationResult.from_dict(r) for e, r in results.items()},  # type: ignore
            significance_level=self.significance_level,
        )
