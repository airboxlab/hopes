import numpy as np

from hopes.policy.policies import Policy


def compute_action_probs_from_policy(policy: Policy, obs: np.ndarray) -> np.ndarray:
    """Compute the action probabilities under a given policy for a given set of observations.

    :param policy: the policy for which to compute the action probabilities.
    :param obs: the observation for which to compute the action probabilities.
    :return: the action probabilities.
    """
    log_likelihoods = policy.log_likelihoods(obs)
    action_probs = np.exp(log_likelihoods)
    return action_probs
