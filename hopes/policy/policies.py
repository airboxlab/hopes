from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):

    @abstractmethod
    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        """Compute the log-likelihoods of the actions under the policy for a given set of observations."""
        raise NotImplementedError

    def compute_action_probs_from_policy(self, obs: np.ndarray) -> np.ndarray:
        """Compute the action probabilities under a given policy for a given set of observations.

        :param policy: the policy for which to compute the action probabilities.
        :param obs: the observation for which to compute the action probabilities.
        :return: the action probabilities.
        """
        assert obs.ndim == 2, "Observations must have shape (batch_size, obs_dim)."
        assert obs.shape[1] > 0, "Observations must have at least one feature."

        log_likelihoods = self.log_likelihoods(obs)
        action_probs = np.exp(log_likelihoods)
        return action_probs


class RandomPolicy(Policy):

    def __init__(self, num_actions: int):
        assert num_actions > 0, "Number of actions must be positive."
        self.num_actions = num_actions

    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        action_probs = np.random.rand(obs.shape[0], self.num_actions)
        action_probs /= action_probs.sum(axis=1, keepdims=True)
        return np.log(action_probs)
