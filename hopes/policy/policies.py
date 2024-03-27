from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):

    @abstractmethod
    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(Policy):

    def __init__(self, num_actions: int):
        assert num_actions > 0, "Number of actions must be positive."
        self.num_actions = num_actions

    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        assert obs.ndim == 2, "Observations must have shape (batch_size, obs_dim)."
        assert obs.shape[1] > 0, "Observations must have shape (batch_size, obs_dim)."

        action_probs = np.random.rand(obs.shape[0], self.num_actions)
        action_probs /= action_probs.sum(axis=1, keepdims=True)
        return np.log(action_probs)

