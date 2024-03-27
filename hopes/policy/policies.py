from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def log_likelihoods(self, obs):
        raise NotImplementedError
