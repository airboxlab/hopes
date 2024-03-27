from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression

from hopes.dev_utils import override


class Policy(ABC):
    @abstractmethod
    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        """Compute the log-likelihoods of the actions under the policy for a given set of
        observations.

        :param obs: the observation for which to compute the log-likelihoods, shape:
            (batch_size, obs_dim).
        """
        raise NotImplementedError

    def compute_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Compute the action probabilities under a given policy for a given set of observations.

        :param obs: the observation for which to compute the action probabilities.
        :return: the action probabilities.
        """
        assert obs.ndim == 2, "Observations must have shape (batch_size, obs_dim)."
        assert obs.shape[1] > 0, "Observations must have at least one feature."

        log_likelihoods = self.log_likelihoods(obs)
        action_probs = np.exp(log_likelihoods)
        return action_probs


class RandomPolicy(Policy):
    """A random policy that selects actions uniformly at random."""

    def __init__(self, num_actions: int):
        assert num_actions > 0, "Number of actions must be positive."
        self.num_actions = num_actions

    @override(Policy)
    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        action_probs = np.random.rand(obs.shape[0], self.num_actions)
        action_probs /= action_probs.sum(axis=1, keepdims=True)
        return np.log(action_probs)


class RegressionBasedPolicy(Policy):
    """A policy that uses a regression model to predict the log-likelihoods of actions given
    observations."""

    def __init__(
        self, obs: np.ndarray, act: np.ndarray, regression_model: str = "logistic"
    ) -> None:
        """
        :param obs: the observations for training the regression model, shape: (batch_size, obs_dim).
        :param act: the actions for training the regression model, shape: (batch_size,).
        :param regression_model: the type of regression model to use. For now, only logistic is supported.
        """
        assert regression_model in ["logistic"], "Only logistic regression is supported for now."
        assert obs.ndim == 2, "Observations must have shape (batch_size, obs_dim)."
        assert obs.shape[0] == act.shape[0], "Number of observations and actions must match."

        self.model_x = obs
        self.model_y = act
        self.model = LogisticRegression()

    def fit(self):
        self.model.fit(self.model_x, self.model_y)

    @override(Policy)
    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        return self.model.predict_log_proba(obs)


class HttpPolicy(Policy):
    """A policy that uses a remote HTTP server that returns log likelihoods for actions given
    observations."""

    def __init__(
        self,
        host: str,
        path: str,
        request_payload_fun: callable = lambda obs: {"obs": obs.tolist()},
        response_payload_fun: callable = lambda response: np.array(
            response.json()["log_likelihoods"]
        ),
        request_method: str = "POST",
        ssl: bool = False,
        port: int = 80,
        verify_ssl: bool = True,
    ) -> None:
        """
        :param host: the host of the HTTP server.
        :param path: the path of the HTTP server.
        :param request_payload_fun: a function that takes observations as input and returns the payload for the request.
        :param response_payload_fun: a function that takes the response from the server and returns the log likelihoods.
        :param request_method: the HTTP request method.
        :param ssl: whether to use SSL.
        :param port: the port of the HTTP server.
        :param verify_ssl: whether to verify the SSL certificate.
        """
        self.host = host
        self.port = port
        self.path = path
        self.request_payload_fun = request_payload_fun
        self.response_payload_fun = response_payload_fun
        self.request_method = request_method
        self.ssl = ssl
        self.verify_ssl = verify_ssl

    @override(Policy)
    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
