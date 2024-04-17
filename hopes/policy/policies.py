"""Policy classes to help model a target or behavior policy."""
from abc import ABC, abstractmethod

import numpy as np
import pwlf
import requests
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from hopes.dev_utils import override
from hopes.policy.utils import bin_actions, deterministic_log_probs


class Policy(ABC):
    """An abstract class for policies.

    The policy must be subclassed and the log_likelihoods method must be implemented.
    """

    _name: str | None = None
    _epsilon: float | None = None

    @property
    def name(self):
        return self._name or self.__class__.__name__

    def with_name(self, name: str) -> "Policy":
        """Set the name of the policy.

        :param name: the name of the policy.
        """
        self._name = name
        return self

    def with_epsilon(self, epsilon: float | None = None) -> "Policy":
        """Set the epsilon value for epsilon-greedy action selection.

        :param epsilon: the epsilon value for epsilon-greedy action selection.
        """
        assert epsilon is None or 0 <= epsilon <= 1, "Epsilon must be in [0, 1]."
        self._epsilon = epsilon
        return self

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
        # epsilon-greedy action selection
        if self._epsilon is not None and (np.random.rand() < self._epsilon):
            action_probs = np.ones_like(action_probs) / action_probs.shape[1]
        return action_probs

    def select_action(self, obs: np.ndarray, deterministic: float = False) -> np.ndarray:
        """Select actions under the policy for given observations.

        :param obs: the observation(s) for which to select an action, shape (batch_size,
            obs_dim).
        :param deterministic: whether to select actions deterministically.
        :return: the selected action(s).
        """
        assert not (
            deterministic and self._epsilon is not None
        ), "Cannot be deterministic and epsilon-greedy at the same time."

        action_probs = self.compute_action_probs(obs)

        # deterministic action selection
        if deterministic:
            return np.argmax(action_probs, axis=1)

        # action selection based on computed action probabilities
        else:
            return np.array([np.random.choice(len(probs), p=probs) for probs in action_probs])


class RandomPolicy(Policy):
    """A random policy that selects actions uniformly at random.

    It can serve as a baseline policy for comparison with other policies.
    """

    def __init__(self, num_actions: int):
        assert num_actions > 0, "Number of actions must be positive."
        self.num_actions = num_actions

    @override(Policy)
    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        """Compute the log-likelihoods of the actions under the random policy for a given set of
        observations."""
        action_probs = np.random.rand(obs.shape[0], self.num_actions)
        action_probs /= action_probs.sum(axis=1, keepdims=True)
        return np.log(action_probs)


class ClassificationBasedPolicy(Policy):
    """A policy that uses a classification model to predict the log-likelihoods of actions given
    observations.

    In absence of an actual control policy, this can be used to train a policy on a dataset
    of (obs, act) pairs that would have been collected offline.

    It currently supports logistic regression, random forest and MLP models.

    Example usage:

    .. code-block:: python

        # train a classification-based policy
        reg_policy = ClassificationBasedPolicy(obs=train_obs, act=train_act, classification_model="random_forest")
        fit_stats = reg_policy.fit()

        # compute action probabilities for new observations
        act_probs = reg_policy.compute_action_probs(obs=new_obs)
    """

    def __init__(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        classification_model: str = "logistic",
        model_params: dict | None = None,
    ) -> None:
        """
        :param obs: the observations for training the classification model, shape: (batch_size, obs_dim).
        :param act: the actions for training the classification model, shape: (batch_size,).
        :param classification_model: the type of classification model to use. For now, only logistic, mlp and
            random_forest are supported.
        :param model_params: optional parameters for the classification model.
        """
        supported_models = ["logistic", "mlp", "random_forest"]
        assert (
            classification_model in supported_models
        ), f"Only {supported_models} supported for now."
        assert obs.ndim == 2, "Observations must have shape (batch_size, obs_dim)."
        assert obs.shape[0] == act.shape[0], "Number of observations and actions must match."

        self.model_obs = obs
        self.model_act = act
        self.num_actions = len(np.unique(act))
        self.classification_model = classification_model
        self.model_params = model_params or {}

        if self.classification_model == "logistic":
            self.model = LogisticRegression()

        elif self.classification_model == "random_forest":
            self.model = RandomForestClassifier(
                max_depth=self.model_params.get("max_depth", 10),
                n_estimators=self.model_params.get("n_estimators", 100),
            )

        elif self.classification_model == "mlp":
            hidden_size = self.model_params.get("hidden_size", 64)
            activation = self.model_params.get("activation", "relu")
            act_cls = torch.nn.ReLU if activation == "relu" else torch.nn.Tanh
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.model_obs.shape[1], hidden_size),
                act_cls(),
                torch.nn.Linear(hidden_size, hidden_size),
                act_cls(),
                torch.nn.Linear(hidden_size, self.num_actions),
            )

    def fit(self) -> dict[str, float]:
        """Fit the classification model on the training data and return performance statistics
        computed on the training data.

        :return: the accuracy and F1 score on the training data.
        """
        if self.classification_model == "mlp":
            num_epochs = self.model_params.get("num_epochs", 1000)
            lr = self.model_params.get("lr", 0.01)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()
            targets = torch.tensor(self.model_act, dtype=torch.float32).view(-1).long()

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                predicted = self.model(torch.tensor(self.model_obs, dtype=torch.float32))
                loss = criterion(predicted, targets)
                loss.backward()
                optimizer.step()

            predicted = self.model(torch.tensor(self.model_obs, dtype=torch.float32)).argmax(dim=1)
            accuracy = (predicted == targets).float().mean().item()
            f1 = f1_score(targets, predicted, average="weighted")
        else:
            self.model.fit(self.model_obs, self.model_act)
            y_pred = self.model.predict(self.model_obs)
            y_true = self.model_act
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")

        return {"accuracy": accuracy, "f1": f1}

    @override(Policy)
    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        """Compute the log-likelihoods of the actions under the classification-based policy for a
        given set of observations."""
        if self.classification_model == "mlp":
            with torch.no_grad():
                output = self.model(torch.Tensor(obs))
                return torch.log_softmax(output, dim=1).numpy()
        else:
            return self.model.predict_log_proba(obs)


class PiecewiseLinearPolicy(Policy):
    """A piecewise linear policy that selects actions based on a set of linear segments defined by
    thresholds and slopes.

    This can be used to estimate a probability distribution over actions drawn from a BMS
    reset rule, for instance an outdoor air reset that is a function of outdoor air
    temperature and is bounded by a minimum and maximum on both axis. This can also be
    helpful to model a simple schedule, where action is a function of time.

    Since the output of a piecewise linear model is deterministic, the log-likelihoods are
    computed by assuming the function is deterministic and assigning a probability of 1 to
    the action returned by the function and an almost zero probability to all other actions.

    Also, the piecewise linear policy output being continuous, we need to discretize the
    action space to compute the log-likelihoods. This is done by binning the actions to the
    nearest action in the discretized action space.
    """

    def __init__(
        self,
        num_segments: int,
        obs: np.ndarray,
        act: np.ndarray,
        actions_bins: list[float | int] | None = None,
    ):
        """
        :param num_segments: the number of segments for the piecewise linear model.
        :param obs: the observations for training the piecewise linear model, shape: (batch_size, obs_dim).
        :param act: the actions for training the piecewise linear model, shape: (batch_size,).
        :param actions_bins: the bins for discretizing the action space. If not provided, we assume the action space
            is already discretized.
        """
        assert num_segments > 0, "Number of segments must be positive."
        assert (
            len(obs.shape) == 1 or obs.shape[1] == 1
        ), "Piecewise linear policy only supports 1D observations."
        assert obs.shape[0] == act.shape[0], "Number of observations and actions must match."

        self.num_segments = num_segments
        self.model_obs = obs.squeeze() if obs.ndim == 2 else obs
        self.model_act = act.squeeze() if act.ndim == 2 else act
        self.model = None

        # bins used to discretize the action space
        self.actions_bins = actions_bins if actions_bins else np.unique(self.model_act)

    def fit(self) -> dict[str, float]:
        """Fit the piecewise linear model on the training data and return the RMSE and R² on the
        training.

        :return: the RMSE ('rmse') and R² ('r2') on the training data.
        """
        # initialize piecewise linear fit
        self.model = pwlf.PiecewiseLinFit(self.model_obs, self.model_act)

        # fit the data for specified number of segments
        self.model.fit(self.num_segments)

        y_pred = self.model.predict(self.model_obs)
        y_true = self.model_act

        # compute RMSE
        diff_res = y_true - y_pred
        ss_res = np.dot(diff_res, diff_res)
        rmse = np.sqrt(ss_res / len(y_true))

        # compute R²
        diff_tot = y_true - np.mean(y_true)
        ss_tot = np.dot(diff_tot, diff_tot)
        r_squared = 1 - (ss_res / ss_tot)

        return {"rmse": rmse, "r2": r_squared}

    @override(Policy)
    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        """Compute the log-likelihoods of the actions under the piecewise linear policy for a given
        set of observations."""
        if obs.ndim == 1:
            raw_actions = self.model.predict(obs)
        else:
            raw_actions = np.array([self.model.predict(o) for o in obs])
        # bin the action to the nearest action using the discretized action space
        actions = bin_actions(raw_actions, self.actions_bins)
        # return the log-likelihoods
        return deterministic_log_probs(actions, self.actions_bins)


class FunctionBasedPolicy(Policy):
    """A policy based on a deterministic function that maps observations to actions.

    Log-likelihoods are computed by assuming the function is deterministic and assigning a
    probability of 1 to the action returned by the function and an almost zero probability
    to all other actions.
    """

    def __init__(self, policy_function: callable, actions_bins: list[float | int]) -> None:
        """
        :param policy_function: a function that takes in observations and returns actions.
        :param actions_bins: the bins for discretizing the action space.
        """
        assert callable(policy_function), "Policy function must be callable."
        assert len(actions_bins) > 0, "Action bins must be non-empty."
        self.policy_function = policy_function
        self.actions_bins = np.array(actions_bins)

    @override(Policy)
    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        """Compute the log-likelihoods of the actions under the function-based policy for a given
        set of observations."""
        raw_actions = np.vectorize(self.policy_function)(obs)
        # bin the action to the nearest action using the discretized action space
        actions = bin_actions(raw_actions, self.actions_bins)
        # return the log-likelihoods
        return deterministic_log_probs(actions, self.actions_bins)


class HttpPolicy(Policy):
    """A policy that uses a remote HTTP server that returns log likelihoods for actions given
    observations.

    The request and response payloads processing is customizable by providing the
    request_payload_fun and response_payload_fun functions.
    """

    def __init__(
        self,
        host: str,
        path: str,
        request_payload_fun: callable = lambda obs: {"obs": obs.tolist()},
        response_payload_fun: callable = lambda response: np.array(
            response.json()["log_likelihoods"]
        ),
        request_method: str = "POST",
        headers: dict = {"content-type": "application/json"},  # noqa
        ssl: bool = False,
        port: int = 80,
        verify_ssl: bool = True,
        batch_size: int = 1,
    ) -> None:
        """
        :param host: the host of the HTTP server.
        :param path: the path of the HTTP server.
        :param request_payload_fun: a function that takes observations as input and returns the payload for the request.
        :param response_payload_fun: a function that takes the response from the server and returns the extracted log probs.
        :param request_method: the HTTP request method.
        :param headers: the headers for the HTTP request.
        :param ssl: whether to use SSL.
        :param port: the port of the HTTP server.
        :param verify_ssl: whether to verify the SSL certificate.
        :param batch_size: the batch size for sending requests to the server.
        """
        self.host = host
        self.port = port
        self.path = path
        self.request_payload_fun = request_payload_fun
        self.response_payload_fun = response_payload_fun
        self.request_method = request_method
        self.headers = headers
        self.ssl = ssl
        self.verify_ssl = verify_ssl
        self.batch_size = batch_size

        assert self.request_method in ["GET", "POST"], "Only GET and POST methods are supported."
        assert callable(self.request_payload_fun), "Request payload function must be callable."
        assert callable(self.response_payload_fun), "Response payload function must be callable."
        assert self.batch_size > 0, "Batch size must be positive."

    @override(Policy)
    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        """Compute the log-likelihoods of the actions under the HTTP policy for a given set of
        observations."""
        all_log_likelihoods = []
        for chunk in np.array_split(obs, len(obs) // self.batch_size):
            # Send HTTP request to server
            response = requests.post(
                f"http{'s' if self.ssl else ''}://{self.host}:{self.port}/{self.path}",
                json=self.request_payload_fun(chunk),
                verify=self.verify_ssl,
                headers=self.headers,
            )
            # Extract log likelihoods from response
            log_likelihoods = self.response_payload_fun(response)
            all_log_likelihoods.append(log_likelihoods)

        return np.concatenate(all_log_likelihoods, axis=0)
