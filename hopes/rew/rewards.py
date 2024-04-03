from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class RewardModel(ABC):
    scaler = None

    @abstractmethod
    def estimate(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        """Estimate the rewards for a given set of observations and actions.

        :param obs: the observations for which to estimate the rewards, shape: (batch_size,
            obs_dim).
        :param act: the actions for which to estimate the rewards, shape: (batch_size,).
        :return: the estimated rewards.
        """
        raise NotImplementedError

    def with_scaler(self, scaler: callable) -> "RewardModel":
        assert callable(scaler), "scaler must be a callable"
        self.scaler = scaler
        return self

    def _scale(self, rew: np.ndarray) -> np.ndarray:
        if self.scaler:
            return self.scaler(rew.reshape(-1, 1) if rew.ndim == 1 else rew)
        else:
            return rew


class RewardFunctionModel(RewardModel):
    """A reward model that uses a given reward function to estimate rewards."""

    def __init__(self, reward_function: callable) -> None:
        """
        :param reward_function: a function that takes in observations and actions and returns rewards.
        """
        assert callable(reward_function), "reward_function must be a callable"
        assert (
            reward_function.__code__.co_argcount == 2
        ), "reward_function must take two arguments (obs, act)"
        self.reward_function = reward_function

    def estimate(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        assert (
            obs.shape[0] == act.shape[0]
        ), "The number of pairs of observations and actions passed to the reward function must be the same."
        act = act.reshape(-1, 1) if act.ndim == 1 else act
        return np.vectorize(self._scale)([self.reward_function(o, a) for o, a in zip(obs, act)])


class RegressionBasedRewardModel(RewardModel):
    """A reward model that uses a fitted regression model to estimate rewards."""

    def __init__(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        regression_model: str = "linear",
        model_params: dict | None = None,
    ) -> None:
        """
        :param obs: the observations for training the reward model, shape: (batch_size, obs_dim).
        :param act: the actions for training the reward model, shape: (batch_size,).
        :param rew: the rewards for training the reward model, shape: (batch_size,).
        :param regression_model: the type of reward model to use. For now, only linear, polynomial, random_forest
            and mlp are supported.
        :param model_params: optional parameters for the reward model.
        """
        supported_reward_models = ["linear", "polynomial", "mlp", "random_forest"]

        assert (
            regression_model in supported_reward_models
        ), f"Only {supported_reward_models} supported for now."
        assert obs.ndim == 2, "Observations must have shape (batch_size, obs_dim)."
        assert (
            obs.shape[0] == act.shape[0] == rew.shape[0]
        ), "The number of observations, actions, and rewards must be the same."

        self.obs = obs
        self.act = act.reshape(-1, 1) if act.ndim == 1 else act
        self.rew = rew.reshape(-1, 1) if rew.ndim == 1 else rew
        self.model_params = model_params or {}
        self.regression_model = regression_model
        self.poly_features = None

        # both linear and polynomial models are implemented using sklearn LinearRegression
        # for polynomial model, we use PolynomialFeatures to generate polynomial features then fit the linear model
        if self.regression_model == "linear" or self.regression_model == "polynomial":
            self.model = LinearRegression()

        # mlp model is implemented using torch. We use a simple feedforward neural network and MSE loss.
        # configuration is basic for now, but can be extended in the future
        elif self.regression_model == "mlp":
            hidden_size = self.model_params.get("hidden_size", 64)
            activation = self.model_params.get("activation", "relu")
            act_cls = torch.nn.ReLU if activation == "relu" else torch.nn.Tanh
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.obs.shape[1] + self.act.shape[1], hidden_size),
                act_cls(),
                torch.nn.Linear(hidden_size, 1),
            )

        elif self.regression_model == "random_forest":
            self.model = RandomForestRegressor(
                max_depth=self.model_params.get("max_depth", 10),
                n_estimators=self.model_params.get("n_estimators", 100),
            )

    def fit(self) -> dict[str, float] | None:
        """Fit the reward model to the training data."""
        model_in = np.concatenate((self.obs, self.act), axis=1)

        if self.regression_model == "mlp":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.model_params.get("lr", 0.01)
            )
            criterion = torch.nn.MSELoss()
            for _ in range(self.model_params.get("num_epochs", 1000)):
                optimizer.zero_grad()
                pred_rew = self.model(torch.tensor(model_in, dtype=torch.float32))
                loss = criterion(pred_rew, torch.tensor(self.rew, dtype=torch.float32))
                loss.backward()
                optimizer.step()

        elif self.regression_model == "polynomial":
            self.poly_features = PolynomialFeatures(degree=self.model_params.get("degree", 2))
            self.model.fit(self.poly_features.fit_transform(model_in), self.rew)

        elif self.regression_model == "linear" or self.regression_model == "random_forest":
            self.model.fit(model_in, self.rew)

        # report RMSE
        pred_rew = self.estimate(self.obs, self.act)
        rmse = np.sqrt(np.mean((pred_rew - self.rew) ** 2))
        return {"rmse": rmse}

    def estimate(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        """Estimate the rewards for a given set of observations and actions.

        :param obs: the observations for which to estimate the rewards, shape: (batch_size,
            obs_dim).
        :param act: the actions for which to estimate the rewards, shape: (batch_size,).
        :return: the estimated rewards, shape: (batch_size,).
        """
        if act.ndim == 1:
            act = act.reshape(-1, 1)

        inputs = np.concatenate((obs, act), axis=1)

        if self.regression_model == "mlp":
            with torch.no_grad():
                rew = self.model(torch.tensor(inputs, dtype=torch.float32)).numpy().flatten()
        else:
            if self.regression_model == "polynomial":
                inputs = self.poly_features.transform(inputs)
            rew = np.squeeze(self.model.predict(inputs))

        return self._scale(rew)
