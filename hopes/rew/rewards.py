from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class RewardModel(ABC):
    @abstractmethod
    def estimate(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        """Estimate the rewards for a given set of observations and actions.

        :param obs: the observations for which to estimate the rewards, shape: (batch_size,
            obs_dim).
        :param act: the actions for which to estimate the rewards, shape: (batch_size,).
        :return: the estimated rewards.
        """
        raise NotImplementedError


class RewardFunctionModel(RewardModel):
    """A reward model that uses a given reward function to estimate rewards."""

    def __init__(self, reward_function: callable) -> None:
        """
        :param reward_function: a function that takes in observations and actions and returns rewards.
        """
        self.reward_function = reward_function

    def estimate(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            return self.reward_function(obs, act)
        else:
            return np.array([self.reward_function(o, a) for o, a in zip(obs, act)])


class RegressionBasedRewardModel(RewardModel):
    def __init__(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        reward_model: str = "linear",
        model_params: dict = {},
    ) -> None:
        """
        :param obs: the observations for training the reward model, shape: (batch_size, obs_dim).
        :param act: the actions for training the reward model, shape: (batch_size,).
        :param rew: the rewards for training the reward model, shape: (batch_size,).
        :param reward_model: the type of reward model to use. For now, only linear, polynomial and mlp are supported.
        :param model_params: optional parameters for the reward model.
        """
        supported_reward_models = ["linear", "polynomial", "mlp"]
        assert (
            reward_model in supported_reward_models
        ), f"Only {supported_reward_models} supported for now."
        assert obs.ndim == 2, "Observations must have shape (batch_size, obs_dim)."
        assert (
            obs.shape[0] == act.shape[0] == rew.shape[0]
        ), "The number of observations, actions, and rewards must be the same."

        self.obs = obs
        if act.ndim == 1:
            act = act.reshape(-1, 1)
        self.act = act
        if rew.ndim == 1:
            rew = rew.reshape(-1, 1)
        self.rew = rew
        self.model_params = model_params
        self.reward_model = reward_model
        self.poly_features = None

        if self.reward_model == "linear" or self.reward_model == "polynomial":
            self.model = LinearRegression()
        elif self.reward_model == "mlp":
            hidden_size = model_params.get("hidden_size", 64)
            activation = model_params.get("activation", "relu")
            act_cls = torch.nn.ReLU if activation == "relu" else torch.nn.Tanh
            self.model = torch.nn.Sequential(
                torch.nn.Linear(obs.shape[1] + act.shape[1], hidden_size),
                act_cls(),
                torch.nn.Linear(hidden_size, 1),
            )

    def fit(self) -> None:
        model_in = np.concatenate((self.obs, self.act), axis=1)

        if self.reward_model == "mlp":
            optimizer = torch.optim.Adam(self.model.parameters())
            criterion = torch.nn.MSELoss()
            for e in range(self.model_params.get("num_epochs", 1000)):
                optimizer.zero_grad()
                pred_rew = self.model(torch.tensor(model_in, dtype=torch.float32))
                loss = criterion(pred_rew, torch.tensor(self.rew, dtype=torch.float32))
                loss.backward()
                optimizer.step()

        elif self.reward_model == "polynomial":
            self.poly_features = PolynomialFeatures(degree=self.model_params.get("degree", 2))
            self.model.fit(self.poly_features.fit_transform(model_in), self.rew)

        elif isinstance(self.model, LinearRegression):
            self.model.fit(np.concatenate((self.obs, self.act), axis=1), self.rew)

    def estimate(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        """Estimate the rewards for a given set of observations and actions.

        :param obs: the observations for which to estimate the rewards, shape: (batch_size,
            obs_dim).
        :param act: the actions for which to estimate the rewards, shape: (batch_size,).
        :return: the estimated rewards, shape: (batch_size,).
        """
        if act.ndim == 1:
            act = act.reshape(-1, 1)

        if isinstance(self.model, torch.nn.Module):
            with torch.no_grad():
                return (
                    self.model(
                        torch.tensor(np.concatenate((obs, act), axis=1), dtype=torch.float32)
                    )
                    .numpy()
                    .flatten()
                )
        else:
            inputs = np.concatenate((obs, act), axis=1)
            if self.reward_model == "polynomial":
                inputs = self.poly_features.transform(inputs)
            return np.squeeze(self.model.predict(inputs))
