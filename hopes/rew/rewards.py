from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
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

        if obs.ndim == 1:
            rew = self.reward_function(obs, act)
        else:
            rew = np.array([self.reward_function(o, a) for o, a in zip(obs, act)])

        return self._scale(rew)


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
        supported_models = ["linear", "polynomial", "mlp", "random_forest"]

        assert regression_model in supported_models, f"Only {supported_models} supported for now."
        assert obs.ndim == 2, "Observations must have shape (batch_size, obs_dim)."
        assert (
            obs.shape[0] == act.shape[0] == rew.shape[0]
        ), "The number of observations, actions, and rewards must be the same."

        self.obs = obs
        self.act = act.reshape(-1, 1) if act.ndim == 1 else act
        self.rew = rew.reshape(-1, 1) if rew.ndim == 1 else rew

        # model configuration
        self.model_params = model_params or {}
        self.regression_model = regression_model
        self.poly_features = None

        # both linear and polynomial models are implemented using sklearn LinearRegression
        # for polynomial model, we use PolynomialFeatures to generate polynomial features then fit the linear model
        if self.regression_model in ["linear", "polynomial"]:
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

    def fit(self) -> dict[str, float]:
        """Fit the reward model to the training data.

        :return: a dictionary containing the RMSE of the fitted model.
        """
        model_in = np.concatenate((self.obs, self.act), axis=1)

        if self.regression_model == "mlp":
            num_epochs = self.model_params.get("num_epochs", 100)
            lr = self.model_params.get("lr", 0.01)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = torch.nn.MSELoss()
            for _ in range(num_epochs):
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


class RTGQModelHGBoost(RewardModel):
    r"""Train a regression model approximating Q(s, a) from logged trajectories.

    The model uses logged trajectories as features and predicts
    return-to-go (RTG) targets. It can then estimate Q-values for specific (obs, act)
    pairs through `estimate(...)`, and for all actions through `predict_q_values(...)`.

    RTG target:

    .. math::

        RTG_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k

        where :math:`\gamma \in [0, 1]` is the discount factor.

    Features used for regression:

    .. math::

        X = [obs, step\_idx, action]

    where:

    - `obs` is the observation vector
    - `step_idx` is the timestep index within the episode
    - `action` is the discrete action index

    After training, the model can evaluate **all possible actions**
    to produce a matrix:

    .. math::

        Q(s,a) \in \mathbb{R}^{N \times A}

    where:

    - `N` = number of samples
    - `A` = number of discrete actions
    """

    def __init__(
        self,
        *,
        steps_per_episode: int,
        num_actions: int,
        discount_factor: float = 1.0,
        model_params: dict | None = None,
        random_state: int = 0,
    ) -> None:
        """Initialize the Q-model trainer.

        :param steps_per_episode: Number of timesteps per episode.
        :param num_actions: Number of discrete actions.
        :param model_params: Optional parameters for `HistGradientBoostingRegressor`.
        :param random_state: Random seed.
        """
        super().__init__()

        self.steps_per_episode = steps_per_episode
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.model_params = model_params or {}
        self.random_state = random_state

        self.model: HistGradientBoostingRegressor | None = None

    def _check_inputs(
        self,
        *,
        obs_flat: np.ndarray,
        act_flat: np.ndarray,
        rew_flat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Check input shapes and types.

        :param obs_flat: Shape (N, obs_dim)
        :param act_flat: Shape (N,), discrete action indices
        :param rew_flat: Shape (N,), rewards
        :return: Tuple of validated (obs_flat, act_flat, rew_flat)
        """
        obs_flat = np.asarray(obs_flat, dtype=np.float32)
        act_flat = np.asarray(act_flat, dtype=np.int64).reshape(-1)
        rew_flat = np.asarray(rew_flat, dtype=np.float32).reshape(-1)

        if self.steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be > 0")

        if self.num_actions <= 1:
            raise ValueError("num_actions must be > 1")

        if not (0.0 <= self.discount_factor <= 1.0):
            raise ValueError("discount_factor must be in [0, 1]")

        if obs_flat.ndim != 2:
            raise ValueError("obs_flat must be (N, obs_dim)")

        n_samples = obs_flat.shape[0]

        if act_flat.shape[0] != n_samples:
            raise ValueError("act_flat length mismatch")

        if rew_flat.shape[0] != n_samples:
            raise ValueError("rew_flat length mismatch")

        if n_samples % self.steps_per_episode != 0:
            raise ValueError("samples must be divisible by steps_per_episode")

        if np.any(act_flat < 0) or np.any(act_flat >= self.num_actions):
            raise ValueError("invalid action index")

        return obs_flat, act_flat, rew_flat

    def _compute_rtg(self, rew_flat: np.ndarray) -> np.ndarray:
        """Compute discounted return-to-go.

        :param rew_flat: Shape (N,), rewards
        :return: Shape (N,), discounted return-to-go values
        """
        n_samples = rew_flat.shape[0]
        num_eps = n_samples // self.steps_per_episode

        rew = rew_flat.reshape(num_eps, self.steps_per_episode)
        rtg = np.zeros_like(rew, dtype=np.float32)

        for t in range(self.steps_per_episode - 1, -1, -1):
            if t == self.steps_per_episode - 1:
                rtg[:, t] = rew[:, t]
            else:
                rtg[:, t] = rew[:, t] + self.discount_factor * rtg[:, t + 1]

        return rtg.reshape(-1).astype(np.float32)

    def _build_state_features(self, obs_flat: np.ndarray) -> np.ndarray:
        """Build state features by appending timestep index to observations.

        :param obs_flat: Shape (N, obs_dim), observations
        :return: Shape (N, obs_dim + 1), augmented state features with step index
        """
        n_samples = obs_flat.shape[0]
        num_eps = n_samples // self.steps_per_episode

        step_idx = (
            np.tile(np.arange(self.steps_per_episode), num_eps).reshape(-1, 1).astype(np.float32)
        )

        return np.concatenate([obs_flat, step_idx], axis=1)

    # training of regression model
    def fit(
        self,
        *,
        obs_flat: np.ndarray,
        act_flat: np.ndarray,
        rew_flat: np.ndarray,
    ) -> HistGradientBoostingRegressor:
        """Fit the regression model.

        :param obs_flat: Shape (N, obs_dim), observations
        :param act_flat: Shape (N,), discrete action indices
        :param rew_flat: Shape (N,), rewards
        :return: Fitted regression model
        """

        obs_flat, act_flat, rew_flat = self._check_inputs(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
        )

        rtg = self._compute_rtg(rew_flat)
        x_state = self._build_state_features(obs_flat)
        action_feat = act_flat.astype(np.float32).reshape(-1, 1)
        x_sa = np.concatenate([x_state, action_feat], axis=1)

        params = dict(
            max_depth=6,
            learning_rate=0.05,
            max_iter=500,
            random_state=self.random_state,
        )
        params.update(self.model_params)

        self.model = HistGradientBoostingRegressor(**params)
        self.model.fit(x_sa, rtg)

        return self.model

    def estimate(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        """Estimate Q-values for provided (obs, act) pairs.

        Assumes the inputs are flattened and ordered consistently by episode,
        so that timestep indices can be reconstructed from `steps_per_episode`.
        :param obs: Shape (N, obs_dim), observations
        :param act: Shape (N,), discrete action indices
        :return: Shape (N,), estimated Q-values for the given (obs, act) pairs
        """
        if self.model is None:
            raise ValueError("model not fitted")

        obs = np.asarray(obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.int64).reshape(-1)

        if obs.ndim != 2:
            raise ValueError("obs must be (N, obs_dim)")

        if obs.shape[0] != act.shape[0]:
            raise ValueError("obs and act must have the same number of samples")

        if obs.shape[0] % self.steps_per_episode != 0:
            raise ValueError("samples must be divisible by steps_per_episode")

        if np.any(act < 0) or np.any(act >= self.num_actions):
            raise ValueError("invalid action index")

        x_state = self._build_state_features(obs)
        action_feat = act.astype(np.float32).reshape(-1, 1)
        x_sa = np.concatenate([x_state, action_feat], axis=1)

        q = np.asarray(self.model.predict(x_sa), dtype=np.float32)
        return self._scale(q)

    # prediction of Q-values for all actions
    def predict_q_values(self, *, obs_flat: np.ndarray) -> np.ndarray:
        """Predict Q-values for all actions given observations.

        :param obs_flat: Shape (N, obs_dim), observations
        :return: Shape (N, num_actions), predicted Q-values for each action
        """
        if self.model is None:
            raise ValueError("model not fitted")

        obs_flat = np.asarray(obs_flat, dtype=np.float32)

        if obs_flat.ndim != 2:
            raise ValueError("obs_flat must be (N, obs_dim)")

        n_samples = obs_flat.shape[0]

        if n_samples % self.steps_per_episode != 0:
            raise ValueError("samples must be divisible by steps_per_episode")

        x_state = self._build_state_features(obs_flat)

        # repeat state for each action
        x_state_rep = np.repeat(x_state, self.num_actions, axis=0)

        all_actions = (
            np.tile(np.arange(self.num_actions), n_samples).reshape(-1, 1).astype(np.float32)
        )

        x_sa = np.concatenate([x_state_rep, all_actions], axis=1)
        q = self.model.predict(x_sa).astype(np.float32)

        return self._scale(q.reshape(n_samples, self.num_actions))
