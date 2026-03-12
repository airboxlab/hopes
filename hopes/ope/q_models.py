from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class QModelTrainResult:
    """Training a regression model to approximate the action-value function Q using logged
    trajectories.

    The model takes the state (observation + timestep) and action as input and predicts the
    return-to-go, providing the Q estimates required by DM and DR OPE estimators.

    Parameters
    ----------
    q_values:
        Estimated Q-values for all actions, shape `(N, A)`.
    model:
        Fitted regression model.
    """

    q_values: np.ndarray
    model: HistGradientBoostingRegressor


class RTGQModelHGBoost:
    """Train a regression model approximating the action-value function Q.

    The model is trained using logged trajectories and predicts
    return-to-go (RTG) from `(state, timestep, action)` features.

    RTG target:

    .. math::

        RTG_t = \\sum_{k=t}^{T-1} r_k

    Features used for regression:

    .. math::

        X = [obs, step\\_idx, action]

    where:

    - `obs` is the observation vector
    - `step_idx` is the timestep index within the episode
    - `action` is the discrete action index

    After training, the model can evaluate **all possible actions**
    to produce a matrix:

    .. math::

        Q(s,a) \\in \\mathbb{R}^{N \times A}

    where:

    - `N` = number of samples
    - `A` = number of discrete actions
    """

    def __init__(
        self,
        *,
        steps_per_episode: int,
        num_actions: int,
        model_params: dict[str, Any] | None = None,
        random_state: int = 0,
    ) -> None:
        """Initialize the Q-model trainer.

        Parameters
        ----------
        steps_per_episode:
            Number of timesteps per episode.
        num_actions:
            Number of discrete actions.
        model_params:
            Optional parameters for `HistGradientBoostingRegressor`.
        random_state:
            Random seed.
        """
        if steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be > 0")

        if num_actions <= 1:
            raise ValueError("num_actions must be > 1")

        self.steps_per_episode = steps_per_episode
        self.num_actions = num_actions
        self.model_params = model_params or {}
        self.random_state = random_state

        self.model: HistGradientBoostingRegressor | None = None

    # -----------------------------------------------------
    # utilities
    # -----------------------------------------------------

    def _check_inputs(
        self,
        *,
        obs_flat: np.ndarray,
        act_flat: np.ndarray,
        rew_flat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_flat = np.asarray(obs_flat, dtype=np.float32)
        act_flat = np.asarray(act_flat, dtype=np.int64).reshape(-1)
        rew_flat = np.asarray(rew_flat, dtype=np.float32).reshape(-1)

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
        """Compute return-to-go."""
        n_samples = rew_flat.shape[0]
        num_eps = n_samples // self.steps_per_episode

        rew = rew_flat.reshape(num_eps, self.steps_per_episode)

        rtg = np.flip(np.cumsum(np.flip(rew, axis=1), axis=1), axis=1)

        return rtg.reshape(-1).astype(np.float32)

    def _build_state_features(self, obs_flat: np.ndarray) -> np.ndarray:
        """Build state features `[obs, step_idx]`."""
        n_samples = obs_flat.shape[0]
        num_eps = n_samples // self.steps_per_episode

        step_idx = (
            np.tile(
                np.arange(self.steps_per_episode),
                num_eps,
            )
            .reshape(-1, 1)
            .astype(np.float32)
        )

        return np.concatenate([obs_flat, step_idx], axis=1)

    # -----------------------------------------------------
    # training
    # -----------------------------------------------------

    def fit(
        self,
        *,
        obs_flat: np.ndarray,
        act_flat: np.ndarray,
        rew_flat: np.ndarray,
    ) -> HistGradientBoostingRegressor:
        """Fit the regression model."""

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

    # -----------------------------------------------------
    # prediction
    # -----------------------------------------------------

    def predict_q_values(
        self,
        *,
        obs_flat: np.ndarray,
    ) -> np.ndarray:
        """Predict Q-values for all actions.

        Returns
        -------
        np.ndarray
            Shape `(N, A)`
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
            np.tile(
                np.arange(self.num_actions),
                n_samples,
            )
            .reshape(-1, 1)
            .astype(np.float32)
        )

        x_sa = np.concatenate([x_state_rep, all_actions], axis=1)

        q = self.model.predict(x_sa).astype(np.float32)

        return q.reshape(n_samples, self.num_actions)

    # -----------------------------------------------------
    # convenience
    # -----------------------------------------------------

    def fit_predict_q_values(
        self,
        *,
        obs_flat: np.ndarray,
        act_flat: np.ndarray,
        rew_flat: np.ndarray,
        return_model: bool = True,
    ) -> tuple[np.ndarray, HistGradientBoostingRegressor | None]:
        """Fit and predict Q-values in a single call."""

        model = self.fit(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
        )

        q_values = self.predict_q_values(obs_flat=obs_flat)

        if return_model:
            return q_values, model

        return q_values, None
