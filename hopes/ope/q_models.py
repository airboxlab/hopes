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
    """

    Q0: np.ndarray  # (N,) float32
    Q1: np.ndarray  # (N,) float32
    model: HistGradientBoostingRegressor


class RTGQModelHGBoost:
    """Regression-based Q-model trained on logged trajectories.

    The model approximates Q(s,a) by supervised learning on step-level samples.

    Target:
        Return-to-go (RTG) at each timestep:
            RTG_t = sum_{k=t..T-1} r_k

    Features:
        X = [obs, step_idx]
        X_sa = [obs, step_idx, action] --> Action appended as last feature.

    After training, the model can be evaluated for both discrete actions (0/1)
    to produce:
        Q0[i] = Q(s_i, 0)
        Q1[i] = Q(s_i, 1)

    Notes
    -----
    - Assumes 2 discrete actions {0,1}.
    - steps_per_episode must divide N.
    """

    def __init__(
        self,
        *,
        steps_per_episode: int,
        model_params: dict[str, Any] | None = None,
        random_state: int = 0,
    ) -> None:
        if steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be > 0")
        self.steps_per_episode = steps_per_episode
        self.model_params = dict(model_params or {})
        self.random_state = random_state

        self.model: HistGradientBoostingRegressor | None = None

    # ---------- internal helpers ----------

    def _check_and_cast_inputs(
        self,
        *,
        obs_flat: np.ndarray,
        act_flat: np.ndarray,
        rew_flat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_flat = np.asarray(obs_flat, dtype=np.float32)
        if obs_flat.ndim != 2:
            raise ValueError(f"obs_flat must be 2D (N, obs_dim), got {obs_flat.shape}")

        act_flat = np.asarray(act_flat).reshape(-1)
        rew_flat = np.asarray(rew_flat).reshape(-1)

        N = obs_flat.shape[0]
        if act_flat.shape[0] != N or rew_flat.shape[0] != N:
            raise ValueError("obs_flat, act_flat, rew_flat must have the same length")

        if N % self.steps_per_episode != 0:
            raise ValueError(
                f"N ({N}) is not divisible by steps_per_episode ({self.steps_per_episode}). "
                "Check episode filtering/flattening."
            )

        return obs_flat, act_flat, rew_flat

    def _compute_rtg_flat(self, rew_flat: np.ndarray) -> np.ndarray:
        """Compute RTG per step, flattened back to (N,)."""
        N = rew_flat.shape[0]
        num_eps = N // self.steps_per_episode

        rew_day = rew_flat.astype(np.float32).reshape(num_eps, self.steps_per_episode)
        rtg_day = np.flip(np.cumsum(np.flip(rew_day, axis=1), axis=1), axis=1)
        return rtg_day.reshape(-1).astype(np.float32)

    # ---- BUILD FEATURES ----
    # The timestep index is included to help the model capture time-dependent effects within the episode
    def _build_state_features(self, obs_flat: np.ndarray) -> np.ndarray:
        """Build X = [obs, step_idx]."""
        N = obs_flat.shape[0]
        num_eps = N // self.steps_per_episode

        step_idx = (
            np.tile(np.arange(self.steps_per_episode), num_eps).reshape(-1, 1).astype(np.float32)
        )
        return np.concatenate([obs_flat, step_idx], axis=1)  # (N, obs_dim+1)

    def fit(
        self,
        *,
        obs_flat: np.ndarray,
        act_flat: np.ndarray,
        rew_flat: np.ndarray,
    ) -> HistGradientBoostingRegressor:
        """Fit Q-model on logged data.

        After calling fit(), you can call predict_q0_q1(obs_flat) to obtain Q0/Q1.
        """
        obs_flat, act_flat, rew_flat = self._check_and_cast_inputs(
            obs_flat=obs_flat, act_flat=act_flat, rew_flat=rew_flat
        )

        y = self._compute_rtg_flat(rew_flat)
        X = self._build_state_features(obs_flat)

        a_feat = act_flat.astype(np.float32).reshape(-1, 1)
        X_sa = np.concatenate([X, a_feat], axis=1)

        # ---- TRAIN MODEL ----
        params = dict(
            max_depth=6,
            learning_rate=0.05,
            max_iter=500,
            random_state=self.random_state,
        )

        if self.model_params:
            params.update(self.model_params)

        self.model = HistGradientBoostingRegressor(**params)
        self.model.fit(X_sa, y)

        return self.model

    def predict_q0_q1(self, *, obs_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict Q(s,0) and Q(s,1) for each step.

        Returns
        -------
        (Q0, Q1): both shape (N,), float32
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit(...) first.")

        obs_flat = np.asarray(obs_flat, dtype=np.float32)
        if obs_flat.ndim != 2:
            raise ValueError(f"obs_flat must be 2D (N, obs_dim), got {obs_flat.shape}")

        N = obs_flat.shape[0]
        if N % self.steps_per_episode != 0:
            raise ValueError(
                f"N ({N}) is not divisible by steps_per_episode ({self.steps_per_episode})."
            )

        X = self._build_state_features(obs_flat)

        a0 = np.zeros((N, 1), dtype=np.float32)
        a1 = np.ones((N, 1), dtype=np.float32)

        Q0 = self.model.predict(np.concatenate([X, a0], axis=1)).astype(np.float32)
        Q1 = self.model.predict(np.concatenate([X, a1], axis=1)).astype(np.float32)

        return Q0, Q1

    def fit_predict_q0_q1(
        self,
        *,
        obs_flat: np.ndarray,
        act_flat: np.ndarray,
        rew_flat: np.ndarray,
        return_model: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, HistGradientBoostingRegressor | None]:
        """Notebook-friendly one-shot API.

        Returns:
            Q0, Q1, model (or None if return_model=False)
        """
        model = self.fit(obs_flat=obs_flat, act_flat=act_flat, rew_flat=rew_flat)
        Q0, Q1 = self.predict_q0_q1(obs_flat=obs_flat)

        if return_model:
            return Q0, Q1, model
        return Q0, Q1, None
