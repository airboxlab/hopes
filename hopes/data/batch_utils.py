from collections.abc import Callable, Hashable
from typing import Any

import numpy as np
import pandas as pd

from hopes.general_utils import (
    get_action_int,
    parse_list,
    softmax_1d,
    to_1d_float32,
    to_1d_int64,
    to_2d_float32,
    to_list,
)


# Function utilities
def build_episode_dict(
    df: pd.DataFrame,
    cols: list[str],
    episode_id_col: str = "episode_id",
) -> dict[Hashable, Any]:
    """Group df by episode_id and keep only selected columns, resetting index per episode."""

    return {
        episode_id: group[cols].reset_index(drop=True)
        for episode_id, group in df.groupby(episode_id_col)
    }


# Since each episode start at midnight (fract hour (frh) = 0.0), but for us in training the obs vector starts each day from frh=0.25
# it's required to remove the first line in each episode to start from frh=0.25 as first state
def drop_first_step_in_each_episode(
    episode_dict: dict[str, pd.DataFrame],
    n_drop: int = 1,
) -> dict[str, pd.DataFrame]:
    """Drop the first n rows of each episode (e.g., to start at frh=0.25 instead of 0.0)."""
    out = {}
    for episode_id, df_ep in episode_dict.items():
        out[episode_id] = df_ep.iloc[n_drop:].reset_index(drop=True)
    return out


# Check for the length of each episode, since we have to check that len is the same as the one required by the training
# If an episode does not respect this constraint --> remove it from the episode_dict and return the list of removed
# episode ids (for logging purposes)
def filter_episodes_by_length(
    episode_dict: dict[str, pd.DataFrame],
    steps_per_episode: int,
    verbose: bool = True,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Remove episodes whose length != steps_per_episode.

    Returns (filtered_episode_dict, removed_episode_ids).
    """
    bad = [eid for eid, df_ep in episode_dict.items() if len(df_ep) != steps_per_episode]

    if verbose:
        print(f"Total number of episodes with invalid length: {len(bad)}")
        if not bad:
            print(f"All episodes have length {steps_per_episode}")

    filtered = dict(episode_dict)
    for eid in bad:
        filtered.pop(eid, None)

    if verbose:
        print(f"Removed {len(bad)} bad episodes.")

    return filtered, bad


def filter_episodes_by_temp_condition(
    episode_dict: dict[str, pd.DataFrame],
    *,
    raw_obs_col: str = "raw_observation",
    idx_zone_temp: int,
    htg_setpoint: float,
    verbose: bool = True,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Whenever the agent decides to turn on (action = 1) but the zone is already at setpoint, the action is ignored and FCU kept.
    So:
    Remove episodes where zone_temp is always >= heating setpoint.
    zone_temp is extracted from raw_observation[idx_zone_temp].

    If an episode fits with the above condition --> remove it from the episode_dict and return the list of
    removed episode ids (for logging purposes).
    Returns (filtered_episode_dict, removed_episode_ids).
    """
    bad = []
    filtered = dict(episode_dict)

    for episode_id, df_ep in episode_dict.items():
        # extract zone temperature from raw_observation
        zone_temp = df_ep[raw_obs_col].apply(lambda x: float(parse_list(x)[idx_zone_temp]))

        # check if temperature is always >= heating setpoint
        if (zone_temp >= htg_setpoint).all():
            bad.append(episode_id)
            if verbose:
                print(f"\nRemoving episode: {episode_id}")
                print("Zone temperature series:")
                print(zone_temp.values)

    # remove bad episodes in-place
    for eid in bad:
        filtered.pop(eid, None)

    if verbose:
        print(f"\nRemoved {len(bad)} episodes where zone_temp is always >= heating setpoint.")

    return filtered, bad


def add_logged_policy_probs(
    episode_dict: dict[str, pd.DataFrame],
    *,
    logits_col: str = "logits",
    action_col: str = "action",
    probs_col: str = "probs",
    p_taken_col: str = "p_taken",
    argmax_action_col: str = "argmax_action",
) -> dict[str, pd.DataFrame]:
    """Reconstructs action probabilities from the logged logits. Logits are parsed from their
    stored format, converted to probabilities via softmax, and used to compute the probability of
    the logged action.

    For each episode df:
      - parse logits
      - compute probs via softmax
      - parse action to int
      - compute p_taken = probs[action]
      - add argmax_action for diagnostics
    Returns updated episode_dict.
    """
    out = {}
    for episode_id, df_ep in episode_dict.items():
        df_ep = df_ep.copy()

        # parse logits from string to a list of float
        df_ep[logits_col] = df_ep[logits_col].apply(to_list)

        # logits -> probs via softmax
        df_ep[probs_col] = df_ep[logits_col].apply(softmax_1d)

        # probability of the taken action
        df_ep[action_col] = df_ep[action_col].apply(get_action_int)

        # p_taken = pi_b(a_t | s_t), used later for IS-based estimators
        df_ep[p_taken_col] = df_ep.apply(
            lambda row: float(row[probs_col][row[action_col]]),
            axis=1,
        )

        # Add argmax action
        # Derives the greedy (argmax) action for basic policy diagnostics.
        df_ep[argmax_action_col] = df_ep[logits_col].apply(lambda l: int(np.argmax(l)))

        out[episode_id] = df_ep

    return out


def add_sticky_action(
    episode_dict: dict[str, pd.DataFrame],
    *,
    action_col: str = "action",
    sticky_col: str = "sticky_action",
    on_value: int = 1,
) -> dict[str, pd.DataFrame]:
    """Developing the effective actiony applied in the system by enforcing action stickiness. Once
    the action switches to 1, it is forced to remain 1 for all subsequent timesteps, matching the
    real-world control logic used during deployment.

    Add sticky action per episode:
    once action becomes `on_value`, it stays `on_value` forever.
    """
    out = {}
    for episode_id, df_ep in episode_dict.items():
        df_ep = df_ep.copy()
        df_ep[sticky_col] = (df_ep[action_col] == on_value).cummax().astype(int)
        out[episode_id] = df_ep
    return out


def global_minmax_scale_reward(
    episode_dict: dict[str, pd.DataFrame],
    *,
    reward_col: str = "reward",
    dtype=np.float32,
    eps: float = 1e-12,
    verbose: bool = True,
) -> tuple[dict[str, pd.DataFrame], float, float]:
    """Compute global min/max over available episodes, then apply min-max scaling in-place per
    episode.

    Returns: (scaled_episode_dict, r_min, r_max)
    """
    all_rewards = np.concatenate(
        [
            df_ep[reward_col].astype(float).to_numpy()
            for df_ep in episode_dict.values()
            if reward_col in df_ep.columns and len(df_ep) > 0
        ]
    )

    if all_rewards.size == 0:
        raise ValueError("No rewards found to scale. Check reward_col or episode_dict content.")

    r_min = float(all_rewards.min())
    r_max = float(all_rewards.max())
    denom = max(r_max - r_min, eps)

    if verbose:
        print("Global reward min/max:", r_min, r_max)

    out = {}
    for eid, df_ep in episode_dict.items():
        df_ep = df_ep.copy()
        r = df_ep[reward_col].astype(float).to_numpy()
        df_ep[reward_col] = ((r - r_min) / denom).astype(dtype)
        out[eid] = df_ep

    return out, r_min, r_max


def build_episode_sequences(
    episode_dict: dict[str, pd.DataFrame],
    *,
    filtered_obs_col: str = "filtered_observation",
    action_col: str = "action",
    sticky_action_col: str = "sticky_action",
    reward_col: str = "reward",
    probs_col: str = "probs",
    p_taken_col: str = "p_taken",
    argmax_action_col: str = "argmax_action",
) -> dict[str, dict[str, Any]]:
    """Build per-episode python sequences (lists) from episode_dict."""
    seqs: dict[str, dict[str, Any]] = {}

    # Each dict entry corresponds to one episode and stores time-aligned sequences
    # This format is used later to build flattened batches for OPE estimators
    for eid, df_ep in episode_dict.items():
        seqs[eid] = {
            # Observations batch (list of lists)
            "obs": df_ep[filtered_obs_col].apply(parse_list).tolist(),
            # Actions batch (list of floats/ints)
            "act": df_ep[action_col].astype(int).tolist(),
            # Sticky actions batch (list of floats/ints)
            "sticky_act": df_ep[sticky_action_col].astype(int).tolist(),
            # Rewards batch (list of floats)
            "rew": df_ep[reward_col].astype(float).tolist(),
            # Probs batch (list of lists, e.g., [p0, p1])
            "probs": df_ep[probs_col].apply(lambda x: list(x)).tolist(),
            # p_taken batch (list of floats)
            "p_taken": df_ep[p_taken_col].astype(float).tolist(),
            # argmax_action batch (list of ints)
            "argmax": df_ep[argmax_action_col].astype(int).tolist(),
        }

    return seqs


def build_numpy_batches(
    episode_seqs: dict[str, dict[str, Any]],
) -> tuple[
    list[np.ndarray],  # obs_batches
    list[np.ndarray],  # act_batches
    list[np.ndarray],  # sticky_act_batches
    list[np.ndarray],  # rew_batches
    list[np.ndarray],  # probs_batches
    list[np.ndarray],  # p_taken_batches
    list[np.ndarray],  # argmax_batches
    list[str],  # episode_ids in the same order
]:
    """Convert per-episode python sequences into typed NumPy arrays. Each episode is mapped to (T,
    ·) tensors for observations and probabilities, and (T,) vectors for actions and rewards,
    preparing the data for vectorized OPE computations.

    Returns lists aligned by episode order:
      obs: (T, obs_dim) float32
      act: (T,) int64
      sticky_act: (T,) int64
      rew: (T,) float32
      probs: (T, num_actions) float32
      p_taken: (T,) float32
      argmax: (T,) int64
    """
    episode_ids = list(episode_seqs.keys())

    obs_batches = []
    act_batches = []
    sticky_act_batches = []
    rew_batches = []
    probs_batches = []
    p_taken_batches = []
    argmax_batches = []

    for eid in episode_ids:
        s = episode_seqs[eid]

        obs_batches.append(to_2d_float32(s["obs"]))
        act_batches.append(to_1d_int64(s["act"]))
        sticky_act_batches.append(to_1d_int64(s["sticky_act"]))
        rew_batches.append(to_1d_float32(s["rew"]))
        probs_batches.append(to_2d_float32(s["probs"]))
        p_taken_batches.append(to_1d_float32(s["p_taken"]))
        argmax_batches.append(to_1d_int64(s["argmax"]))

    assert (
        len(obs_batches)
        == len(act_batches)
        == len(sticky_act_batches)
        == len(rew_batches)
        == len(p_taken_batches)
    ), "Batch length mismatch"

    return (
        obs_batches,
        act_batches,
        sticky_act_batches,
        rew_batches,
        probs_batches,
        p_taken_batches,
        argmax_batches,
        episode_ids,
    )


def flatten_episode_batches(
    *,
    obs_batches: list[np.ndarray],
    act_batches: list[np.ndarray],
    sticky_act_batches: list[np.ndarray],
    rew_batches: list[np.ndarray],
    p_taken_batches: list[np.ndarray],
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten per-episode batches into contiguous step-level arrays. Episode batches are
    concatenated to form global tensors for observations, actions (raw and sticky), rewards, and
    behavior propensities which are the inputs expected by step-wise and trajectory-wise OPE
    estimators.

    Returns:
      obs_flat: (N, obs_dim) float32
      act_flat: (N,) int64
      sticky_act_flat: (N,) int64
      rew_flat: (N,) float32
      p_b_taken_flat: (N,) float32
    """

    # vstack/hstack are used to move from episode-wise storage to a single step-wise dataset
    obs_flat = np.vstack(obs_batches).astype(np.float32)
    act_flat = np.hstack(act_batches).astype(np.int64)
    sticky_act_flat = np.hstack(sticky_act_batches).astype(np.int64)
    rew_flat = np.hstack(rew_batches).astype(np.float32)

    # p_b_taken_flat represents pi_b(a_t | s_t) for each logged transition
    p_b_taken_flat = np.hstack(p_taken_batches).astype(np.float32)

    # Sanity checks
    assert (
        p_b_taken_flat.shape[0] == act_flat.shape[0] == rew_flat.shape[0] == obs_flat.shape[0]
    ), "Flattened arrays have inconsistent lengths"

    # Sanity check on the shape of p_b_taken_flat
    if verbose:
        N = obs_flat.shape[0]
        print(
            "N:",
            N,
            "obs_flat:",
            obs_flat.shape,
            "act_flat:",
            act_flat.shape,
            "rew_flat:",
            rew_flat.shape,
        )
        print(
            "p_b_taken_flat:",
            p_b_taken_flat.shape,
            "min/mean/max:",
            float(p_b_taken_flat.min()),
            float(p_b_taken_flat.mean()),
            float(p_b_taken_flat.max()),
        )

    return obs_flat, act_flat, sticky_act_flat, rew_flat, p_b_taken_flat


def apply_stickiness_correction_to_rho(
    rho: np.ndarray,  # importance ratios, shape (n_eps, T)
    sticky_act_day: np.ndarray,  # sticky actions applied, shape (n_eps, T), values in {0,1}
    value_after_switch: float = 1.0,  # value assigned to rho after the first switch.
) -> np.ndarray:
    """Function for apply stickiness correction to importance ratios This function returns the
    corrected rho If sticky_act_day[i] ever becomes 1, then for that episode i:

    rho[i, t_star+1:] = value_after_switch where t_star is the first timestep where
    sticky_act_day[i, t] == 1.
    """
    assert rho.shape == sticky_act_day.shape, "rho and sticky_act_day must have same shape"

    has_switch = (sticky_act_day == 1).any(axis=1)
    first_switch = np.argmax(
        sticky_act_day == 1, axis=1
    )  # safe even if no switch; guarded by has_switch

    T = rho.shape[1]
    for i in range(rho.shape[0]):
        if has_switch[i]:
            t_star = int(first_switch[i])
            if t_star + 1 < T:
                rho[i, t_star + 1 :] = value_after_switch

    return rho


def validate_batches(
    *,
    obs_batches: list[np.ndarray],
    act_batches: list[np.ndarray],
    sticky_act_batches: list[np.ndarray],
    rew_batches: list[np.ndarray],
    p_taken_batches: list[np.ndarray],
    T_expected: int,
    obs_dim_expected: int,
    verbose: bool = True,
) -> None:
    """Run shape/dtype checks similar to the notebook assertions."""
    for i, (o, a, st_a, r, p) in enumerate(
        zip(obs_batches, act_batches, sticky_act_batches, rew_batches, p_taken_batches)
    ):
        # Observation checks
        assert isinstance(o, np.ndarray)
        assert o.dtype == np.float32
        assert o.ndim == 2, f"obs[{i}] is not 2D"
        assert o.shape[1] == obs_dim_expected, f"obs[{i}] has wrong obs_dim"
        assert o.shape[0] == T_expected, f"obs[{i}] has wrong episode length"

        # Action checks
        assert isinstance(a, np.ndarray)
        assert a.dtype == np.int64
        assert a.ndim == 1, f"act[{i}] is not 1D"
        assert a.shape[0] == T_expected, f"act[{i}] has wrong episode length"

        # Sticky action checks
        assert isinstance(st_a, np.ndarray)
        assert st_a.dtype == np.int64
        assert st_a.ndim == 1, f"sticky_act[{i}] is not 1D"
        assert st_a.shape[0] == T_expected, f"sticky_act[{i}] has wrong episode length"

        # Reward checks
        assert isinstance(r, np.ndarray)
        assert r.dtype == np.float32
        assert r.ndim == 1, f"rew[{i}] is not 1D"
        assert r.shape[0] == T_expected, f"rew[{i}] has wrong episode length"

        # p_taken checks
        assert isinstance(p, np.ndarray)
        assert p.dtype == np.float32, f"p_taken[{i}] dtype is {p.dtype}, expected float32"
        assert p.ndim == 1, f"p_taken[{i}] is not 1D"
        assert p.shape[0] == T_expected, f"p_taken[{i}] has wrong episode length"
        assert np.all(np.isfinite(p)), f"p_taken[{i}] contains NaN/Inf"
        assert np.all((p >= 0.0) & (p <= 1.0)), f"p_taken[{i}] not in [0,1]"

    if verbose:
        print("Batch shape and dtype checks passed.")


RewardSeriesFn = Callable[[pd.DataFrame], pd.Series]


def add_zone_temp_and_reward(
    episode_dict: dict[str, pd.DataFrame],
    *,
    reward_series_fn: RewardSeriesFn,
    raw_obs_col: str = "raw_observation",
    zone_temp_col: str = "zone_temperature",
    reward_col: str = "reward",
    idx_zone_temp: int,
) -> dict[str, pd.DataFrame]:
    """# Function implementation to obtain reward per each episode and compute zone_temperature for
    checking results in episode_dict.

    Reward computation is injected via reward_series_fn(df_ep) -> pd.Series, so customer-
    specific reward logic can live outside the library (e.g. notebook).
    """
    out = {}
    for eid, df_ep in episode_dict.items():
        df_ep = df_ep.copy()

        df_ep[zone_temp_col] = df_ep[raw_obs_col].apply(
            lambda x: float(parse_list(x)[idx_zone_temp])
        )

        r = reward_series_fn(df_ep)
        if not isinstance(r, pd.Series):
            raise TypeError("reward_series_fn must return a pandas Series")

        if len(r) != len(df_ep):
            raise ValueError(
                f"reward series length mismatch for episode {eid}: " f"{len(r)} != {len(df_ep)}"
            )

        # align index defensively
        r = r.reset_index(drop=True)
        df_ep = df_ep.reset_index(drop=True)
        df_ep[reward_col] = r.values

        out[eid] = df_ep

    return out
