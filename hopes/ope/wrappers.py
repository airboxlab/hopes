import numpy as np

from hopes.data.pre_processing import build_stepwise_importance_ratios_with_stickiness
from hopes.ope.estimators import (
    SelfNormalizedPerDecisionImportanceSampling,
    SequentialDoublyRobust,
)


def wpdis_daily(
    num_days: int,
    steps_per_episode: int,
    rew_flat: np.ndarray,
    act_flat: np.ndarray,
    p_b_taken_flat: np.ndarray,
    p_e_taken_flat: np.ndarray,
    sticky_act_flat: np.ndarray,
    eps: float = 1e-12,
    clip: float = 20.0,
    num_bootstrap_samples: int = 2000,
    significance_level: float = 0.05,
    normalization: str = "per_timestep",
) -> tuple[float, float, float]:
    """Estimate daily self-normalized per-decision IS with optional stickiness correction.

    This wrapper adapts notebook-style flat arrays of taken propensities to the estimator API.
    Since the estimators now expect either:
    - full target/behavior action probability matrices, or
    - precomputed step-wise importance ratios,

    this function builds minimal valid probability matrices from the logged actions and taken
    propensities, then computes corrected step-wise importance ratios in preprocessing.

    Parameters
    ----------
    num_days:
        Number of episodes / days.
    steps_per_episode:
        Number of timesteps per episode.
    rew_flat:
        Flat reward array of shape `(N,)`.
    act_flat:
        Logged action indices of shape `(N,)`.
    p_b_taken_flat:
        Behavior policy probabilities on logged actions, shape `(N,)`.
    p_e_taken_flat:
        Target policy probabilities on logged actions, shape `(N,)`.
    sticky_act_flat:
        Sticky action indicator, shape `(N,)`.
    eps:
        Numerical stabilizer.
    clip:
        Optional symmetric clipping threshold for importance ratios.
    num_bootstrap_samples:
        Number of bootstrap samples used for the confidence interval.
    significance_level:
        Significance level used for the confidence interval.
    normalization:
        Normalization method for WIS. Can be "global" or "per_timestep".

    Returns
    -------
    tuple[float, float, float]
        `(mean, lower_bound, upper_bound)`.
    """
    rew_flat = np.asarray(rew_flat, dtype=np.float32).reshape(-1)
    act_flat = np.asarray(act_flat, dtype=np.int64).reshape(-1)
    p_b_taken_flat = np.asarray(p_b_taken_flat, dtype=np.float32).reshape(-1)
    p_e_taken_flat = np.asarray(p_e_taken_flat, dtype=np.float32).reshape(-1)
    sticky_act_flat = np.asarray(sticky_act_flat, dtype=np.int64).reshape(-1)

    n_samples = rew_flat.shape[0]

    if n_samples != num_days * steps_per_episode:
        raise ValueError("N must equal num_days * steps_per_episode.")
    if act_flat.shape[0] != n_samples:
        raise ValueError("act_flat must have length N.")
    if p_b_taken_flat.shape[0] != n_samples or p_e_taken_flat.shape[0] != n_samples:
        raise ValueError("p_b_taken_flat and p_e_taken_flat must have length N.")
    if sticky_act_flat.shape[0] != n_samples:
        raise ValueError("sticky_act_flat must have length N.")

    num_actions = int(act_flat.max()) + 1
    if num_actions < 2:
        raise ValueError("num_actions inferred from act_flat must be >= 2.")

    def build_probs_from_taken(p_taken: np.ndarray) -> np.ndarray:
        """Build a minimal valid probability matrix from taken propensities.

        The logged action receives `p_taken`, and the remaining mass is distributed
        uniformly over the other actions.
        """
        p_taken = np.clip(np.asarray(p_taken, dtype=np.float32), eps, 1.0 - eps)

        probs = np.zeros((n_samples, num_actions), dtype=np.float32)
        leftover = 1.0 - p_taken
        fill_value = leftover / float(num_actions - 1)

        probs[:] = fill_value.reshape(-1, 1)
        probs[np.arange(n_samples), act_flat] = p_taken

        probs = np.clip(probs, eps, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    p_b = build_probs_from_taken(p_b_taken_flat)
    p_e = build_probs_from_taken(p_e_taken_flat)

    rho = build_stepwise_importance_ratios_with_stickiness(
        target_policy_action_probabilities=p_e,
        behavior_policy_action_probabilities=p_b,
        logged_actions=act_flat,
        steps_per_episode=steps_per_episode,
        eps=eps,
        clip=clip,
        apply_stickiness=True,
        sticky_actions=sticky_act_flat,
        value_after_switch=1.0,
    )

    estimator = SelfNormalizedPerDecisionImportanceSampling(
        steps_per_episode=steps_per_episode,
        discount_factor=1.0,
        normalization=normalization,
        eps=eps,
    )

    estimator.set_parameters(
        target_policy_action_probabilities=p_e,
        behavior_policy_action_probabilities=p_b,
        rewards=rew_flat,
    )
    estimator.set_importance_ratios(rho)

    ci = estimator.estimate_policy_value_with_confidence_interval(
        method="bootstrap",
        significance_level=significance_level,
        num_samples=num_bootstrap_samples,
    )

    return (
        ci["mean"],
        ci["lower_bound"],
        ci["upper_bound"],
    )


def compute_stepwise_ips_wis(
    p_b_taken_flat: np.ndarray,
    p_e_taken_flat: np.ndarray,
    rew_flat: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, np.ndarray | float]:
    r"""Compute step-wise IPS and WIS diagnostics from taken propensities.

    Parameters
    ----------
    p_b_taken_flat:
        Behavior propensities :math:`\pi_b(a_t \mid s_t)`, shape `(N,)`.
    p_e_taken_flat:
        Target propensities :math:`\pi_e(a_t \mid s_t)`, shape `(N,)`.
    rew_flat:
        Rewards :math:`r_t`, shape `(N,)`.
    eps:
        Numerical stabilizer.

    Returns
    -------
    dict[str, np.ndarray | float]
        Dictionary containing:
        - `weights`
        - `ips`
        - `wis`
        - `w_max`
        - `w_p99`
    """
    rew_flat = np.asarray(rew_flat, dtype=np.float32).reshape(-1)
    p_b_taken_flat = np.asarray(p_b_taken_flat, dtype=np.float32).reshape(-1)
    p_e_taken_flat = np.asarray(p_e_taken_flat, dtype=np.float32).reshape(-1)

    if rew_flat.shape[0] != p_b_taken_flat.shape[0] or rew_flat.shape[0] != p_e_taken_flat.shape[0]:
        raise ValueError("rew_flat, p_b_taken_flat, and p_e_taken_flat must have the same length.")

    pb = np.maximum(p_b_taken_flat, eps)
    pe = p_e_taken_flat

    if not np.all((pe >= 0.0) & (pe <= 1.0)):
        raise ValueError("p_e_taken_flat must be in [0, 1].")
    if not np.all((pb >= 0.0) & (pb <= 1.0)):
        raise ValueError("p_b_taken_flat must be in [0, 1].")

    weights = pe / pb

    if not np.isfinite(weights).all():
        raise ValueError("Importance weights contain NaN or Inf.")

    ips = float(np.mean(weights * rew_flat))
    wis = float(np.sum(weights * rew_flat) / (np.sum(weights) + eps))

    return {
        "weights": weights.astype(np.float32),
        "ips": ips,
        "wis": wis,
        "w_max": float(weights.max()),
        "w_p99": float(np.quantile(weights, 0.99)),
    }


def dr_step_daily(
    *,
    num_days: int,
    steps_per_episode: int,
    rewards_flat: np.ndarray,
    target_policy_action_probabilities: np.ndarray,
    behavior_policy_action_probabilities: np.ndarray,
    logged_actions: np.ndarray,
    q_values: np.ndarray,
    sticky_actions: np.ndarray | None = None,
    eps: float = 1e-12,
    clip: float = 20.0,
    apply_stickiness: bool = False,
    value_after_switch: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute daily Sequential DR estimates and the step-wise ratios used by the estimator.

    Parameters
    ----------
    num_days:
        Number of episodes / days.
    steps_per_episode:
        Number of timesteps per episode.
    rewards_flat:
        Flat reward array of shape `(N,)`.
    target_policy_action_probabilities:
        Target policy action probabilities, shape `(N, A)`.
    behavior_policy_action_probabilities:
        Behavior policy action probabilities, shape `(N, A)`.
    logged_actions:
        Logged action indices, shape `(N,)`.
    q_values:
        Estimated Q-values for all actions, shape `(N, A)`.
    sticky_actions:
        Sticky action indicator, shape `(N,)`. Required when `apply_stickiness=True`.
    eps:
        Numerical stabilizer.
    clip:
        Optional symmetric clipping threshold.
    apply_stickiness:
        Whether to apply stickiness correction in preprocessing.
    value_after_switch:
        Replacement value after the first sticky switch. Required when
        `apply_stickiness=True`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - `dr_day`: daily DR estimates, shape `(num_days,)`
        - `rho`: step-wise importance ratios, shape `(N,)`
    """
    rewards_flat = np.asarray(rewards_flat, dtype=np.float32).reshape(-1)
    logged_actions = np.asarray(logged_actions, dtype=np.int64).reshape(-1)
    target_policy_action_probabilities = np.asarray(
        target_policy_action_probabilities,
        dtype=np.float32,
    )
    behavior_policy_action_probabilities = np.asarray(
        behavior_policy_action_probabilities,
        dtype=np.float32,
    )
    q_values = np.asarray(q_values, dtype=np.float32)

    n_samples = rewards_flat.shape[0]

    if n_samples != num_days * steps_per_episode:
        raise ValueError("N must equal num_days * steps_per_episode.")
    if logged_actions.shape[0] != n_samples:
        raise ValueError("logged_actions must have length N.")
    if target_policy_action_probabilities.shape[0] != n_samples:
        raise ValueError("target_policy_action_probabilities must have shape (N, A).")
    if behavior_policy_action_probabilities.shape[0] != n_samples:
        raise ValueError("behavior_policy_action_probabilities must have shape (N, A).")
    if q_values.shape != target_policy_action_probabilities.shape:
        raise ValueError(
            "q_values must have the same shape as target_policy_action_probabilities: (N, A)."
        )

    rho = build_stepwise_importance_ratios_with_stickiness(
        target_policy_action_probabilities=target_policy_action_probabilities,
        behavior_policy_action_probabilities=behavior_policy_action_probabilities,
        logged_actions=logged_actions,
        steps_per_episode=steps_per_episode,
        eps=eps,
        clip=clip,
        apply_stickiness=apply_stickiness,
        sticky_actions=sticky_actions,
        value_after_switch=value_after_switch,
    )

    estimator = SequentialDoublyRobust(
        steps_per_episode=steps_per_episode,
        discount_factor=1.0,
        eps=eps,
        clip=clip,
    )
    estimator.set_parameters(
        target_policy_action_probabilities=target_policy_action_probabilities,
        behavior_policy_action_probabilities=behavior_policy_action_probabilities,
        rewards=rewards_flat,
    )
    estimator.set_logged_actions(logged_actions)
    estimator.set_model_predictions(q_values=q_values)
    estimator.set_importance_ratios(rho)

    dr_day = estimator.estimate_weighted_rewards().reshape(-1)

    return dr_day.astype(np.float32), rho.astype(np.float32)
