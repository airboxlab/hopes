import numpy as np

from hopes.ope.estimators import WeightedPerDecisionImportanceSampling


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
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Daily wrapper around BaseEstimator-style WPDIS.

    This wrapper adapts notebook-style flat arrays (including taken propensities) to the
    BaseEstimator API which expects full (N, A) probability matrices.

    Since we only have taken propensities, we build "minimal" probability matrices that
    place p_taken at the logged action index and distribute the remaining mass uniformly
    across other actions (for numerical validity).

    Returns (mean, lo, hi) from episode-level bootstrap CI.
    """

    # rewards per day
    rew_flat = np.asarray(rew_flat, dtype=np.float32).reshape(-1)
    act_flat = np.asarray(act_flat, dtype=np.int64).reshape(-1)

    # behavior + target propensities per step
    p_b_taken_flat = np.asarray(p_b_taken_flat, dtype=np.float32).reshape(-1)
    p_e_taken_flat = np.asarray(p_e_taken_flat, dtype=np.float32).reshape(-1)

    # stickiness correction
    sticky_act_flat = np.asarray(sticky_act_flat, dtype=np.int64).reshape(-1)

    N = rew_flat.shape[0]
    if N != num_days * steps_per_episode:
        raise ValueError("N must equal num_days * steps_per_episode")
    if act_flat.shape[0] != N:
        raise ValueError("act_flat must have length N")
    if p_b_taken_flat.shape[0] != N or p_e_taken_flat.shape[0] != N:
        raise ValueError("p_*_taken_flat must have length N")
    if sticky_act_flat.shape[0] != N:
        raise ValueError("sticky_act_flat must have length N")

    # Infer number of actions from act_flat (assumes actions are 0..A-1)
    num_actions = int(act_flat.max()) + 1
    if num_actions < 2:
        raise ValueError("num_actions inferred from act_flat must be >= 2")

    # Build minimal valid distributions (N, A) from taken propensities:
    # - put p_taken on the logged action
    # - distribute leftover mass uniformly to other actions (positive + rows sum to 1)
    def build_probs_from_taken(p_taken: np.ndarray) -> np.ndarray:
        p_taken = np.clip(p_taken.astype(np.float32), eps, 1.0 - eps)
        P = np.full((N, num_actions), 0.0, dtype=np.float32)

        leftover = 1.0 - p_taken
        fill = leftover / float(num_actions - 1)

        P[:] = fill.reshape(-1, 1)
        P[np.arange(N), act_flat] = p_taken
        # ensure strictly positive
        P = np.clip(P, eps, 1.0)
        # re-normalize exactly
        P /= P.sum(axis=1, keepdims=True)
        return P

    P_b = build_probs_from_taken(p_b_taken_flat)
    P_e = build_probs_from_taken(p_e_taken_flat)

    # estimate + CI
    est = WeightedPerDecisionImportanceSampling(
        steps_per_episode=steps_per_episode,
        eps=eps,
        clip=clip,
        apply_stickiness=True,
    )
    est.set_logged_data(actions=act_flat, sticky_actions=sticky_act_flat)
    est.set_parameters(
        target_policy_action_probabilities=P_e,
        behavior_policy_action_probabilities=P_b,
        rewards=rew_flat,
    )

    ci = est.estimate_policy_value_with_confidence_interval(
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
    return ci["mean"], ci["lower_bound"], ci["upper_bound"]


### IPS/WIS step-wise
def compute_stepwise_ips_wis(
    p_b_taken_flat: np.ndarray,
    p_e_taken_flat: np.ndarray,
    rew_flat: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, np.ndarray | float]:
    """Computing step-wise IPS and self-normalized IS (WIS) estimates as basic off-policy
    baselines. Importance weights are built from the ratio between new-policy and behavior-policy
    propensities on logged actions, and clipping is used to inspect sensitivity to high-variance
    weights.

    Args:
        p_b_taken_flat: behavior propensities π_b(a_t|s_t), shape (N,)
        p_e_taken_flat: target propensities π_e(a_t|s_t), shape (N,)
        rew_flat: rewards r_t, shape (N,)
        eps: numerical stability constant

    Returns:
        dict with:
            weights: importance weights per step
            ips: step-wise IPS estimate
            wis: step-wise WIS estimate
    """

    # pb and pe correspond to pi_b(a_t | s_t) and pi_e(a_t | s_t) respectively
    # Behavior propensity (from logits -> softmax)
    pb = np.maximum(p_b_taken_flat.astype(float), eps)

    # Target propensity (new policy evaluated on logged actions)
    pe = p_e_taken_flat.astype(float)

    if not np.all((pe >= 0.0) & (pe <= 1.0)):
        raise ValueError("p_e_taken_flat must be in [0,1]")

    if not np.all((pb >= 0.0) & (pb <= 1.0)):
        raise ValueError("p_b_taken_flat must be in [0,1]")

    # Importance weights (per-step)
    w = pe / pb

    if not np.isfinite(w).all():
        raise ValueError("Importance weights contain NaN or Inf")

    # Step-wise IPS / WIS (keep as diagnostics)
    ips = float(np.mean(w * rew_flat))
    wis = float(np.sum(w * rew_flat) / (np.sum(w) + eps))

    return {
        "weights": w,
        "ips": ips,
        "wis": wis,
        "w_max": float(w.max()),
        "w_p99": float(np.quantile(w, 0.99)),
    }


def dr_step_daily(
    *,
    num_days: int,
    steps_per_episode: int,
    rtg_flat: np.ndarray,  # (N,)
    P_new: np.ndarray,  # (N, A)
    act_flat: np.ndarray,  # (N,)
    p_b_taken_flat: np.ndarray,  # (N,)
    Q0: np.ndarray,  # (N,)
    Q1: np.ndarray,  # (N,)
    eps: float = 1e-12,
    cap: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Doubly Robust (per-step) aggregated daily Computing a step-wise DR estimate of the daily
    return.

    The DR estimator combines the DR baseline with importance-weighted corrections on the logged actions,
    and optional clipping is applied to stabilize large weights.

    Returns
    -------
    dr_day: (num_days,)  daily DR estimate taken at t=0
    w_logged: (N,)       clipped importance weights on logged actions
    """

    rtg_flat = np.asarray(rtg_flat, dtype=np.float32).reshape(-1)
    act_flat = np.asarray(act_flat).reshape(-1).astype(np.int64)
    p_b_taken_flat = np.asarray(p_b_taken_flat, dtype=np.float32).reshape(-1)

    P_new = np.asarray(P_new, dtype=np.float32)
    Q0 = np.asarray(Q0, dtype=np.float32).reshape(-1)
    Q1 = np.asarray(Q1, dtype=np.float32).reshape(-1)

    N = rtg_flat.shape[0]
    assert P_new.shape[0] == N
    assert Q0.shape[0] == N and Q1.shape[0] == N
    assert act_flat.shape[0] == N and p_b_taken_flat.shape[0] == N
    assert N == num_days * steps_per_episode, "N must be num_days * steps_per_episode"

    idx = np.arange(N)

    # w_logged = pi_e(a_logged|s) / pi_b(a_logged|s)
    p_e_taken_flat = P_new[idx, act_flat]
    w_logged = p_e_taken_flat / np.maximum(p_b_taken_flat, eps)

    # Clipping is used to limit the impact of rare transitions with very small behavior propensities
    w_logged = np.clip(w_logged, 1.0 / cap, cap).astype(np.float32)

    # Q(s, a_logged)
    Q_taken = np.where(act_flat == 0, Q0, Q1).astype(np.float32)

    # V_hat(s) = sum_a pi(a|s) Q(s,a)
    V_hat = (P_new[:, 0] * Q0 + P_new[:, 1] * Q1).astype(np.float32)

    # DR per step: V_hat(s) + w*(G - Q_taken)
    dr_step = V_hat + w_logged * (rtg_flat - Q_taken)

    # The daily value is taken at t=0 for consistency with the DM formulation used above
    dr_day = dr_step.reshape(num_days, steps_per_episode)[:, 0]

    return dr_day.astype(np.float32), w_logged
