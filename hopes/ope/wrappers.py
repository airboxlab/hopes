import numpy as np


def compute_stepwise_ips_wis(
    p_b_taken_flat: np.ndarray,
    p_e_taken_flat: np.ndarray,
    rew_flat: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, np.ndarray | float]:
    """Compute step-wise IPS and WIS diagnostics from taken propensities.

    :param p_b_taken_flat: Shape (N,), behavior policy probabilities on logged actions.
    :param p_e_taken_flat: Shape (N,), target policy probabilities on logged actions.
    :param rew_flat: Shape (N,), rewards.
    :param eps: Numerical stabilizer.
    :return: Dictionary containing:
    - `weights`: Shape (N,), importance weights.
    - `ips`: Scalar, step-wise IPS estimate.
    - `wis`: Scalar, step-wise WIS estimate.
    - `w_max`: Scalar, maximum importance weight.
    - `w_p99`: Scalar, 99th percentile of importance weights.
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
