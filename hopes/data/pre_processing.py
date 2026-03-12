import numpy as np

from hopes.ope.utils import apply_stickiness_correction_to_rho


def extract_logged_action_probabilities(
    action_probabilities: np.ndarray,
    logged_actions: np.ndarray,
) -> np.ndarray:
    """Extract the probability of each logged action from per-step action distributions.

    Parameters
    ----------
    action_probabilities:
        Array of action probabilities with shape (n_samples, n_actions).
    logged_actions:
        Logged actions with shape (n_samples,).

    Returns
    -------
    np.ndarray
        Probability assigned to each logged action, shape (n_samples,).

    Raises
    ------
    ValueError
        If shapes are invalid or action indices are out of bounds.
    """
    action_probabilities = np.asarray(action_probabilities, dtype=np.float32)
    logged_actions = np.asarray(logged_actions, dtype=np.int64).reshape(-1)

    if action_probabilities.ndim != 2:
        raise ValueError("action_probabilities must be a 2D array of shape (n_samples, n_actions).")

    if logged_actions.ndim != 1:
        raise ValueError("logged_actions must be a 1D array.")

    if action_probabilities.shape[0] != logged_actions.shape[0]:
        raise ValueError("logged_actions length must match the number of samples.")

    n_samples, n_actions = action_probabilities.shape

    if np.any(logged_actions < 0) or np.any(logged_actions >= n_actions):
        raise ValueError("logged_actions contains invalid action indices.")

    row_indices = np.arange(n_samples, dtype=np.int64)
    return action_probabilities[row_indices, logged_actions].astype(np.float32)


def build_stepwise_importance_ratios(
    target_policy_action_probabilities: np.ndarray,
    behavior_policy_action_probabilities: np.ndarray,
    logged_actions: np.ndarray,
    eps: float = 1e-12,
    clip: float | None = None,
) -> np.ndarray:
    """Build step-wise importance ratios for the logged actions.

    For each sample t, this computes:

    .. math::
        \\rho_t = \\frac{\\pi_e(a_t \\mid s_t)}{\\pi_b(a_t \\mid s_t)}

    where :math:`a_t` is the logged action.

    Parameters
    ----------
    target_policy_action_probabilities:
        Target policy action probabilities, shape (n_samples, n_actions).
    behavior_policy_action_probabilities:
        Behavior policy action probabilities, shape (n_samples, n_actions).
    logged_actions:
        Logged actions, shape (n_samples,).
    eps:
        Numerical stabilizer used in the denominator.
    clip:
        Optional symmetric clipping threshold applied as
        ``np.clip(rho, 1 / clip, clip)``.
        Must be greater than or equal to 1.0 when provided.

    Returns
    -------
    np.ndarray
        Step-wise importance ratios, shape (n_samples,).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0.")

    if clip is not None and clip < 1.0:
        raise ValueError("clip must be >= 1.0 when provided.")

    p_e_taken = extract_logged_action_probabilities(
        action_probabilities=target_policy_action_probabilities,
        logged_actions=logged_actions,
    )
    p_b_taken = extract_logged_action_probabilities(
        action_probabilities=behavior_policy_action_probabilities,
        logged_actions=logged_actions,
    )

    rho = p_e_taken / np.maximum(p_b_taken, eps)

    if clip is not None:
        rho = np.clip(rho, 1.0 / clip, clip)

    return rho.astype(np.float32)


def apply_stickiness_correction(
    importance_ratios: np.ndarray,
    sticky_actions: np.ndarray,
    steps_per_episode: int,
    value_after_switch: float,
) -> np.ndarray:
    """Apply stickiness correction to precomputed step-wise importance ratios.

    Parameters
    ----------
    importance_ratios:
        Step-wise importance ratios, shape (n_samples,).
    sticky_actions:
        Sticky action indicator, shape (n_samples,).
    steps_per_episode:
        Number of steps per episode.
    value_after_switch:
        Replacement ratio value after the first sticky switch within an episode.

    Returns
    -------
    np.ndarray
        Corrected step-wise importance ratios, shape (n_samples,).

    Raises
    ------
    ValueError
        If shapes are invalid.
    """
    importance_ratios = np.asarray(importance_ratios, dtype=np.float32).reshape(-1)
    sticky_actions = np.asarray(sticky_actions, dtype=np.int64).reshape(-1)

    if steps_per_episode <= 0:
        raise ValueError("steps_per_episode must be > 0.")

    if importance_ratios.shape[0] != sticky_actions.shape[0]:
        raise ValueError("sticky_actions length must match importance_ratios length.")

    if importance_ratios.shape[0] % steps_per_episode != 0:
        raise ValueError("Number of samples must be divisible by steps_per_episode.")

    rho_2d = importance_ratios.reshape(-1, steps_per_episode)
    sticky_2d = sticky_actions.reshape(-1, steps_per_episode)

    corrected = apply_stickiness_correction_to_rho(
        rho=rho_2d,
        sticky_act_day=sticky_2d,
        value_after_switch=value_after_switch,
    )

    return corrected.reshape(-1).astype(np.float32)


def build_stepwise_importance_ratios_with_stickiness(
    target_policy_action_probabilities: np.ndarray,
    behavior_policy_action_probabilities: np.ndarray,
    logged_actions: np.ndarray,
    steps_per_episode: int,
    eps: float = 1e-12,
    clip: float | None = None,
    apply_stickiness: bool = False,
    sticky_actions: np.ndarray | None = None,
    value_after_switch: float | None = None,
) -> np.ndarray:
    """Build step-wise importance ratios with optional stickiness correction.

    Parameters
    ----------
    target_policy_action_probabilities:
        Target policy action probabilities, shape (n_samples, n_actions).
    behavior_policy_action_probabilities:
        Behavior policy action probabilities, shape (n_samples, n_actions).
    logged_actions:
        Logged actions, shape (n_samples,).
    steps_per_episode:
        Number of steps per episode.
    eps:
        Numerical stabilizer used in the denominator.
    clip:
        Optional symmetric clipping threshold applied as
        ``np.clip(rho, 1 / clip, clip)``.
    apply_stickiness:
        Whether to apply stickiness correction.
    sticky_actions:
        Sticky action indicator, required when ``apply_stickiness=True``.
    value_after_switch:
        Replacement ratio value after the first sticky switch, required when
        ``apply_stickiness=True``.

    Returns
    -------
    np.ndarray
        Step-wise importance ratios, shape (n_samples,).

    Raises
    ------
    ValueError
        If stickiness correction is requested but required inputs are missing.
    """
    rho = build_stepwise_importance_ratios(
        target_policy_action_probabilities=target_policy_action_probabilities,
        behavior_policy_action_probabilities=behavior_policy_action_probabilities,
        logged_actions=logged_actions,
        eps=eps,
        clip=clip,
    )

    if not apply_stickiness:
        return rho

    if sticky_actions is None:
        raise ValueError("sticky_actions must be provided when apply_stickiness=True.")

    if value_after_switch is None:
        raise ValueError("value_after_switch must be provided when apply_stickiness=True.")

    return apply_stickiness_correction(
        importance_ratios=rho,
        sticky_actions=sticky_actions,
        steps_per_episode=steps_per_episode,
        value_after_switch=value_after_switch,
    )
