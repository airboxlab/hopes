import numpy as np


def apply_stickiness_correction_to_rho(
    rho: np.ndarray,  # importance ratios, shape (n_eps, T)
    sticky_act_day: np.ndarray,  # sticky actions applied, shape (n_eps, T), values in {0,1}
    value_after_switch: float = 1.0,  # value assigned to rho after the first switch.
) -> np.ndarray:
    """Apply stickiness correction to importance ratios.

    If the sticky action indicator becomes 1 within an episode, all subsequent
    importance ratios for that episode are replaced with ``value_after_switch``.

    For each episode ``i``, let ``t_star`` be the first timestep such that
    ``sticky_act_day[i, t_star] == 1``. For all ``t > t_star`` we set:

        rho[i, t] = value_after_switch

    This reflects the assumption that once the sticky action is triggered
    (e.g. HVAC turned ON), the control logic becomes deterministic and both
    policies effectively coincide.

    :param importance_ratios: Step-wise importance ratios of shape (n_episodes, T).
    :param sticky_act_day: Sticky action indicator of shape (n_episodes, T).
    :param value_after_switch: Value assigned to ratios after the first sticky switch.
    :return: Corrected importance ratios with the same shape as the input.
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
                # After the first sticky switch the control becomes deterministic
                # (HVAC remains ON). Since both policies coincide, the importance
                # ratios are set to 1 to avoid unnecessary variance.
                rho[i, t_star + 1 :] = value_after_switch
    return rho


def extract_logged_action_probabilities(
    action_probabilities: np.ndarray,
    logged_actions: np.ndarray,
) -> np.ndarray:
    """Extract the probability of each logged action from per-step action distributions.

    :param action_probabilities: action probability matrix of shape `(n_samples,
        n_actions)`.
    :param logged_actions: logged action indices of shape `(n_samples,)`.
    :return: probability assigned to each logged action, shape `(n_samples,)`.
    :raises ValueError: if shapes are invalid or action indices are out of bounds.
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

    For each sample :math:`t`, this computes:

    .. math::
        \rho_t = \frac{\\pi_e(a_t \\mid s_t)}{\\pi_b(a_t \\mid s_t)}

    where :math:`a_t` is the logged action.

    :param target_policy_action_probabilities: target policy action probabilities of shape
        `(n_samples, n_actions)`.
    :param behavior_policy_action_probabilities: behavior policy action probabilities of shape
        `(n_samples, n_actions)`.
    :param logged_actions: logged action indices of shape `(n_samples,)`.
    :param eps: numerical stabilizer used in the denominator.
    :param clip: optional symmetric clipping threshold applied as
        ``np.clip(rho, 1 / clip, clip)``. Must be greater than or equal to 1.0 when provided.
    :return: step-wise importance ratios, shape `(n_samples,)`.
    :raises ValueError: if inputs are invalid.
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

    :param importance_ratios: step-wise importance ratios of shape `(n_samples,)`.
    :param sticky_actions: sticky action indicator of shape `(n_samples,)`.
    :param steps_per_episode: number of steps per episode.
    :param value_after_switch: replacement ratio value after the first sticky switch within
        an episode.
    :return: corrected step-wise importance ratios, shape `(n_samples,)`.
    :raises ValueError: if shapes are invalid.
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

    :param target_policy_action_probabilities: target policy action probabilities of shape
        `(n_samples, n_actions)`.
    :param behavior_policy_action_probabilities: behavior policy action probabilities of shape
        `(n_samples, n_actions)`.
    :param logged_actions: logged action indices of shape `(n_samples,)`.
    :param steps_per_episode: number of steps per episode.
    :param eps: numerical stabilizer used in the denominator.
    :param clip: optional symmetric clipping threshold applied as
        ``np.clip(rho, 1 / clip, clip)``.
    :param apply_stickiness: whether to apply stickiness correction.
    :param sticky_actions: sticky action indicator, required when ``apply_stickiness=True``.
    :param value_after_switch: replacement ratio value after the first sticky switch, required when
        ``apply_stickiness=True``.
    :return: step-wise importance ratios, shape `(n_samples,)`.
    :raises ValueError: if stickiness correction is requested but required inputs are missing.
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
