import numpy as np


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
