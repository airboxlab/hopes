import matplotlib.pyplot as plt
import numpy as np

## Make the result actionable (where does the gap come from?)

# Plots for indicating:
# - At which steps do OLD and NEW differ the most in action distribution?
# - At which steps does the model predict higher/lower reward under NEW?


def plot_action_probabilities_over_day(P_b, P_new, num_days, T, title_suffix=""):
    """Plots mean P(a=1|s) over the day for behavior vs new."""
    assert P_b.shape == P_new.shape
    assert P_b.shape[0] == num_days * T

    P_b_day = P_b.reshape(num_days, T, -1)
    P_new_day = P_new.reshape(num_days, T, -1)

    p0_b = P_b_day[:, :, 0].mean(axis=0)
    p0_new = P_new_day[:, :, 0].mean(axis=0)
    p1_b = P_b_day[:, :, 1].mean(axis=0)
    p1_new = P_new_day[:, :, 1].mean(axis=0)

    plt.figure()
    plt.plot(p1_b, label="behavior: P(a=1|s)")
    plt.plot(p1_new, label="new: P(a=1|s)")
    plt.xlabel("step in day")
    plt.ylabel("probability")
    plt.title(f"Average probability of action=1 over the day {title_suffix}".strip())
    plt.legend()
    plt.show()

    # plot 1b - 1 - P(a=0|s)
    plt.figure()
    plt.plot(1.0 - p0_b, label="behavior: 1-P(a=0|s)")
    plt.plot(1.0 - p0_new, label="new: 1-P(a=0|s)")
    plt.xlabel("step in day")
    plt.ylabel("probability")
    plt.title("Average 1 - P(a=0|s) over the day")
    plt.legend()
    plt.show()

    print(
        "Mean P_behavior(a=1):",
        float(P_b[:, 1].mean()),
        "Mean P_new(a=1):",
        float(P_new[:, 1].mean()),
    )


def plot_expected_rtg_over_day(P_b, P_new, Q0, Q1, num_days, T, title_suffix=""):
    """Plots expected RTG under Q-model for behavior vs new."""
    assert P_b.shape == P_new.shape
    assert Q0.shape == Q1.shape == (num_days * T,)

    V_b = (P_b[:, 0] * Q0 + P_b[:, 1] * Q1).reshape(num_days, T).mean(axis=0)
    V_new = (P_new[:, 0] * Q0 + P_new[:, 1] * Q1).reshape(num_days, T).mean(axis=0)

    plt.figure()
    plt.plot(V_b, label="behavior: E[RTG | behavior]")
    plt.plot(V_new, label="new: E[RTG | new]")
    plt.xlabel("step in day")
    plt.ylabel("expected RTG (model)")
    plt.title(f"Model-predicted expected return-to-go over the day {title_suffix}".strip())
    plt.legend()
    plt.show()


def plot_immediate_reward_proxy_from_rtg(P_b, P_new, Q0, Q1, num_days, T, title_suffix=""):
    """
    Plots proxy immediate reward from RTG: rhat_t = V_t - V_{t+1}, and its cumulative gap.
    """
    assert P_b.shape == P_new.shape
    assert Q0.shape == Q1.shape == (num_days * T,)

    V_b = (P_b[:, 0] * Q0 + P_b[:, 1] * Q1).reshape(num_days, T).mean(axis=0)
    V_new = (P_new[:, 0] * Q0 + P_new[:, 1] * Q1).reshape(num_days, T).mean(axis=0)

    rhat_b = V_b[:-1] - V_b[1:]
    rhat_new = V_new[:-1] - V_new[1:]
    steps = np.arange(T - 1)

    plt.figure()
    plt.plot(steps, rhat_b, label="behavior: approx E[r_t]")
    plt.plot(steps, rhat_new, label="new: approx E[r_t]")
    plt.xlabel("step in day")
    plt.ylabel("approx expected immediate reward (model)")
    plt.title(f"Approx expected immediate reward over the day {title_suffix}".strip())
    plt.legend()
    plt.show()

    gap = rhat_new - rhat_b
    cum_gap = np.cumsum(gap)

    plt.figure()
    plt.plot(steps, cum_gap, label="cumsum(Δ approx E[r_t])")
    plt.xlabel("step in day")
    plt.ylabel("cumulative gap (approx)")
    plt.title(f"Cumulative model-predicted gap over the day {title_suffix}".strip())
    plt.legend()
    plt.show()


def plot_sticky_on_fraction(sticky_act_flat, num_days, T, title_suffix=""):
    """Plots fraction of episodes where sticky_action is ON at each step."""
    a_sticky_day = sticky_act_flat.reshape(num_days, T)
    p_on = (a_sticky_day == 1).mean(axis=0)

    plt.figure()
    plt.plot(p_on, label="P(sticky_action=1) (applied)")
    plt.xlabel("step in day")
    plt.ylabel("fraction ON")
    plt.title(f"Fraction of episodes with HVAC ON (sticky action) {title_suffix}".strip())
    plt.legend()
    plt.show()


def plot_raw_vs_sticky_mismatch(act_flat, sticky_act_flat, num_days, T, title_suffix=""):
    """Plots fraction of timesteps where stickiness overrides the raw action."""
    a_raw_day = act_flat.reshape(num_days, T)
    a_sticky_day = sticky_act_flat.reshape(num_days, T)

    mismatch = (a_raw_day != a_sticky_day).mean(axis=0)

    plt.figure()
    plt.plot(mismatch, label="P(raw != sticky)")
    plt.xlabel("step in day")
    plt.ylabel("fraction mismatch")
    plt.title(f"Fraction of timesteps where stickiness overrides raw action {title_suffix}".strip())
    plt.legend()
    plt.show()


def plot_mean_rho_before_after_stickiness(
    p_b_taken_flat, p_e_taken_flat, sticky_act_flat, num_days, T, eps=1e-12, title_suffix=""
):
    """Plots mean importance ratio per step, before and after stickiness correction."""
    pb_day = p_b_taken_flat.reshape(num_days, T)
    pe_day = p_e_taken_flat.reshape(num_days, T)

    rho_unc = pe_day / np.maximum(pb_day, eps)

    rho_corr = rho_unc.copy()
    a_sticky_day = sticky_act_flat.reshape(num_days, T)
    has_switch = (a_sticky_day == 1).any(axis=1)
    first_switch = np.argmax(a_sticky_day == 1, axis=1)

    for i in range(num_days):
        if has_switch[i]:
            t = int(first_switch[i])
            if t + 1 < T:
                rho_corr[i, t + 1 :] = 1.0

    plt.figure()
    plt.plot(rho_unc.mean(axis=0), label="mean rho (uncorrected)")
    plt.plot(rho_corr.mean(axis=0), label="mean rho (sticky-corrected)")
    plt.xlabel("step in day")
    plt.ylabel("mean rho")
    plt.title(f"Mean importance ratio per step (before/after stickiness) {title_suffix}".strip())
    plt.legend()
    plt.show()


def plot_all_diagnostics_for_agent(
    agent_name: str,
    P_b,
    P_new,
    Q0,
    Q1,
    act_flat,
    sticky_act_flat,
    p_b_taken_flat,
    p_e_taken_flat,
    num_days,
    T,
    eps=1e-12,
):
    """Convenience wrapper to generate all plots for one agent/policy."""
    suffix = f"({agent_name})"

    plot_action_probabilities_over_day(P_b, P_new, num_days, T, title_suffix=suffix)
    plot_expected_rtg_over_day(P_b, P_new, Q0, Q1, num_days, T, title_suffix=suffix)
    plot_immediate_reward_proxy_from_rtg(P_b, P_new, Q0, Q1, num_days, T, title_suffix=suffix)
    plot_sticky_on_fraction(sticky_act_flat, num_days, T, title_suffix=suffix)
    plot_raw_vs_sticky_mismatch(act_flat, sticky_act_flat, num_days, T, title_suffix=suffix)
    plot_mean_rho_before_after_stickiness(
        p_b_taken_flat, p_e_taken_flat, sticky_act_flat, num_days, T, eps=eps, title_suffix=suffix
    )
