import re

import numpy as np

from hopes.data.batch_utils import apply_stickiness_correction_to_rho
from hopes.data.diagnostics import identity_test, probs_diagnostics
from hopes.ope.estimators import StickySequentialDoublyRobust, StickyTrajectoryWiseIS
from hopes.ope.wrappers import compute_stepwise_ips_wis, dr_step_daily, wpdis_daily
from hopes.policy.onnx import OnnxRunner, probs_from_runner
from hopes.policy.utils import (
    load_latest_model_onnx,
    prepare_onnx_model,
    write_onnx_bytes,
)
from hopes.visualization.plot_diagnostics import plot_all_diagnostics_for_agent

#### Scale check: step-level vs day-level returns
# Target metric: daily return --> defined as the sum of rewards over a full episode (1 day, 34 steps).
# The behavior daily return provides the reference scale for all day-level estimates.
# -IPS and WIS are interpreted at the step level and mainly used as diagnostics,
#  while daily aggregation is required to compare policy performance meaningfully.


def run_ope_for_one_prefix(
    *,
    prefix_new: str,
    checkpoint_model_regex,
    tag: str,
    bucket: str,
    obs_batches,
    obs_flat,
    act_batches,
    act_flat,
    sticky_act_flat,
    rew_flat,
    p_b_taken_flat,
    probs_batches,
    Q0,
    Q1,
    steps_per_episode: int = 34,
    reassignment_threshold: float = 1e-3,
    eps: float = 1e-12,
    cap: int = 20,
    n_boot: int = 2000,
    seed: int = 0,
    gamma: float = 1.0,
    plot_diagnostics: bool = False,
    verbose: bool = False,
    diagnostics: bool = False,
):
    """Evaluate one evaluation policy (latest ONNX under prefix_new) using OPE on fixed logged
    behavior data. Assumes behavior propensities p_b_taken_flat are already computed from logged
    logits.

    Returns a dictionary of OPE metrics used for ranking / promotion.
    """

    # 1) Load latest ONNX for this run ---
    model_onnx_bytes, _, _ = load_latest_model_onnx(bucket, prefix_new, checkpoint_model_regex)

    # write local ONNX (unique filename)
    safe_tag = re.sub(r"[^a-zA-Z0-9_\-]+", "_", tag)

    # Evaluating the new policy on the logged trajectories to compute its action probabilities in the same states visited
    # by the behavior policy.
    # For each timestep, it extracts the full action distribution and the probability assigned
    # by the new policy to the logged action, which are later used by IS- and DR-based OPE estimators.
    # Write ONNX bytes to local file (raw)
    onnx_path_raw = write_onnx_bytes(model_onnx_bytes, f"eval_{safe_tag}_raw")

    # Prepare processed ONNX for inference (graph cleanup, etc.)
    onnx_path_proc = onnx_path_raw.replace(".onnx", "_processed.onnx")
    prepare_onnx_model(model_in=onnx_path_raw, model_out=onnx_path_proc)

    # 2) Run new policy on the logged trajectories ---
    new_runner = OnnxRunner(
        onnx_path_proc,
        T=10,
        obs_dim=obs_flat.shape[1],
        act_dim=2,
    )

    P_new, logp_new_taken = probs_from_runner(new_runner, obs_batches, act_batches)

    if verbose:
        print("row sums new:", P_new.sum(axis=1).min(), P_new.sum(axis=1).max())

    # Validate P_new rows sum to 1
    assert np.allclose(P_new.sum(axis=1), 1.0, atol=1e-4), "P_new rows do not sum to 1"

    # 3) Target propensity on logged actions: p_e_taken_flat = pi_e(a_t_logged | s_t),
    # used to build importance sampling ratios
    N = act_flat.shape[0]
    idx = np.arange(N)

    p_e_taken_flat = P_new[idx, act_flat].astype(np.float32)
    # Sanity check on the shape of p_e_taken_flat that must be equal to p_b_taken_flat
    assert p_e_taken_flat.shape == p_b_taken_flat.shape, "p_e_taken_flat shape mismatch"
    if verbose:
        print(
            "p_e_taken_flat percentiles:", np.percentile(p_e_taken_flat, [0, 1, 5, 50, 95, 99, 100])
        )
    assert np.allclose(P_new.sum(axis=1), 1.0, atol=1e-4)

    # 4) Episode structure and stickiness-corrected ratios (for WPDIS / traj-IS)
    T = steps_per_episode
    if N % T != 0:
        raise ValueError("N must be multiple of steps_per_episode")
    num_days = N // T

    r_day = rew_flat.reshape(num_days, T)
    pb_day = p_b_taken_flat.reshape(num_days, T)
    pe_day = p_e_taken_flat.reshape(num_days, T)

    # rho = pi_e(a_t | s_t) / pi_b(a_t | s_t)
    rho = pe_day / np.maximum(pb_day, eps)

    # After the first HVAC activation, the behavior and evaluation policies are assumed to coincide, so importance ratios are set to 1.0
    # for the remaining steps. This limits weight explosion while preserving the contribution of the decision
    # point where the policies may differ.

    # Stickiness correction (rho=1 after first switch to 1)
    sticky_day = sticky_act_flat.reshape(num_days, T)

    # After the first switch to action=1, both policies effectively follow the same control logic
    # Setting rho=1 beyond this point avoids accumulating unnecessary variance
    rho = apply_stickiness_correction_to_rho(rho, sticky_day)

    # Behavior policy propensities: from logits -> softmax
    if diagnostics:
        # Ratio diagnostics
        ratio = p_e_taken_flat / np.maximum(p_b_taken_flat, eps)
        print("p_b_taken percentiles:", np.percentile(p_b_taken_flat, [0, 1, 5, 50, 95, 99, 100]))
        print("fraction p_b_taken < 1e-3:", np.mean(p_b_taken_flat < 1e-3))
        # Target policy propensities --> new policy on logged actions
        print("p_e_taken percentiles:", np.percentile(p_e_taken_flat, [0, 1, 5, 50, 95, 99, 100]))
        # IS will explode?
        print("ratio percentiles:", np.percentile(ratio, [50, 90, 95, 99, 99.5, 100]))
        print("fraction ratio > 100:", np.mean(ratio > 100))

    # Sanity: p_b_taken_flat should match probs[action]
    # (assumes you already flattened probs_batches into probs_flat)
    probs_flat = np.vstack(probs_batches).astype(np.float32)
    a_flat = np.concatenate(act_batches).astype(int)
    assert np.allclose(p_b_taken_flat, probs_flat[np.arange(N), a_flat], atol=1e-6)

    # 5) Step-wise IPS/WIS diagnostics
    results_ips_wis_stepwise = compute_stepwise_ips_wis(
        p_b_taken_flat, p_e_taken_flat, rew_flat, eps
    )

    # Clipped WIS is reported to assess how sensitive the estimate is to large importance weights
    if diagnostics:
        for caps in [5, 10, 20, 50]:
            w_clip = np.clip(results_ips_wis_stepwise["weights"], 1.0 / caps, caps)
            wis_clip = float(np.sum(w_clip * rew_flat) / (np.sum(w_clip) + eps))
            print(f"WIS clipped (cap={caps}):", wis_clip)

    # 6) Trajectory-wise IS + SNIS (sticky-corrected)
    traj_is = StickyTrajectoryWiseIS(steps_per_episode=steps_per_episode, eps=eps)
    traj_is.set_parameters(
        p_e_taken_flat=p_e_taken_flat,
        p_b_taken_flat=p_b_taken_flat,
        rew_flat=rew_flat,
        sticky_act_flat=sticky_act_flat,
    )
    G = traj_is.estimate_weighted_rewards()  # sum of rewards per episode
    V_IS = traj_is.estimate_policy_value()
    V_SNIS = traj_is.estimate_self_normalized_value()

    # Clipped variants are reported to diagnose sensitivity to large trajectory weights
    if diagnostics:
        for caps in [5, 10, 20, 50]:
            rho_c = np.clip(rho, 1.0 / caps, caps)
            logW_c = np.sum(np.log(np.maximum(rho_c, eps)), axis=1)
            W_c = np.exp(np.clip(logW_c, -50, 50))
            V_c = float(np.sum(W_c * G) / np.maximum(np.sum(W_c), eps))
            print(f"Trajectory-SNIS clipped (cap={caps}) (sticky-corrected):", V_c)

    # 7) WPDIS daily for evaluation and behaviour policies
    # Checking WPDIS daily for new policies (evaluation)
    wpdis_e_mean, wpdis_e_lo, wpdis_e_hi = wpdis_daily(
        num_days=num_days,
        steps_per_episode=steps_per_episode,
        rew_flat=rew_flat,
        p_b_taken_flat=p_b_taken_flat,
        p_e_taken_flat=p_e_taken_flat,
        sticky_act_flat=sticky_act_flat,
        eps=eps,
        clip=float(cap),
        n_boot=n_boot,
        alpha=0.05,
        seed=seed,
    )

    print(f"WPDIS eval policy(cap={cap}) mean:", wpdis_e_mean, "CI:", (wpdis_e_lo, wpdis_e_hi))

    # Checking WPDIS daily for old policy (behaviour)
    wpdis_b_mean, wpdis_b_lo, wpdis_b_hi = wpdis_daily(
        num_days=num_days,
        steps_per_episode=steps_per_episode,
        rew_flat=rew_flat,
        p_b_taken_flat=p_b_taken_flat,
        p_e_taken_flat=p_b_taken_flat,  # identity case since we are checking on behaviour policy
        sticky_act_flat=sticky_act_flat,
        eps=eps,
        clip=float(cap),
        n_boot=n_boot,
        alpha=0.05,
        seed=seed,
    )

    print(f"WPDIS behav policy(cap={cap}) mean:", wpdis_b_mean, "CI:", (wpdis_b_lo, wpdis_b_hi))

    # Compute delta to check re-assignment
    delta_lb = wpdis_e_lo - wpdis_b_mean

    # Checking if the evaluation agent could be elected for managing FCUs as condition is respected, including a threshold
    promote = wpdis_e_lo > wpdis_b_mean * (1 - reassignment_threshold)

    print("PROMOTE?", promote, "| delta =", wpdis_e_lo - wpdis_b_mean)

    if diagnostics:
        ## Other sanity checks
        probs_diagnostics(P_new, p_b_taken_flat, p_e_taken_flat, eps, verbose)

        #### Identity test --> weights must be all ones
        identity_test(rew_flat, p_e_taken_flat, eps, verbose)

    # Mismatch diagnostics (NEW vs BEHAVIOR / STICKY) --> optional depending on diagnostics variable.
    # Probability under NEW of taking the BEHAVIOR action
    if diagnostics:
        p_e_on_behavior = P_new[idx, act_flat].astype(np.float32)

        # Greedy action under NEW
        a_new_greedy = np.argmax(P_new, axis=1).astype(np.int64)
        mismatch_new_vs_raw = float(np.mean(a_new_greedy != act_flat))

        # Probability under NEW of taking the STICKY (applied) action
        p_e_on_sticky = P_new[idx, sticky_act_flat].astype(np.float32)
        mismatch_new_vs_sticky = float(np.mean(a_new_greedy != sticky_act_flat))

        # Ratio diagnostics (can explode if pb is small)
        w = p_e_on_behavior / np.maximum(p_b_taken_flat, eps)

        print(
            "p_e(a_behavior|s) percentiles:",
            np.percentile(p_e_on_behavior, [0, 1, 5, 50, 95, 99, 100]),
        )
        print("fraction p_e(a_behavior|s) < 1e-3:", float(np.mean(p_e_on_behavior < 1e-3)))
        print("fraction p_e(a_behavior|s) < 1e-2:", float(np.mean(p_e_on_behavior < 1e-2)))
        print("greedy mismatch rate (new vs behavior):", mismatch_new_vs_raw)

        print(
            "p_e(a_sticky|s) percentiles:", np.percentile(p_e_on_sticky, [0, 1, 5, 50, 95, 99, 100])
        )
        print("fraction p_e(a_sticky|s) < 1e-3:", float(np.mean(p_e_on_sticky < 1e-3)))
        print("greedy mismatch rate (new vs sticky):", mismatch_new_vs_sticky)

        print("w percentiles:", np.percentile(w, [50, 90, 95, 99, 99.5, 100]))
        print("w max:", float(w.max()))

    # 9) Build RETURN-TO-GO (RTG) for DR (step-wise DR uses RTG as y)
    rew_day = rew_flat.reshape(num_days, steps_per_episode)

    # Behavior daily returns (ground truth scale)
    G_behavior = rew_flat.reshape(num_days, steps_per_episode).sum(axis=1)

    # G_t = sum_{k=t..T-1} r_k
    rtg_day = np.flip(np.cumsum(np.flip(rew_day, axis=1), axis=1), axis=1)
    rtg_flat = rtg_day.reshape(-1).astype(np.float32)

    # 10) Estimator validation: DM behavior vs true behavior
    P_b = np.vstack(probs_batches).astype(np.float32)
    assert P_b.shape == P_new.shape
    assert np.allclose(P_b.sum(axis=1), 1.0, atol=1e-6)

    # Direct Method under behavior policy (same Q-model) for daily performance: sum over the day of E_a[Q(s_t,a)] under NEW
    V_step_behavior = P_b[:, 0] * Q0 + P_b[:, 1] * Q1
    dm_behavior_day = V_step_behavior.reshape(num_days, steps_per_episode)[:, 0]

    # DM under new policy
    V_step_new = P_new[:, 0] * Q0 + P_new[:, 1] * Q1
    dm_day = V_step_new.reshape(num_days, steps_per_episode)[:, 0]

    if verbose:
        print(
            "DM behavior daily (mean/std):",
            float(dm_behavior_day.mean()),
            float(dm_behavior_day.std()),
        )
        print("True behavior daily (mean/std):", float(G_behavior.mean()), float(G_behavior.std()))
        print("DM daily (mean/std):", float(dm_day.mean()), float(dm_day.std()))

    # 11) Doubly Robust (per-step) aggregated daily
    # Computing a step-wise DR estimate of the daily return.
    # The DR estimator combines the DR baseline with importance-weighted corrections on the logged actions,
    # and optional clipping is applied to stabilize large weights

    dr_day, w_logged = dr_step_daily(
        num_days=num_days,
        rtg_flat=rtg_flat,
        P_new=P_new,
        act_flat=act_flat,
        p_b_taken_flat=p_b_taken_flat,
        eps=eps,
        cap=cap,
        Q0=Q0,
        Q1=Q1,
        steps_per_episode=steps_per_episode,  # <-- ensure wrapper signature includes this
    )

    if verbose:
        print("DR daily: mean", float(dr_day.mean()), "std", float(dr_day.std()))
        print(
            "w_logged diagnostics: max",
            float(w_logged.max()),
            "p99",
            float(np.quantile(w_logged, 0.99)),
        )

    # 12) Doubly Robust sequential TD-form estimator (sticky corrected, returns episode-level (daily) values)
    dr_td = StickySequentialDoublyRobust(
        steps_per_episode=steps_per_episode, eps=eps, cap=float(cap), gamma=float(gamma)
    )
    dr_td.set_parameters(
        rew_flat=rew_flat,
        act_flat=act_flat,
        sticky_act_flat=sticky_act_flat,
        p_b_taken_flat=p_b_taken_flat,
        p_e_taken_flat=p_e_taken_flat,
        P_new=P_new,
        Q0=Q0,
        Q1=Q1,
    )

    # Components for diagnostics if you want to log them
    rho_td, W_t_td, dr_episode = dr_td.estimate_components()
    DR = float(np.mean(dr_episode))

    if verbose:
        print("DR daily (step-wise): mean", float(dr_day.mean()), "std", float(dr_day.std()))
        print(
            "w_logged diagnostics: max",
            float(w_logged.max()),
            "p99",
            float(np.quantile(w_logged, 0.99)),
        )
        print("DR (TD form):", DR)
        print("rho diagnostics: max", float(rho_td.max()), "p99", float(np.quantile(rho_td, 0.99)))
        print("W_t diagnostics: max", float(W_t_td.max()), "p99", float(np.quantile(W_t_td, 0.99)))

    # 12) Optional: plot diagnostics for this agent
    if plot_diagnostics:
        plot_all_diagnostics_for_agent(
            agent_name=tag,
            P_b=P_b,
            P_new=P_new,
            Q0=Q0,
            Q1=Q1,
            act_flat=act_flat,
            sticky_act_flat=sticky_act_flat,
            p_b_taken_flat=p_b_taken_flat,
            p_e_taken_flat=p_e_taken_flat,
            num_days=num_days,
            T=steps_per_episode,
        )

    # 13) Build output dictionary with all relevant metrics for ranking and monitoring (including diagnostics if needed)
    out = {
        "agent_centroid": tag,
        # Main OPE: evaluation policy
        "wpdis_cap_mean_e": float(wpdis_e_mean),
        "wpdis_cap_ci_lo_e": float(wpdis_e_lo),
        "wpdis_cap_ci_hi_e": float(wpdis_e_hi),
        # Main OPE: behavior policy
        "wpdis_cap_mean_b": float(wpdis_b_mean),
        "wpdis_cap_ci_lo_b": float(wpdis_b_lo),
        "wpdis_cap_ci_hi_b": float(wpdis_b_hi),
        # Check criteria for re-assignment
        "promote_lb_gt_behavior_mean": bool(promote),
        "delta_lb_vs_bmean": float(delta_lb),
        # Trajectory-level diagnostics
        "traj_is": float(V_IS),
        "traj_snis": float(V_SNIS),
        # step-wise baselines (diagnostic)
        "ips_step": float(results_ips_wis_stepwise["ips"]),
        "wis_step": float(results_ips_wis_stepwise["wis"]),
        # DM / DR
        "dm_day_mean": float(np.mean(dm_day)),
        "dm_day_std": float(np.std(dm_day)),
        "dr_day_mean": float(np.mean(dr_day)),
        "dr_day_std": float(np.std(dr_day)),
        "dr_td": float(DR),
        # Scale reference (behavior)
        "behavior_daily_mean": float(G_behavior.mean()),
        "behavior_daily_std": float(G_behavior.std()),
        # Optional: store mean/std of DM behavior for monitoring
        "dm_behavior_day_mean": float(np.mean(dm_behavior_day)),
        "dm_behavior_day_std": float(np.std(dm_behavior_day)),
    }

    return out
