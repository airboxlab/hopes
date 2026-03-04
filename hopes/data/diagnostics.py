import numpy as np

# To explicitly validate the main components of the pipeline.
# These checks are needed to ensure that:
# - the ONNX policy replay behaves as expected,
# - the action probabilities are well-formed,
# - the importance sampling machinery is consistent,
# - and the estimators operate on the correct scale.

# If any of the checks below fails, the OPE estimates should be considered unreliable until the issue is understood and fixed.


#### Action probability diagnostics
def probs_diagnostics(P_new, p_b_taken_flat, p_e_taken_flat, eps, verbose=False):
    """Verify that the action probabilities produced by the ONNX policies are OK (from the values
    point of view).

    Checking:
    - probabilities are finite (no NaNs or infs),
    - each row sums to one,
    - and no systematic saturation or collapse occurs.
    """

    # 1) New policy probs diagnostics
    if verbose:
        print("P_new min/max:", float(np.min(P_new)), float(np.max(P_new)))
        print(
            "P_new row sums (min/max):",
            float(P_new.sum(axis=1).min()),
            float(P_new.sum(axis=1).max()),
        )
    assert np.allclose(P_new.sum(axis=1), 1.0, atol=1e-4)

    # 2) Behavior/target taken-prob diagnostics
    if verbose:
        print(
            "p_b_taken_flat percentiles:", np.percentile(p_b_taken_flat, [0, 1, 5, 50, 95, 99, 100])
        )
        print(
            "p_e_taken_flat percentiles:", np.percentile(p_e_taken_flat, [0, 1, 5, 50, 95, 99, 100])
        )

    assert np.all(np.isfinite(p_b_taken_flat)) and np.all(np.isfinite(p_e_taken_flat))
    assert np.all((p_b_taken_flat >= 0.0) & (p_b_taken_flat <= 1.0))
    assert np.all((p_e_taken_flat >= 0.0) & (p_e_taken_flat <= 1.0))

    # 3) Ratio diagnostics --> IS is going to explode?
    w = p_e_taken_flat / np.maximum(p_b_taken_flat, eps)
    if verbose:
        print("weights w min/max:", float(w.min()), float(w.max()))
        print("weights w percentiles:", np.percentile(w, [50, 90, 95, 99, 99.5, 100]))
        print("fraction p_b_taken < 1e-3:", float(np.mean(p_b_taken_flat < 1e-3)))


#### Identity test
def identity_test(rew_flat, p_e_taken_flat, eps, verbose=False):
    """Evaluating OPE by setting the target policy equal to the behavior policy.

    In this case:
    - all importance weights should be equal to 1,
    - IPS and WIS should reduce to simple averages of the logged rewards,
    - and the estimated return should match the scale of the behavior data.
    """

    w_id = p_e_taken_flat / np.maximum(p_e_taken_flat, eps)
    assert np.allclose(w_id, 1.0, atol=1e-6)

    ips_id = float(np.mean(w_id * rew_flat))
    wis_id = float(np.sum(w_id * rew_flat) / (np.sum(w_id) + eps))
    if verbose:
        print("Identity weights (min/max):", float(w_id.min()), float(w_id.max()))
        print("IPS identity (per-step):", ips_id)
        print("WIS identity (per-step):", wis_id)
