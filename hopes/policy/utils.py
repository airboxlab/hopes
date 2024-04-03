import numpy as np


def deterministic_log_probs(
    actions: np.ndarray, actions_bins: np.ndarray, lamb: float = 1e-6
) -> np.ndarray:
    """Compute the log probabilities of a given set of actions, assuming a deterministic policy.

    We assign a probability of ~1 to the action returned by the function and an almost zero
    probability to all other actions (note: sum of log probs must be 1)
    """
    top = np.log(1.0 - (lamb * (len(actions_bins) - 1)))
    others = np.log(lamb)
    return np.array([[top if a == action else others for a in actions_bins] for action in actions])


def bin_actions(actions: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Bin the given actions into the given bins."""
    return np.array([min(bins, key=lambda x: abs(x - ra)) for ra in actions])


def piecewise_linear(x, left_cp, right_cp, slope, y0, y1) -> np.ndarray:
    r"""Define a piecewise linear function with 3 segments, such as:

     y0 --- \ (1)
             \ slope
              \
           (2) \ --- y1

    (1) left_cp (2) right_cp
    Note: the slope is not necessarily negative, the 2nd segment function can be increasing or decreasing.

    :param x: the input variable.
    :param left_cp: the left change point.
    :param right_cp: the right change point.
    :param slope: the slope of the linear segment.
    :param y0: the base value of the left segment.
    :param y1: the base value of the right segment.
    """
    # define the conditions for each segment
    conditions = [x < left_cp, (x >= left_cp) & (x <= right_cp), x > right_cp]
    # first segment is flat until lcp
    # second segment is linear between lcp and rcp
    # third segment is flat after rcp
    funcs = [
        lambda _: y0,
        lambda v: slope * (v - left_cp) + y0,
        lambda _: y1,
    ]
    return np.piecewise(x, conditions, funcs)
