import numpy as np


def discretize_action_space(
    actions: np.ndarray, bins: int | list[int] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize a (possibly) continuous action space into `bins` bins.

    :param actions: the actions to discretize.
    :param bins: the number of bins to discretize the actions into, or the bin edges. If None, the Freedman-Diaconis estimator is used.
    :return: computed bins, the discretized actions.
    """
    assert isinstance(actions, np.ndarray), "actions must be a numpy array"
    assert bins is None or isinstance(
        bins, (int, list)
    ), "bins must be an integer or a list of bin edges"

    if bins is None or isinstance(bins, int):
        # if bins is an integer, compute the bin edges using the number of bins.
        # otherwsie, compute the bin edges using Frieedman-Diaconis estimator
        # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
        bins = np.histogram_bin_edges(actions, bins=bins or "fd")
        # remove the last bin edge to make the bins and binned actions have the same length
        bins = bins[:-1]

    # bin the actions
    binned = np.digitize(actions, bins) - 1
    # convert the given actions to the discretized actions
    discretized = np.vectorize(lambda x: bins[x])(binned)
    return bins, discretized
