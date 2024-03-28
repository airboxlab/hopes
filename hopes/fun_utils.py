import numpy as np


def piecewise_linear(x, left_cp, right_cp, slope, y0, y1) -> np.ndarray:
    r"""Define a piecewise linear function with 3 segments, such as:

     y0 --- \ left_cp
             \ slope
              \ right_cp
               \ --- y1

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
