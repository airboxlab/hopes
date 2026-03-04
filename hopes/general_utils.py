import ast
from typing import Any

import numpy as np

# Since in the aws s3 the traj are all stores as string, this function allows to modify the format in order
# to obtain the one required in further analyses


def parse_list(x: Any) -> list:
    """Parse raw_observation if stored as string."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x)
    raise TypeError(f"Unsupported type: {type(x)}")


def to_list(v):
    """Logits/actions may be stored as strings in the CSV, so they are parsed back to lists."""
    if isinstance(v, str):
        return ast.literal_eval(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return list(v)


def softmax_1d(x):
    """Stable softmax for a single action-logit vector."""
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)  # numerical stability to avoid overflow
    e = np.exp(x)
    return e / np.sum(e)


def log_softmax(x, axis=-1):
    """log_softmax is used instead of softmax to improve numerical stability."""
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def get_action_int(a: Any) -> int:
    """Ensure action is an int (handles scalar, list-like, or string-stored list)."""
    a_parsed = to_list(a) if isinstance(a, str) else a
    if isinstance(a_parsed, (list, tuple, np.ndarray)):
        return int(float(a_parsed[0]))
    return int(float(a_parsed))


# Functions for converting per-episode Python lists into typed NumPy arrays with consistent shapes and dtypes
# Explicit dtype casting ensures numerical stability and consistent behavior across estimators
def to_2d_float32(x):
    return np.asarray(x, dtype=np.float32)


def to_1d_int64(x):
    """Reshaping to 1D vectors avoids shape mismatches when flattening episodes."""
    return np.asarray(x, dtype=np.int64).reshape(-1)


def to_1d_float32(x):
    return np.asarray(x, dtype=np.float32).reshape(-1)
