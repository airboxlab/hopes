import numpy as np


def check_array(
    array: np.ndarray,
    name: str,
    expected_ndim: int = 1,
    expected_dtype: type | tuple[type, ...] | None = None,
) -> None:
    """Input validation on array.

    :param array: The array to validate.
    :param name: The name of the array.
    :param expected_ndim: The expected number of dimensions of the array.
    :param expected_dtype: The expected dtype of the array.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be a np.ndarray, but got {type(array)}")

    if array.ndim != expected_ndim:
        raise ValueError(f"{name} must be {expected_ndim}D array, but got {array.ndim}D array")

    if expected_dtype is not None:
        if isinstance(array, expected_dtype):
            raise ValueError(
                f"The elements of {name} must be {expected_dtype}, but got {array.dtype}"
            )
