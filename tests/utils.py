import numpy as np


def assert_log_probs(log_probs: np.ndarray, expected_shape: tuple):
    assert np.all(np.isfinite(log_probs))
    assert np.all(log_probs <= 0.0)
    assert_act_probs(np.exp(log_probs), expected_shape)


def assert_act_probs(act_probs: np.ndarray, expected_shape: tuple):
    assert isinstance(act_probs, np.ndarray)
    assert act_probs.shape == expected_shape
    assert np.all(act_probs >= 0.0)
    assert np.all(act_probs <= 1.0)
    assert np.allclose(act_probs.sum(axis=1), 1.0, atol=1e-3)
