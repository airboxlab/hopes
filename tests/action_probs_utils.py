import numpy as np


def generate_action_probs(traj_length: int, num_actions: int) -> np.ndarray:
    """Generate random action probabilities with shape (traj_length, num_actions).

    Action probabilities are normalized to sum to 1.
    """
    action_probs = np.random.rand(traj_length, num_actions)
    action_probs /= action_probs.sum(axis=1, keepdims=True)
    return action_probs
