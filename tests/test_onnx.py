import unittest
from pathlib import Path

import numpy as np

from hopes.policy.onnx import OnnxModelBasedPolicy
from tests.utils import assert_log_probs


class TestOnnxPolicy(unittest.TestCase):
    def test_onnx_policy(self):
        onnx_file_path = Path(__file__).parent / "resources" / "onnx" / "model.onnx"
        policy = OnnxModelBasedPolicy(
            onnx_model_path=onnx_file_path,
            obs_input=("default_policy/obs:0", np.float32),
            state_dim=(1, 10, 32),
            seq_len=10,
            prev_n_actions=10,
            prev_n_rewards=0,
            state_input=("default_policy/state_in_0:0", np.float32),
            seq_len_input=("default_policy/seq_lens:0", np.int32),
            prev_actions_input=("default_policy/prev_actions:0", np.int64),
            state_output_name="default_policy/Reshape_5:0",
            action_output_name="default_policy/cond_1/Merge:0",
            action_probs_output_name=None,
            action_log_probs_output_name=None,
            action_dist_inputs_output_name="default_policy/model_2/dense_6/BiasAdd:0",
        )

        obs = np.random.rand(1, 15)
        log_probs = policy.log_probabilities(obs=obs)
        assert_log_probs(log_probs, expected_shape=(1, 2))
