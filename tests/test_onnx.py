import unittest
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import numpy as np

from hopes.policy.onnx import OnnxModelBasedPolicy, OnnxRunner
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

    def test_onnx_runner_step_log_probs_shape_and_sum(self):
        onnx_file_path = Path(__file__).parent / "resources" / "onnx" / "model.onnx"

        # NOTE: this model expects obs dim 15 (as per the policy test above)
        runner = OnnxRunner(
            onnx_path=str(onnx_file_path),
            T=10,
            obs_dim=15,
            act_dim=2,
        )

        obs_t = np.random.rand(15)
        logp = runner.step_log_probs(obs_t)

        # shape check + validity (log-probabilities should sum to 1 after exp)
        assert_log_probs(logp, expected_shape=(1, 2))

    def test_onnx_runner_reset_and_prev_actions_update(self):
        onnx_file_path = Path(__file__).parent / "resources" / "onnx" / "model.onnx"
        T = 10

        runner = OnnxRunner(
            onnx_path=str(onnx_file_path),
            T=T,
            obs_dim=15,
            act_dim=2,
        )

        # After reset, prev_actions should be all zeros
        runner.reset()
        self.assertEqual(runner.prev_actions.shape, (1, T))
        self.assertTrue(np.all(runner.prev_actions == 0))

        # Update with a known action, verify shift-left + new action at end
        runner.update_prev_actions(1)
        self.assertTrue(np.all(runner.prev_actions[0, :-1] == 0))
        self.assertEqual(int(runner.prev_actions[0, -1]), 1)

        # Another update: last two positions become [1, 0] -> actually shift again then append
        runner.update_prev_actions(0)
        self.assertEqual(int(runner.prev_actions[0, -2]), 1)
        self.assertEqual(int(runner.prev_actions[0, -1]), 0)

    def test_onnx_runner_state_updates(self):
        onnx_file_path = Path(__file__).parent / "resources" / "onnx" / "model.onnx"
        T = 10

        runner = OnnxRunner(
            onnx_path=str(onnx_file_path),
            T=T,
            obs_dim=15,
            act_dim=2,
        )

        runner.reset()
        state_before = runner.state.copy()

        obs_t = np.random.rand(15)
        _ = runner.step_log_probs(obs_t)

        # State should have been updated by the ONNX output
        self.assertEqual(runner.state.shape, state_before.shape)
        self.assertFalse(np.allclose(runner.state, state_before))

    def test_onnx_runner_io_names(self):
        onnx_file_path = Path(__file__).parent / "resources" / "onnx" / "model.onnx"
        runner = OnnxRunner(
            onnx_path=str(onnx_file_path),
            T=10,
            obs_dim=15,
            act_dim=2,
        )

        input_names = {i.name for i in runner.sess.get_inputs()}
        output_names = {o.name for o in runner.sess.get_outputs()}

        required_inputs = {
            runner.obs_name,
            runner.state_in_name,
            runner.seq_lens_name,
            runner.prev_actions_name,
        }
        required_outputs = {runner.logits_name, runner.state_out_name}

        missing_in = required_inputs - input_names
        missing_out = required_outputs - output_names

        self.assertFalse(missing_in, f"Missing ONNX inputs: {missing_in}")
        self.assertFalse(missing_out, f"Missing ONNX outputs: {missing_out}")

    def test_onnx_runner_determinism_under_reset(self):
        onnx_file_path = Path(__file__).parent / "resources" / "onnx" / "model.onnx"
        runner = OnnxRunner(
            onnx_path=str(onnx_file_path),
            T=10,
            obs_dim=15,
            act_dim=2,
        )

        obs_t = np.random.rand(15).astype(np.float32)

        runner.reset()
        logp1 = runner.step_log_probs(obs_t).copy()

        runner.reset()
        logp2 = runner.step_log_probs(obs_t).copy()

        self.assertTrue(
            np.allclose(logp1, logp2, atol=1e-6),
            "Non-deterministic outputs under reset",
        )

    def test_onnx_runner_episode_replay(self):
        onnx_file_path = Path(__file__).parent / "resources" / "onnx" / "model.onnx"
        runner = OnnxRunner(
            onnx_path=str(onnx_file_path),
            T=10,
            obs_dim=15,
            act_dim=2,
        )

        ep_len = 7
        obs_ep = np.random.rand(ep_len, 15).astype(np.float32)
        act_ep = np.random.randint(0, 2, size=(ep_len,), dtype=np.int64)

        runner.reset()
        probs = []
        for t in range(ep_len):
            logp = runner.step_log_probs(obs_ep[t])
            p = np.exp(logp[0]).astype(np.float32)
            probs.append(p)
            runner.update_prev_actions(int(act_ep[t]))

        probs = np.vstack(probs)
        self.assertEqual(probs.shape, (ep_len, 2))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))

    def test_probs_from_runner_shapes_and_consistency(self):
        from hopes.policy.onnx import probs_from_runner  # import here to keep top clean

        onnx_file_path = Path(__file__).parent / "resources" / "onnx" / "model.onnx"
        runner = OnnxRunner(
            onnx_path=str(onnx_file_path),
            T=10,
            obs_dim=15,
            act_dim=2,
        )

        # Build 2 toy episodes with different lengths
        ep1_T = 5
        ep2_T = 3
        obs_ep1 = np.random.rand(ep1_T, 15).astype(np.float32)
        obs_ep2 = np.random.rand(ep2_T, 15).astype(np.float32)

        # Logged actions must be valid ints in [0, act_dim-1]
        act_ep1 = np.random.randint(0, 2, size=(ep1_T,), dtype=np.int64)
        act_ep2 = np.random.randint(0, 2, size=(ep2_T,), dtype=np.int64)

        probs, logp_taken = probs_from_runner(
            runner=runner,
            obs_batches=[obs_ep1, obs_ep2],
            act_batches=[act_ep1, act_ep2],
        )

        total_T = ep1_T + ep2_T

        # Shape checks
        self.assertEqual(probs.shape, (total_T, 2))
        self.assertEqual(logp_taken.shape, (total_T,))

        # Probabilities should sum to 1 per row (within tolerance)
        row_sums = probs.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, np.ones(total_T, dtype=np.float32), atol=1e-5))

        # logp_taken should correspond to the probability of the taken action
        # i.e., exp(logp_taken[t]) == probs[t, a_logged[t]] where a_logged is concatenated actions
        a_all = np.concatenate([act_ep1, act_ep2]).astype(np.int64)
        p_taken = probs[np.arange(total_T), a_all]
        self.assertTrue(np.allclose(np.exp(logp_taken), p_taken, atol=1e-5))

    def _make_mock_policy_for_log_probs_branch(self, branch: str):
        """Create an OnnxModelBasedPolicy instance bypassing __init__ to test log_probabilities
        branches without requiring multiple real ONNX models."""
        from hopes.policy.onnx import (
            OnnxModelBasedPolicy,  # local import to avoid circulars
        )

        policy = OnnxModelBasedPolicy.__new__(OnnxModelBasedPolicy)

        # Minimal attributes used by map_inputs() and log_probabilities()
        policy.session = Mock()
        policy.obs_input = ("obs", np.float32)

        # Disable state/prev actions/rewards logic for this unit test
        policy.state_input = None
        policy.seq_len_input = None
        policy.prev_actions_input = None
        policy.prev_rewards_input = None
        policy.state_output_name = None
        policy.state_dim = None
        policy.seq_len = None
        policy.prev_n_actions = None
        policy.prev_n_rewards = None
        policy.state = None
        policy.prev_actions = None
        policy.prev_rewards = None

        # Outputs layout
        policy.action_output_name = "action_out"

        if branch == "log_probs":
            policy.action_log_probs_output_name = "logp_out"
            policy.action_probs_output_name = None
            policy.action_dist_inputs_output_name = None
            output_names = ["action_out", "logp_out"]

            action_out = np.array([[1]], dtype=np.int64)
            logp_out = np.array([[-0.7, -0.3]], dtype=np.float32)

            policy.session.run.return_value = [action_out, logp_out]

        elif branch == "probs":
            policy.action_log_probs_output_name = None
            policy.action_probs_output_name = "probs_out"
            policy.action_dist_inputs_output_name = None
            output_names = ["action_out", "probs_out"]

            action_out = np.array([[0]], dtype=np.int64)
            probs_out = np.array([[0.25, 0.75]], dtype=np.float32)

            policy.session.run.return_value = [action_out, probs_out]

        elif branch == "dist_inputs":
            policy.action_log_probs_output_name = None
            policy.action_probs_output_name = None
            policy.action_dist_inputs_output_name = "dist_out"
            output_names = ["action_out", "dist_out"]

            action_out = np.array([[0]], dtype=np.int64)
            dist_inputs = np.array([[1.0, 2.0]], dtype=np.float32)  # logits

            policy.session.run.return_value = [action_out, dist_inputs]

        else:
            raise ValueError(f"Unknown branch: {branch}")

        return policy, output_names

    def test_onnx_policy_log_probabilities_branch_log_probs_output(self):
        policy, output_names = self._make_mock_policy_for_log_probs_branch("log_probs")

        with patch.object(
            type(policy), "output_names", new_callable=PropertyMock
        ) as mock_out_names:
            mock_out_names.return_value = output_names

            obs = np.random.rand(1, 15).astype(np.float32)
            logp = policy.log_probabilities(obs=obs)

            self.assertEqual(logp.shape, (1, 2))
            self.assertEqual(logp.dtype, np.float32)
            # Should match what session returned
            self.assertTrue(np.allclose(logp, np.array([[-0.7, -0.3]], dtype=np.float32)))

    def test_onnx_policy_log_probabilities_branch_probs_output(self):
        policy, output_names = self._make_mock_policy_for_log_probs_branch("probs")

        with patch.object(
            type(policy), "output_names", new_callable=PropertyMock
        ) as mock_out_names:
            mock_out_names.return_value = output_names

            obs = np.random.rand(1, 15).astype(np.float32)
            logp = policy.log_probabilities(obs=obs)

            self.assertEqual(logp.shape, (1, 2))
            self.assertEqual(logp.dtype, np.float32)

            expected = np.log(np.array([[0.25, 0.75]], dtype=np.float32))
            self.assertTrue(np.allclose(logp, expected, atol=1e-6))

    def test_onnx_policy_log_probabilities_branch_dist_inputs_output(self):
        policy, output_names = self._make_mock_policy_for_log_probs_branch("dist_inputs")

        with patch.object(
            type(policy), "output_names", new_callable=PropertyMock
        ) as mock_out_names:
            mock_out_names.return_value = output_names

            obs = np.random.rand(1, 15).astype(np.float32)
            logp = policy.log_probabilities(obs=obs)

            self.assertEqual(logp.shape, (1, 2))
            self.assertEqual(logp.dtype, np.float32)

            # Manual log-softmax for logits [1, 2]
            logits = np.array([1.0, 2.0], dtype=np.float32)
            expected = logits - np.logaddexp.reduce(logits)
            self.assertTrue(np.allclose(logp, expected.reshape(1, -1), atol=1e-6))

    def test_onnx_policy_init_rejects_non_onnx_extension(self):
        from hopes.policy.onnx import OnnxModelBasedPolicy

        with self.assertRaises(AssertionError):
            OnnxModelBasedPolicy(
                onnx_model_path="not_a_model.txt",
                obs_input=("default_policy/obs:0", np.float32),
                action_output_name="x",
                action_dist_inputs_output_name="y",
            )

    def test_onnx_policy_init_rejects_missing_file(self):
        from hopes.policy.onnx import OnnxModelBasedPolicy

        with self.assertRaises(AssertionError):
            OnnxModelBasedPolicy(
                onnx_model_path="missing_model.onnx",
                obs_input=("default_policy/obs:0", np.float32),
                action_output_name="x",
                action_dist_inputs_output_name="y",
            )
