import os
from pathlib import Path

import numpy as np
import onnxruntime as rt

from hopes.dev_utils import override
from hopes.policy.policies import Policy


class OnnxModelBasedPolicy(Policy):
    """A policy that uses an existing ONNX model to predict the log-likelihoods of actions given
    observations."""

    def __init__(
        self,
        onnx_model_path: str | Path,
        obs_input: tuple[str, np.dtype],
        state_dim: tuple[int, int, int] | None = None,
        seq_len: int | None = None,
        prev_n_actions: int | None = None,
        prev_n_rewards: int | None = None,
        state_input: tuple[str, np.dtype] | None = None,
        seq_len_input: tuple[str, np.dtype] | None = None,
        prev_actions_input: tuple[str, np.dtype] | None = None,
        prev_rewards_input: tuple[str, np.dtype] | None = None,
        state_output_name: str | None = None,
        action_output_name: str | None = None,
        action_probs_output_name: str | None = None,
        action_log_probs_output_name: str | None = None,
        action_dist_inputs_output_name: str | None = None,
    ) -> None:
        """
        :param onnx_model_path: the path to the ONNX model.
        :param obs_input: the name and data type of the observations input.
        :param state_dim: the dimensions of the state input.
        :param seq_len: the sequence length for the state input.
        :param prev_n_actions: the number of previous actions to consider.
        :param prev_n_rewards: the number of previous rewards to consider.
        :param state_input: the name and data type of the state input.
        :param seq_len_input: the name and data type of the sequence length input.
        :param prev_actions_input: the name and data type of the previous actions input.
        :param prev_rewards_input: the name and data type of the previous rewards input.
        :param state_output_name: the name of the state output.
        :param action_output_name: the name of the action output.
        :param action_probs_output_name: the name of the action probabilities output.
        :param action_log_probs_output_name: the name of the action log probabilities output.
        :param action_dist_inputs_output_name: the name of the action distribution inputs output.
        """
        onnx_model_path = str(onnx_model_path)
        assert onnx_model_path.endswith(".onnx"), "ONNX model must have .onnx extension."
        assert os.path.exists(onnx_model_path), f"ONNX model path {onnx_model_path} does not exist."

        self.onnx_model_path = onnx_model_path
        self.session = rt.InferenceSession(self.onnx_model_path)
        self.obs_input = obs_input
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.prev_n_actions = prev_n_actions
        self.prev_n_rewards = prev_n_rewards
        self.state_input = state_input
        self.seq_len_input = seq_len_input
        self.prev_actions_input = prev_actions_input
        self.prev_rewards_input = prev_rewards_input
        self.state_output_name = state_output_name
        self.action_output_name = action_output_name
        self.action_probs_output_name = action_probs_output_name
        self.action_log_probs_output_name = action_log_probs_output_name
        self.action_dist_inputs_output_name = action_dist_inputs_output_name

        self.state: np.ndarray | None = None
        self.prev_actions: np.ndarray | None = None
        self.prev_rewards: np.ndarray | None = None

        # check if observations input is present in the model
        model_inputs = [node.name for node in self.session.get_inputs()]
        assert (
            self.obs_input is not None and self.obs_input[0] in model_inputs
        ), "Observations input not found in model."

        input_names = [self.obs_input[0]]
        if self.state_input is not None:
            input_names += [self.state_input[0], self.seq_len_input[0]]
        if self.prev_actions_input is not None:
            input_names += [self.prev_actions_input[0]]
        if self.prev_rewards_input is not None:
            input_names += [self.prev_rewards_input[0]]

        # check number of defined inputs equals number of inputs in the model
        assert len(input_names) == len(model_inputs), (
            "Number of inputs in the model does not match the expected inputs.\n"
            f"List of expected inputs: {model_inputs}.\n"
            f"List of provided inputs: {input_names}.\n"
        )

        # check all expected inputs are present in the model
        for expected_input_name in [
            self.state_input,
            self.seq_len_input,
            self.prev_actions_input,
            self.prev_rewards_input,
        ]:
            if expected_input_name is not None:
                assert (
                    expected_input_name[0] in model_inputs
                ), f"{expected_input_name[0]} input not found in model. List of model inputs: {model_inputs}."

        if self.state_input is not None:
            assert (
                self.state_output_name is not None
            ), "State output must be provided for state input."
            assert self.state_dim is not None, "State dimensions must be provided for state input."
            assert (
                self.seq_len_input is not None
            ), "Sequence length input must be provided for state input."
            assert self.seq_len is not None, "Sequence length must be provided for state input."

        # check outputs
        assert (
            self.action_output_name is not None and self.action_output_name in self.output_names
        ), "Action output not found in model."
        assert (
            (
                self.action_probs_output_name is not None
                and self.action_probs_output_name in self.output_names
            )
            or (
                self.action_log_probs_output_name is not None
                and self.action_log_probs_output_name in self.output_names
            )
            or (
                self.action_dist_inputs_output_name is not None
                and self.action_dist_inputs_output_name in self.output_names
            )
        ), "One of action probs, action log probs or action dist inputs must be provided."

        if self.state_dim is not None:
            self.reset_state()

    @property
    def output_names(self) -> list[str]:
        """The names of the outputs of the ONNX model.

        By default, returns all output names. It can be overridden to return a subset of
        output names.
        """
        return [output.name for output in self.session.get_outputs()]

    def compute_reward(self, obs: np.ndarray, action: int) -> float:
        """Compute the reward for the given observations and actions. Only necessary if
        prev_n_rewards is > 0.

        :param obs: the observations for which to compute the reward.
        :param action: the action for which to compute the reward.
        :return: the computed reward.
        """
        raise NotImplementedError

    def map_inputs(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Prepare the inputs for the ONNX model.

        :param obs: the observations for which to prepare the inputs.
        :return: the inputs for the ONNX model.
        """
        inputs = {self.obs_input[0]: obs.astype(self.obs_input[1])}

        if self.state_input is not None:
            inputs[self.state_input[0]] = self.state.astype(self.state_input[1])
            inputs[self.seq_len_input[0]] = np.array([self.seq_len]).astype(self.seq_len_input[1])

        if self.prev_actions_input is not None:
            inputs[self.prev_actions_input[0]] = self.prev_actions.astype(
                self.prev_actions_input[1]
            )

        if self.prev_rewards_input is not None:
            inputs[self.prev_rewards_input[0]] = self.prev_rewards.astype(
                self.prev_rewards_input[1]
            )

        return inputs

    def reset_state(self) -> None:
        """Reset the state of the policy."""
        if self.state_input is not None:
            num_transformers, memory, attention_dim = self.state_dim
            self.state = np.array(
                [np.zeros((memory, attention_dim), np.float32) for _ in range(num_transformers)]
            )

        if self.prev_actions_input is not None:
            self.prev_actions = np.array([np.array([0] * self.prev_n_actions)])

        if self.prev_rewards_input is not None:
            self.prev_rewards = np.array([np.array([0] * self.prev_n_rewards)])

    @override(Policy)
    def log_likelihoods(self, obs: np.ndarray) -> np.ndarray:
        # get the output of the ONNX model
        output = self.session.run(input_feed=self.map_inputs(obs), output_names=self.output_names)

        # update the state if state input is provided
        if self.state_input is not None:
            # get index of state output in output names
            state_out_idx = None
            if self.state_output_name is not None:
                state_out_idx = self.output_names.index(self.state_output_name)

            self.state = [
                np.concatenate([self.state[i], output[state_out_idx]], axis=0)[1:]
                for i in range(len(self.state))
            ]

        action_idx = self.output_names.index(self.action_output_name)
        action = output[action_idx]

        # update previous actions if provided
        if self.prev_actions_input is not None:
            self.prev_actions = [np.concatenate([self.prev_actions[0], action], axis=0)[1:]]

        # update previous rewards if provided
        if self.prev_rewards_input is not None:
            self.prev_rewards = [
                np.concatenate([self.prev_rewards[0], self.compute_reward(obs, action)], axis=0)[1:]
            ]

        # get the action log-likelihoods from the output, from action probs, action log probs or action dist inputs
        if self.action_log_probs_output_name is not None:
            logp_index = self.output_names.index(self.action_log_probs_output_name)
            return output[logp_index]
        elif self.action_probs_output_name is not None:
            actp_index = self.output_names.index(self.action_probs_output_name)
            return np.log(output[actp_index])
        else:
            dist_index = self.output_names.index(self.action_dist_inputs_output_name)
            action_dist_inputs = np.array(output[dist_index]).squeeze()
            log_probs = action_dist_inputs - np.logaddexp.reduce(action_dist_inputs, axis=-1)
            return log_probs.reshape(1, -1)
