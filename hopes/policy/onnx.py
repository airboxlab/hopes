import os
from pathlib import Path

import numpy as np
import onnxruntime as rt

from hopes.dev_utils import override
from hopes.policy.policies import Policy
from hopes.policy.utils import log_softmax


class OnnxModelBasedPolicy(Policy):
    """A policy that uses an existing ONNX model to predict the log-probabilities of actions given
    observations.

    This class makes some opinionated assumptions about the structure of the ONNX model. You may need to override some
    methods if your model does not fit this structure.

    It supports models with attention mechanisms, where the state of the model is updated at each step.
    The action log probabilities are computed from the output of the model, with 3 options depending on the output layer
    of the underlying model:

    - from the action probabilities output.
    - from the action log probabilities output.
    - from the action distribution inputs output.

    Example of usage, based on a pre-trained model in Ray RLlib, saved using :meth:`ray.rllib.algorithms.algorithm.Algorithm.export_policy_model`.
    This model uses an Attention-based Transformer model and passes 10 previous actions as inputs to the model. The action
    log probabilities are computed from the action distribution inputs output.

    .. code-block:: python

            onnx_file_path = "model.onnx"
            policy = OnnxModelBasedPolicy(
                onnx_model_path=onnx_file_path,
                obs_input=("default_policy/obs:0", np.float32),
                state_dim=(1, 10, 32),  # (num_transformers, memory, attention_dim)
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

            policy.log_probabilities(obs=np.random.rand(1, 15))
    """

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
    def log_probabilities(self, obs: np.ndarray) -> np.ndarray:
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

        # get the action log-probabilities from the output, from action probs, action log probs or action dist inputs
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


class OnnxRunner:
    """A lightweight ONNX inference wrapper used to replay a recurrent policy on logged
    trajectories.

    This helper is designed for *offline replay*: given a sequence of observations from a logged episode,
    it runs an exported ONNX policy step-by-step and returns action log-probabilities (and optionally
    probabilities for all actions). This is typically used to compute importance sampling ratios for
    OPE estimators (e.g., IS / WIS / DR), where the target policy must be evaluated on the same states
    visited by the behavior policy.

    This class is intentionally lower-level than :class:`~hopes.policy.onnx.OnnxModelBasedPolicy`:
    it exposes explicit control of the recurrent state buffer and the previous-actions buffer
    (common for RLlib attention/RNN exports), and provides a single-step method returning
    log-probabilities.

    Assumptions / conventions
    -------------------------
    - The ONNX model exposes inputs for:
        * observations (``obs``)
        * a recurrent/attention state tensor (``state_in``)
        * sequence lengths (``seq_lens``)
        * a buffer of previous actions (``prev_actions``)
    - The model exposes outputs for:
        * action distribution inputs (logits) OR action probabilities/log-probabilities
        * updated recurrent/attention state (``state_out``)
    - The runner maintains:
        * an internal recurrent state tensor (``self.state``)
        * an internal previous-actions buffer (``self.prev_actions``)
      which are reset at the start of each episode.

    Exploration handling
    --------------------
    Some exported policies include an additional ``is_exploring`` input. When present,
    the runner forces exploration OFF by feeding a ``False``/``0`` value, ensuring
    deterministic evaluation during offline replay.

    Typical usage
    -------------
    .. code-block:: python

        runner = OnnxRunner(
            onnx_path="policy.onnx",
            T=10,
            obs_dim=16,
            act_dim=2,
        )

        # obs_batches: list[np.ndarray] with shapes [(T_ep, obs_dim), ...]
        # act_batches: list[np.ndarray] with shapes [(T_ep,), ...]
        probs, logp_taken = probs_from_runner(runner, obs_batches, act_batches)

        # probs shape: (sum_t over all episodes, act_dim)
        # logp_taken shape: (sum_t over all episodes,)

    Notes
    -----
    - The exact input/output node names are model-specific. This implementation uses
      opinionated defaults (RLlib-style export names). If your model differs, consider
      making these names configurable via constructor arguments or a small config object.
    - ``step_log_probs`` returns log-probabilities for *all* actions at a single timestep.
      To match the original execution context for recurrent models, callers should update
      ``prev_actions`` with the *logged* action after each step.
    """

    def __init__(self, onnx_path: str, T: int = 10, obs_dim: int = 16, act_dim: int = 2):
        self.sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.T = T
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.obs_name = "default_policy/obs:0"
        self.state_in_name = "default_policy/state_in_0:0"
        self.seq_lens_name = "default_policy/seq_lens:0"
        self.prev_actions_name = "default_policy/prev_actions:0"

        self.logits_name = "default_policy/model_2/dense_6/BiasAdd:0"
        self.state_out_name = "default_policy/Reshape_5:0"

        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.is_exploring_name = None
        self.is_exploring_dtype = None

        for inp in self.sess.get_inputs():
            if "is_exploring" in inp.name:
                self.is_exploring_name = inp.name
                self.is_exploring_dtype = inp.type
                break

        self.reset()

    def reset(self):
        self.state = np.zeros((1, self.T, 32), dtype=np.float32)
        self.prev_actions = np.zeros((1, self.T), dtype=np.int64)
        self.seq_lens = np.array([self.T], dtype=np.int32)

    # The runner feeds observations and recurrent state to the exported policy, retrieves action logits,
    # and converts them into log-probabilities while updating the internal RNN state and previous-action buffer.
    # The runner maintains the recurrent hidden state and the prev_actions buffer
    # to match the original policy execution context during offline replay
    def step_log_probs(self, obs_t: np.ndarray) -> np.ndarray:
        obs_t = np.asarray(obs_t, dtype=np.float32).reshape(1, self.obs_dim)

        feed = {
            self.obs_name: obs_t,
            self.state_in_name: self.state,
            self.seq_lens_name: self.seq_lens,
            self.prev_actions_name: self.prev_actions,
        }

        # Force exploration to OFF
        if self.is_exploring_name is not None:
            if self.is_exploring_dtype == "tensor(bool)":
                feed[self.is_exploring_name] = np.array(False, dtype=np.bool_)
            else:
                # fallback to int
                feed[self.is_exploring_name] = np.array([0], dtype=np.int64)

        logits, state_out = self.sess.run([self.logits_name, self.state_out_name], feed)

        logits = np.asarray(logits, dtype=np.float32).reshape(1, self.act_dim)
        logp = log_softmax(logits, axis=1)

        state_out = np.asarray(state_out, dtype=np.float32)

        if state_out.ndim == 3 and state_out.shape[2] == 32:
            self.state = state_out
        elif state_out.ndim == 2 and state_out.shape[1] == 32:
            self.state = np.concatenate([self.state[:, 1:, :], state_out.reshape(1, 1, 32)], axis=1)
        else:
            raise RuntimeError(f"Unexpected state_out shape: {state_out.shape}")

        return logp

    def update_prev_actions(self, a_t: int):
        a = np.array([[a_t]], dtype=np.int64)
        self.prev_actions = np.concatenate([self.prev_actions[:, 1:], a], axis=1)


def probs_from_runner(runner: OnnxRunner, obs_batches, act_batches):
    """Evaluating the new policy on the logged trajectories to compute its action probabilities in
    the same states visited by the behavior policy. For each timestep, it extracts the full action
    distribution and the probability assigned by the new policy to the logged action, which are
    later used by IS- and DR-based OPE estimators.

    :param runner: the OnnxRunner instance used to compute log-probabilities.
    :param obs_batches: list of np.ndarray with shapes [(T_ep, obs_dim), ...] containing the observations for each episode.
    :param act_batches: list of np.ndarray with shapes [(T_ep,), ...] containing the logged actions for each episode.
    :return: a tuple (probs_all, logp_taken_all) where:
    """
    probs_all = []
    logp_taken_all = []  # optional

    for obs_ep, act_ep in zip(obs_batches, act_batches):
        assert obs_ep.shape[0] == act_ep.shape[0], "obs/actions length mismatch in episode"

        # The runner is reset at the beginning of each episode to ensure consistent recurrent state
        runner.reset()

        # prev_actions is updated with the logged action to match the original execution context
        for t in range(obs_ep.shape[0]):
            lp = runner.step_log_probs(obs_ep[t])[0]  # log π(.|s_t, prev_actions_buffer)
            probs = np.exp(lp).astype(np.float32)

            # 1) save action probability for logged actions (for IS/DR)
            a_logged = int(act_ep[t])
            probs_all.append(probs)
            logp_taken_all.append(float(lp[a_logged]))

            # update prev_actions with the LOGGED action
            runner.update_prev_actions(a_logged)

    return np.vstack(probs_all), np.array(logp_taken_all, dtype=np.float32)
