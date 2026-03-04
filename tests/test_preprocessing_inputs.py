import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from hopes.data.preprocessing_inputs import prepare_behavior_inputs_from_df


class TestPreprocessingInputs(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"episode_id": ["ep0", "ep0", "ep1", "ep1"], "dummy": [1, 2, 3, 4]})
        self.kw = dict(
            cols=["raw_observation", "filtered_observation", "logits", "action", "reward"],
            steps_per_episode=4,
            idx_zone_temp=0,
            idx_fract_time=1,
            target_start_occ=6.0,
            htg_setpoint=20.0,
            clg_setpoint=26.0,
            temp_margin=0.25,
        )

    @patch("hopes.data.preprocessing_inputs.RTGQModelHGBoost")
    @patch("hopes.data.preprocessing_inputs.flatten_episode_batches")
    @patch("hopes.data.preprocessing_inputs.build_numpy_batches")
    @patch("hopes.data.preprocessing_inputs.build_episode_sequences")
    @patch("hopes.data.preprocessing_inputs.global_minmax_scale_reward")
    @patch("hopes.data.preprocessing_inputs.add_zone_temp_and_reward")
    @patch("hopes.data.preprocessing_inputs.add_sticky_action")
    @patch("hopes.data.preprocessing_inputs.add_logged_policy_probs")
    @patch("hopes.data.preprocessing_inputs.filter_episodes_by_temp_condition")
    @patch("hopes.data.preprocessing_inputs.filter_episodes_by_length")
    @patch("hopes.data.preprocessing_inputs.drop_first_step_in_each_episode")
    @patch("hopes.data.preprocessing_inputs.build_episode_dict")
    def test_prepare_behavior_inputs_happy_path(
        self,
        m_build_episode_dict,
        m_drop_first,
        m_filter_len,
        m_filter_temp,
        m_add_logged,
        m_add_sticky,
        m_add_zone_reward,
        m_scale,
        m_build_seqs,
        m_build_batches,
        m_flatten,
        m_qmodel_cls,
    ):
        # episode_dict chain
        episode_dict = {"ep0": pd.DataFrame(), "ep1": pd.DataFrame()}
        m_build_episode_dict.return_value = episode_dict
        m_drop_first.return_value = episode_dict
        m_filter_len.return_value = (episode_dict, [])
        m_filter_temp.return_value = (episode_dict, [])

        m_add_logged.return_value = episode_dict
        m_add_sticky.return_value = episode_dict
        m_add_zone_reward.return_value = episode_dict
        m_scale.return_value = (episode_dict, 0.0, 1.0)

        # batches
        obs_batches = [np.zeros((4, 3), dtype=np.float32), np.zeros((4, 3), dtype=np.float32)]
        act_batches = [np.zeros((4,), dtype=np.int64), np.ones((4,), dtype=np.int64)]
        sticky_batches = [np.zeros((4,), dtype=np.int64), np.ones((4,), dtype=np.int64)]
        rew_batches = [np.zeros((4,), dtype=np.float32), np.zeros((4,), dtype=np.float32)]
        probs_batches = [
            np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (4, 1)) for _ in range(2)
        ]
        p_taken_batches = [np.full((4,), 0.5, dtype=np.float32) for _ in range(2)]
        argmax_batches = [np.zeros((4,), dtype=np.int64) for _ in range(2)]
        episode_ids = ["ep0", "ep1"]

        m_build_seqs.return_value = {"ep0": {}, "ep1": {}}
        m_build_batches.return_value = (
            obs_batches,
            act_batches,
            sticky_batches,
            rew_batches,
            probs_batches,
            p_taken_batches,
            argmax_batches,
            episode_ids,
        )

        # flatten
        obs_flat = np.zeros((8, 3), dtype=np.float32)
        act_flat = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        sticky_flat = act_flat.copy()
        rew_flat = np.zeros((8,), dtype=np.float32)
        p_b_taken_flat = np.full((8,), 0.5, dtype=np.float32)
        m_flatten.return_value = (obs_flat, act_flat, sticky_flat, rew_flat, p_b_taken_flat)

        # Q-model
        q_inst = MagicMock()
        Q0 = np.zeros((8,), dtype=np.float32)
        Q1 = np.ones((8,), dtype=np.float32)
        q_inst.fit_predict_q0_q1.return_value = (Q0, Q1, None)
        m_qmodel_cls.return_value = q_inst

        out = prepare_behavior_inputs_from_df(self.df, **self.kw)

        self.assertIsInstance(out, dict)
        self.assertEqual(out["n_episodes"], 2)
        self.assertEqual(out["steps_per_episode"], 4)
        self.assertEqual(out["Q0"].shape, (8,))
        self.assertEqual(out["Q1"].shape, (8,))

        m_qmodel_cls.assert_called_once_with(steps_per_episode=4, random_state=0)
        q_inst.fit_predict_q0_q1.assert_called_once()

    @patch("hopes.data.preprocessing_inputs.filter_episodes_by_temp_condition")
    @patch("hopes.data.preprocessing_inputs.filter_episodes_by_length")
    @patch("hopes.data.preprocessing_inputs.drop_first_step_in_each_episode")
    @patch("hopes.data.preprocessing_inputs.build_episode_dict")
    def test_prepare_behavior_inputs_returns_none_if_no_episodes_after_temp_filter(
        self,
        m_build_episode_dict,
        m_drop_first,
        m_filter_len,
        m_filter_temp,
    ):
        episode_dict = {"ep0": pd.DataFrame()}
        m_build_episode_dict.return_value = episode_dict
        m_drop_first.return_value = episode_dict
        m_filter_len.return_value = (episode_dict, [])
        m_filter_temp.return_value = ({}, ["ep0"])  # everything removed

        out = prepare_behavior_inputs_from_df(self.df, **self.kw)
        self.assertIsNone(out)

    @patch(
        "hopes.data.preprocessing_inputs.global_minmax_scale_reward",
        side_effect=ValueError("no rewards"),
    )
    @patch("hopes.data.preprocessing_inputs.add_zone_temp_and_reward")
    @patch("hopes.data.preprocessing_inputs.add_sticky_action")
    @patch("hopes.data.preprocessing_inputs.add_logged_policy_probs")
    @patch("hopes.data.preprocessing_inputs.filter_episodes_by_temp_condition")
    @patch("hopes.data.preprocessing_inputs.filter_episodes_by_length")
    @patch("hopes.data.preprocessing_inputs.drop_first_step_in_each_episode")
    @patch("hopes.data.preprocessing_inputs.build_episode_dict")
    def test_prepare_behavior_inputs_returns_none_if_scaling_raises(
        self,
        m_build_episode_dict,
        m_drop_first,
        m_filter_len,
        m_filter_temp,
        m_add_logged,
        m_add_sticky,
        m_add_zone_reward,
        m_scale,
    ):
        episode_dict = {"ep0": pd.DataFrame(), "ep1": pd.DataFrame()}
        m_build_episode_dict.return_value = episode_dict
        m_drop_first.return_value = episode_dict
        m_filter_len.return_value = (episode_dict, [])
        m_filter_temp.return_value = (episode_dict, [])

        m_add_logged.return_value = episode_dict
        m_add_sticky.return_value = episode_dict
        m_add_zone_reward.return_value = episode_dict

        out = prepare_behavior_inputs_from_df(self.df, **self.kw)
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
