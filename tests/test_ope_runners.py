import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hopes.ope.pipeline import run_ope_for_one_prefix


class TestRunOpeForOnePrefix(unittest.TestCase):
    def _make_fake_inputs(
        self, *, num_days: int, steps_per_episode: int, obs_dim: int, act_dim: int
    ):
        """Creates consistent fake inputs for the pipeline.

        NOTE: sticky_action uses numpy.maximum.accumulate (np has no cummax).
        """
        T = steps_per_episode
        N = num_days * T

        # Episode-wise obs/actions
        obs_batches = []
        act_batches = []
        sticky_act_batches = []
        rew_batches = []
        probs_batches = []

        rng = np.random.default_rng(0)

        for _ in range(num_days):
            obs_ep = rng.normal(size=(T, obs_dim)).astype(np.float32)
            act_ep = rng.integers(0, act_dim, size=(T,), dtype=np.int64)

            # sticky: once 1, always 1
            sticky_ep = np.maximum.accumulate((act_ep == 1).astype(np.int64)).astype(np.int64)

            rew_ep = rng.normal(size=(T,)).astype(np.float32)

            # behavior probs: simple valid distribution per step
            probs_ep = np.full((T, act_dim), 1.0 / act_dim, dtype=np.float32)

            obs_batches.append(obs_ep)
            act_batches.append(act_ep)
            sticky_act_batches.append(sticky_ep)
            rew_batches.append(rew_ep)
            probs_batches.append(probs_ep)

        obs_flat = np.vstack(obs_batches).astype(np.float32)
        act_flat = np.hstack(act_batches).astype(np.int64)
        sticky_act_flat = np.hstack(sticky_act_batches).astype(np.int64)
        rew_flat = np.hstack(rew_batches).astype(np.float32)

        probs_flat = np.vstack(probs_batches).astype(np.float32)
        p_b_taken_flat = probs_flat[np.arange(N), act_flat].astype(np.float32)

        # Q0/Q1 fake (must be (N,))
        Q0 = rng.normal(size=(N,)).astype(np.float32)
        Q1 = rng.normal(size=(N,)).astype(np.float32)

        return {
            "num_days": num_days,
            "steps_per_episode": steps_per_episode,
            "obs_batches": obs_batches,
            "obs_flat": obs_flat,
            "act_batches": act_batches,
            "act_flat": act_flat,
            "sticky_act_flat": sticky_act_flat,
            "rew_flat": rew_flat,
            "p_b_taken_flat": p_b_taken_flat,
            "probs_batches": probs_batches,
            "Q0": Q0,
            "Q1": Q1,
        }

    # -------------------------
    # 1) HAPPY PATH (no S3 / no onnxruntime)
    # -------------------------
    @patch("hopes.ope.pipeline.plot_all_diagnostics_for_agent")
    @patch("hopes.ope.pipeline.StickySequentialDoublyRobust")
    @patch("hopes.ope.pipeline.StickyTrajectoryWiseIS")
    @patch("hopes.ope.pipeline.compute_stepwise_ips_wis")
    @patch("hopes.ope.pipeline.wpdis_daily")
    @patch("hopes.ope.pipeline.probs_from_runner")
    @patch("hopes.ope.pipeline.OnnxRunner")  # <-- IMPORTANT: avoid onnxruntime
    @patch("hopes.ope.pipeline.prepare_onnx_model")
    @patch("hopes.ope.pipeline.write_onnx_bytes")
    @patch("hopes.ope.pipeline.load_latest_model_onnx")
    def test_run_ope_for_one_prefix_happy_path(
        self,
        m_load_latest,
        m_write_onnx,
        m_prepare,
        m_runner_cls,
        m_probs_from_runner,
        m_wpdis_daily,
        m_stepwise,
        m_traj_cls,
        m_dr_cls,
        m_plot,
    ):
        data = self._make_fake_inputs(num_days=3, steps_per_episode=4, obs_dim=6, act_dim=2)
        N = data["act_flat"].shape[0]
        num_days = data["num_days"]
        T = data["steps_per_episode"]

        # --- Mock ONNX loading/writing (avoid S3 + filesystem dependency)
        m_load_latest.return_value = (b"fake_onnx_bytes", "model.onnx", {"dummy": True})
        m_write_onnx.return_value = "/tmp/fake_raw.onnx"
        m_prepare.return_value = None

        # --- Mock runner creation (avoid onnxruntime InferenceSession)
        m_runner_cls.return_value = MagicMock()

        # --- Mock probs_from_runner output
        probs_flat = np.vstack(data["probs_batches"]).astype(np.float32)
        P_new = probs_flat.copy()
        P_new = np.clip(P_new + 1e-4, 1e-6, 1.0)
        P_new = P_new / P_new.sum(axis=1, keepdims=True)

        act_flat = data["act_flat"]
        logp_taken = np.log(P_new[np.arange(N), act_flat]).astype(np.float32)
        m_probs_from_runner.return_value = (P_new, logp_taken)

        # --- Mock wpdis_daily outputs (called twice)
        m_wpdis_daily.side_effect = [
            (1.0, 0.9, 1.1),  # eval
            (0.95, 0.85, 1.05),  # behavior
        ]

        # --- Mock compute_stepwise_ips_wis
        fake_weights = (
            P_new[np.arange(N), act_flat] / np.maximum(data["p_b_taken_flat"], 1e-12)
        ).astype(np.float32)
        m_stepwise.return_value = {"weights": fake_weights, "ips": 0.1, "wis": 0.2}

        # --- Mock StickyTrajectoryWiseIS instance
        traj_inst = MagicMock()
        traj_inst.estimate_weighted_rewards.return_value = (
            data["rew_flat"].reshape(num_days, T).sum(axis=1)
        )
        traj_inst.estimate_policy_value.return_value = 123.0
        traj_inst.estimate_self_normalized_value.return_value = 45.0
        m_traj_cls.return_value = traj_inst

        # --- Mock StickySequentialDoublyRobust instance
        dr_inst = MagicMock()
        rho_td = np.ones((num_days, T), dtype=np.float32)
        W_t_td = np.ones((num_days, T), dtype=np.float32)
        dr_episode = np.full((num_days,), 2.5, dtype=np.float32)
        dr_inst.estimate_components.return_value = (rho_td, W_t_td, dr_episode)
        m_dr_cls.return_value = dr_inst

        out = run_ope_for_one_prefix(
            prefix_new="s3/prefix",
            checkpoint_model_regex=None,
            tag="agentA",
            bucket="bucket",
            obs_batches=data["obs_batches"],
            obs_flat=data["obs_flat"],
            act_batches=data["act_batches"],
            act_flat=data["act_flat"],
            sticky_act_flat=data["sticky_act_flat"],
            rew_flat=data["rew_flat"],
            p_b_taken_flat=data["p_b_taken_flat"],
            probs_batches=data["probs_batches"],
            Q0=data["Q0"],
            Q1=data["Q1"],
            steps_per_episode=T,
            plot_diagnostics=False,
            verbose=False,
            diagnostics=False,
        )

        expected_keys = {
            "agent_centroid",
            "wpdis_cap_mean_e",
            "wpdis_cap_ci_lo_e",
            "wpdis_cap_ci_hi_e",
            "wpdis_cap_mean_b",
            "wpdis_cap_ci_lo_b",
            "wpdis_cap_ci_hi_b",
            "promote_lb_gt_behavior_mean",
            "delta_lb_vs_bmean",
            "traj_is",
            "traj_snis",
            "ips_step",
            "wis_step",
            "dm_day_mean",
            "dm_day_std",
            "dr_day_mean",
            "dr_day_std",
            "dr_td",
            "behavior_daily_mean",
            "behavior_daily_std",
            "dm_behavior_day_mean",
            "dm_behavior_day_std",
        }
        self.assertTrue(expected_keys.issubset(set(out.keys())))
        self.assertEqual(out["agent_centroid"], "agentA")
        self.assertIsInstance(out["promote_lb_gt_behavior_mean"], bool)
        self.assertAlmostEqual(out["dr_td"], float(np.mean(dr_episode)), places=7)

        # plot should not be called
        m_plot.assert_not_called()

    # -------------------------
    # 2) RAISES if N not multiple of steps_per_episode (no S3 / no onnxruntime)
    # -------------------------
    @patch("hopes.ope.pipeline.probs_from_runner")
    @patch("hopes.ope.pipeline.OnnxRunner")
    @patch("hopes.ope.pipeline.prepare_onnx_model")
    @patch("hopes.ope.pipeline.write_onnx_bytes")
    @patch("hopes.ope.pipeline.load_latest_model_onnx")
    def test_run_ope_for_one_prefix_raises_if_N_not_multiple_of_steps(
        self,
        m_load_latest,
        m_write_onnx,
        m_prepare,
        m_runner_cls,
        m_probs_from_runner,
    ):
        data = self._make_fake_inputs(num_days=3, steps_per_episode=4, obs_dim=6, act_dim=2)

        # Break N so it is NOT divisible by T
        # Example: drop last element from act_flat (and keep consistency on required arrays)
        act_flat_bad = data["act_flat"][:-1].copy()
        sticky_act_flat_bad = data["sticky_act_flat"][:-1].copy()
        rew_flat_bad = data["rew_flat"][:-1].copy()
        p_b_taken_flat_bad = data["p_b_taken_flat"][:-1].copy()
        obs_flat_bad = data["obs_flat"][:-1, :].copy()

        N_bad = act_flat_bad.shape[0]

        # --- patch S3 and filesystem/onnxruntime calls
        m_load_latest.return_value = (b"fake_onnx_bytes", "model.onnx", {"dummy": True})
        m_write_onnx.return_value = "/tmp/fake_raw.onnx"
        m_prepare.return_value = None
        m_runner_cls.return_value = MagicMock()

        # Need P_new with shape (N_bad, 2) so the code can reach the N%T check
        P_new = np.full((N_bad, 2), 0.5, dtype=np.float32)
        logp_taken = np.log(np.full((N_bad,), 0.5, dtype=np.float32))
        m_probs_from_runner.return_value = (P_new, logp_taken)

        with self.assertRaises(ValueError):
            run_ope_for_one_prefix(
                prefix_new="s3/prefix",
                checkpoint_model_regex=None,
                tag="agentA",
                bucket="bucket",
                obs_batches=data["obs_batches"],
                obs_flat=obs_flat_bad,
                act_batches=data["act_batches"],
                act_flat=act_flat_bad,
                sticky_act_flat=sticky_act_flat_bad,
                rew_flat=rew_flat_bad,
                p_b_taken_flat=p_b_taken_flat_bad,
                probs_batches=data["probs_batches"],
                Q0=data["Q0"][:N_bad],
                Q1=data["Q1"][:N_bad],
                steps_per_episode=4,
                plot_diagnostics=False,
                verbose=False,
                diagnostics=False,
            )
