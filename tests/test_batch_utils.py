import unittest

import numpy as np
import pandas as pd

from hopes.data.batch_utils import (
    add_logged_policy_probs,
    add_sticky_action,
    add_zone_temp_and_reward,
    apply_stickiness_correction_to_rho,
    build_episode_dict,
    build_episode_sequences,
    build_numpy_batches,
    drop_first_step_in_each_episode,
    filter_episodes_by_length,
    filter_episodes_by_temp_condition,
    flatten_episode_batches,
    global_minmax_scale_reward,
    validate_batches,
)


class TestBatchUtils(unittest.TestCase):
    def setUp(self):
        # Two episodes, 4 steps each
        rows = []
        for eid in ["ep1", "ep2"]:
            for t in range(4):
                # raw_observation: zone temp at index 0
                zone_temp = 19.0 + t if eid == "ep1" else 21.0 + t  # ep2 always >= 21
                raw_obs = f"[{zone_temp}, 0.0, 1.0]"

                # filtered_observation: e.g. 2-dim observation vector
                filt_obs = f"[{float(t)}, {float(t+1)}]"

                # logits as string
                # make them vary with time to avoid degenerate probs
                logits = f"[{0.1 * t}, {1.0 + 0.1 * t}]"

                # action sometimes as string (to exercise get_action_int / parsing)
                action = "1" if (t % 2 == 0) else 0

                # reward
                reward = float(t) if eid == "ep1" else float(t + 10)

                rows.append(
                    {
                        "episode_id": eid,
                        "t": t,
                        "raw_observation": raw_obs,
                        "filtered_observation": filt_obs,
                        "logits": logits,
                        "action": action,
                        "reward": reward,
                    }
                )

        self.df = pd.DataFrame(rows)

    def test_build_episode_dict(self):
        ep = build_episode_dict(
            self.df, cols=["t", "action", "reward", "raw_observation"], episode_id_col="episode_id"
        )
        self.assertEqual(set(ep.keys()), {"ep1", "ep2"})
        self.assertEqual(len(ep["ep1"]), 4)
        self.assertEqual(list(ep["ep1"].columns), ["t", "action", "reward", "raw_observation"])

        # index should be reset per episode
        self.assertEqual(list(ep["ep1"].index), [0, 1, 2, 3])

    def test_drop_first_step_in_each_episode(self):
        ep = build_episode_dict(self.df, cols=["t", "action"], episode_id_col="episode_id")
        ep2 = drop_first_step_in_each_episode(ep, n_drop=1)

        self.assertEqual(len(ep2["ep1"]), 3)
        self.assertEqual(int(ep2["ep1"].iloc[0]["t"]), 1)  # first kept row was t=1
        self.assertEqual(list(ep2["ep1"].index), [0, 1, 2])

    def test_filter_episodes_by_length(self):
        ep = build_episode_dict(self.df, cols=["t", "action"], episode_id_col="episode_id")

        # Make ep2 wrong length by dropping one row
        ep_bad = dict(ep)
        ep_bad["ep2"] = ep_bad["ep2"].iloc[:-1].reset_index(drop=True)

        filtered, removed = filter_episodes_by_length(ep_bad, steps_per_episode=4, verbose=False)
        self.assertEqual(set(filtered.keys()), {"ep1"})
        self.assertEqual(removed, ["ep2"])

    def test_filter_episodes_by_temp_condition(self):
        ep = build_episode_dict(self.df, cols=["raw_observation", "t"], episode_id_col="episode_id")

        # Condition: remove episodes where zone_temp always >= 21.0
        # ep2 starts at 21.0 and increases, so it should be removed
        filtered, removed = filter_episodes_by_temp_condition(
            ep,
            raw_obs_col="raw_observation",
            idx_zone_temp=0,
            htg_setpoint=21.0,
            verbose=False,
        )
        self.assertEqual(set(filtered.keys()), {"ep1"})
        self.assertEqual(removed, ["ep2"])

    def test_add_logged_policy_probs(self):
        ep = build_episode_dict(self.df, cols=["logits", "action"], episode_id_col="episode_id")
        out = add_logged_policy_probs(
            ep,
            logits_col="logits",
            action_col="action",
            probs_col="probs",
            p_taken_col="p_taken",
            argmax_action_col="argmax_action",
        )

        df_ep1 = out["ep1"]
        self.assertIn("probs", df_ep1.columns)
        self.assertIn("p_taken", df_ep1.columns)
        self.assertIn("argmax_action", df_ep1.columns)

        # probs should be length-2 and sum to 1
        p0 = df_ep1.loc[0, "probs"]
        self.assertEqual(len(p0), 2)
        self.assertTrue(np.isfinite(p0).all())
        self.assertAlmostEqual(float(np.sum(p0)), 1.0, places=6)

        # p_taken should match probs[action]
        for i in range(len(df_ep1)):
            a = int(df_ep1.loc[i, "action"])
            p_taken = float(df_ep1.loc[i, "p_taken"])
            probs = df_ep1.loc[i, "probs"]
            self.assertAlmostEqual(p_taken, float(probs[a]), places=6)

        # argmax_action should be 1 for logits [small, bigger]
        self.assertTrue((df_ep1["argmax_action"] == 1).all())

    def test_add_sticky_action(self):
        ep = build_episode_dict(self.df, cols=["action"], episode_id_col="episode_id")

        # Ensure actions are ints here (add_sticky_action expects comparable values)
        for k in ep:
            ep[k]["action"] = ep[k]["action"].apply(lambda x: int(x))

        out = add_sticky_action(ep, action_col="action", sticky_col="sticky_action", on_value=1)
        st = out["ep1"]["sticky_action"].to_numpy()

        # Once it hits 1, it stays 1
        self.assertTrue(np.all(np.diff(st) >= 0))

        # If first action is 1 (t=0), then all should be 1
        # In our data: action is 1 on even t => t=0 action=1
        self.assertTrue(np.all(st == 1))

    def test_global_minmax_scale_reward(self):
        ep = build_episode_dict(self.df, cols=["reward"], episode_id_col="episode_id")
        scaled, r_min, r_max = global_minmax_scale_reward(ep, reward_col="reward", verbose=False)

        self.assertLessEqual(r_min, r_max)
        for eid, df_ep in scaled.items():
            r = df_ep["reward"].to_numpy()
            self.assertTrue(np.all(r >= -1e-6))
            self.assertTrue(np.all(r <= 1.0 + 1e-6))
            self.assertEqual(r.dtype, np.float32)

    def test_build_episode_sequences_and_numpy_batches_and_flatten(self):
        # Start from a dict that includes all needed columns
        cols = [
            "filtered_observation",
            "raw_observation",
            "logits",
            "action",
            "reward",
        ]
        ep = build_episode_dict(self.df, cols=cols, episode_id_col="episode_id")

        # add probs/p_taken/argmax
        ep = add_logged_policy_probs(ep)

        # ensure action is int then add sticky action
        for k in ep:
            ep[k]["action"] = ep[k]["action"].apply(lambda x: int(x))
        ep = add_sticky_action(ep)

        # scale reward to avoid negatives in other parts of pipeline (optional)
        ep, _, _ = global_minmax_scale_reward(ep, reward_col="reward", verbose=False)

        seqs = build_episode_sequences(
            ep,
            filtered_obs_col="filtered_observation",
            action_col="action",
            sticky_action_col="sticky_action",
            reward_col="reward",
            probs_col="probs",
            p_taken_col="p_taken",
            argmax_action_col="argmax_action",
        )

        self.assertEqual(set(seqs.keys()), {"ep1", "ep2"})
        self.assertIn("obs", seqs["ep1"])
        self.assertEqual(len(seqs["ep1"]["obs"]), 4)
        self.assertEqual(len(seqs["ep1"]["probs"]), 4)

        (
            obs_batches,
            act_batches,
            sticky_act_batches,
            rew_batches,
            probs_batches,
            p_taken_batches,
            argmax_batches,
            episode_ids,
        ) = build_numpy_batches(seqs)

        self.assertEqual(episode_ids, ["ep1", "ep2"])
        self.assertEqual(len(obs_batches), 2)

        # dtype/shape checks per episode
        self.assertEqual(obs_batches[0].dtype, np.float32)
        self.assertEqual(act_batches[0].dtype, np.int64)
        self.assertEqual(sticky_act_batches[0].dtype, np.int64)
        self.assertEqual(rew_batches[0].dtype, np.float32)
        self.assertEqual(probs_batches[0].dtype, np.float32)
        self.assertEqual(p_taken_batches[0].dtype, np.float32)
        self.assertEqual(argmax_batches[0].dtype, np.int64)

        self.assertEqual(obs_batches[0].shape, (4, 2))  # filtered_observation has 2 dims
        self.assertEqual(probs_batches[0].shape, (4, 2))  # 2 actions

        # Flatten and check lengths
        obs_flat, act_flat, sticky_flat, rew_flat, p_b_taken_flat = flatten_episode_batches(
            obs_batches=obs_batches,
            act_batches=act_batches,
            sticky_act_batches=sticky_act_batches,
            rew_batches=rew_batches,
            p_taken_batches=p_taken_batches,
            verbose=False,
        )

        self.assertEqual(obs_flat.shape, (8, 2))
        self.assertEqual(act_flat.shape, (8,))
        self.assertEqual(sticky_flat.shape, (8,))
        self.assertEqual(rew_flat.shape, (8,))
        self.assertEqual(p_b_taken_flat.shape, (8,))

        # p_b_taken should be in (0,1)
        self.assertTrue(np.all(p_b_taken_flat > 0.0))
        self.assertTrue(np.all(p_b_taken_flat <= 1.0))

    def test_action_stickiness_monotonicity_and_rho_correction(self):
        """Validating the stickiness correction applied to importance ratios.

        It compares uncorrected and corrected ratios and asserts that, after the first
        activation, all ratios are set to 1 as expected.
        """

        steps_per_episode = 6
        num_days = 10
        eps = 1e-12

        # Build synthetic sticky actions
        # once 1 appears, stays 1
        sticky_act_day = np.zeros((num_days, steps_per_episode), dtype=np.int64)
        for i in range(num_days):
            switch_t = np.random.randint(0, steps_per_episode)
            sticky_act_day[i, switch_t:] = 1

        sticky_act_flat = sticky_act_day.reshape(-1)

        # Monotonicity check
        self.assertTrue(
            np.all(np.diff(sticky_act_day, axis=1) >= 0),
            "sticky_action decreases in some episodes",
        )

        # Build synthetic behavior/target probs
        p_b_taken_flat = np.random.uniform(0.1, 1.0, size=num_days * steps_per_episode)
        p_e_taken_flat = np.random.uniform(0.1, 1.0, size=num_days * steps_per_episode)

        pb_day = p_b_taken_flat.reshape(num_days, steps_per_episode)
        pe_day = p_e_taken_flat.reshape(num_days, steps_per_episode)

        rho_unc = pe_day / np.maximum(pb_day, eps)

        rho_corr = apply_stickiness_correction_to_rho(
            rho_unc.copy(),
            sticky_act_day,
        )

        has_switch = (sticky_act_day == 1).any(axis=1)
        first_switch = np.argmax(sticky_act_day == 1, axis=1)

        # After switch, rho must be exactly 1.0
        for i in np.where(has_switch)[0]:
            t_star = int(first_switch[i])
            if t_star + 1 < steps_per_episode:
                self.assertTrue(
                    np.allclose(rho_corr[i, t_star + 1 :], 1.0),
                    f"rho not 1 after switch in ep {i}",
                )

    def test_validate_batches_ok(self):
        T = 5
        obs_dim = 3

        obs_batches = [np.zeros((T, obs_dim), dtype=np.float32)]
        act_batches = [np.zeros((T,), dtype=np.int64)]
        sticky_act_batches = [np.zeros((T,), dtype=np.int64)]
        rew_batches = [np.zeros((T,), dtype=np.float32)]
        p_taken_batches = [np.full((T,), 0.5, dtype=np.float32)]

        validate_batches(
            obs_batches=obs_batches,
            act_batches=act_batches,
            sticky_act_batches=sticky_act_batches,
            rew_batches=rew_batches,
            p_taken_batches=p_taken_batches,
            T_expected=T,
            obs_dim_expected=obs_dim,
            verbose=False,
        )

    def test_validate_batches_wrong_obs_dim(self):
        T = 5
        obs_dim = 3

        obs_batches = [np.zeros((T, obs_dim + 1), dtype=np.float32)]
        act_batches = [np.zeros((T,), dtype=np.int64)]
        sticky_act_batches = [np.zeros((T,), dtype=np.int64)]
        rew_batches = [np.zeros((T,), dtype=np.float32)]
        p_taken_batches = [np.full((T,), 0.5, dtype=np.float32)]

        with self.assertRaises(AssertionError):
            validate_batches(
                obs_batches=obs_batches,
                act_batches=act_batches,
                sticky_act_batches=sticky_act_batches,
                rew_batches=rew_batches,
                p_taken_batches=p_taken_batches,
                T_expected=T,
                obs_dim_expected=obs_dim,
                verbose=False,
            )

    def test_validate_batches_invalid_probability_range(self):
        T = 5
        obs_dim = 3

        obs_batches = [np.zeros((T, obs_dim), dtype=np.float32)]
        act_batches = [np.zeros((T,), dtype=np.int64)]
        sticky_act_batches = [np.zeros((T,), dtype=np.int64)]
        rew_batches = [np.zeros((T,), dtype=np.float32)]
        p_taken_batches = [np.array([0.5, 1.2, 0.5, -0.1, 0.3], dtype=np.float32)]

        with self.assertRaises(AssertionError):
            validate_batches(
                obs_batches=obs_batches,
                act_batches=act_batches,
                sticky_act_batches=sticky_act_batches,
                rew_batches=rew_batches,
                p_taken_batches=p_taken_batches,
                T_expected=T,
                obs_dim_expected=obs_dim,
                verbose=False,
            )

    def test_add_zone_temp_and_reward_happy_path(self):
        # Two episodes, each 3 steps
        ep1 = pd.DataFrame(
            {
                "raw_observation": [
                    str([21.5, 0.25]),
                    str([21.7, 0.50]),
                    str([22.0, 0.75]),
                ],
                "sticky_action": [0, 0, 1],
            }
        )
        ep2 = pd.DataFrame(
            {
                "raw_observation": [
                    str([19.0, 0.25]),
                    str([19.3, 0.50]),
                    str([19.8, 0.75]),
                ],
                "sticky_action": [0, 1, 1],
            }
        )

        episode_dict = {"ep1": ep1, "ep2": ep2}

        def reward_series_fn(df_ep: pd.DataFrame) -> pd.Series:
            # simple deterministic reward: reward = 1 if sticky_action==1 else 0
            return pd.Series(
                df_ep["sticky_action"].astype(float).to_numpy(), index=df_ep.index, name="reward"
            )

        out = add_zone_temp_and_reward(
            episode_dict,
            reward_series_fn=reward_series_fn,
            raw_obs_col="raw_observation",
            zone_temp_col="zone_temperature",
            reward_col="reward",
            idx_zone_temp=0,
        )

        self.assertEqual(set(out.keys()), {"ep1", "ep2"})

        # Check columns exist and shapes preserved
        for eid in ["ep1", "ep2"]:
            self.assertIn("zone_temperature", out[eid].columns)
            self.assertIn("reward", out[eid].columns)
            self.assertEqual(len(out[eid]), 3)

        # zone_temperature parsed from raw_observation[0]
        self.assertTrue(
            np.allclose(
                out["ep1"]["zone_temperature"].to_numpy(), np.array([21.5, 21.7, 22.0], dtype=float)
            )
        )
        self.assertTrue(
            np.allclose(
                out["ep2"]["zone_temperature"].to_numpy(), np.array([19.0, 19.3, 19.8], dtype=float)
            )
        )

        # reward from sticky_action
        self.assertTrue(
            np.allclose(out["ep1"]["reward"].to_numpy(), np.array([0.0, 0.0, 1.0], dtype=float))
        )
        self.assertTrue(
            np.allclose(out["ep2"]["reward"].to_numpy(), np.array([0.0, 1.0, 1.0], dtype=float))
        )

    def test_add_zone_temp_and_reward_defensive_index_alignment(self):
        # Non-default index to ensure reset_index(drop=True) path is exercised
        ep = pd.DataFrame(
            {
                "raw_observation": [
                    str([20.0, 0.25]),
                    str([20.5, 0.50]),
                    str([21.0, 0.75]),
                ],
                "sticky_action": [0, 1, 1],
            },
            index=[100, 200, 300],
        )
        episode_dict = {"ep": ep}

        def reward_series_fn(df_ep: pd.DataFrame) -> pd.Series:
            # Return series with a different index to ensure alignment would break without reset
            r = pd.Series([10.0, 11.0, 12.0], index=[0, 1, 2], name="reward")
            return r

        out = add_zone_temp_and_reward(
            episode_dict,
            reward_series_fn=reward_series_fn,
            raw_obs_col="raw_observation",
            zone_temp_col="zone_temperature",
            reward_col="reward",
            idx_zone_temp=0,
        )

        # After defensive reset, rewards must be assigned positionally (no NaNs)
        self.assertTrue(
            np.allclose(out["ep"]["reward"].to_numpy(), np.array([10.0, 11.0, 12.0], dtype=float))
        )
        self.assertFalse(np.isnan(out["ep"]["reward"].to_numpy()).any())

        # And the df index has been reset to RangeIndex
        self.assertTrue(isinstance(out["ep"].index, pd.RangeIndex))
        self.assertEqual(list(out["ep"].index), [0, 1, 2])

    def test_add_zone_temp_and_reward_reward_fn_must_return_series(self):
        ep = pd.DataFrame({"raw_observation": [str([20.0, 0.25])], "sticky_action": [0]})
        episode_dict = {"ep": ep}

        def bad_reward_fn(_df_ep: pd.DataFrame):
            return [0.0]  # not a Series

        with self.assertRaises(TypeError):
            _ = add_zone_temp_and_reward(
                episode_dict,
                reward_series_fn=bad_reward_fn,
                idx_zone_temp=0,
            )

    def test_add_zone_temp_and_reward_length_mismatch_raises(self):
        ep = pd.DataFrame(
            {
                "raw_observation": [
                    str([20.0, 0.25]),
                    str([20.5, 0.50]),
                ],
                "sticky_action": [0, 1],
            }
        )
        episode_dict = {"ep": ep}

        def reward_series_fn(_df_ep: pd.DataFrame) -> pd.Series:
            return pd.Series([1.0])  # wrong length

        with self.assertRaises(ValueError):
            _ = add_zone_temp_and_reward(
                episode_dict,
                reward_series_fn=reward_series_fn,
                idx_zone_temp=0,
            )
