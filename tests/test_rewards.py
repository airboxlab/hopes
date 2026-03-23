import unittest

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

from hopes.rew.rewards import (
    RegressionBasedRewardModel,
    RewardFunctionModel,
    RTGQModelHGBoost,
)


class TestRewards(unittest.TestCase):
    def test_polynomial_reward_model(self):
        self._test_reward_model("polynomial")

    def test_linear_reward_model(self):
        self._test_reward_model("linear")

    def test_mlp_reward_model(self):
        self._test_reward_model("mlp")

    def test_rf_reward_model(self):
        self._test_reward_model("random_forest")

    def _test_reward_model(self, model_type: str) -> None:
        num_actions = 3
        num_obs = 5
        num_samples = 100
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)
        rew = np.random.rand(num_samples)

        mlp_reward_model = RegressionBasedRewardModel(
            obs=obs, act=act, rew=rew, regression_model=model_type
        )
        print(f"{model_type} stats: ", mlp_reward_model.fit())

        new_obs = np.random.rand(10, num_obs)
        new_act = np.random.randint(num_actions, size=10)
        rewards = mlp_reward_model.estimate(obs=new_obs, act=new_act)

        self.assertEqual(rewards.shape, (10,))

    def test_reward_fun(self):
        def reward_fun(obs_, act_):
            return np.sum(obs_, axis=0) + act_

        num_actions = 3
        num_obs = 5
        num_samples = 100
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)
        rew = [reward_fun(o, a) for o, a in zip(obs, act) if o.ndim == 1]

        reward_model = RewardFunctionModel(reward_function=reward_fun)
        rewards = reward_model.estimate(obs=obs, act=act)

        np.testing.assert_array_almost_equal(rew, rewards.squeeze())

    def test_scaler(self):
        def neg_reward_fun(obs_, act_):
            return -(np.sum(obs_, axis=0) + act_)

        num_actions = 3
        num_obs = 5
        num_samples = 100
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)
        rew = np.array([neg_reward_fun(o, a) for o, a in zip(obs, act) if o.ndim == 1])

        rew_scaled = lambda x: (x - rew.min()) / (rew.max() - rew.min())

        reward_model = RewardFunctionModel(reward_function=neg_reward_fun).with_scaler(rew_scaled)
        rewards = reward_model.estimate(obs=obs, act=act)

        self.assertTrue(np.all(rewards >= 0) and np.all(rewards <= 1))

    def test_external_scaler(self):
        def neg_reward_fun(obs_, act_):
            return -(np.sum(obs_, axis=0) + act_)

        num_actions = 3
        num_obs = 5
        num_samples = 100
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)
        rew = np.array([neg_reward_fun(o, a) for o, a in zip(obs, act) if o.ndim == 1])

        rew_scaled = MinMaxScaler(clip=True)
        rew_scaled.fit(rew.reshape(-1, 1))

        reward_model = RewardFunctionModel(reward_function=neg_reward_fun).with_scaler(
            rew_scaled.transform
        )
        rewards = reward_model.estimate(obs=obs, act=act)

        self.assertTrue(np.all(rewards >= 0) and np.all(rewards <= 1))

    def test_fit_raises_on_invalid_steps_per_episode(self):
        model = RTGQModelHGBoost(
            steps_per_episode=0,
            num_actions=2,
        )

        obs_flat = np.random.rand(8, 3).astype(np.float32)
        act_flat = np.random.randint(0, 2, size=8)
        rew_flat = np.random.rand(8).astype(np.float32)

        with self.assertRaises(ValueError):
            model.fit(
                obs_flat=obs_flat,
                act_flat=act_flat,
                rew_flat=rew_flat,
            )

    def test_fit_raises_on_invalid_num_actions(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=1,
        )

        obs_flat = np.random.rand(8, 3).astype(np.float32)
        act_flat = np.zeros(8, dtype=np.int64)
        rew_flat = np.random.rand(8).astype(np.float32)

        with self.assertRaises(ValueError):
            model.fit(
                obs_flat=obs_flat,
                act_flat=act_flat,
                rew_flat=rew_flat,
            )

    def test_fit_raises_on_obs_ndim(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=2,
        )

        obs_flat = np.random.rand(8).astype(np.float32)
        act_flat = np.random.randint(0, 2, size=8)
        rew_flat = np.random.rand(8).astype(np.float32)

        with self.assertRaises(ValueError):
            model.fit(
                obs_flat=obs_flat,
                act_flat=act_flat,
                rew_flat=rew_flat,
            )

    def test_fit_raises_on_discount_factor_greater_than_one(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=2,
            discount_factor=1.5,
        )

        obs_flat = np.random.rand(8, 3).astype(np.float32)
        act_flat = np.random.randint(0, 2, size=8)
        rew_flat = np.random.rand(8).astype(np.float32)

        with self.assertRaises(ValueError):
            model.fit(
                obs_flat=obs_flat,
                act_flat=act_flat,
                rew_flat=rew_flat,
            )

    def test_fit_raises_on_negative_discount_factor(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=2,
            discount_factor=-0.1,
        )

        obs_flat = np.random.rand(8, 3).astype(np.float32)
        act_flat = np.random.randint(0, 2, size=8)
        rew_flat = np.random.rand(8).astype(np.float32)

        with self.assertRaises(ValueError):
            model.fit(
                obs_flat=obs_flat,
                act_flat=act_flat,
                rew_flat=rew_flat,
            )

    def test_fit_raises_on_length_mismatch(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=2,
        )

        obs_flat = np.random.rand(8, 3).astype(np.float32)
        act_flat = np.random.randint(0, 2, size=7)
        rew_flat = np.random.rand(8).astype(np.float32)

        with self.assertRaises(ValueError):
            model.fit(
                obs_flat=obs_flat,
                act_flat=act_flat,
                rew_flat=rew_flat,
            )

    def test_fit_raises_on_invalid_action_index(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=2,
        )

        obs_flat = np.random.rand(8, 3).astype(np.float32)
        act_flat = np.array([0, 1, 0, 1, 0, 1, 0, 2], dtype=np.int64)
        rew_flat = np.random.rand(8).astype(np.float32)

        with self.assertRaises(ValueError):
            model.fit(
                obs_flat=obs_flat,
                act_flat=act_flat,
                rew_flat=rew_flat,
            )

    def test_fit_raises_if_samples_not_divisible_by_steps(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=2,
        )

        obs_flat = np.random.rand(10, 3).astype(np.float32)
        act_flat = np.random.randint(0, 2, size=10)
        rew_flat = np.random.rand(10).astype(np.float32)

        with self.assertRaises(ValueError):
            model.fit(
                obs_flat=obs_flat,
                act_flat=act_flat,
                rew_flat=rew_flat,
            )

    def test_fit_returns_model(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=3,
            random_state=0,
        )

        obs_flat = np.random.rand(12, 5).astype(np.float32)
        act_flat = np.random.randint(0, 3, size=12)
        rew_flat = np.random.rand(12).astype(np.float32)

        fitted = model.fit(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
        )

        self.assertIsInstance(fitted, HistGradientBoostingRegressor)
        self.assertIs(model.model, fitted)

    def test_estimate_raises_if_not_fitted(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=3,
        )

        obs = np.random.rand(12, 5).astype(np.float32)
        act = np.random.randint(0, 3, size=12)

        with self.assertRaises(ValueError):
            model.estimate(obs=obs, act=act)

    def test_estimate_shape(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=3,
            random_state=0,
        )

        obs_flat = np.random.rand(12, 5).astype(np.float32)
        act_flat = np.random.randint(0, 3, size=12)
        rew_flat = np.random.rand(12).astype(np.float32)

        model.fit(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
        )

        q_logged = model.estimate(obs=obs_flat, act=act_flat)

        self.assertIsInstance(q_logged, np.ndarray)
        self.assertEqual(q_logged.shape, (12,))
        self.assertTrue(np.issubdtype(q_logged.dtype, np.floating))
        self.assertTrue(np.all(np.isfinite(q_logged)))

    def test_estimate_raises_on_obs_act_length_mismatch(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=3,
            random_state=0,
        )

        obs_flat = np.random.rand(12, 5).astype(np.float32)
        act_flat = np.random.randint(0, 3, size=12)
        rew_flat = np.random.rand(12).astype(np.float32)

        model.fit(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
        )

        with self.assertRaises(ValueError):
            model.estimate(
                obs=np.random.rand(12, 5).astype(np.float32),
                act=np.random.randint(0, 3, size=11),
            )

    def test_estimate_raises_on_invalid_action_index(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=3,
            random_state=0,
        )

        obs_flat = np.random.rand(12, 5).astype(np.float32)
        act_flat = np.random.randint(0, 3, size=12)
        rew_flat = np.random.rand(12).astype(np.float32)

        model.fit(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
        )

        bad_actions = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 3], dtype=np.int64)

        with self.assertRaises(ValueError):
            model.estimate(obs=obs_flat, act=bad_actions)

    def test_estimate_uses_scaler(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=3,
            random_state=0,
        )

        obs_flat = np.random.rand(12, 5).astype(np.float32)
        act_flat = np.random.randint(0, 3, size=12)
        rew_flat = np.random.rand(12).astype(np.float32)

        model.fit(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
        )

        unscaled = model.estimate(obs=obs_flat, act=act_flat)
        scaled = model.with_scaler(lambda x: x + 1.0).estimate(obs=obs_flat, act=act_flat)

        self.assertEqual(unscaled.shape, (12,))
        self.assertEqual(scaled.shape, (12, 1))
        np.testing.assert_allclose(scaled.squeeze(), unscaled + 1.0)

    def test_predict_q_values_raises_if_not_fitted(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=3,
        )

        obs_flat = np.random.rand(12, 5).astype(np.float32)

        with self.assertRaises(ValueError):
            model.predict_q_values(obs_flat=obs_flat)

    def test_predict_q_values_shape(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=3,
            random_state=0,
        )

        obs_flat = np.random.rand(12, 5).astype(np.float32)
        act_flat = np.random.randint(0, 3, size=12)
        rew_flat = np.random.rand(12).astype(np.float32)

        model.fit(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
        )

        q_values = model.predict_q_values(obs_flat=obs_flat)

        self.assertIsInstance(q_values, np.ndarray)
        self.assertEqual(q_values.shape, (12, 3))
        self.assertTrue(np.issubdtype(q_values.dtype, np.floating))
        self.assertTrue(np.all(np.isfinite(q_values)))

    def test_predict_q_values_raises_on_invalid_obs_shape(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=3,
            random_state=0,
        )

        obs_flat = np.random.rand(12, 5).astype(np.float32)
        act_flat = np.random.randint(0, 3, size=12)
        rew_flat = np.random.rand(12).astype(np.float32)

        model.fit(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
        )

        with self.assertRaises(ValueError):
            model.predict_q_values(obs_flat=np.random.rand(12).astype(np.float32))

    def test_predict_q_values_raises_if_samples_not_divisible_by_steps(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=3,
            random_state=0,
        )

        obs_flat = np.random.rand(12, 5).astype(np.float32)
        act_flat = np.random.randint(0, 3, size=12)
        rew_flat = np.random.rand(12).astype(np.float32)

        model.fit(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
        )

        with self.assertRaises(ValueError):
            model.predict_q_values(obs_flat=np.random.rand(10, 5).astype(np.float32))

    def test_predict_q_values_uses_scaler(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=3,
            random_state=0,
        ).with_scaler(lambda x: x * 0.0)

        obs_flat = np.random.rand(12, 5).astype(np.float32)
        act_flat = np.random.randint(0, 3, size=12)
        rew_flat = np.random.rand(12).astype(np.float32)

        model.fit(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
        )

        q_values = model.predict_q_values(obs_flat=obs_flat)

        self.assertEqual(q_values.shape, (12, 3))
        self.assertTrue(np.allclose(q_values, 0.0))

    def test_compute_rtg_structure(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=2,
            discount_factor=0.5,
        )

        rew_flat = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,  # episode 1
                5.0,
                6.0,
                7.0,
                8.0,  # episode 2
            ],
            dtype=np.float32,
        )

        rtg = model._compute_rtg(rew_flat)

        expected = np.array(
            [
                3.25,
                4.5,
                5.0,
                4.0,
                10.75,
                11.5,
                11.0,
                8.0,
            ],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(rtg, expected))

    def test_build_state_features_shape(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=2,
        )

        obs_flat = np.random.rand(8, 3).astype(np.float32)

        x_state = model._build_state_features(obs_flat)

        self.assertEqual(x_state.shape, (8, 4))
        self.assertTrue(np.all(np.isfinite(x_state)))
