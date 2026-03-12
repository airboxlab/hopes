import unittest

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from hopes.ope.q_models import RTGQModelHGBoost


class TestRTGQModelHGBoost(unittest.TestCase):
    def test_init_invalid_steps_per_episode_raises(self):
        with self.assertRaises(ValueError):
            RTGQModelHGBoost(
                steps_per_episode=0,
                num_actions=2,
            )

    def test_init_invalid_num_actions_raises(self):
        with self.assertRaises(ValueError):
            RTGQModelHGBoost(
                steps_per_episode=4,
                num_actions=1,
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

    def test_fit_predict_q_values_returns_q_values_and_model(self):
        model = RTGQModelHGBoost(
            steps_per_episode=3,
            num_actions=4,
            random_state=0,
        )

        obs_flat = np.random.rand(15, 6).astype(np.float32)
        act_flat = np.random.randint(0, 4, size=15)
        rew_flat = np.random.rand(15).astype(np.float32)

        q_values, fitted_model = model.fit_predict_q_values(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
            return_model=True,
        )

        self.assertIsInstance(q_values, np.ndarray)
        self.assertEqual(q_values.shape, (15, 4))
        self.assertIsInstance(fitted_model, HistGradientBoostingRegressor)

    def test_fit_predict_q_values_returns_none_model_when_requested(self):
        model = RTGQModelHGBoost(
            steps_per_episode=3,
            num_actions=4,
            random_state=0,
        )

        obs_flat = np.random.rand(15, 6).astype(np.float32)
        act_flat = np.random.randint(0, 4, size=15)
        rew_flat = np.random.rand(15).astype(np.float32)

        q_values, fitted_model = model.fit_predict_q_values(
            obs_flat=obs_flat,
            act_flat=act_flat,
            rew_flat=rew_flat,
            return_model=False,
        )

        self.assertIsInstance(q_values, np.ndarray)
        self.assertEqual(q_values.shape, (15, 4))
        self.assertIsNone(fitted_model)

    def test_compute_rtg_structure(self):
        model = RTGQModelHGBoost(
            steps_per_episode=4,
            num_actions=2,
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
                10.0,
                9.0,
                7.0,
                4.0,
                26.0,
                21.0,
                15.0,
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

        self.assertEqual(x_state.shape, (8, 4))  # 3 obs dims + 1 timestep
        self.assertTrue(np.all(np.isfinite(x_state)))
