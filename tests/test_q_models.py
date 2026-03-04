import unittest

import numpy as np

from hopes.ope.q_models import RTGQModelHGBoost


class TestRTGQModelHGBoost(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)

        self.steps_per_episode = 5
        self.num_eps = 10
        self.obs_dim = 4

        self.N = self.steps_per_episode * self.num_eps

        self.obs_flat = rng.normal(size=(self.N, self.obs_dim)).astype(np.float32)
        self.act_flat = rng.integers(0, 2, size=self.N, dtype=np.int64)
        self.rew_flat = rng.normal(size=self.N).astype(np.float32)

    def test_fit_predict_q0_q1_shapes(self):
        model = RTGQModelHGBoost(steps_per_episode=self.steps_per_episode)

        Q0, Q1, gbr = model.fit_predict_q0_q1(
            obs_flat=self.obs_flat,
            act_flat=self.act_flat,
            rew_flat=self.rew_flat,
            return_model=True,
        )

        self.assertEqual(Q0.shape, (self.N,))
        self.assertEqual(Q1.shape, (self.N,))

        self.assertEqual(Q0.dtype, np.float32)
        self.assertEqual(Q1.dtype, np.float32)

        self.assertIsNotNone(gbr)

    def test_fit_then_predict(self):
        model = RTGQModelHGBoost(steps_per_episode=self.steps_per_episode)

        model.fit(
            obs_flat=self.obs_flat,
            act_flat=self.act_flat,
            rew_flat=self.rew_flat,
        )

        Q0, Q1 = model.predict_q0_q1(obs_flat=self.obs_flat)

        self.assertEqual(Q0.shape, (self.N,))
        self.assertEqual(Q1.shape, (self.N,))
        self.assertTrue(np.all(np.isfinite(Q0)))
        self.assertTrue(np.all(np.isfinite(Q1)))

    def test_predict_without_fit_raises(self):
        model = RTGQModelHGBoost(steps_per_episode=self.steps_per_episode)

        with self.assertRaises(ValueError):
            model.predict_q0_q1(obs_flat=self.obs_flat)

    def test_invalid_episode_length(self):
        model = RTGQModelHGBoost(steps_per_episode=self.steps_per_episode)

        bad_obs = self.obs_flat[:-1]
        bad_act = self.act_flat[:-1]
        bad_rew = self.rew_flat[:-1]

        with self.assertRaises(ValueError):
            model.fit(
                obs_flat=bad_obs,
                act_flat=bad_act,
                rew_flat=bad_rew,
            )

    def test_rtg_computation(self):
        """Rewards increasing linearly inside each episode.

        RTG should follow a known pattern.
        """
        steps = 4
        num_eps = 2
        N = steps * num_eps

        obs = np.zeros((N, 3), dtype=np.float32)
        act = np.zeros(N, dtype=np.int64)

        # rewards per episode: [1,2,3,4]
        rew = np.tile(np.array([1, 2, 3, 4], dtype=np.float32), num_eps)

        model = RTGQModelHGBoost(steps_per_episode=steps)

        rtg = model._compute_rtg_flat(rew)

        expected = np.array(
            [10, 9, 7, 4, 10, 9, 7, 4],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(rtg, expected))
