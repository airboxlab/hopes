import unittest

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from hopes.rew.rewards import RegressionBasedRewardModel, RewardFunctionModel


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
