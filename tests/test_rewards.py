import unittest

import numpy as np

from hopes.rew.rewards import RegressionBasedRewardModel, RewardFunctionModel


class TestRewards(unittest.TestCase):
    def test_polynomial_reward_model(self):
        num_actions = 3
        num_obs = 5
        num_samples = 100
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)
        rew = np.random.rand(num_samples)

        poly_reward_model = RegressionBasedRewardModel(
            obs=obs, act=act, rew=rew, reward_model="polynomial"
        )
        poly_reward_model.fit()

        new_obs = np.random.rand(10, num_obs)
        new_act = np.random.randint(num_actions, size=10)
        rewards = poly_reward_model.estimate(obs=new_obs, act=new_act)

        self.assertEqual(rewards.shape, (10,))

    def test_linear_reward_model(self):
        num_actions = 3
        num_obs = 5
        num_samples = 100
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)
        rew = np.random.rand(num_samples)

        linear_reward_model = RegressionBasedRewardModel(
            obs=obs, act=act, rew=rew, reward_model="linear"
        )
        linear_reward_model.fit()

        new_obs = np.random.rand(10, num_obs)
        new_act = np.random.randint(num_actions, size=10)
        rewards = linear_reward_model.estimate(obs=new_obs, act=new_act)

        self.assertEqual(rewards.shape, (10,))

    def test_mlp_reward_model(self):
        num_actions = 3
        num_obs = 5
        num_samples = 100
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)
        rew = np.random.rand(num_samples)

        mlp_reward_model = RegressionBasedRewardModel(obs=obs, act=act, rew=rew, reward_model="mlp")
        mlp_reward_model.fit()

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

        np.testing.assert_array_almost_equal(rew, rewards)
