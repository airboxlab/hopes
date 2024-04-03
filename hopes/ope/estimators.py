from abc import ABC, abstractmethod

import numpy as np

from hopes.assert_utils import check_array
from hopes.dev_utils import override
from hopes.rew.rewards import RegressionBasedRewardModel


class BaseEstimator(ABC):
    """Base class for all estimators."""

    def __init__(self):
        self.target_policy_action_probabilities: np.ndarray | None = None
        self.behavior_policy_action_probabilities: np.ndarray | None = None
        self.rewards: np.ndarray | None = None

    def set_parameters(
        self,
        target_policy_action_probabilities: np.ndarray,
        behavior_policy_action_probabilities: np.ndarray | None,
        rewards: np.ndarray | None,
    ) -> None:
        """Set the parameters for estimating the policy value.

        :param target_policy_action_probabilities: the probabilities of taking actions under
            the target policy.
        :param behavior_policy_action_probabilities: the probabilities of taking actions
            under the behavior policy.
        :param rewards: the rewards received under the behavior policy.
        :return: None
        """
        self.target_policy_action_probabilities = target_policy_action_probabilities
        self.behavior_policy_action_probabilities = behavior_policy_action_probabilities
        self.rewards = rewards

        self.check_parameters()

    def check_parameters(self) -> None:
        """Check if the estimator parameters are valid.

        This method should be called before estimating the policy value. It can be
        overridden by subclasses to add additional checks.

        Importance sampling estimators have several assumptions that must be met:
        - coverage: the target and behavior policies must have non-zero probability of taking the actions.
            This is not 100% necessary, i.e. if both policies have zero probability of taking an action under some
            state, but we enforce this assumption here to avoid numerical issues.
        - positivity: the rewards must be non-negative to be able to get a lower bound estimate of the
            target policy.
        """

        if self.behavior_policy_action_probabilities is None:
            behavior_policy_action_probabilities = np.ones_like(
                self.target_policy_action_probabilities
            )
            behavior_policy_action_probabilities /= behavior_policy_action_probabilities.sum(
                axis=1, keepdims=True
            )
        else:
            behavior_policy_action_probabilities = self.behavior_policy_action_probabilities

        if self.rewards is None:
            rewards = np.zeros(self.target_policy_action_probabilities.shape[0])
        else:
            rewards = self.rewards

        if (
            self.target_policy_action_probabilities is None
            or behavior_policy_action_probabilities is None
            or rewards is None
        ):
            raise ValueError("You must set the parameters before estimating the policy value.")

        for array, name, ndims in zip(
            [
                self.target_policy_action_probabilities,
                behavior_policy_action_probabilities,
                rewards,
            ],
            [
                "target_policy_action_probabilities",
                "behavior_policy_action_probabilities",
                "rewards",
            ],
            [2, 2, 1],
        ):
            check_array(array=array, name=name, expected_ndim=ndims, expected_dtype=float)

        if not (
            self.target_policy_action_probabilities.shape[0]
            == behavior_policy_action_probabilities.shape[0]
            == rewards.shape[0]
        ):
            raise ValueError("The number of samples must be the same for all parameters.")

        if not (
            self.target_policy_action_probabilities.shape[1]
            == behavior_policy_action_probabilities.shape[1]
        ):
            raise ValueError(
                "The probabilities of taking actions under the target and "
                "behavior policies must have the same shape."
            )

        if np.any(self.target_policy_action_probabilities <= 0):
            raise ValueError("The target policy action probabilities must be positive.")

        if np.any(behavior_policy_action_probabilities <= 0):
            raise ValueError("The behavior policy action probabilities must be positive.")

        if np.any(rewards < 0):
            raise ValueError(
                "The rewards must be non-negative to be able to get a lower bound estimate of the target policy. "
                "Use a positive reward estimator or an appropriate scaler (i.e MinMaxScaler) to scale the rewards."
            )

        for array, name in zip(
            [self.target_policy_action_probabilities, behavior_policy_action_probabilities],
            ["target_policy_action_probabilities", "behavior_policy_action_probabilities"],
        ):
            if not np.allclose(np.sum(array, axis=1), np.ones(array.shape[0], dtype=float)):
                raise ValueError(f"The {name} must sum to 1 on each sample.")

    @abstractmethod
    def estimate_policy_value(self) -> float:
        pass


class InverseProbabilityWeighting(BaseEstimator):
    r"""Inverse Probability Weighting (IPW) estimator.

    V_{IPW}(\pi_e, D)=\frac {1}{n} \sum_{t=1}^n p(s_t,a_t) r_t

    Where:
        - D is the offline collected dataset.
        - p(s_t,a_t) is the importance weight defined as p(s_t,a_t)=\frac{\pi_e(a_t|s_t)}{\pi_b(a_t|s_t)}.
        - pi_e is the target policy and pi_b is the behavior policy.
        - r_t is the reward at time t.
        - n is the number of samples.

    This estimator has generally high variance, especially on small datasets, and can be improved by using
    self-normalized importance weights.

    References:
        https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs
    """

    def __init__(self) -> None:
        super().__init__()

        # computed importance weights, saved for reuse by subclasses
        # shape: (n, num_actions)
        self.importance_weights: np.ndarray | None = None

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        self.importance_weights = None
        self.check_parameters()

        self.importance_weights = (
            self.target_policy_action_probabilities / self.behavior_policy_action_probabilities
        )
        return np.mean(self.importance_weights * self.rewards.reshape(-1, 1))


class SelfNormalizedInverseProbabilityWeighting(InverseProbabilityWeighting):
    r"""Self-Normalized Inverse Probability Weighting (SNIPW) estimator.

    V_{SNIPW}(\pi_e, D)= \frac {\sum_{t=1}^n p(s_t,a_t) r_t}{\sum_{t=1}^n p(s_t,a_t)}

    Where:
        - D is the offline collected dataset.
        - p(s_t,a_t) is the importance weight defined as p(s_t,a_t)=\frac{\pi_e(a_t|s_t)}{\pi_b(a_t|s_t)}.
        - pi_e is the target policy and pi_b is the behavior policy.
        - r_t is the reward at time t.
        - n is the number of samples.

    References:
        https://papers.nips.cc/paper_files/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html
    """

    def __init__(self) -> None:
        super().__init__()

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        super().estimate_policy_value()

        return np.sum(self.importance_weights * self.rewards.reshape(-1, 1)) / np.sum(
            self.importance_weights
        )


class DirectMethod(BaseEstimator):
    r"""Direct Method (DM) estimator.

    V_{DM}(\pi_e, D, Q)=\frac {1}{n} \sum_{t=1}^n \sum_{a \in A} \pi_e(a|s_0) Q(s_0, a)

    Where:
        - D is the offline collected dataset.
        - pi_e is the target policy.
        - Q is the Q model trained to estimate the expected reward of the initial state under the behavior policy.
        - n is the number of samples.
        - a is the action taken.

    This estimator trains a Q model using supervised learning, then uses it to estimate the expected reward of the
    initial state under the target policy. The performance of this estimator depends on the quality of the Q model.
    """

    def __init__(
        self,
        q_model_cls: type[RegressionBasedRewardModel],
        behavior_policy_obs: np.ndarray,
        behavior_policy_act: np.ndarray,
        behavior_policy_rewards: np.ndarray,
        steps_per_episode: int,
        q_model_type: str = "random_forest",
        q_model_params: dict | None = None,
    ) -> None:
        """Initialize the Direct Method estimator.

        :param q_model_cls: the class of the Q model to use.
        :param behavior_policy_obs: the observations for training the Q model, shape:
            (batch_size, obs_dim).
        :param behavior_policy_act: the actions for training the Q model, shape:
            (batch_size,).
        :param behavior_policy_rewards: the rewards for training the Q model, shape:
            (batch_size,).
        :param steps_per_episode: the number of steps per episode. The number of samples
            must be divisible by this number.
        :param q_model_type: the type of regression model to use for the Q model.
        :param q_model_params: the parameters of the regression model.
        """
        super().__init__()

        assert issubclass(
            q_model_cls, RegressionBasedRewardModel
        ), "The Q model must be a subclass of RegressionBasedRewardModel."
        assert (
            behavior_policy_obs.ndim == 2
        ), "The observations must have shape (batch_size, obs_dim)."
        assert behavior_policy_act.ndim == 1, "The actions must have shape (batch_size,)."
        assert behavior_policy_rewards.ndim == 1, "The rewards must have shape (batch_size,)."
        assert (
            behavior_policy_obs.shape[0]
            == behavior_policy_act.shape[0]
            == behavior_policy_rewards.shape[0]
        ), "The number of samples must be the same for all parameters."
        assert steps_per_episode > 0, "The number of steps per episode must be positive."
        assert (
            behavior_policy_obs.shape[0] % steps_per_episode == 0
        ), "The number of samples must be divisible by the number of steps per episode."

        self.q_model_cls = q_model_cls
        self.q_model_type = q_model_type
        self.q_model_params = q_model_params
        self.q_model: RegressionBasedRewardModel | None = None
        self.behavior_policy_obs = behavior_policy_obs
        self.behavior_policy_act = behavior_policy_act
        self.behavior_policy_rewards = behavior_policy_rewards
        self.steps_per_episode = steps_per_episode

    def fit(self) -> dict[str, float] | None:
        self.q_model = self.q_model_cls(
            obs=self.behavior_policy_obs,
            act=self.behavior_policy_act,
            rew=self.behavior_policy_rewards,
            regression_model=self.q_model_type,
            model_params=self.q_model_params,
        )
        return self.q_model.fit()

    def check_parameters(self) -> None:
        super().check_parameters()

        assert (
            self.behavior_policy_obs.shape[0] == self.target_policy_action_probabilities.shape[0]
        ), "The number of samples must be the same for the behavior policy and the target policy."

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        self.check_parameters()

        # use the Q model to predict the expected rewards
        state_action_value_prediction = self.q_model.estimate(
            obs=self.behavior_policy_obs,
            act=self.behavior_policy_act,
        ).reshape(-1, 1)

        # compute the expected reward of the initial state under the target policy
        state_value = (
            (state_action_value_prediction * self.target_policy_action_probabilities)
            .sum(axis=1)
            .reshape(-1, self.steps_per_episode)
        )
        initial_state_value = state_value[:, 0]
        return np.mean(initial_state_value)
