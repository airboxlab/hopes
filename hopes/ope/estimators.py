import re
from abc import ABC, abstractmethod

import numpy as np
import scipy

from hopes.assert_utils import check_array
from hopes.dev_utils import override
from hopes.ope.utils import apply_stickiness_correction_to_rho
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
            check_array(
                array=array, name=name, expected_ndim=ndims, expected_dtype=(float, np.float32)
            )

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

    def estimate_policy_value_with_confidence_interval(
        self,
        method: str = "bootstrap",
        significance_level: float = 0.05,
        num_samples: int = 1000,
    ) -> dict[str, float]:
        r"""Estimate the confidence interval of the policy value.

        The `bootstrap` method uses bootstrapping to estimate the confidence interval of the policy value. Bootstrapping
        consists in resampling the data with replacement to infer the distribution of the estimated weighted rewards.
        The confidence interval is then computed as the quantiles of the bootstrapped samples.

        The `t-test` method (or `Student's t-test`) uses the t-distribution of the estimated weighted rewards - assuming
        that the weighted rewards are normally distributed - to estimate the confidence interval of the policy value.
        It follows the t-distribution formula :math:`t = \frac{\hat{\mu} - \mu}{\hat{\sigma} / \sqrt{n}}`, where
        :math:`\hat{\mu}` is the mean of the weighted rewards, :math:`\mu` is the true mean of the weighted rewards,
        :math:`\hat{\sigma}` is the standard deviation of the weighted rewards, and :math:`n` is the number of samples.
        The confidence interval is then computed as:

        .. math::
            [\hat{\mu} - t_{\mathrm{test}}(1 - \alpha, n-1) \frac{\hat{\sigma}}{\sqrt{n}},
            \hat{\mu} + t_{\mathrm{test}}(1 - \alpha, n-1) \frac{\hat{\sigma}}{\sqrt{n}}]

        The input data is sampled from the estimated weighted rewards, using :meth:`estimate_weighted_rewards`.

        Example:

        .. code-block:: python

            ipw = InverseProbabilityWeighting()
            ipw.set_parameters(
                target_policy_action_probabilities=target_policy_action_probabilities,
                behavior_policy_action_probabilities=behavior_policy_action_probabilities,
                rewards=rewards,
            )
            metrics = ipw.estimate_policy_value_with_confidence_interval(
                method="bootstrap", significance_level=0.05
            )
            print(metrics)

        Should output something like:

        .. code-block:: python

            {
                "lower_bound": 10.2128,
                "upper_bound": 10.6167,
                "mean": 10.4148,
                "std": 6.72408,
            }

        :param method: the method to use for estimating the confidence interval. Currently, only "bootstrap" and
            "t-test" are supported.
        :param significance_level: the significance level of the confidence interval.
        :param num_samples: the number of bootstrap samples to use. Only used when `method` is "bootstrap".
        :return: a dictionary containing the confidence interval of the policy value. The keys are:

            - "lower_bound": the lower bound of the policy value, given the significance level.
            - "upper_bound": the upper bound of the policy value, given the significance level.
            - "mean": the mean of the policy value.
            - "std": the standard deviation of the policy value.
        """
        assert method in ["bootstrap", "t-test"], "The method must be 'bootstrap' or 't-test'."
        assert 0 < significance_level < 1, "The significance level must be in (0, 1)."

        weighted_rewards = self.estimate_weighted_rewards()
        assert (
            weighted_rewards is not None and len(weighted_rewards) > 0
        ), "The weighted rewards must not be empty."

        weighted_rewards = weighted_rewards.reshape(-1)

        if method == "bootstrap":
            boot_samples = [
                np.mean(
                    np.random.choice(weighted_rewards, size=weighted_rewards.shape[0], replace=True)
                )
                for _ in np.arange(num_samples)
            ]

            lower_bound = np.quantile(boot_samples, significance_level / 2)
            upper_bound = np.quantile(boot_samples, 1 - significance_level / 2)

            return {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "mean": np.mean(boot_samples),
                "std": np.std(boot_samples),
            }

        elif method == "t-test":
            num_samples = weighted_rewards.shape[0]
            mean = np.mean(weighted_rewards)
            # compute the standard deviation of the weighted rewards, using degrees of freedom = num_samples - 1
            std = np.std(weighted_rewards, ddof=1)
            # compute t, with alpha = significance_level / 2 and degrees of freedom = num_samples - 1
            t = scipy.stats.t.ppf(1 - significance_level / 2, num_samples - 1)
            # compute the confidence interval
            ci = t * std / np.sqrt(num_samples)

            return {
                "lower_bound": mean - ci,
                "upper_bound": mean + ci,
                "mean": mean,
                "std": std,
            }

    def short_name(self) -> str:
        """Return the short name of the estimator.

        This method can be overridden by subclasses to customize the short name.

        :return: the short name of the estimator. By default, it returns the abbreviation of
            the class name, ie "IPW".
        """
        return re.sub("[^A-Z]", "", self.__class__.__name__)

    @abstractmethod
    def estimate_weighted_rewards(self) -> np.ndarray:
        """Estimate the weighted rewards.

        This method should be overridden by subclasses to implement the specific estimator.

        :return: the weighted rewards.
        """
        pass

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the value of the target policy.

        This method should be overridden by subclasses to implement the specific estimator. The typical implementation
        should call :meth:`estimate_weighted_rewards` to compute the weighted rewards, then compute the policy value.

        :return: the estimated value of the target policy.
        """
        pass


class InverseProbabilityWeighting(BaseEstimator):
    r"""Inverse Probability Weighting (IPW) estimator.

    :math:`V_{IPW}(\pi_e, D)=\frac {1}{n} \sum_{t=1}^n p(s_t,a_t) r_t`

    Where:
        - :math:`D` is the offline collected dataset.
        - :math:`p(s_t,a_t)` is the importance weight defined as :math:`p(s_t,a_t)=\frac{\pi_e(a_t|s_t)}{\pi_b(a_t|s_t)}`.
        - :math:`\pi_e` is the target policy and :math:`\pi_b` is the behavior policy.
        - :math:`r_t` is the reward observed at time :math:`t` for the behavior policy.
        - :math:`n` is the number of samples.

    This estimator has generally high variance, especially on small datasets, and can be improved by using
    self-normalized importance weights.

    .. rubric:: References

    https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs
    """

    def __init__(self) -> None:
        super().__init__()

        # computed importance weights, saved for reuse by subclasses
        # shape: (n, num_actions)
        self.importance_weights: np.ndarray | None = None

    @override(BaseEstimator)
    def estimate_weighted_rewards(self) -> np.ndarray:
        """Estimate the weighted rewards using the IPW estimator."""
        self.importance_weights = None
        self.check_parameters()

        self.importance_weights = (
            self.target_policy_action_probabilities / self.behavior_policy_action_probabilities
        )
        return self.importance_weights * self.rewards.reshape(-1, 1)

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        """Estimate the value of the target policy using the IPW estimator."""
        return np.mean(self.estimate_weighted_rewards())


class SelfNormalizedInverseProbabilityWeighting(InverseProbabilityWeighting):
    r"""Self-Normalized Inverse Probability Weighting (SNIPW) estimator.

    :math:`V_{SNIPW}(\pi_e, D)= \frac {\sum_{t=1}^n p(s_t,a_t) r_t}{\sum_{t=1}^n p(s_t,a_t)}`

    Where:
        - :math:`D` is the offline collected dataset.
        - :math:`p(s_t,a_t)` is the importance weight defined as :math:`p(s_t,a_t)=\frac{\pi_e(a_t|s_t)}{\pi_b(a_t|s_t)}`.
        - :math:`\pi_e` is the target policy and :math:`\pi_b` is the behavior policy.
        - :math:`r_t` is the reward at time :math:`t`.
        - :math:`n` is the number of samples.

    .. rubric:: References

    https://papers.nips.cc/paper_files/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html
    """

    def __init__(self) -> None:
        super().__init__()

    @override(BaseEstimator)
    def estimate_weighted_rewards(self) -> np.ndarray:
        """Estimate the weighted rewards using the SNIPW estimator."""
        super().estimate_weighted_rewards()

        weighted_rewards = self.importance_weights * self.rewards.reshape(-1, 1)
        return weighted_rewards / self.importance_weights

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        """Estimate the value of the target policy using the SNIPW estimator."""
        super().estimate_weighted_rewards()

        weighted_rewards = self.importance_weights * self.rewards.reshape(-1, 1)
        return np.sum(weighted_rewards) / np.sum(self.importance_weights)


class DirectMethod(BaseEstimator):
    r"""Direct Method (DM) estimator.

    :math:`V_{DM}(\pi_e, D, Q)=\frac {1}{n} \sum_{i=1}^n \sum_{a \in A} \pi_e(a|s^i_0) Q(s^i_0, a)`

    Where:
        - :math:`D = \{\{ (s_t, a_t, r_t) \}^{T-1}_{t=0}\}^n_{i=1}` is the offline collected dataset
          consisting of n trajectories.
        - :math:`\pi_e` is the target policy.
        - :math:`Q(s^i_0, a)` is the Q model trained to estimate the expected discounted sum of rewards
          from the initial state :math:`s^i_0` when taking action :math:`a` under the behavior policy.
        - :math:`n` is the number of episodes/trajectories.
        - :math:`a` is the action taken in the set of actions :math:`A`.
        - :math:`s^i_0` is the initial state of the i-th trajectory.

    This estimator trains a Q model using supervised learning on initial states and their corresponding
    discounted cumulative returns, then uses it to estimate the expected value under the target policy.
    The performance of this estimator depends on the quality of the Q model.
    """

    def __init__(
        self,
        q_model_cls: type[RegressionBasedRewardModel],
        behavior_policy_obs: np.ndarray,
        behavior_policy_act: np.ndarray,
        behavior_policy_rewards: np.ndarray,
        steps_per_episode: int,
        discount_factor: float = 1.0,
        q_model_type: str = "random_forest",
        q_model_params: dict | None = None,
    ) -> None:
        """Initialize the Direct Method estimator.

        :param q_model_cls: the class of the Q model to use.
        :param behavior_policy_obs: the observations for training the Q model, shape:
            (batch_size, obs_dim). These should be observations from all timesteps.
        :param behavior_policy_act: the actions for training the Q model, shape:
            (batch_size,). These should be actions from all timesteps.
        :param behavior_policy_rewards: the rewards for training the Q model, shape:
            (batch_size,). These should be rewards from all timesteps.
        :param steps_per_episode: the number of steps per episode. The number of samples
            must be divisible by this number.
        :param discount_factor: the discount factor for computing cumulative returns. Must
            be in [0, 1].
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
        assert 0 <= discount_factor <= 1, "The discount factor must be in [0, 1]."

        self.q_model_cls = q_model_cls
        self.q_model_type = q_model_type
        self.q_model_params = q_model_params
        self.q_model: RegressionBasedRewardModel | None = None
        self.behavior_policy_obs = behavior_policy_obs
        self.behavior_policy_act = behavior_policy_act
        self.behavior_policy_rewards = behavior_policy_rewards
        self.steps_per_episode = steps_per_episode
        self.discount_factor = discount_factor

    def fit(self) -> dict[str, float] | None:
        """Fit the Q model to estimate the expected discounted sum of rewards from the initial
        state.

        The Q model is trained on (initial_state, initial_action) pairs with their corresponding
        discounted cumulative returns computed as:

        :math:`G_0 = r_0 + γ*r_1 + γ²*r_2 + ... + γ^(T-1)*r_(T-1)`

        :return: the fit statistics of the Q model.
        """
        # Reshape data into episodes
        num_episodes = self.behavior_policy_obs.shape[0] // self.steps_per_episode
        obs_episodes = self.behavior_policy_obs.reshape(num_episodes, self.steps_per_episode, -1)
        act_episodes = self.behavior_policy_act.reshape(num_episodes, self.steps_per_episode)
        rew_episodes = self.behavior_policy_rewards.reshape(num_episodes, self.steps_per_episode)

        # Extract initial states and actions
        initial_obs = obs_episodes[:, 0, :]  # shape: (num_episodes, obs_dim)
        initial_act = act_episodes[:, 0]  # shape: (num_episodes,)

        # Compute discounted cumulative returns from initial state
        # G_0 = r_0 + γ*r_1 + γ²*r_2 + ... + γ^(T-1)*r_(T-1)
        discount_powers = np.power(self.discount_factor, np.arange(self.steps_per_episode))
        cumulative_returns = np.sum(
            rew_episodes * discount_powers, axis=1
        )  # shape: (num_episodes,)

        # Train Q model on initial states and cumulative returns
        self.q_model = self.q_model_cls(
            obs=initial_obs,
            act=initial_act,
            rew=cumulative_returns,
            regression_model=self.q_model_type,
            model_params=self.q_model_params,
        )
        return self.q_model.fit()

    def check_parameters(self) -> None:
        """Check if the estimator parameters are valid.

        Base estimator checks plus additional checks for the Q model.
        """
        super().check_parameters()

        assert (
            self.behavior_policy_obs.shape[0] == self.target_policy_action_probabilities.shape[0]
        ), "The number of samples must be the same for the behavior policy and the target policy."

    @override(BaseEstimator)
    def estimate_weighted_rewards(self) -> np.ndarray:
        """Estimate the weighted rewards using the Direct Method estimator.

        For each episode `i`, computes: :math:`V(s^i_0) = Σ_{a∈A} π_e(a|s^i_0) * Q(s^i_0, a)`

        Where :math:`Q(s^i_0, a)` is the predicted discounted cumulative return from the initial state :math:`s^i_0`.

        :return: the estimated values for each episode/trajectory.
        """
        self.check_parameters()

        # Extract initial states and their action probabilities
        num_episodes = self.behavior_policy_obs.shape[0] // self.steps_per_episode
        obs_episodes = self.behavior_policy_obs.reshape(num_episodes, self.steps_per_episode, -1)
        initial_obs = obs_episodes[:, 0, :]  # shape: (num_episodes, obs_dim)

        # Extract initial state action probabilities
        target_policy_probs_episodes = self.target_policy_action_probabilities.reshape(
            num_episodes, self.steps_per_episode, -1
        )
        initial_action_probs = target_policy_probs_episodes[
            :, 0, :
        ]  # shape: (num_episodes, num_actions)

        num_actions = initial_action_probs.shape[1]

        # Predict Q(s_0, a) for all actions at initial states
        # Expand initial states to evaluate all actions
        initial_obs_expanded = np.repeat(
            initial_obs, num_actions, axis=0
        )  # shape: (num_episodes * num_actions, obs_dim)
        all_actions = np.tile(
            np.arange(num_actions), num_episodes
        )  # shape: (num_episodes * num_actions,)

        # Get Q values for all (s_0, a) pairs
        q_values = self.q_model.estimate(
            obs=initial_obs_expanded,
            act=all_actions,
        )  # shape: (num_episodes * num_actions,)

        # Reshape Q values to (num_episodes, num_actions)
        q_values = q_values.reshape(num_episodes, num_actions)

        # Compute expected value: V(s_0) = Σ_a π_e(a|s_0) * Q(s_0, a)
        initial_state_value = np.sum(
            initial_action_probs * q_values, axis=1
        )  # shape: (num_episodes,)

        return initial_state_value

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        """Estimate the value of the target policy using the Direct Method estimator."""
        return np.mean(self.estimate_weighted_rewards())


class TrajectoryPerDecisionMixin(ABC):
    """Mixin for trajectory-wise and per-decision estimators.

    It provides a method to compute the weighted rewards for both trajectory-wise and per-
    decision estimators.
    """

    def compute_weighted_rewards(
        self,
        target_policy_action_probabilities: np.ndarray,
        behavior_policy_action_probabilities: np.ndarray,
        rewards: np.ndarray,
        steps_per_episode: int,
        discount_factor: float,
        is_per_decision: bool,
    ) -> np.ndarray:
        """Compute the weighted rewards for given type of estimator. See the specific estimator for
        more details.

        :param target_policy_action_probabilities: the probabilities of taking actions under
            the target policy.
        :param behavior_policy_action_probabilities: the probabilities of taking actions
            under the behavior policy.
        :param rewards: the rewards received under the behavior policy.
        :param steps_per_episode: the number of steps per episode.
        :param discount_factor: the discount factor.
        :param is_per_decision: whether the estimator is per decision or per trajectory.
        """

        # compute importance ratios
        importance_weights = (
            target_policy_action_probabilities / behavior_policy_action_probabilities
        )
        num_actions = target_policy_action_probabilities.shape[1]
        # shape: (n, T * num_actions)
        importance_weights = importance_weights.reshape(-1, steps_per_episode * num_actions)

        # compute the importance weights per decision, which is the cumulative product of the
        # importance weights over the trajectory; at each step t, we'll have the product from 0 to t-1.
        # shape: (n, T * num_actions)
        importance_weights = np.cumprod(importance_weights, axis=1)

        if not is_per_decision:
            # in trajectory-wise estimators, we only need the last importance weight, which is the product
            # over the whole trajectory.
            # shape: (n, 1)
            importance_weights = importance_weights[:, -1].reshape(-1, 1)

        # normalize the importance weights. Normalization technique depends on the estimator (see implementations).
        importance_weights = self.normalize(importance_weights)

        # rewards, shape: (n, T)
        rewards = rewards.reshape(-1, steps_per_episode)

        # discount factors
        # compute a matrix of discount factors, shape: (n, T)
        num_trajectories = rewards.shape[0]
        discount_factors = np.full((num_trajectories, steps_per_episode), discount_factor)
        # compute the discount factor at each step as
        # [gamma^0, gamma^1, ..., gamma^(T-1)] = [gamma^1, gamma^2, ..., gamma^T] / gamma
        discount_factors = np.cumprod(discount_factors, axis=1) / discount_factor

        # if the estimator is per decision, we need to repeat the discount factors and rewards for each action.
        # shape: (n, T * num_actions)
        if is_per_decision:
            discount_factors = np.tile(discount_factors, (1, num_actions))
            rewards = np.tile(rewards, (1, num_actions))

        # compute the weighted rewards per trajectory.
        # shape: (n, 1)
        weighted_rewards = np.sum(
            # element-wise product
            # trajectory wise: (n, 1) * (n, T) * (n, T) -> (n, T)
            # per decision: (n, T * num_actions) * (n, T * num_actions) * (n, T * num_actions) -> (n, T * num_actions)
            importance_weights * discount_factors * rewards,
            # sum weights over the trajectory length
            axis=1,
        ).reshape(-1, 1)

        return weighted_rewards

    def normalize(self, weights: np.ndarray) -> np.ndarray:
        """Normalize the importance weights. This method can be overridden by subclasses to
        implement a specific normalization strategy.

        :param weights: the importance weights to normalize.
        :return: the normalized importance weights.
        """
        return weights


class TrajectoryWiseImportanceSampling(BaseEstimator, TrajectoryPerDecisionMixin):
    r"""Trajectory-wise Importance Sampling (TIS) estimator.

    :math:`V_{TIS} (\pi_e, D) = \frac {1}{n} \sum_{i=1}^ n\sum_{t=0}^{T-1} \gamma^t w^{(i)}_{0:T-1} r_t^{(i)}`

    Where:

    - :math:`D` is the offline collected dataset.
    - :math:`w^{(i)}_{0:T-1}` is the importance weight of the trajectory :math:`i` defined as :math:`w_{0:T-1} = \prod_{t=0}^{T-1} \frac {\pi_e(a_t|s_t)} {\pi_b(a_t|s_t)}`
    - :math:`\pi_e` is the target policy and :math:`\pi_b` is the behavior policy.
    - :math:`n` is the number of trajectories.
    - :math:`T` is the length of the trajectory.
    - :math:`\gamma_t` is the discount factor at time :math:`t`.
    - :math:`r_t^{(i)}` is the reward at time :math:`t` of trajectory :math:`i`.

    TIS can suffer from high variance due to the product operation of the importance weights, also when action space is
    large.

    .. rubric:: References

    https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs
    """

    def __init__(self, steps_per_episode: int, discount_factor: float = 1.0) -> None:
        super().__init__()

        assert steps_per_episode > 0, "The number of steps per episode must be positive."
        assert 0 <= discount_factor <= 1, "The discount factor must be in [0, 1]."

        self.steps_per_episode = steps_per_episode
        self.discount_factor = discount_factor

        self.importance_weights: np.ndarray | None = None

    @override(BaseEstimator)
    def short_name(self) -> str:
        return "TIS"

    @override(BaseEstimator)
    def check_parameters(self) -> None:
        """Check if the estimator parameters are valid."""
        super().check_parameters()

        assert (
            self.target_policy_action_probabilities.shape[0] % self.steps_per_episode == 0
        ), "The number of samples must be divisible by the number of steps per episode."

    @override(BaseEstimator)
    def estimate_weighted_rewards(self) -> np.ndarray:
        """Estimate the weighted rewards using the Trajectory-wise Importance Sampling estimator.

        :return: the weighted rewards, or here the policy value per trajectory.
        """
        self.check_parameters()

        return self.compute_weighted_rewards(
            target_policy_action_probabilities=self.target_policy_action_probabilities,
            behavior_policy_action_probabilities=self.behavior_policy_action_probabilities,
            rewards=self.rewards,
            steps_per_episode=self.steps_per_episode,
            discount_factor=self.discount_factor,
            is_per_decision=False,
        )

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        """Estimate the value of the target policy using the Trajectory-wise Importance Sampling
        estimator."""
        return np.mean(self.estimate_weighted_rewards())


class SelfNormalizedTrajectoryWiseImportanceSampling(TrajectoryWiseImportanceSampling):
    r"""Self-Normalized Trajectory-wise Importance Sampling (TIS) estimator.

    .. math::
        V_{TIS} (\pi_e, D) = \frac {1}{n} \sum_{i=1}^n \sum_{t=0}^{T-1}
            \gamma^t \frac {w^{(i)}_{0:T-1}} {\frac {1}{n} \sum_{j=1}^n w^{(j)}_{0:T-1}} r_t^{(i)}

    Where:

    - :math:`D` is the offline collected dataset.
    - :math:`w^{(i)}_{0:T-1}` is the importance weight of the trajectory :math:`i` defined as :math:`w_{0:T-1} = \prod_{t=0}^{T-1} \frac {\pi_e(a_t|s_t)} {\pi_b(a_t|s_t)}`
    - :math:`\pi_e` is the target policy and :math:`\pi_b` is the behavior policy.
    - :math:`n` is the number of trajectories.
    - :math:`T` is the length of the trajectory.
    - :math:`\gamma_t` is the discount factor at time :math:`t`.
    - :math:`r_t^{(i)}` is the reward at time :math:`t` of trajectory :math:`i`.

    SNTIS is a variance reduction technique for TIS. It divides the weighted rewards by the mean of the importance
    weights of the trajectories.

    .. rubric:: References

    https://arxiv.org/abs/1906.03735
    """

    @override(TrajectoryWiseImportanceSampling)
    def short_name(self) -> str:
        return "SNTIS"

    @override(TrajectoryPerDecisionMixin)
    def normalize(self, weights: np.ndarray) -> np.ndarray:
        """Normalize the importance weights using the self-normalization strategy.

        It uses self-normalization to reduce the variance of the estimator, using the mean
        of the importance weights over the trajectories.

        :param weights: the importance weights to normalize.
        :return: the normalized importance weights.
        """
        return weights / (np.mean(weights) + 1e-10)


class PerDecisionImportanceSampling(BaseEstimator, TrajectoryPerDecisionMixin):
    r"""Per-Decision Importance Sampling (PDIS) estimator.

    :math:`V_{PDIS} (\pi_e, D) = \frac {1}{n} \sum_{i=1}^n \sum_{t=0}^{T-1} \gamma^t w^{(i)}_{t} r_t^{(i)}`

    Where:

    - :math:`D` is the offline collected dataset.
    - :math:`w^{(i)}_{t}` is the importance weight of the decision :math:`t` of trajectory :math:`i` defined as :math:`w_{t} = \frac {\pi_e(a_t|s_t)} {\pi_b(a_t|s_t)}`
    - :math:`\pi_e` is the target policy and :math:`\pi_b` is the behavior policy.
    - :math:`n` is the number of trajectories.
    - :math:`T` is the length of the trajectory.
    - :math:`\gamma_t` is the discount factor at time :math:`t`.
    - :math:`r_t^{(i)}` is the reward at time :math:`t` of trajectory :math:`i`.

    .. rubric:: References

    https://arxiv.org/abs/1906.03735
    """

    def __init__(self, steps_per_episode: int, discount_factor: float = 1.0) -> None:
        super().__init__()

        assert steps_per_episode > 0, "The number of steps per episode must be positive."
        assert 0 <= discount_factor <= 1, "The discount factor must be in [0, 1]."

        self.steps_per_episode = steps_per_episode
        self.discount_factor = discount_factor

        self.importance_weights: np.ndarray | None = None

    @override(BaseEstimator)
    def check_parameters(self) -> None:
        """Check if the estimator parameters are valid."""
        super().check_parameters()

        assert (
            self.target_policy_action_probabilities.shape[0] % self.steps_per_episode == 0
        ), "The number of samples must be divisible by the number of steps per episode."

    @override(BaseEstimator)
    def estimate_weighted_rewards(self) -> np.ndarray:
        """Estimate the weighted rewards using the Trajectory-wise Importance Sampling estimator.

        :return: the weighted rewards, or here the policy value per trajectory.
        """
        self.check_parameters()

        return self.compute_weighted_rewards(
            target_policy_action_probabilities=self.target_policy_action_probabilities,
            behavior_policy_action_probabilities=self.behavior_policy_action_probabilities,
            rewards=self.rewards,
            steps_per_episode=self.steps_per_episode,
            discount_factor=self.discount_factor,
            is_per_decision=True,
        )

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        """Estimate the value of the target policy using the Trajectory-wise Importance Sampling
        estimator."""
        return np.mean(self.estimate_weighted_rewards())


class SelfNormalizedPerDecisionImportanceSampling(PerDecisionImportanceSampling):
    r"""Self-Normalized Per-Decision Importance Sampling (PDIS) estimator.

    .. math::
        V_{PDIS} (\pi_e, D) = \frac {1}{n} \sum_{i=1}^n \sum_{t=0}^{T-1}
            \gamma^t \frac {w^{(i)}_{t}} {\frac {1}{n} \sum_{j=1}^n w^{(j)}_{t}} r_t^{(i)}

    Where:

    - :math:`D` is the offline collected dataset.
    - :math:`w^{(i)}_{t}` is the importance weight of the decision :math:`t` of trajectory :math:`i` defined as :math:`w_{t} = \frac {\pi_e(a_t|s_t)} {\pi_b(a_t|s_t)}`
    - :math:`\pi_e` is the target policy and :math:`\pi_b` is the behavior policy.
    - :math:`n` is the number of trajectories.
    - :math:`T` is the length of the trajectory.
    - :math:`\gamma_t` is the discount factor at time :math:`t`.
    - :math:`r_t^{(i)}` is the reward at time :math:`t` of trajectory :math:`i`.

    SNPDIS is a variance reduction technique for PDIS.

    .. rubric:: References

    https://arxiv.org/abs/1906.03735
    """

    @override(TrajectoryPerDecisionMixin)
    def normalize(self, weights: np.ndarray) -> np.ndarray:
        """Normalize the importance weights using the self-normalization strategy.

        It uses self-normalization to reduce the variance of the estimator, using the mean
        of the importance weights over the trajectories.

        :param weights: the importance weights to normalize.
        :return: the normalized importance weights.
        """
        return weights / (np.mean(weights) + 1e-10)


class WeightedPerDecisionImportanceSampling(BaseEstimator):
    """Weighted Per-Decision Importance Sampling (WPDIS) estimator. Computing the WPDIS estimate on
    daily trajectories using per-decision importance weights. Step-level arrays rare reshaped into
    (episode, timestep) format, builds per-step ratios from behavior vs new propensities, and
    applies the stickiness correction by setting ratios to 1 after the first activation.

    This estimator computes the self-normalized per-decision IS estimate over all episode-steps:

        rho_{i,t} = pi_e(a_{i,t} | s_{i,t}) / pi_b(a_{i,t} | s_{i,t})
        W_{i,t}   = Π_{k<=t} rho_{i,k}

        V = (Σ_i Σ_t W_{i,t} r_{i,t}) / (Σ_i Σ_t W_{i,t})

    Optionally applies "stickiness correction" to rho: after the first sticky switch (sticky_action==1),
    ratios are set to 1.0 for subsequent timesteps.

    Notes
    -----
    - Requires logged actions to extract taken propensities from the (N, A) probability matrices.
    - For confidence intervals, uses episode-level bootstrap (resampling episodes with replacement),
      which matches typical trajectory-level OPE uncertainty estimation.
    """

    def __init__(
        self,
        *,
        steps_per_episode: int,
        eps: float = 1e-12,
        clip: float | None = None,
        apply_stickiness: bool = True,
        stickiness_value_after_switch: float = 1.0,
    ) -> None:
        super().__init__()
        assert steps_per_episode > 0, "steps_per_episode must be > 0"
        assert eps > 0.0, "eps must be > 0"

        if clip is not None:
            assert clip > 1.0, "clip must be > 1.0"

        self.steps_per_episode = steps_per_episode
        self.eps = eps
        self.clip = clip
        self.apply_stickiness = apply_stickiness
        self.stickiness_value_after_switch = stickiness_value_after_switch

        # Extra required logged data (not part of BaseEstimator)
        self.logged_actions: np.ndarray | None = None  # shape (N,)
        self.logged_sticky_actions: np.ndarray | None = None  # shape (N,)

    def set_logged_data(
        self,
        *,
        actions: np.ndarray,
        sticky_actions: np.ndarray | None = None,
    ) -> None:
        """Set logged actions (and optionally sticky actions) needed for taken-propensity
        extraction and correction.

        Args:
            actions: logged actions a_t, shape (N,)
            sticky_actions: sticky action applied (0/1), shape (N,)
        """
        self.logged_actions = np.asarray(actions)
        if sticky_actions is not None:
            self.logged_sticky_actions = np.asarray(sticky_actions)
        else:
            self.logged_sticky_actions = None

    @override(BaseEstimator)
    def short_name(self) -> str:
        return "WPDIS"

    @override(BaseEstimator)
    def check_parameters(self) -> None:
        """
        Override base checks because:
        - we need logged actions and optionally sticky actions
        - and we need divisibility by steps_per_episode
        """
        if self.target_policy_action_probabilities is None:
            raise ValueError("target_policy_action_probabilities not set")
        if self.behavior_policy_action_probabilities is None:
            raise ValueError("behavior_policy_action_probabilities not set")
        if self.rewards is None:
            raise ValueError("rewards not set")

        # Prob matrices must be 2D and sum to 1 row-wise
        for arr, name in [
            (self.target_policy_action_probabilities, "target_policy_action_probabilities"),
            (self.behavior_policy_action_probabilities, "behavior_policy_action_probabilities"),
        ]:
            check_array(array=arr, name=name, expected_ndim=2, expected_dtype=(float, np.float32))

            row_sums = np.sum(arr, axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-5):
                raise ValueError(
                    f"{name} rows must sum to 1 (min={row_sums.min()}, max={row_sums.max()})"
                )

            if np.any(arr <= 0):
                raise ValueError(f"{name} must be strictly positive (for numerical stability)")

        # Rewards must be 1D
        check_array(
            array=self.rewards, name="rewards", expected_ndim=1, expected_dtype=(float, np.float32)
        )

        N = self.rewards.shape[0]
        if self.target_policy_action_probabilities.shape[0] != N:
            raise ValueError("target_policy_action_probabilities and rewards must have same N")
        if self.behavior_policy_action_probabilities.shape[0] != N:
            raise ValueError("behavior_policy_action_probabilities and rewards must have same N")

        if N % self.steps_per_episode != 0:
            raise ValueError("N must be divisible by steps_per_episode")

        if self.logged_actions is None:
            raise ValueError("logged actions not set. Call set_logged_data(actions=...)")

        self.logged_actions = np.asarray(self.logged_actions, dtype=np.int64).reshape(-1)
        if self.logged_actions.shape[0] != N:
            raise ValueError("logged actions length must match N")

        if self.apply_stickiness:
            if self.logged_sticky_actions is None:
                raise ValueError(
                    "stickiness is enabled but sticky_actions is None. Provide it in set_logged_data()."
                )
            self.logged_sticky_actions = np.asarray(
                self.logged_sticky_actions, dtype=np.int64
            ).reshape(-1)
            if self.logged_sticky_actions.shape[0] != N:
                raise ValueError("sticky_actions length must match N")

    def _compute_r_and_rho_day(self) -> tuple[np.ndarray, np.ndarray]:
        """Build (n_eps, T) reward matrix and (n_eps, T) per-step ratios rho for the taken action.

        Applies clipping and optional stickiness correction.
        """
        self.check_parameters()

        N = self.rewards.shape[0]
        n_eps = N // self.steps_per_episode
        T = self.steps_per_episode

        # reshape rewards
        r_day = self.rewards.reshape(n_eps, T).astype(np.float32)

        # taken propensities from full distributions + logged actions
        a = self.logged_actions
        idx = np.arange(N, dtype=np.int64)

        p_b_taken = self.behavior_policy_action_probabilities[idx, a].astype(np.float32)
        p_e_taken = self.target_policy_action_probabilities[idx, a].astype(np.float32)

        pb_day = p_b_taken.reshape(n_eps, T)
        pe_day = p_e_taken.reshape(n_eps, T)

        rho = pe_day / np.maximum(pb_day, self.eps)

        # optional clipping
        if self.clip is not None:
            rho = np.clip(rho, 1.0 / self.clip, self.clip)

        # optional stickiness correction
        if self.apply_stickiness:
            sticky_day = self.logged_sticky_actions.reshape(n_eps, T)
            rho = apply_stickiness_correction_to_rho(
                rho=rho,
                sticky_act_day=sticky_day,
                value_after_switch=self.stickiness_value_after_switch,
            )

        return r_day, rho

    @override(BaseEstimator)
    def estimate_weighted_rewards(self) -> np.ndarray:
        """Return per-episode WPDIS numerator contribution (not yet normalized by global denom).

        We return an (n_eps, 1) vector to be compatible with downstream patterns. The final
        policy value uses the global self-normalization denom.
        """
        r_day, rho = self._compute_r_and_rho_day()
        W = np.cumprod(rho, axis=1)  # (n_eps, T)

        # per-episode numerator contribution Σ_t W_{i,t} r_{i,t}
        num_i = np.sum(W * r_day, axis=1).reshape(-1, 1).astype(np.float32)
        return num_i

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        """
        Compute the self-normalized WPDIS estimate:
            V = Σ_i Σ_t W_{i,t} r_{i,t} / Σ_i Σ_t W_{i,t}
        """
        r_day, rho = self._compute_r_and_rho_day()
        W = np.cumprod(rho, axis=1)

        num = float(np.sum(W * r_day))
        den = float(np.sum(W))

        return float(num / np.maximum(den, self.eps))

    def estimate_policy_value_with_confidence_interval(
        self,
        *,
        n_boot: int = 2000,
        alpha: float = 0.05,
        seed: int = 0,
    ) -> dict[str, float]:
        """Episode-level bootstrap CI for WPDIS.

        Returns:
            {"mean": ..., "lower_bound": ..., "upper_bound": ...}
        """
        assert n_boot > 0, "n_boot must be > 0"
        assert 0.0 < alpha < 1.0, "alpha must be in (0,1)"

        r_day, rho = self._compute_r_and_rho_day()
        n_eps = r_day.shape[0]

        rng = np.random.default_rng(seed)
        vals = np.empty(n_boot, dtype=np.float32)

        for b in range(n_boot):
            idx = rng.integers(0, n_eps, size=n_eps)
            r_b = r_day[idx]
            rho_b = rho[idx]

            W = np.cumprod(rho_b, axis=1)
            num = np.sum(W * r_b)
            den = np.sum(W)
            vals[b] = float(num / np.maximum(den, self.eps))

        lo = float(np.quantile(vals, alpha / 2))
        hi = float(np.quantile(vals, 1 - alpha / 2))

        return {
            "mean": float(vals.mean()),
            "lower_bound": lo,
            "upper_bound": hi,
        }


# class WeightedPerDecisionImportanceSampling:
#     """
#     Weighted Per-Decision Importance Sampling (WPDIS) estimator.
#     Computing the WPDIS estimate on daily trajectories using per-decision importance weights.
#     Step-level arrays rare reshaped into (episode, timestep) format, builds per-step ratios from behavior
#     vs new propensities, and applies the stickiness correction by setting ratios to 1 after the first activation.
#
#     This implementation matches the notebook logic:
#         - Optional clipping on per-step importance ratios rho
#         - Per-decision cumulative weights:
#               W_{i,t} = Π_{k<=t} rho_{i,k}
#         - Self-normalized estimate over all episode-steps:
#               V = Σ_i Σ_t W_{i,t} r_{i,t} / Σ_i Σ_t W_{i,t}
#
#     Bootstrapping is performed at the episode level by resampling episodes with replacement.
#
#     Parameters
#     ----------
#     eps:
#         Small constant to avoid division by zero in the normalization.
#     """
#
#     def __init__(self, eps: float = 1e-12) -> None:
#         assert eps > 0, "eps must be > 0"
#         self.eps = eps
#
#     def short_name(self) -> str:
#         return "WPDIS"
#
#     def estimate(
#         self,
#         r: np.ndarray,
#         rho: np.ndarray,
#         clip: float | None = None,
#     ) -> float:
#         """
#         Compute the WPDIS point estimate.
#
#         Parameters
#         ----------
#         r:
#             Rewards, shape (n_eps, T).
#         rho:
#             Per-step importance ratios for the logged actions, shape (n_eps, T).
#         clip:
#             If provided, clip rho to [1/clip, clip].
#
#         Returns
#         -------
#         float
#             WPDIS estimate.
#         """
#         r = np.asarray(r, dtype=np.float32)
#         rho = np.asarray(rho, dtype=np.float32)
#
#         assert r.ndim == 2 and rho.ndim == 2, "r and rho must be 2D arrays (n_eps, T)"
#         assert r.shape == rho.shape, "r and rho must have the same shape"
#
#         if clip is not None:
#             assert clip > 1.0, "clip must be > 1.0"
#             rho = np.clip(rho, 1.0 / clip, clip)
#
#         W = np.cumprod(rho, axis=1)  # (n_eps, T)
#         num = np.sum(W * r)
#         den = np.sum(W)
#
#         return float(num / np.maximum(den, self.eps))
#
#     def estimate_with_ci(
#         self,
#         r: np.ndarray,
#         rho: np.ndarray,
#         clip: float = 20.0,
#         n_boot: int = 2000,
#         alpha: float = 0.05,
#         seed: int = 0,
#     ) -> dict[str, float]:
#         """
#         Estimate WPDIS with an episode-level bootstrap confidence interval.
#         Estimating uncertainty on the WPDIS value using bootstrap resampling over episodes.
#         By repeatedly resampling episodes with replacement, it provides an empirical confidence
#         interval for the off-policy return estimate.
#         Parameters
#         ----------
#         r:
#             Rewards, shape (n_eps, T).
#         rho:
#             Per-step importance ratios, shape (n_eps, T).
#         clip:
#             Clipping for rho in [1/clip, clip].
#         n_boot:
#             Number of bootstrap resamples.
#         alpha:
#             Significance level (e.g., 0.05 for 95% CI).
#         seed:
#             RNG seed.
#
#         Returns
#         -------
#         dict[str, float]
#                 A dictionary containing:
#                 - "mean": the mean WPDIS estimate across bootstrap samples.
#                 - "lower_bound": the lower bound of the confidence interval.
#                 - "upper_bound": the upper bound of the confidence interval.
#         """
#
#         assert n_boot > 0, "n_boot must be > 0"
#         assert 0.0 < alpha < 1.0, "alpha must be in (0, 1)"
#
#         r = np.asarray(r, dtype=np.float32)
#         rho = np.asarray(rho, dtype=np.float32)
#
#         assert r.ndim == 2 and rho.ndim == 2
#         assert r.shape == rho.shape
#
#         rng = np.random.default_rng(seed)
#         n = r.shape[0]
#         vals = np.empty(n_boot, dtype=np.float32)
#
#         for b in range(n_boot):
#             idx = rng.integers(0, n, size=n)
#             vals[b] = self.estimate(r[idx], rho[idx], clip=clip)
#
#         lo = float(np.quantile(vals, alpha / 2))
#         hi = float(np.quantile(vals, 1 - alpha / 2))
#
#         return {
#             "mean": float(vals.mean()),
#             "lower_bound": lo,
#             "upper_bound": hi,
#         }


### IS trajectory-wise
class StickyTrajectoryWiseIS(BaseEstimator):
    """
    Computing trajectory-level IS and self-normalized IS (SNIS) estimates for daily returns with a correction for action stickiness.
    Importance ratios are adjusted to account for action stickiness: after the first HVAC activation,
    policies are assumed to coincide and ratios are set to 1.

    For each episode i:
        W_i = ∏_t ρ_{i,t}

    where:
        ρ_{i,t} = π_e(a_t|s_t) / π_b(a_t|s_t)

    After the first sticky switch, ρ is set to 1 to avoid
    accumulating ratios in forced-control regions.
    """

    def __init__(
        self, *, steps_per_episode: int, eps: float = 1e-12, log_weight_clip: float = 50.0
    ):
        super().__init__()
        self.steps_per_episode = steps_per_episode
        self.eps = eps
        self.log_weight_clip = log_weight_clip

        self.p_e_taken_flat = None
        self.p_b_taken_flat = None
        self.rew_flat = None
        self.sticky_act_flat = None

    def set_parameters(
        self,
        *,
        p_e_taken_flat: np.ndarray,
        p_b_taken_flat: np.ndarray,
        rew_flat: np.ndarray,
        sticky_act_flat: np.ndarray,
    ) -> None:
        self.p_e_taken_flat = p_e_taken_flat
        self.p_b_taken_flat = p_b_taken_flat
        self.rew_flat = rew_flat
        self.sticky_act_flat = sticky_act_flat

    def estimate_weighted_rewards(self) -> np.ndarray:
        """
        Returns episode-level weighted returns: (num_eps, 1)
        so that BaseEstimator CI utilities (bootstrap / t-test) can operate.
        """
        if any(
            v is None
            for v in [
                self.p_e_taken_flat,
                self.p_b_taken_flat,
                self.rew_flat,
                self.sticky_act_flat,
            ]
        ):
            raise ValueError("Estimator parameters not set.")

        N = self.rew_flat.shape[0]
        if N % self.steps_per_episode != 0:
            raise ValueError("Total steps not divisible by steps_per_episode")

        num_eps = N // self.steps_per_episode

        r_day = self.rew_flat.reshape(num_eps, self.steps_per_episode)
        pb_day = self.p_b_taken_flat.reshape(num_eps, self.steps_per_episode)
        pe_day = self.p_e_taken_flat.reshape(num_eps, self.steps_per_episode)
        sticky_day = self.sticky_act_flat.reshape(num_eps, self.steps_per_episode)

        rho = pe_day / np.maximum(pb_day, self.eps)
        rho = apply_stickiness_correction_to_rho(rho, sticky_day)

        logW = np.sum(np.log(np.maximum(rho, self.eps)), axis=1)
        logW = np.clip(logW, -self.log_weight_clip, self.log_weight_clip)
        W = np.exp(logW)

        G = r_day.sum(axis=1)  # episode return

        # Return shape (num_eps, 1) like the other trajectory estimators
        return (W * G).reshape(-1, 1).astype(np.float32)

    def estimate_policy_value(self) -> float:
        return float(np.mean(self.estimate_weighted_rewards()))

    def estimate_self_normalized_value(self) -> float:
        """Self-normalized trajectory-wise IS (SNIS)"""

        N = self.rew_flat.shape[0]
        num_eps = N // self.steps_per_episode

        r_day = self.rew_flat.reshape(num_eps, self.steps_per_episode)
        pb_day = self.p_b_taken_flat.reshape(num_eps, self.steps_per_episode)
        pe_day = self.p_e_taken_flat.reshape(num_eps, self.steps_per_episode)
        sticky_day = self.sticky_act_flat.reshape(num_eps, self.steps_per_episode)

        rho = pe_day / np.maximum(pb_day, self.eps)
        rho = apply_stickiness_correction_to_rho(rho, sticky_day)

        logW = np.sum(np.log(np.maximum(rho, self.eps)), axis=1)
        logW = np.clip(logW, -self.log_weight_clip, self.log_weight_clip)
        W = np.exp(logW)

        G = r_day.sum(axis=1)

        return float(np.sum(W * G) / (np.sum(W) + self.eps))


class StickySequentialDoublyRobust(BaseEstimator):
    """Computing a per-decision DR estimate using a TD-style formulation. Building cumulative
    importance weights from behavior vs new propensities, applies the stickiness correction by
    setting ratios to 1 after the first activation, and then aggregates the weighted TD residuals
    over the episode to estimate the policy value.

    V_DR = V_hat(s0) + Σ_t W_t * (r_t + γ V_hat(s_{t+1}) - Q(s_t,a_t))

    where
        W_t = ∏_{k≤t} rho_k
        rho_t = π_e(a_t|s_t) / π_b(a_t|s_t)

    After the first sticky switch, rho is set to 1.
    """

    def __init__(
        self,
        *,
        steps_per_episode: int,
        gamma: float = 1.0,
        cap: float = 20.0,
        eps: float = 1e-12,
        log_weight_clip: float = 50.0,
    ):
        super().__init__()
        self.steps_per_episode = steps_per_episode
        self.gamma = gamma
        self.cap = cap
        self.eps = eps
        self.log_weight_clip = log_weight_clip

        self.rew_flat = None
        self.act_flat = None
        self.p_b_taken_flat = None
        self.P_new = None
        self.sticky_act_flat = None
        self.Q0 = None
        self.Q1 = None

    @override(BaseEstimator)
    def set_parameters(
        self,
        *,
        rew_flat: np.ndarray,
        act_flat: np.ndarray,
        p_b_taken_flat: np.ndarray,
        P_new: np.ndarray,
        sticky_act_flat: np.ndarray,
        Q0: np.ndarray,
        Q1: np.ndarray,
    ) -> None:
        self.rew_flat = np.asarray(rew_flat, dtype=np.float32).reshape(-1)
        self.act_flat = np.asarray(act_flat, dtype=np.int64).reshape(-1)
        self.p_b_taken_flat = np.asarray(p_b_taken_flat, dtype=np.float32).reshape(-1)
        self.P_new = np.asarray(P_new, dtype=np.float32)
        self.sticky_act_flat = np.asarray(sticky_act_flat, dtype=np.int64).reshape(-1)
        self.Q0 = np.asarray(Q0, dtype=np.float32).reshape(-1)
        self.Q1 = np.asarray(Q1, dtype=np.float32).reshape(-1)

        N = self.rew_flat.shape[0]
        if any(
            arr.shape[0] != N
            for arr in [self.act_flat, self.p_b_taken_flat, self.sticky_act_flat, self.Q0, self.Q1]
        ):
            raise ValueError("All flat arrays must have same length N")
        if self.P_new.shape[0] != N or self.P_new.shape[1] != 2:
            raise ValueError("P_new must have shape (N, 2)")
        if N % self.steps_per_episode != 0:
            raise ValueError("Total steps not divisible by steps_per_episode")

    def estimate_components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (rho, W_t, dr_episode) exactly like the notebook.

        Shapes:
          rho: (num_eps, T)
          W_t: (num_eps, T)
          dr_episode: (num_eps,)
        """
        if any(
            v is None
            for v in [
                self.rew_flat,
                self.act_flat,
                self.p_b_taken_flat,
                self.P_new,
                self.sticky_act_flat,
                self.Q0,
                self.Q1,
            ]
        ):
            raise ValueError("Estimator parameters not set.")

        N = self.rew_flat.shape[0]
        T = self.steps_per_episode
        num_eps = N // T

        r_day = self.rew_flat.reshape(num_eps, T)
        a_day = self.act_flat.reshape(num_eps, T)
        sticky_day = self.sticky_act_flat.reshape(num_eps, T)

        pb_day = self.p_b_taken_flat.reshape(num_eps, T)

        idx = np.arange(N)
        p_e_taken_flat = self.P_new[idx, self.act_flat]  # (N,)
        pe_day = p_e_taken_flat.reshape(num_eps, T)

        rho = pe_day / np.maximum(pb_day, self.eps)
        rho = apply_stickiness_correction_to_rho(rho, sticky_day)
        rho = np.clip(rho, 1.0 / self.cap, self.cap)

        # W_t are cumulative products of per-step ratios --> W_t = prod_{k=0..t} rho_k (log-stable)
        # They are the main source of variance in trajectory-based OPE
        log_rho = np.log(np.maximum(rho, self.eps))
        logW = np.cumsum(log_rho, axis=1)
        W_t = np.exp(np.clip(logW, -self.log_weight_clip, self.log_weight_clip)).astype(np.float32)

        # Q predictions reshaped
        Q0_day = self.Q0.reshape(num_eps, T)
        Q1_day = self.Q1.reshape(num_eps, T)
        Q_taken = np.where(a_day == 0, Q0_day, Q1_day).astype(np.float32)

        # V_hat(s_t) = sum_a pi_new(a|s_t) Q(s_t,a)
        V_hat = (self.P_new[:, 0] * self.Q0 + self.P_new[:, 1] * self.Q1).astype(np.float32)
        V_hat_day = V_hat.reshape(num_eps, T)

        # V_next = V(s_{t+1}) (last step -> 0)
        V_next = np.concatenate(
            [V_hat_day[:, 1:], np.zeros((num_eps, 1), dtype=np.float32)],
            axis=1,
        )

        # DR per episode (daily)
        # TD residual (r + gamma*V_next - Q_taken) is used for the DR correction term
        dr_episode = (
            V_hat_day[:, 0] + np.sum(W_t * (r_day + self.gamma * V_next - Q_taken), axis=1)
        ).astype(np.float32)

        return rho.astype(np.float32), W_t, dr_episode

    def estimate_weighted_rewards(self) -> np.ndarray:
        """Returns per-episode DR estimates shaped (num_eps, 1), consistent with BaseEstimator
        expectations."""
        rho, W_t, dr_episode = self.estimate_components()
        return np.asarray(dr_episode, dtype=np.float32).reshape(-1, 1)

    def estimate_policy_value(self) -> float:
        return float(np.mean(self.estimate_weighted_rewards()))
