import re
from abc import ABC, abstractmethod

import numpy as np
import scipy

from hopes.assert_utils import check_array
from hopes.dev_utils import override
from hopes.rew.rewards import RegressionBasedRewardModel


class BaseEstimator(ABC):
    """Base class for all estimators."""

    def __init__(self):
        self.target_policy_action_probabilities: np.ndarray | None = None
        self.behavior_policy_action_probabilities: np.ndarray | None = None
        self.rewards: np.ndarray | None = None
        self.importance_ratios: np.ndarray | None = None

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

    def set_importance_ratios(self, importance_ratios: np.ndarray | None) -> None:
        """Set precomputed step-wise importance ratios used by the estimator. This is useful when
        the taken-action ratios are built in a preprocessing step, for example after applying
        stickiness correction outside the estimator.

        param importance_ratios: Precomputed importance ratios for the logged actions.
            Supported shapes are ``(n_samples,)`` or ``(n_episodes, steps_per_episode)``.
        :raises ValueError: If the shape of ``importance_ratios`` is invalid.
        """

        if importance_ratios is None:
            self.importance_ratios = None
        else:
            self.importance_ratios = np.asarray(importance_ratios, dtype=np.float32)

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

        if self.importance_ratios is not None:
            if self.importance_ratios.ndim not in (1, 2):
                raise ValueError("importance_ratios must be 1D or 2D.")

            if self.rewards is not None:
                n_samples = self.rewards.shape[0]

                if (
                    self.importance_ratios.ndim == 1
                    and self.importance_ratios.shape[0] != n_samples
                ):
                    raise ValueError("1D importance_ratios length must match rewards length.")

                if self.importance_ratios.ndim == 2:
                    if (
                        self.importance_ratios.shape[0] * self.importance_ratios.shape[1]
                        != n_samples
                    ):
                        raise ValueError("2D importance_ratios size must match rewards length.")

    def _bootstrap_sample_policy_value(
        self,
        weighted_rewards: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        """Return one bootstrap estimate of the policy value.

        By default, we assume that the estimator can be expressed as the mean of per-episode
        weighted rewards. In that case, a bootstrap sample is obtained by resampling these
        values and taking their mean.

        Subclasses can override this method when the estimator is not a simple average of
        independent episode-level contributions (e.g. ratio estimators with a global
        normalization term).

        :param weighted_rewards: Episode-level weighted rewards.
        :param rng: Random number generator for reproducibility.
        :return: One bootstrap estimate of the policy value.
        """
        return float(
            np.mean(rng.choice(weighted_rewards, size=weighted_rewards.shape[0], replace=True))
        )

    def estimate_policy_value_with_confidence_interval(
        self,
        method: str = "bootstrap",
        significance_level: float = 0.05,
        num_samples: int = 1000,
        random_state: int | None = None,
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
        :param random_state: the random state to use for reproducibility. Only used when `method` is "bootstrap".
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
            # set the random state for reproducibility when using the bootstrap method
            rng = np.random.default_rng(random_state)

            # Delegate the computation of each bootstrap sample to a hook.
            # This allows subclasses to override only the statistic computation
            # without duplicating the whole CI logic.
            boot_samples = [
                self._bootstrap_sample_policy_value(weighted_rewards, rng)
                for _ in np.arange(num_samples)
            ]

            lower_bound = np.quantile(boot_samples, significance_level / 2)
            upper_bound = np.quantile(boot_samples, 1 - significance_level / 2)

            return {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "mean": float(np.mean(boot_samples)),
                "std": float(np.std(boot_samples)),
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
                "lower_bound": float(mean - ci),
                "upper_bound": float(mean + ci),
                "mean": float(mean),
                "std": float(std),
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
        importance_ratios: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute weighted rewards for trajectory-wise or per-decision estimators.
        When `importance_ratios` is provided, the estimator uses these precomputed
        step-wise ratios directly instead of recomputing them from policy probabilities.

        :param target_policy_action_probabilities: Target policy action probabilities,
            shape ``(n_samples, n_actions)``.
        :param behavior_policy_action_probabilities: Behavior policy action probabilities,
            shape ``(n_samples, n_actions)``.
        :param rewards: Logged rewards, shape ``(n_samples,)``.
        :param steps_per_episode: Number of steps per episode.
        :param discount_factor: Discount factor in ``[0, 1]``.
        :param is_per_decision: Whether to compute per-decision or trajectory-wise weighting.
        :param importance_ratios: Optional precomputed step-wise importance ratios for the
            logged actions. Supported shapes are ``(n_samples,)`` or
            ``(n_episodes, steps_per_episode)``.
        :return: Weighted rewards per episode, shape ``(n_episodes, 1)``.
        """

        # rewards, shape: (n, T)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, steps_per_episode)

        # discount factors
        # compute a matrix of discount factors, shape: (n, T)
        num_trajectories = rewards.shape[0]

        discount_factors = np.full(
            (num_trajectories, steps_per_episode), discount_factor, dtype=np.float32
        )
        # compute the discount factor at each step as
        # [gamma^0, gamma^1, ..., gamma^(T-1)] = [gamma^1, gamma^2, ..., gamma^T] / gamma
        discount_factors = np.cumprod(discount_factors, axis=1) / discount_factor

        # Support an alternate input path where step-wise importance ratios are
        # precomputed upstream (e.g. after stickiness correction in preprocessing).
        # This keeps the estimator logic generic while avoiding estimator-specific
        # handling of sticky actions.
        if importance_ratios is not None:
            importance_ratios = np.asarray(importance_ratios, dtype=np.float32)

            if importance_ratios.ndim == 1:
                step_weights = importance_ratios.reshape(num_trajectories, steps_per_episode)
            else:
                step_weights = importance_ratios

            # if the estimator is per decision, we need to repeat the discount factors and rewards for each action.
            # shape: (n, T * num_actions)
            if is_per_decision:
                importance_weights = np.cumprod(step_weights, axis=1)
            else:
                importance_weights = np.prod(step_weights, axis=1, keepdims=True)

            importance_weights = self.normalize(importance_weights)

            weighted_rewards = np.sum(
                importance_weights * discount_factors * rewards,
                axis=1,
            ).reshape(-1, 1)

            return weighted_rewards.astype(np.float32)

        # compute importance ratios
        importance_weights = (
            target_policy_action_probabilities / behavior_policy_action_probabilities
        )
        num_actions = target_policy_action_probabilities.shape[1]

        # compute the importance weights per decision, which is the cumulative product of the
        # importance weights over the trajectory; at each step t, we'll have the product from 0 to t-1.
        # shape: (n, T * num_actions)
        importance_weights = importance_weights.reshape(-1, steps_per_episode * num_actions)
        importance_weights = np.cumprod(importance_weights, axis=1)

        if not is_per_decision:
            # in trajectory-wise estimators, we only need the last importance weight, which is the product
            # over the whole trajectory.
            # shape: (n, 1)
            importance_weights = importance_weights[:, -1].reshape(-1, 1)

        # normalize the importance weights. Normalization technique depends on the estimator (see implementations).
        importance_weights = self.normalize(importance_weights)

        # if the estimator is per decision, we need to repeat the discount factors and rewards for each action.
        # shape: (n, T * num_actions)
        if is_per_decision:
            # discount factors
            # compute a matrix of discount factors, shape: (n, T)
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

        return weighted_rewards.astype(np.float32)

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
            importance_ratios=self.importance_ratios,
        )

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        """Estimate the value of the target policy using the Trajectory-wise Importance Sampling
        estimator."""
        return float(np.mean(self.estimate_weighted_rewards()))


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
            is_per_decision=False,
            importance_ratios=self.importance_ratios,
        )

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        """Estimate the value of the target policy using the Trajectory-wise Importance Sampling
        estimator."""
        return float(np.mean(self.estimate_weighted_rewards()))


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

    def __init__(
        self,
        *,
        steps_per_episode: int,
        discount_factor: float = 1.0,
        normalization: str = "per_timestep",
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            steps_per_episode=steps_per_episode,
            discount_factor=discount_factor,
        )
        self.normalization = normalization
        self.eps = eps

    """
    :param steps_per_episode: the number of steps per episode. The number of samples must be divisible by this number.
    :param discount_factor: the discount factor for computing cumulative returns. Must be in [0, 1].
    :param normalization: the normalization strategy to use. Supported values are "per_timestep" and "global".
        1) "per_timestep" normalizes the importance weights at each timestep by the mean of the importance weights at that timestep.
        2) "global" normalizes the importance weights by the mean of the importance weights over the entire trajectory.
    :param eps: a small value to avoid division by zero when normalizing the importance weights.
    :return: the estimated value of the target policy.
    """

    @override(BaseEstimator)
    def check_parameters(self) -> None:
        super().check_parameters()

        if self.normalization not in {"per_timestep", "global"}:
            raise ValueError("normalization must be 'per_timestep' or 'global'.")

        if self.eps <= 0:
            raise ValueError("eps must be > 0.")

    @override(TrajectoryPerDecisionMixin)
    def normalize(self, weights: np.ndarray) -> np.ndarray:
        """Normalize the importance weights using the self-normalization strategy.

        It uses self-normalization to reduce the variance of the estimator, using the mean
        of the importance weights over the trajectories.

        :param weights: the importance weights to normalize.
        :return: the normalized importance weights.
        """
        return weights / (np.mean(weights) + 1e-10)

    @override(BaseEstimator)
    def estimate_weighted_rewards(self) -> np.ndarray:
        # The standard self-normalized per-timestep case is already handled by the base implementation in TrajectoryPerDecisionMixin,
        # including the path where importance ratios are precomputed upstream.
        if self.normalization == "per_timestep":
            return super().estimate_weighted_rewards()

        # Only the custom global normalization case needs to be handled here, where the normalization
        # is done at the trajectory level instead of per timestep.
        self.check_parameters()

        if self.importance_ratios is None:
            raise ValueError("importance_ratios must be provided for global normalization.")

        rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1, self.steps_per_episode)
        n_episodes, horizon = rewards.shape

        rho = np.asarray(self.importance_ratios, dtype=np.float32)
        if rho.ndim == 1:
            rho = rho.reshape(n_episodes, horizon)

        discount_factors = np.full(
            (n_episodes, horizon),
            self.discount_factor,
            dtype=np.float32,
        )
        discount_factors = np.cumprod(discount_factors, axis=1) / self.discount_factor

        W = np.cumprod(rho, axis=1)

        # Global normalization: compute the sum of importance weights over the entire trajectory
        # and normalize the weighted rewards by this sum.
        num_i = np.sum(W * discount_factors * rewards, axis=1)
        den = float(np.sum(W))

        weighted_rewards = (n_episodes * num_i / np.maximum(den, self.eps)).reshape(-1, 1)

        return weighted_rewards.astype(np.float32)

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        return float(np.mean(self.estimate_weighted_rewards()))

    @override(BaseEstimator)
    def _bootstrap_sample_policy_value(
        self,
        weighted_rewards: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        r"""For standard per-timestep SNPDIS, we can reuse the base implementation, where the
        estimator is an average of episode-level contributions. However, for global normalization,
        the estimator is a ratio where the denominator depends on all samples jointly, so it's
        needed to recompute it for each bootstrap resample.

        .. math::
        \frac{\sum_{i=1}^n \sum_{t=0}^{T-1} W_{i,t} r_{i,t}}
             {\sum_{i=1}^n \sum_{t=0}^{T-1} W_{i,t}}
        """

        if self.normalization != "global" or self.importance_ratios is None:
            return super()._bootstrap_sample_policy_value(weighted_rewards, rng)

        rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1, self.steps_per_episode)
        n_episodes, horizon = rewards.shape

        rho = np.asarray(self.importance_ratios, dtype=np.float32)
        if rho.ndim == 1:
            rho = rho.reshape(n_episodes, horizon)

        discount_factors = np.full(
            (n_episodes, horizon),
            self.discount_factor,
            dtype=np.float32,
        )
        discount_factors = np.cumprod(discount_factors, axis=1) / self.discount_factor

        # Resample episodes with replacement
        idx = rng.choice(n_episodes, size=n_episodes, replace=True)

        rewards_b = rewards[idx]
        rho_b = rho[idx]
        discount_b = discount_factors[idx]

        # Recompute cumulative importance weights on the bootstrap sample
        W_b = np.cumprod(rho_b, axis=1)

        # Compute numerator and denominator of the ratio estimator
        num = float(np.sum(W_b * discount_b * rewards_b))
        den = float(np.sum(W_b))

        return float(num / np.maximum(den, self.eps))


class SequentialDoublyRobust(BaseEstimator):
    r"""Sequential Doubly Robust estimator.

    This estimator computes a per-decision doubly robust estimate using a
    temporal-difference-style formulation. It combines model-based predictions
    with cumulative importance weights built from behavior and target policy
    action probabilities.

    The per-episode estimate is computed as:

    .. math::
        \hat{V}_{\mathrm{DR}}^{(i)} =
        \hat{V}(s_{i,0}) +
        \sum_{t=0}^{T-1}
        W_{i,t}
        \left(
            r_{i,t}
            + \gamma \hat{V}(s_{i,t+1})
            - \hat{Q}(s_{i,t}, a_{i,t})
        \right)

    where:

    .. math::
        W_{i,t} = \prod_{k=0}^{t} \rho_{i,k}

    and

    .. math::
        \rho_{i,t} =
        \frac{\pi_e(a_{i,t} \mid s_{i,t})}{\pi_b(a_{i,t} \mid s_{i,t})}

    with:

    - :math:`i` denoting the episode index,
    - :math:`t` denoting the timestep index,
    - :math:`r_{i,t}` the observed reward at timestep :math:`t`,
    - :math:`\hat{Q}(s_{i,t}, a_{i,t})` the estimated action-value for the logged action,
    - :math:`\hat{V}(s_{i,t})` the estimated state value under the target policy,
    - :math:`\gamma` the discount factor,
    - :math:`\pi_e` the target policy,
    - :math:`\pi_b` the behavior policy.

    If precomputed step-wise importance ratios are provided, they are used directly.
    Otherwise, the ratios are constructed from the target and behavior policy action
    probabilities and the logged actions.

    Stickiness handling, when needed, must be applied upstream during preprocessing.
    """

    def __init__(
        self,
        *,
        steps_per_episode: int,
        discount_factor: float = 1.0,
        eps: float = 1e-12,
        clip: float | None = None,
    ) -> None:
        """Initialize the Sequential Doubly Robust estimator.

        :param steps_per_episode: Number of timesteps in each episode.
        :param discount_factor: Discount factor
        :math:`\\gamma` used in the TD correction term. Must be in
        :math:`[0, 1]`.
        :param eps: Numerical stabilizer used in importance-ratio computation to avoid
            division by zero.
        :param clip: Optional symmetric clipping threshold applied to step-wise importance
            ratios as
        :math:`\rho_t \\leftarrow \\mathrm{clip}(\rho_t, 1 / c, c)`. When provided, it must
            satisfy
        :math:`c \\geq 1`.
        """
        super().__init__()

        assert steps_per_episode > 0, "The number of steps per episode must be positive."
        assert 0 <= discount_factor <= 1, "The discount factor must be in [0, 1]."

        self.steps_per_episode = steps_per_episode
        self.discount_factor = discount_factor
        self.eps = eps
        self.clip = clip

        self.logged_actions: np.ndarray | None = None
        self.q_values: np.ndarray | None = None

    def set_logged_actions(self, logged_actions: np.ndarray) -> None:
        """Set logged actions.

        :param logged_actions: Logged action indices with shape `(n_samples,)`.
        """
        self.logged_actions = np.asarray(logged_actions, dtype=np.int64).reshape(-1)

    def set_model_predictions(self, *, q_values: np.ndarray) -> None:
        r"""Set model-based predictions used by the sequential DR estimator.

        :param q_values: Estimated action-values for all actions, shape `(n_samples,
            n_actions)`. Each row must contain the estimated action-values
        :math:`[\hat{Q}(s_t, a)]_{a \in \mathcal{A}}` for the corresponding state.
        """
        self.q_values = np.asarray(q_values, dtype=np.float32)

    @override(BaseEstimator)
    def short_name(self) -> str:
        return "SDR"

    @override(BaseEstimator)
    def check_parameters(self) -> None:
        """Check if the estimator parameters are valid."""
        super().check_parameters()

        if self.eps <= 0:
            raise ValueError("eps must be > 0.")

        if self.clip is not None and self.clip < 1.0:
            raise ValueError("clip must be >= 1.0 when provided.")

        n_samples = self.rewards.shape[0]
        n_actions = self.target_policy_action_probabilities.shape[1]

        if n_samples % self.steps_per_episode != 0:
            raise ValueError("The number of samples must be divisible by steps_per_episode.")

        if self.q_values is None:
            raise ValueError("q_values not set. Call set_model_predictions(...).")

        if self.q_values.ndim != 2:
            raise ValueError("q_values must be a 2D array of shape (n_samples, n_actions).")

        if self.q_values.shape != (n_samples, n_actions):
            raise ValueError(
                "q_values must have shape (n_samples, n_actions), matching "
                "target_policy_action_probabilities."
            )

        if self.logged_actions is None:
            raise ValueError("logged_actions must be provided.")

        if self.logged_actions.shape[0] != n_samples:
            raise ValueError("logged_actions length must match rewards length.")

        if np.any(self.logged_actions < 0) or np.any(self.logged_actions >= n_actions):
            raise ValueError("logged_actions contains invalid action indices.")

    def _get_stepwise_importance_ratios(self) -> np.ndarray:
        """Get the step-wise importance ratios.

        :return: Step-wise importance ratios, shape `(n_episodes, steps_per_episode)`.
        """

        n_samples = self.rewards.shape[0]
        n_episodes = n_samples // self.steps_per_episode

        if self.importance_ratios is not None:
            rho = np.asarray(self.importance_ratios, dtype=np.float32)
            if rho.ndim == 1:
                rho = rho.reshape(n_episodes, self.steps_per_episode)
            return rho

        idx = np.arange(n_samples, dtype=np.int64)
        actions = np.asarray(self.logged_actions, dtype=np.int64).reshape(-1)

        p_e_taken = self.target_policy_action_probabilities[idx, actions].astype(np.float32)
        p_b_taken = self.behavior_policy_action_probabilities[idx, actions].astype(np.float32)

        rho = p_e_taken / np.maximum(p_b_taken, self.eps)

        if self.clip is not None:
            rho = np.clip(rho, 1.0 / self.clip, self.clip)

        return rho.reshape(n_episodes, self.steps_per_episode)

    @override(BaseEstimator)
    def estimate_weighted_rewards(self) -> np.ndarray:
        """Estimate episode-level sequential DR contributions.

        :return: Episode-level DR estimates, shape `(n_episodes, 1)`.
        """
        self.check_parameters()

        rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1, self.steps_per_episode)
        q_values = np.asarray(self.q_values, dtype=np.float32)
        rho = self._get_stepwise_importance_ratios()

        n_episodes, horizon = rewards.shape
        n_samples = n_episodes * horizon

        logged_actions = np.asarray(self.logged_actions, dtype=np.int64).reshape(-1)
        idx = np.arange(n_samples, dtype=np.int64)

        q_logged = q_values[idx, logged_actions].reshape(n_episodes, horizon)

        v_values = np.sum(
            self.target_policy_action_probabilities * q_values,
            axis=1,
        ).reshape(n_episodes, horizon)

        v_next = np.zeros_like(v_values)
        v_next[:, :-1] = v_values[:, 1:]
        v_next[:, -1] = 0.0

        cumulative_weights = np.cumprod(rho, axis=1)

        episode_estimates = v_values[:, 0].copy()

        for t in range(horizon):
            td_correction = rewards[:, t] + self.discount_factor * v_next[:, t] - q_logged[:, t]
            episode_estimates += cumulative_weights[:, t] * td_correction

        return episode_estimates.reshape(-1, 1).astype(np.float32)

    @override(BaseEstimator)
    def estimate_policy_value(self) -> float:
        """Estimate the value of the target policy."""
        return float(np.mean(self.estimate_weighted_rewards()))
