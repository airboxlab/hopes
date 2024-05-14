Hopes: finding the best policy
==============================

What's off-policy policy evaluation?
------------------------------------

TODO

How does importance sampling work?
----------------------------------

In the previous section, we have seen that off-policy policy evaluation is based on the principle that the value of a
policy can be evaluated by using another policy, and the rewards obtained by the other policy. Doesn't it sound strange?

There's actually a good reason for that. It stands in the **importance sampling** definition.

Let's say we have a probability distribution :math:`p(x)` for the variable :math:`x`, and we want to compute
the expectation of a function :math:`f(x)` under this distribution. To evaluate the expected value of :math:`f(x)`
under :math:`p(x)`, we should sample from :math:`p(x)` and compute the average of :math:`f(x)` over the samples if x is
discrete, or compute the integral of :math:`f(x)` over :math:`p(x)` if x is continuous.

In the continuous case, the expected value of :math:`f(x)` under :math:`p(x)` is:

.. math::

    E_{x \sim p(x)}[f(x)] = \int p(x) f(x) dx

Now, let's say we don't have access to samples from :math:`p(x)`, but we have samples from another distribution :math:`q(x)`.

We can still compute the expected value of :math:`f(x)` by using the samples from :math:`q(x)`:

.. math::

    E_{x \sim p(x)}[f(x)] = \int p(x) f(x) dx
                       = \int p(x) \frac{q(x)}{q(x)} f(x) dx
                       = \int q(x) \frac{p(x)}{q(x)} f(x) dx
                       = E_{x \sim q(x)}[\frac{p(x)}{q(x)} f(x)]

Wrapping that up, :math:`E_{x \sim p(x)}[f(x)] = E_{x \sim q(x)}[\frac{p(x)}{q(x)} f(x)]`, which means
the expected value of :math:`f(x)` under :math:`p(x)` is equal to the expected value of :math:`\frac{p(x)}{q(x)} f(x)`
under :math:`q(x)`.

This is very convenient in reinforcement learning, because we can evaluate the value of a policy by using the samples
from another policy. Let's see how this works in the context of reinforcement learning.

Let's say we have trained a policy :math:`\pi_e` and we want to evaluate the value of this policy over trajectories
:math:`\tau` of length :math:`T`, collected by another policy :math:`\pi_b`, the behavior policy. This behavior policy
can be any policy, a rule-based policy, a policy trained with a different algorithm, or a policy trained with the same
algorithm but with different parameters [#]_. It generated the trajectories :math:`\tau` by interacting with the
environment and collecting the rewards :math:`r_t` at each time step :math:`t`.

The value of a policy here is the sum of rewards obtained by the policy over the trajectory :math:`\tau`.

We can use importance sampling to estimate the value of :math:`\pi_e`:

.. math::

    V_{\pi_e} = E_{\tau \sim \pi_e}[\sum_{t=0}^T r_t]
              = E_{\tau \sim \pi_b}[\sum_{t=0}^T \frac{\pi_e(a_t|s_t)}{\pi_b(a_t|s_t)} r_t]
              \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \frac{\pi_e(a_t^i|s_t^i)}{\pi_b(a_t^i|s_t^i)} r_t^i

Intuitively, we can see it as a way to weight the rewards obtained by the behavior policy by the ratio of the probabilities
of taking the action under the evaluation policy and the behavior policy. This ratio is called the importance weight.
The more likely the target policy :math:`\pi_e` is to take an action, and the less likely the behavior policy :math:`\pi_b`
is to take the same action, the higher the importance weight will be, and the more influential the reward will be in the
estimation of the value of the policy.

This estimator is unbiased, since the obtained result is the true expectation of the sum of rewards over trajectories.
However, it can have high variance, especially if trajectories are long, due to the sum of ratios.
Note that if trajectory length is equal to 1 (bandit setting), the value of the policy can be expressed as:

.. math::

    V_{\pi_e} \approx \frac{1}{N} \sum_{i=1}^N \frac{\pi_e(a^i|s^i)}{\pi_b(a^i|s^i)} r^i

which is the definition of the Inverse Probability Weighting (IPW) estimator.

When can't we use importance sampling?
--------------------------------------

Among other generic considerations, there are two assumptions that must be satisfied to use importance sampling:

- *coverage*: the behavior policy must have a non-zero probability of taking all the actions that the evaluation policy
  could take. In Hopes, we deal with this as much as possible by ensuring a small probability of taking all the actions
  in the behavior policy, especially in the deterministic case.
- *positivity*: the rewards must be non-negative to be able to get a lower bound estimate of the target policy. In Hopes,
  you'll find a way to rescale the rewards to make them positive (using MinMaxScaler).

.. [#] in the context of off-policy policy gradient methods, but that's out of the scope of this project.
