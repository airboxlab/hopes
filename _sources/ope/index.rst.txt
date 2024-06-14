Hopes: Estimators
=================

Roadmap
-------

- [x] Implement Inverse Probability Weighting (IPW) estimator
- [x] Implement Self-Normalized Inverse Probability Weighting (SNIPW) estimator
- [x] Implement Direct Method (DM) estimator
- [x] Implement Trajectory-wise Importance Sampling (TIS) estimator
- [x] Implement Self-Normalized Trajectory-wise Importance Sampling (SNTIS) estimator
- [x] Implement Per-Decision Importance Sampling (PDIS) estimator
- [x] Implement Self-Normalized Per-Decision Importance Sampling (SNPDIS) estimator
- [ ] Implement Doubly Robust (DR) estimator

Implemented estimators
-----------------------

Currently, the following estimators are implemented:

.. autosummary::
   :nosignatures:

   hopes.ope.estimators.BaseEstimator
   hopes.ope.estimators.InverseProbabilityWeighting
   hopes.ope.estimators.SelfNormalizedInverseProbabilityWeighting
   hopes.ope.estimators.DirectMethod
   hopes.ope.estimators.TrajectoryWiseImportanceSampling
   hopes.ope.estimators.SelfNormalizedTrajectoryWiseImportanceSampling
   hopes.ope.estimators.PerDecisionImportanceSampling
   hopes.ope.estimators.SelfNormalizedPerDecisionImportanceSampling

Estimators documentation
------------------------

.. autoclass:: hopes.ope.estimators.InverseProbabilityWeighting
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: hopes.ope.estimators.SelfNormalizedInverseProbabilityWeighting
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: hopes.ope.estimators.DirectMethod
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: hopes.ope.estimators.TrajectoryWiseImportanceSampling
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: hopes.ope.estimators.SelfNormalizedTrajectoryWiseImportanceSampling
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: hopes.ope.estimators.PerDecisionImportanceSampling
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: hopes.ope.estimators.SelfNormalizedPerDecisionImportanceSampling
    :members:
    :undoc-members:
    :show-inheritance:

Implementing a new estimator
----------------------------

To implement a new estimator, you need to subclass :class:`hopes.ope.estimators.BaseEstimator` and implement:

- :meth:`hopes.ope.estimators.BaseEstimator.estimate_weighted_rewards`. It should return the estimated weighted rewards.
- :meth:`hopes.ope.estimators.BaseEstimator.estimate_policy_value`. It should return the estimated value of the target policy. It typically uses the estimated weighted rewards.

Optionally, you can implement :meth:`hopes.ope.estimators.BaseEstimator.short_name` to provide a short name for the estimator.
When not implemented, the uppercase letters of the class name are used.

Below is the `BaseEstimator` class documentation.

.. autoclass:: hopes.ope.estimators.BaseEstimator
    :members: estimate_weighted_rewards, estimate_policy_value, short_name
