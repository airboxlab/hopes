Off-Policy Evaluation: Estimators
=================================

Roadmap
-------

- [x] Implement Inverse Probability Weighting (IPW) estimator
- [x] Implement Self-Normalized Inverse Probability Weighting (SNIPW) estimator
- [x] Implement Direct Method (DM) estimator
- [ ] Implement Doubly Robust (DR) estimator
- [ ] Implement Trajectory-Wise Importance Sampling (TWIS) estimator
- [ ] Implement Per-Decision Importance Sampling (PDIS) estimator

Implemented estimators
-----------------------

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

Implementing a new estimator
----------------------------

To implement a new estimator, you need to subclass :class:`hopes.ope.estimators.BaseEstimator` and implement the :meth:`hopes.ope.estimators.BaseEstimator.estimate_policy_value`. It should return the estimated value of the target policy value function.

Below is the `BaseEstimator` class documentation.

.. autoclass:: hopes.ope.estimators.BaseEstimator
    :members:
    :undoc-members:
    :show-inheritance:
