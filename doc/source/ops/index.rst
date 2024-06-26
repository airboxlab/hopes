Hopes: Selection
================

Roadmap
-------

- [x] Confidence Interval estimation using Bootstrap
- [x] Confidence Interval estimation using t-test

Introduction
------------

Running an Off-Policy Evaluation (OPE) experiment and then a selection of the best policies with Hopes is simple.

Example with a synthetic, random, dataset.

.. code-block:: python

    # create the behavior policy
    behavior_policy = ClassificationBasedPolicy(
        obs=obs, act=act, classification_model="logistic"
    )
    behavior_policy.fit()

    # create the target policies
    target_policy_1 = RandomPolicy(num_actions=num_actions).with_name("p1")
    target_policy_2 = RandomPolicy(num_actions=num_actions).with_name("p2")
    target_policy_3 = ClassificationBasedPolicy(
        obs=obs, act=act, classification_model="random_forest"
    ).with_name("p3")
    target_policy_3.fit()

    # initialize the estimators
    estimators = [
        InverseProbabilityWeighting(),
        SelfNormalizedInverseProbabilityWeighting(),
    ]

    # run the off-policy evaluation
    ope = OffPolicyEvaluation(
        obs=obs,
        rewards=rew,
        behavior_policy=behavior_policy,
        estimators=estimators,
        fail_fast=True,
        ci_method="t-test",
        ci_significance_level=0.1,
    )

    results = [
        ope.evaluate(target_policy)
        for target_policy in [target_policy_1, target_policy_2, target_policy_3]
    ]

    # select the top k policies based on lower bound (confidence interval +-90%)
    top_k_results = OffPolicySelection.select_top_k(results, metric="lower_bound", top_k=1)
    print(top_k_results[0])

This should produce an output similar to:

.. code-block:: python

    Policy: p2
    Confidence interval: +- 90.0%
    =====  ========  ==========  =============  =============
    ..         mean         std    lower_bound    upper_bound
    =====  ========  ==========  =============  =============
    IPW    0.510251  0.00788465       0.497324       0.522907
    SNIPW  0.499158  0.00523288       0.490235       0.507513
    =====  ========  ==========  =============  =============

Note that confidence interval (CI) calculation can be based on several methods:

- `bootstrap` (default)
- `t-test`

The documentation of the CI calculation can be found in `BaseEstimator.estimate_policy_value_with_confidence_interval`. See
implementation details for more information.

Implementation details
----------------------

.. autoclass:: hopes.ope.evaluation.OffPolicyEvaluation
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: hopes.ope.selection.OffPolicySelection
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: hopes.ope.estimators.BaseEstimator
    :no-index:
    :members: estimate_policy_value_with_confidence_interval
