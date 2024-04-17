Hopes: Policies
===============================

Implemented policies
--------------------

The base, abstract policy class is :class:`Policy <hopes.policy.policies.Policy>`.

Following policies implement it:

.. autosummary::
   :nosignatures:

   hopes.policy.RandomPolicy
   hopes.policy.ClassificationBasedPolicy
   hopes.policy.PiecewiseLinearPolicy
   hopes.policy.FunctionBasedPolicy
   hopes.policy.HttpPolicy
   hopes.policy.OnnxModelBasedPolicy

They aim to provide an integration with actual policies, which can be used in off-policy evaluation.

Policies documentation
------------------------

.. autoclass:: hopes.policy.ClassificationBasedPolicy
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: hopes.policy.PiecewiseLinearPolicy
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: hopes.policy.OnnxModelBasedPolicy
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: hopes.policy.FunctionBasedPolicy
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: hopes.policy.RandomPolicy
    :members:
    :undoc-members:
    :show-inheritance:

Implementing a new policy
-------------------------

To implement a new policy, you need to subclass :class:`Policy <hopes.policy.policies.Policy>` and implement the
:meth:`hopes.policy.Policy.log_likelihoods` method.

.. autoclass:: hopes.policy.Policy
    :members:
    :undoc-members:
    :show-inheritance:
