.. title:: Welcome to Hopes!

.. toctree::
   :hidden:

   Overview <overview/index>
   Estimators <ope/index>
   Policies <policy/index>
   Selection <ops/index>

Hopes
=====

What's in the box?
------------------

**HOPES** - HVAC Off-policy Policy Evaluation and Selection - is a Python package for evaluating and selecting RL-based
control policies. It offers a set of estimators and tools to evaluate the performance of a target policy,
compared to a baseline policy (characterized by an offline logged dataset), using off-policy evaluation
techniques. It's particularly suited for the context of HVAC control, where the target policy is an RL-based controller
and the baseline policies are rule-based controllers.

Why Hopes?
----------

Hopes is designed to be a flexible and easy-to-use package for evaluating and selecting RL-based control policies.
Imagine you have a dataset of logged actions and observations from a building HVAC system, and you want to evaluate the
performance of one or several RL-based controller. Hopes provides a set of tools to help you do that, including:

- Estimators for evaluating the performance of a target policy compared to a baseline policy.
- Tools for selecting the best policy among a set of candidate policies.
- Tools for visualizing the results of the evaluation and selection process.
- Dataset preprocessing tools to prepare the data for evaluation.

Installation
------------

Supported Python versions: 3.10+

*From PyPI*

.. code-block:: bash
    :linenos:

    pip install hopes

*From source (development version)*

.. code-block:: bash
    :linenos:

    git clone https://github.com/airboxlab/hopes.git
    cd hopes
    # using poetry
    poetry install
    # using pip
    pip install -r requirements.txt
