Overview of Hopes package
==========================

What's in the box?
------------------

**HOPES** - HVAC Off-Policy Evaluation and Selection - is a Python package for evaluating and selecting RL-based
control policies. It offers a set of estimators and tools to evaluate the performance of a target policy,
compared to a baseline policy (characterized by an offline logged dataset), using off-policy evaluation
techniques. It's particularly suited for the context of HVAC control, where the target policy is an RL-based controller
and the baseline policies are rule-based controllers.


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
