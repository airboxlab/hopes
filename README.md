[![tests](https://github.com/airboxlab/hopes/actions/workflows/tests.yml/badge.svg)](https://github.com/airboxlab/hopes/actions/workflows/tests.yml)
[![coverage](https://github.com/airboxlab/hopes/blob/main/coverage.svg)](<>)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# HOPES

## What is HOPES?

**HOPES** (HVAC Off-Policy Evaluation) is a Python package for evaluating and selecting RL-based HVAC
control policies. It offers a set of estimators and tools to evaluate the performance of a target policy,
compared to a set of baseline policies (characterized by an offline logged dataset), using off-policy evaluation
techniques.

## Installation

Supported Python versions: 3.10+

### From PyPI

```bash
pip install hopes
```

### From source (development version)

```bash
git clone https://github.com/airboxlab/hopes.git
cd hopes
# using poetry
poetry install
# using pip
pip install -r requirements.txt
```
