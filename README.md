# gRESIT

[![view - Documentation](https://img.shields.io/badge/view-Documentation-blue?style=for-the-badge)](https://boschresearch.github.io/gresit/)

[![Made with Python](https://img.shields.io/badge/Python->=3.10-blue?logo=python&logoColor=white)](https://python.org "Go to Python homepage")
[![PyPI - maintained](https://img.shields.io/badge/PyPI-maintained-green?logo=pypi)](https://test.pypi.org/project/gresit/1.0.0/)
[![tests - passing](https://img.shields.io/badge/tests-passing-green)](https://github.com/boschresearch/gresit/tree/main/tests)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)

This repo aims at learning and representing causal graphs based on grouped data.
Theoretical details are presented in the [paper](http://www.arxiv.org/abs/2506.05120)

    @misc{goebler2025,
      title={Nonlinear Causal Discovery for Grouped Data},
      author={Konstantin G\"obler and Tobias Windisch and Mathias Drton},
      year={2025},
      eprint={2506.05120},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={<https://arxiv.org/abs/2506.05120}>,
    }

## Authors

- [Konstantin Göbler (TU Munich, Bosch)](mailto:konstantin.goebler@tum.de)
- [Tobias Windisch (HS Kempten)](mailto:tobias.windisch@hs-kempten.de)

**Maintainer*:* [Martin Roth (Bosch)](mailto:martin.roth2@de.bosch.com)

## Table of contents

- [Documentation](#documentation)
- [How to install](#installing)
- [How to build](#building)
- [How to use](#using)
- [How to test](#testing)
- [Github Actions](#actions)

## <a name="documentation">Documentation </a>

The documentation can be found [here]().

## <a name="installing">How to install</a>

The package can be installed with

    pip install gresit

## <a name="building">How to build</a>

Using the [Makefile](Makefile) the package can be installed in an editable way like this:

    make sync-venv

To use the `pre-commit` hooks, one has to enable them in the venv, by

    pre-commit install

Then these hooks are excecuted before every commit. You can run the hooks for all files also separately

    pre-commit run --all-files

or to disable the `pip-compile` hook, which takes some time

    SKIP=pip-compile pre-commit run --all-files

or equivalent

    make pre-commit

## <a name="using">How to use</a>

Consider the following example. We refer to the [documentation](#documentation) for more detailed information.

### Generating Synthetic Data

We first generate synthetic data using an Erdős–Rényi random graph model. Each group of variables is defined with a specified size and edge density.

```python
from gresit.synthetic_data import GenERData

data_gen = GenERData(
    number_of_nodes=10,
    group_size=2,
    edge_density=0.2,
)

data_dict, _ = data_gen.generate_data(num_samples=1000)
```

The output `data_dict` is a dictionary where each key corresponds to a group, and the values are the observed samples.

### Fitting a Graph Model

We now fit a `gRESIT` model using `Multioutcome_MLP` as the regressor and `HSIC` as independence test.

```python
from gresit.group_resit import GroupResit
from gresit.independence_tests import HSIC
from gresit.torch_models import Multioutcome_MLP

model = GroupResit(
    regressor=Multioutcome_MLP(),
    test=HSIC,
    pruning_method="murgs",
)
learned_dag = model.learn_graph(data_dict=data_dict)

# Show the learned graph:
learned_dag.show()
# or show interactive mode:
model.show_interactive()
```

### Accessing the Learned Graph

The learned adjacency matrix representing the estimated group-level graph and a causal ordering can be accessed via:

```python
model.adjacency_matrix
model.causal_ordering
```

## <a name="testing">How to test</a>

In general we use pytest and the test suite can be executed locally via

    python -m pytest

## <a name="actions">Github Actions </a>

### <a name="mkdocs">Documentation with mkdocs </a>

We use mkdocs for building the documentation, this is the corresponding [workflow](.github/workflows/publish_docu.yml).

### Automated issue workflow

With this [workflow](.github/workflows/add_issues.yml) newly created issues are automatically added to our MFD2 project.

### Pre-commit

With this [workflow](.github/workflows/pre-commit.yml) the pre-commit rules, specified in [.pre-commit-config.yaml](https://github.com/bosch-cc-mfd/python_test/blob/main/.pre-commit-config.yaml), are executed.

To use pre-commit locally, please use

    pre-commit install

### Testing

With this [workflow](.github/workflows/test_package.yml) the tests are executed.

## <a name="3rd-party-licenses">Third-Party Licenses</a>

### Runtime dependencies

| Name                                                        | License                                                                                  | Type       |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ---------- |
| [numpy](https://numpy.org/)                                 | [BSD-3-Clause License](https://github.com/numpy/numpy/blob/master/LICENSE.txt)           | Dependency |
| [pandas](https://pandas.pydata.org/)                        | [BSD-3-Clause License](https://github.com/pandas-dev/pandas/blob/master/LICENSE)         | Dependency |
| [scikit-learn](https://scikit-learn.org/)                   | [BSD-3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)   | Dependency |
| [statsmodels](https://www.statsmodels.org/)                 | [BSD-3-Clause License](https://github.com/statsmodels/statsmodels/blob/main/LICENSE.txt) | Dependency |
| [plotly](https://plotly.com/python/)                        | [MIT License](https://github.com/plotly/plotly.py/blob/master/LICENSE.txt)               | Dependency |
| [xgboost](https://github.com/dmlc/xgboost)                  | [Apache License 2.0](https://github.com/dmlc/xgboost/blob/master/LICENSE)                | Dependency |
| [torch](https://pytorch.org/)                               | [BSD-3-Clause License](https://github.com/pytorch/pytorch/blob/main/LICENSE)             | Dependency |
| [seaborn](https://seaborn.pydata.org/)                      | [BSD-3-Clause License](https://github.com/mwaskom/seaborn/blob/master/LICENSE)           | Dependency |
| [pyspark](https://spark.apache.org/docs/latest/api/python/) | [Apache License 2.0](https://github.com/apache/spark/blob/master/LICENSE)                | Dependency |
| [scikit-misc](https://github.com/has2k1/scikit-misc)        | [BSD-3-Clause License](https://github.com/has2k1/scikit-misc/blob/master/LICENSE)        | Dependency |
| [gadjid](https://github.com/xunzheng/gadjid)                | [MIT License](https://github.com/xunzheng/gadjid/blob/main/LICENSE)                      | Dependency |
| [tqdm](https://github.com/tqdm/tqdm)                        | [MIT License](https://github.com/tqdm/tqdm/blob/master/LICENCE)                          | Dependency |
| [dcor](https://github.com/vnmabus/dcor)                     | [MIT License](https://github.com/vnmabus/dcor/blob/master/LICENSE.txt)                   | Dependency |
| [llvmlite](https://github.com/numba/llvmlite)               | [BSD-2-Clause License](https://github.com/numba/llvmlite/blob/main/LICENSE)              | Dependency |
| [causal-learn](https://github.com/cmu-phil/causal-learn)    | [MIT License](https://github.com/cmu-phil/causal-learn/blob/main/LICENSE)                | Dependency |
| [gcastle](https://github.com/huawei-noah/trustworthyAI)     | [Apache License 2.0](https://github.com/huawei-noah/trustworthyAI/blob/master/LICENSE)   | Dependency |
| [gpytorch](https://gpytorch.ai/)                            | [MIT License](https://github.com/cornellius-gp/gpytorch/blob/master/LICENSE)             | Dependency |

### Development dependency

| Name                                                            | License                                                                           | Type     |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------- |
| [mike](https://github.com/jimporter/mike)                       | [BSD-3-Clause License](https://github.com/jimporter/mike/blob/master/LICENSE)     | Optional |
| [mkdocs](https://www.mkdocs.org/)                               | [BSD-2-Clause License](https://github.com/mkdocs/mkdocs/blob/master/LICENSE)      | Optional |
| [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) | [MIT License](https://github.com/squidfunk/mkdocs-material/blob/master/LICENSE)   | Optional |
| [mkdocstrings](https://github.com/mkdocstrings/mkdocstrings)    | [ISC License](https://github.com/mkdocstrings/mkdocstrings/blob/main/LICENSE)     | Optional |
| [pip-licenses](https://github.com/raimon49/pip-licenses)        | [MIT License](https://github.com/raimon49/pip-licenses/blob/master/LICENSE)       | Optional |
| [pip-tools](https://github.com/jazzband/pip-tools)              | [BSD-3-Clause License](https://github.com/jazzband/pip-tools/blob/master/LICENSE) | Optional |
| [pre-commit](https://pre-commit.com/)                           | [MIT License](https://github.com/pre-commit/pre-commit/blob/main/LICENSE)         | Optional |
| [pytest](https://pytest.org/)                                   | [MIT License](https://github.com/pytest-dev/pytest/blob/main/LICENSE)             | Optional |
| [pytest-cov](https://github.com/pytest-dev/pytest-cov)          | [MIT License](https://github.com/pytest-dev/pytest-cov/blob/master/LICENSE)       | Optional |
| [ruff](https://github.com/astral-sh/ruff)                       | [MIT License](https://github.com/astral-sh/ruff/blob/main/LICENSE)                | Optional |
| [uv](https://github.com/astral-sh/uv)                           | [MIT License](https://github.com/astral-sh/uv/blob/main/LICENSE)                  | Optional |
