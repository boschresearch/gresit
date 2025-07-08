# Algorithms

This is a short overview of the basis algorithms implemented and benchmarked for the grouped
setting. Here, we use the synthetic dataset:

```python
from gresit.synthetic_data import GenERData
data_gen = GenERData(number_of_nodes=4, group_size=2)
data_dict, _ = data_gen.generate_data(num_samples=100)
```

## gRESIT

```python
from gresit.group_resit import GroupResit
from gresit.torch_models import Multioutcome_MLP
from gresit.independence_tests import HSIC

alg = GroupResit(
    pruning_method='murgs',
    test=HSIC,
    regressor=Multioutcome_MLP(),
)
graph = alg.learn_graph(data_dict=data_dict)
```

See [here](gresit.md) for more details on its hyperparameters.

## GroupPC

```python

from gresit.group_pc import GroupPC
alg = GroupPC(alpha=0.1)
graph = alg.learn_graph(data_dict=data_dict)
```

## Grouped GraNDAG

```python
from gresit.group_grandag import GroupGraNDAG
alg = GroupGraNDAG(n_iterations=10, with_group_constraint=True)
graph = alg.learn_graph(data_dict)
```

## Grouped LiNGAM

```python
from gresit.group_lingam import GroupLiNGAM
alg = GroupLiNGAM()
graph = alg.learn_graph(data_dict)
```
