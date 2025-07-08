# Getting Started

This example shows how to generate synthetic data and learn a causal graph from group data alone using `gRESIT`.

## Generating Synthetic Data

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

The output data_dict is a dictionary where each key corresponds to a group, and the values are the observed samples. See [data](data.md) for more details on the expected data format.

## Fitting a Graph Model

We now fit a group RESIT model using a [`Multioutcome_MLP`][gresit.torch_models.Multioutcome_MLP] as the regressor and [`HSIC`][gresit.independence_tests.HSIC] as the independence test.

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

## Accessing the Learned Graph

The learned adjacency matrix representing the estimated group-level graph and a causal ordering can be accessed via:

```python
model.adjacency_matrix
model.causal_ordering
```
