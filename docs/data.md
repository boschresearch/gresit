
# Data and `gRESIT` algorithm

The data required by `gRESIT` should be structured in a way that each key in the `data_dict` corresponds to a group, and the value is a `numpy.ndarray` containing the samples for that group. We do *not* require the `numpy.ndarray`'s to be of the same shape, but they should contain the same number of samples.

```python
import numpy as np

rng = np.random.default_rng(42)  # Set seed for reproducibility

X_1 = rng.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=2000)
X_2 = rng.multivariate_normal(mean=np.zeros(3), cov=np.eye(3), size=2000)

X_3 = np.column_stack(
    [X_1[:, 0] * X_2[:, 0] + X_1[:, 1] * X_2[:, 1], X_1[:, 1] * X_2[:, 2]]
) + 0.1 * rng.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=2000)

data_dict = {
    "X_1": X_1,
    "X_2": X_2,
    "X_3": X_3,
}
```

| Key       | Shape     | Dtype     | Example Values                 |
|-----------|-----------|-----------|--------------------------------|
| `X_1` | (2000, 2) | float64 | `[[0.305, -1.04], [0.75, 0.941]]` |
| `X_2` | (2000, 3) | float64 | `[[0.253, 0.895, 0.273], [2.239, 1.43, -0.308]]` |
| `X_3` | (2000, 2) | float64 | `[[-0.836, -0.194], [2.878, -0.318]]` |

Given this data, we can run the `gRESIT` algorithm as follows:

```python
from gresit.group_resit import GroupResit
from gresit.independence_tests import HSIC
from gresit.torch_models import Multioutcome_MLP

gresit = GroupResit(regressor=Multioutcome_MLP(), test=HSIC)
gresit.learn_graph(data_dict)
gresit.show_interactive()
```

Which produces the following interactive graph:

<iframe src="html_plots/graph.html" width="100%" height="600px" style="border:none;"></iframe>

In the section [gRESIT](gresit.md) you will find details on all arguments and hyperparameters for `gRESIT`.
