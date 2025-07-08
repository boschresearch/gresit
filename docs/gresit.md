# gRESIT arguments and hyperparameters

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

## Pruning method

Here, the valid arguments for `pruning_method` are `murgs` and `independence`.

`murgs` is the default pruning method, which uses *multiresponse group sparse additive models* (MURGS) to prune away extraneous edges in the super-graph implied by the causal ordering. `independence` on the other hand uses independence tests as proposed by [Peters et al. (2014)](https://jmlr.org/papers/v15/peters14a.html) to prune the super-graph. Note that while `murgs` is order independent, `independence` strongly depends on the order of which the independence tests are applied.

## `murgs` arguments

- `local_regression_method`:  Type of local linear smoother to use. Options are currently `loess`, `kernel`. Defaults to `kernel`.

## `independence` arguments

- `alpha`: The significance level for the independence tests. Default is `0.01`.

## Regressor

The `regressor` argument specifies the regression technique used in `gRESIT`. It should be an instance of the [`MultiRegressor`][gresit.regression_techniques.MultiRegressor] base class. Currently, the following regression techniques are available:

- [`Multioutcome_MLP`][gresit.torch_models.Multioutcome_MLP]: A multi-output MLP regressor.
    - `rng`: Random number generator to control seed for data splitting. Defaults to`np.random.default_rng(seed=2024)`.
    - `loss`: Standard `mse` loss is default. Other options are `hsic` and `disco`.

    - `dropout_proba`: Dropout probability. Defaults to 0.6.
    - `n_epochs`: Number of times the data gets passed through the MLP. Defaults to 6.
    - `patience`: Minimal number of epochs to train before early stopping applies. Defaults to 60
    - `learning_rate`: Defaults to 1e-3.
    - `val_size`: Relative size of the validation dataset. Defaults to 0.2.
    - `batch_size`: Batch size for training. Defaults to 200.
    - `es`: Early stopping. Defaults to true.

- [Simultaneous Linear Regression][gresit.regression_techniques.SimultaneousLinearModel]: A simultaneous linear regression model.
    - `alpha`: Regularization parameter for the ridge regression. Defaults to 0.1.

- [Reduced Rank Regression][gresit.regression_techniques.ReducedRankRegressor]: Kernel Reduced Rank Ridge Regression.
    - `alpha`: Regularization parameter for the ridge type regularization. Defaults to 1.0.

- [CurdsWhey][gresit.regression_techniques.CurdsWhey]: Breiman and Friedman's curds and whey multivariate regression model.
- [BoostedRegressionTrees][gresit.regression_techniques.BoostedRegressionTrees]: A boosted regression tree model.

## Independence tests

The following independence tests are implemented in [`gresit.independence_tests`][gresit.independence_tests]:

- [`KernelCI`][gresit.independence_tests.KernelCI]
    - Kernel HSIC for conditional independence testing wrapper
     around the `causal-learn` functionality.
- [`FisherZVEC`][gresit.independence_tests.FisherZVec]
    - Fisher Z test for conditional independence.
- [`HSIC`][gresit.independence_tests.HSIC]
    - Hilbert-Schmidt Independence Criterion for unconditional independence testing.
- [`DISCO`][gresit.independence_tests.DISCO]
    - Distance Correlation (DISCO) which equals zero if and only if the vectors
     considered are unconditionally independent.

In `gRESIT`, we only require unconditional independence tests, so the `test` argument should be one of the following:

- `HSIC`
- `DISCO`

## Additional arguments

- `test_size`: The relative size of the test set. If chosen to be larger than
    zero, the regression model is trained on the training set and the subsequent
    independence test is performed on the test set only. Defaults to 0.2.
