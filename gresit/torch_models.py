"""Utility classes and functions related to gresit.

Copyright (c) 2025 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from copy import deepcopy

import gpytorch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from gresit.losses import loss_hsic, loss_mse
from gresit.regression_techniques import MultiRegressor


class MLP(nn.Module):  # type: ignore
    """Torch MLP with single-hidden layer and sigmoid non-linearity.

    Not too fancy but does what it is supposed to do.
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 100, dropout: float = 0.0
    ) -> None:
        """Initializes the NN.

        Args:
            input_dim (int): dim of input layer
            output_dim (int, optional): dim of output layer.
            hidden_dim (int, optional): dim of hidden layer. Defaults to 100.
            dropout (float, optional): dropout probability. Defaults to 0.0.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # define the hidden layer and final layers
        self.hidden1 = nn.Linear(self.input_dim, self.hidden_dim, dtype=torch.float32)
        self.hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim, dtype=torch.float32)
        self.final = nn.Linear(self.hidden_dim, self.output_dim, dtype=torch.float32, bias=False)
        self.bias = nn.Parameter(
            data=torch.Tensor([0] * self.output_dim),
            requires_grad=False,
        ).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward passing the model.

        Args:
            x (torch.Tensor): training data.

        Returns:
            torch.Tensor: prediction after pass through nn.
        """
        x = self.hidden1(x)  # apply hidden layer
        x = torch.tanh(x)  # apply sigmoid non-linearity
        x = nn.Dropout(p=self.dropout)(x)
        x = self.hidden2(x)  # apply hidden layer
        x = nn.Dropout(p=self.dropout)(x)
        x = torch.tanh(x)  # apply sigmoid non-linearity
        x = nn.Dropout(p=self.dropout)(x)
        x = self.final(x)  # apply final layer
        x = x + self.bias  # correct for bias

        return x

    def update_bias(self, bias_value: torch.Tensor) -> None:
        """Update bias due to HSIC location invariance.

        Args:
            bias_value (torch.Tensor): _description_

        Raises:
            ValueError: _description_
        """
        if bias_value.shape == self.bias.shape:
            self.bias.copy_(bias_value)
        else:
            raise ValueError("Shape mismatch")


def make_preds_single(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Helper function for predicting.

    Args:
        model (nn.Model): nn model instance
        X (torch.Tensor): test data

    Returns:
        _type_: _description_
    """
    # helper function to make predictions for a model
    with torch.no_grad():
        y_hat = model(X)
    return y_hat


class Multioutcome_MLP(MultiRegressor):
    """Fit simple MLP with one hidden layer."""

    def __init__(
        self,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
        loss: str = "mse",
        dropout_proba: float = 0.6,
        n_epochs: int = 300,
        patience: int = 50,
        learning_rate: float = 0.01,
        val_size: float = 0.2,
        batch_size: int = 200,
        es: bool = True,
    ) -> None:
        """Initialize MLP.

        Args:
            rng (np.random.Generator, optional): _description_.
                Defaults to np.random.default_rng(seed=2024).
            loss (str, optional): Standard mse loss is default.
                Other options are `hsic` and `disco`.
                Defaults to "mse".
            dropout_proba (float, optional): _description_. Defaults to 0.6.
            n_epochs (int): number of times the data gets passed trough the MLP.
                Defaults to 6.
            patience (int, optional): Minimal number of epochs to train before early stopping
                applies
            learning_rate (float, optional): _description_. Defaults to 1e-3.
            val_size (float): Relative size of the validation dataset
            batch_size (int): Batch size.
            es (bool, optional): Early stopping. Defaults to true.
        """
        super().__init__(rng)

        self.training_info: pd.DataFrame
        self.prediction: torch.Tensor
        self.input_dim: int
        self.output_dim: int
        self._Y_test: np.ndarray | torch.Tensor
        self._X_test: np.ndarray | torch.Tensor
        self._model: nn.Module
        self.dropout_proba = dropout_proba
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_size = val_size
        self.es = es

        self.loss_name = loss

        has_mps = torch.backends.mps.is_built()
        self.device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"

    def _compute_bias(
        self,
        model: nn.Module,
        y_true: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        y_pred_interim: torch.Tensor = model(x)
        return y_true.mean(dim=0) - y_pred_interim.mean(dim=0)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(dim=0)) / x.std(dim=0)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        idx_train: np.ndarray | None = None,
        idx_test: np.ndarray | None = None,
    ) -> None:
        """Fit the MLP model.

        Args:
            X (np.ndarray): Input data
            Y (np.ndarray): Target data
            idx_train (np.ndarray): training indices
            idx_test (np.ndarray): test indices

        Returns:
            int: epoch at which training was stopped.
        """
        if idx_train is None and idx_test is None:
            X_train, X_val, Y_train, Y_val = self.split_and_standardize(
                X=X,
                Y=Y,
                test_size=self.val_size,
            )
        else:
            X_train, X_val, Y_train, Y_val = (X[idx_train], X[idx_test], Y[idx_train], Y[idx_test])
            X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0))
            Y_train = (Y_train - Y_train.mean(axis=0)) / (Y_train.std(axis=0))
            X_val = (X_val - X_train.mean(axis=0)) / (X_train.std(axis=0))
            Y_val = (Y_val - Y_train.mean(axis=0)) / (Y_train.std(axis=0))

        if self.loss_name == "hisc":
            loss_fn = loss_hsic
        else:
            loss_fn = loss_mse

        self._X_train = torch.from_numpy(X_train).float().to(self.device)
        self._Y_train = torch.from_numpy(Y_train).float().to(self.device)

        self._X_val = torch.from_numpy(X_val).float().to(self.device)
        self._Y_val = torch.from_numpy(Y_val).float().to(self.device)

        model = MLP(
            input_dim=self._X_train.shape[1],
            output_dim=self._Y_train.shape[1],
            dropout=self.dropout_proba,
        ).to(self.device)

        if self.es:
            es = EarlyStopping(patience=self.patience)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            amsgrad=True,
        )

        dataset_train = TensorDataset(self._X_train, self._Y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)

        history_train = []
        history_val = []
        trainstep = 0
        for i in tqdm(range(self.n_epochs), ncols=100):
            for j, (x_batch, y_batch) in enumerate(dataloader_train):
                Y_pred = model(x_batch)
                loss = loss_fn(
                    self._standardize(x_batch),
                    self._standardize(Y_pred),
                    self._standardize(y_batch),
                    device=self.device,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                history_train.append(
                    {
                        "epoch": i,
                        "minibatch": j,
                        "trainstep": trainstep,
                        "task": "regression",
                        "loss": loss.detach().cpu().numpy(),
                    }
                )
                trainstep += 1

            # Evaluate Model
            with torch.no_grad():
                if self.loss_name != "mse":
                    bias = self._compute_bias(model=model, y_true=self._Y_val, x=self._X_val)
                    model.update_bias(bias)
                Y_pred = model(self._X_val)
                vloss = loss_fn(
                    self._standardize(self._X_val),
                    self._standardize(Y_pred),
                    self._standardize(self._Y_val),
                    device=self.device,
                )

                history_val.append(
                    {
                        "epoch": i,
                        "trainstep": trainstep,
                        "task": "regression",
                        # "lr": lr_scheduler.get_last_lr()[0],
                        "lr": self.learning_rate,
                        "loss": vloss.detach().cpu().numpy(),
                    }
                )
                if self.es:
                    if es(model, vloss):
                        print(f"{es.status}")
                        break

        self.training_info = pd.DataFrame(history_train)
        self.validation_info = pd.DataFrame(history_val)
        self.prediction = make_preds_single(model, self._X_val)
        self._model = model
        self.final_loss = (
            es.best_loss.detach().cpu().numpy() if self.es and es.best_loss else np.infty
        )

    def predict(self, X_test: np.ndarray | None = None) -> np.ndarray:
        """Make predictions.

        Returns:
            np.ndarray: Predictions
        """
        if X_test is None:
            prediction = self.prediction.detach().numpy()
        else:
            X_torch_test = torch.from_numpy(X_test).float().to(self.device)
            pred = make_preds_single(self._model, X_torch_test)
            prediction = pred.detach().cpu().numpy()
        return prediction

    def plot_training_info(self) -> None:
        """Plot some summary over the MSE during training."""
        sns.lmplot(
            x="trainstep",
            y="loss",
            hue="task",
            lowess=True,
            scatter_kws={"alpha": 0.5},
            line_kws={"linewidth": 2},
            data=self.training_info,
        )

        sns.lmplot(
            x="epoch",
            y="loss",
            hue="task",
            lowess=True,
            scatter_kws={"alpha": 0.5},
            line_kws={"linewidth": 2},
            data=self.validation_info,
        )


class EarlyStopping:
    """Early stopping to conserve compute resources."""

    def __init__(
        self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True
    ) -> None:
        """Initializes the EarlyStopping class.

        Args:
            patience (int, optional): Number of epochs to wait for the validation error to improve.
                Defaults to 5.
            min_delta (float, optional): Minimum change that should be considered an improvement.
                Defaults to 0.0.
            restore_best_weights (bool, optional): Restores the weights to the values they were when
                the validation set was best. Defaults to True.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss: torch.Tensor | None = None
        self.counter = 0
        self.status = ""

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
        """Stopping actions.

        Args:
            model (nn.Module): NN model
            val_loss (float): Value of loss function.

        Returns:
            bool: True if training may be concluded.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = "Early stopping triggered"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


class MultitaskGPModel(gpytorch.models.ExactGP):  # type: ignore
    """Multitask GPR."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods._GaussianLikelihoodBase,
    ) -> None:
        """Inits the model.

        Args:
            train_x (_type_): _description_
            train_y (_type_): _description_
            likelihood (_type_): _description_
        """
        super().__init__(train_x, train_y, likelihood)
        num_tasks = train_y.size(1)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class MultioutcomeGPR(MultiRegressor):
    """MultioutcomeGPR Gaussian Process Regression class."""

    def __init__(
        self,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
        n_epochs: int = 300,
        patience: int = 50,
        learning_rate: float = 0.01,
        val_size: float = 0.2,
        batch_size: int | None = None,
        es: bool = True,
    ) -> None:
        """Initialize MLP.

        Args:
            rng (np.random.Generator, optional): _description_.
                Defaults to np.random.default_rng(seed=2024).
            n_epochs (int): number of times the data gets passed trough the MLP.
                Defaults to 6.
            patience (int, optional): Minimal number of epochs to train before early stopping
                applies
            learning_rate (float, optional): _description_. Defaults to 1e-3.
            val_size (float): Relative size of the validation dataset
            batch_size (int): Batch size.
            es (bool, optional): Early stopping.
        """
        super().__init__(rng)

        self.training_info: pd.DataFrame
        self.prediction: torch.Tensor
        self._Y_test: np.ndarray | torch.Tensor
        self._X_test: np.ndarray | torch.Tensor
        self._model: nn.Module
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_size = val_size
        self.es = es

        has_mps = torch.backends.mps.is_built()
        self.device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        idx_train: np.ndarray | None = None,
        idx_test: np.ndarray | None = None,
    ) -> None:
        """Fit the MLP model.

        Args:
            X (np.ndarray): Input data
            Y (np.ndarray): Target data
            idx_train (np.ndarray): training indices
            idx_test (np.ndarray): test indices

        Returns:
            int: epoch at which training was stopped.
        """
        if idx_train is None and idx_test is None:
            X_train, X_val, Y_train, Y_val = self.split_and_standardize(
                X=X,
                Y=Y,
                test_size=self.val_size,
            )
        else:
            X_train, X_val, Y_train, Y_val = (X[idx_train], X[idx_test], Y[idx_train], Y[idx_test])

        self._X_train = torch.from_numpy(X_train).float().to(self.device)
        self._Y_train = torch.from_numpy(Y_train).float().to(self.device)

        self._X_val = torch.from_numpy(X_val).float().to(self.device)
        self._Y_val = torch.from_numpy(Y_val).float().to(self.device)

        num_tasks = self._Y_train.size(1)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        model = MultitaskGPModel(self._X_train, self._Y_train, likelihood).to(self.device)
        if self.es:
            es = EarlyStopping(patience=self.patience)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        history_val = []
        for i in range(self.n_epochs):
            optimizer.zero_grad()
            output = model(self._X_train)
            loss = -mll(output, self._Y_train)
            loss.backward()
            optimizer.step()

            # Evaluate Model
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                model.eval().to(self.device)
                likelihood.eval().to(self.device)
                predictions = likelihood(model(self._X_val))
                Y_pred = predictions.mean
                vloss = loss_mse(self._X_val, Y_pred, self._Y_val, device=self.device)

                history_val.append(
                    {
                        "epoch": i,
                        "task": "regression",
                        "lr": self.learning_rate,
                        "loss": vloss.detach().cpu().numpy(),
                    }
                )
                if self.es:
                    if es(model, vloss):
                        print(f"{es.status}")
                        break

        self.validation_info = pd.DataFrame(history_val)
        self.prediction = make_preds_single(model, self._X_val)
        self._model = model
        self._likelihood = likelihood
        self.final_loss = es.best_loss if self.es else None

    def predict(self, X_test: np.ndarray | None = None) -> np.ndarray:
        """Make predictions.

        Returns:
            np.ndarray: Predictions
        """
        if X_test is None:
            prediction = self.prediction.detach().numpy()
        else:
            X_torch_test = torch.from_numpy(X_test).float().to(self.device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_model = self._model.eval().to(self.device)
                pred_likelihood = self._likelihood.eval().to(self.device)
                predictions = pred_likelihood(pred_model(X_torch_test))
                pred = predictions.mean
            prediction = pred.detach().cpu().numpy()
        return prediction

    def plot_training_info(self) -> None:
        """Plot some summary over the MSE during training."""
        sns.lmplot(
            x="trainstep",
            y="loss",
            hue="task",
            lowess=True,
            scatter_kws={"alpha": 0.5},
            line_kws={"linewidth": 2},
            data=self.training_info,
        )

        sns.lmplot(
            x="epoch",
            y="loss",
            hue="task",
            lowess=True,
            scatter_kws={"alpha": 0.5},
            line_kws={"linewidth": 2},
            data=self.validation_info,
        )
