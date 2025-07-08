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

from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import sklearn
import sklearn.linear_model
import xgboost as xgb
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MultiRegressor(metaclass=ABCMeta):
    """Abstract class for multivariate regression."""

    def __init__(
        self,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
        **kwargs: dict[str, Any],
    ) -> None:
        """Base class for regressors."""
        self.rng = rng

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit the model.

        Args:
            X (np.ndarray): Matrix of predictors
            Y (np.ndarray): Matrix of responses
        """

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict given data matrix.

        Args:
            X_test (np.ndarray): data matrix to predict on.

        Returns:
            np.ndarray: (Matrix of) predicted values
        """

    def mse(self, Y_test: np.ndarray, X_test: np.ndarray) -> float:
        """Mean squared error.

        Args:
            Y_test (np.ndarray): Test response
            X_test (np.ndarray): Test predictors

        Returns:
            float: MSE
        """
        Yhat = self.predict(X_test)
        mse: float = (np.square(Y_test - Yhat) / np.prod(Y_test.shape)).mean()
        return mse

    def standardize(self, a: np.ndarray) -> np.ndarray:
        """Standardize data.

        Args:
            a (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        return (a - np.mean(a, axis=0)) / np.std(a, axis=0)

    def split_and_standardize(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Train test split and standardize data.

        Args:
            X (np.ndarray): predictors
            Y (np.ndarray): targets
            test_size (float): Size of test data

        Returns:
            tuple[np.ndarray]: X_train, X_test, Y_train, Y_test
        """
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=2024
        )

        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        return X_train, X_test, Y_train, Y_test


class SimultaneousLinearModel(MultiRegressor):
    """Class for performing multivariate linear Regression."""

    def __init__(
        self,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
        alpha: float = 0.1,
    ) -> None:
        """Initializes with a ridge penalty equal to 0.1.

        Args:
            rng (np.random.Generator): A random generator
            alpha (float, optional): Penalty term. Defaults to 0.1.
        """
        super().__init__(rng)

        self.alpha: float = alpha
        self.reg: Ridge

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit multivariate linear regression.

        Args:
            X (np.ndarray): Matrix of predictors
            Y (np.ndarray): Matrix of responses
        """
        self._X_test: np.ndarray = X
        self.reg = Ridge(alpha=self.alpha).fit(X, Y)

    def __str__(self) -> str:
        """Class description."""
        return "Multivariate Linear Regression"

    def predict(self, X_test: np.ndarray | None = None) -> np.ndarray:
        """Predict given data matrix.

        Args:
            X_test (np.ndarray): data matrix to predict on.

        Returns:
            np.ndarray: (Matrix of) predicted values
        """
        if X_test is None:
            X_test = self._X_test
        return self.reg.predict(X_test)


class CurdsWhey(MultiRegressor):
    """Breiman and Friedman's curds and whey multivariate regression model.

    When regression problem is of multivariate nature and the outcome variables are
    related among another, more accurate predictions may be obtained by using a linear combination
    of the OLS predictors.
    """

    def __init__(self) -> None:
        """Initializes C&W linear shrinkage method."""
        self._ols: sklearn.linear_model.LinearRegression
        self._T: np.ndarray
        self._D: np.ndarray
        self._response_transformed: StandardScaler

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fits the C&W model to the multivariate X and Y.

        Args:
            X (np.ndarray): Matrix of predictors
            Y (np.ndarray): Matrix of responses
        """
        n = X.shape[0]
        p = X.shape[1]
        r = p / n
        q = Y.shape[1]

        Y_transform = StandardScaler().fit(Y)
        Y_std = Y_transform.transform(Y)

        cca = CCA(n_components=q)
        cca.fit(X, Y_std)
        X_c, Y_c = cca.transform(X, Y_std)
        c_k = np.corrcoef(X_c.T, Y_c.T).diagonal(offset=q)
        T = cca.y_rotations_

        # Computing Diagonal Shrinkage
        denom_1 = (c_k**2) * ((1 - r) ** 2)
        denom_2 = (1 - c_k**2) * (r**2)
        numer = (1 - r) * (c_k**2 - r)
        ds = numer / (denom_1 + denom_2)
        ds[ds < 0] = 0
        D = np.diag(ds)

        # Predicting values
        ols_cw = LinearRegression().fit(X, Y_std)
        self._ols = ols_cw
        self._T = T
        self._D = D
        self._response_transformed = Y_transform
        self._X_test = X

    def predict(self, X_test: np.ndarray | None = None) -> np.ndarray:
        """Predict response matrix based on X data.

        Args:
            X_test (np.ndarray): Matrix of responses

        Returns:
            np.ndarray: predicted values.
        """
        if X_test is None:
            X_test = self._X_test
        Y_tilde = self._ols.predict(X=X_test)
        return self._response_transformed.inverse_transform(
            Y_tilde @ self._T @ self._D @ np.linalg.inv(self._T)
        )


class ReducedRankRegressor(MultiRegressor):  # , BaseEstimator):
    """Kernel Reduced Rank Ridge Regression."""

    def __init__(self, rank: int, alpha: np.float64 = 1.0) -> None:
        """Initializes the model.

        Args:
            rank (int): Rank constraint.
            alpha (np.float64, optional): Regularization parameter. Defaults to 1.0.
        """
        self.rank = rank
        self.alpha = alpha
        self._P_rr: np.ndarray
        self._Q_fr: np.ndarray
        self._X_train: np.ndarray
        self._X_test: np.ndarray
        self._Y_test: np.ndarray | None = None

    def __str__(self) -> str:
        """Method print."""
        return f"kernel Reduced Rank Ridge Regression \
            by Mukherjee (rank = f{self.rank})"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit kRRR model to data.

        Args:
            X (np.ndarray): training data predictors
            Y (np.ndarray): training (multivariate) response
        """
        K_X: np.ndarray = np.dot(X, X.T)
        tmp_1 = self.alpha * np.identity(K_X.shape[0]) + K_X
        Q_fr = np.linalg.solve(tmp_1, Y)
        P_fr = np.linalg.eig(np.dot(Y.T, np.dot(K_X, Q_fr)))[1].real
        P_rr = np.dot(P_fr[:, 0 : self.rank], P_fr[:, 0 : self.rank].T)

        self._Q_fr = Q_fr
        self._P_rr = P_rr
        self._X_train = X
        self._X_test = X

    def predict(self, X_test: np.ndarray | None = None) -> np.ndarray:
        """Predict fitted kRRR model.

        Args:
            X_test (np.ndarray): Test data to predict on.

        Returns:
            np.ndarray: Predicted values.
        """
        if X_test is None:
            X_test = self._X_test
        K_Xx = np.dot(X_test, self._X_train.T)

        return np.dot(K_Xx, np.dot(self._Q_fr, self._P_rr))


class BoostedRegressionTrees(MultiRegressor):
    """Boosted multi-outcome regression trees.

    This is simply a wrapper around the xgboost `XGBRegressor` class.
    """

    def __init__(self) -> None:
        """Initializes the `XGBRegressor` object."""
        self.clf = xgb.XGBRegressor(tree_method="hist", multi_strategy="multi_output_tree")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit boosted multi-outcome regression trees to training data.

        Args:
            X (np.ndarray): training data predictors
            Y (np.ndarray): training data (multi-) targets
        """
        self.clf.fit(X, Y)
        self._X_test = X

    def predict(self, X_test: np.ndarray | None = None) -> np.ndarray:
        """Predict using fitted boosted multi-outcome regression trees.

        Args:
            X_test (np.ndarray): Test data to predict on.

        Returns:
            np.ndarray: Predicted values.
        """
        if X_test is None:
            X_test = self._X_test
        return self.clf.predict(X_test)
