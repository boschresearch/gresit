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
from itertools import product

import numpy as np
import pandas as pd
from causallearn.utils.KCI import KCI
from dcor import u_distance_covariance_sqr
from scipy.stats import gamma, norm, rankdata
from statsmodels.nonparametric import bandwidths


class Itest(metaclass=ABCMeta):
    """Abstract meta class for independence tests."""

    def _check_input(
        self,
        x_data: np.ndarray | pd.DataFrame | pd.Series,
        y_data: np.ndarray | pd.DataFrame | pd.Series,
    ) -> None:
        if not isinstance(x_data, np.ndarray | pd.DataFrame | pd.Series):
            raise TypeError("x_data must be of type np.ndarray, pd.DataFrame, or pd.Series")
        if not isinstance(y_data, np.ndarray | pd.DataFrame | pd.Series):
            raise TypeError("y_data must be of type np.ndarray, pd.DataFrame, or pd.Series")

    @abstractmethod
    def test(
        self,
        x_data: np.ndarray | pd.DataFrame | pd.Series,
        y_data: np.ndarray | pd.DataFrame | pd.Series,
    ) -> tuple[float, float | str]:
        """Abstract method for independence tests.

        Args:
            x_data (np.ndarray | pd.DataFrame | pd.Series): Variables involved in the test
            y_data (np.ndarray | pd.DataFrame | pd.Series): Variables involved in the test


        Returns:
            tuple[float, float | str]: Test statistic and corresponding pvalue (Test decision).
        """


def xi_vec_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the correlation coefficient between x and y.

    according to the xi coefficient defined by Chatterjee.

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_
    """

    def rank_order(vector: np.ndarray) -> list[np.ndarray]:
        random_index = np.random.choice(np.arange(length), length, replace=False)
        randomized_vector = vector[random_index]
        ranked_vector = rankdata(randomized_vector, method="ordinal")
        answer = [ranked_vector[j] for _, j in sorted(zip(random_index, range(length)))]
        return answer

    def compute_d_sequence(y: np.ndarray) -> float:
        ell = rankdata([-i for i in y], method="max")
        return float(np.sum(ell * (length - ell)) / (length**3))

    def compute_xi_coefficient(vector: np.ndarray) -> float:
        mean_absolute = np.sum(np.abs([a - b for a, b in zip(vector[:-1], vector[1:])]))
        return float(1 - mean_absolute / (2 * (length**2) * d_sequence))

    def distance_transform(x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        m = n * (n + 1) / 2

        if x.ndim == 1:
            diff = x[np.newaxis] - x[..., np.newaxis]
            a = np.linalg.norm(diff[..., np.newaxis], axis=-1)
        else:
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            a = np.linalg.norm(diff, axis=-1)

        H = np.eye(n) - 1.0 / n * np.ones((n, n))
        A = H @ a @ H
        indices = np.triu_indices(A.shape[0])
        D_kx = A[indices]
        if D_kx.shape[0] != m:
            raise ValueError("Shapes do not agree,")
        return D_kx

    x_transformed = distance_transform(x)
    y_transformed = distance_transform(y)

    length = len(x_transformed)
    x_ordered = np.argsort(rank_order(x_transformed))
    y_rank_max = rankdata(y_transformed, method="max")
    x_ordered_max_rank = y_rank_max[x_ordered]
    d_sequence = compute_d_sequence(y_transformed)
    correlation = compute_xi_coefficient(x_ordered_max_rank)

    return float(correlation)


class CItest(metaclass=ABCMeta):
    """Abstract meta class for independence tests."""

    def _check_input(
        self,
        x_data: np.ndarray | pd.DataFrame | pd.Series,
        y_data: np.ndarray | pd.DataFrame | pd.Series,
        z_data: np.ndarray | pd.DataFrame | pd.Series | None = None,
    ) -> None:
        if not isinstance(x_data, np.ndarray | pd.DataFrame | pd.Series):
            raise TypeError("x_data must be of type np.ndarray, pd.DataFrame, or pd.Series")
        if not isinstance(y_data, np.ndarray | pd.DataFrame | pd.Series):
            raise TypeError("y_data must be of type np.ndarray, pd.DataFrame, or pd.Series")
        if not isinstance(z_data, np.ndarray | pd.DataFrame | pd.Series) or z_data is None:
            raise TypeError("y_data must be of type np.ndarray, pd.DataFrame, or pd.Series")

    @abstractmethod
    def test(
        self,
        x_data: np.ndarray | pd.DataFrame | pd.Series,
        y_data: np.ndarray | pd.DataFrame | pd.Series,
        z_data: np.ndarray | pd.DataFrame | pd.Series | None = None,
    ) -> tuple[float, float]:
        """Abstract method for independence tests.

        Args:
            x_data (np.ndarray | pd.DataFrame | pd.Series): Variables involved in the test
            y_data (np.ndarray | pd.DataFrame | pd.Series): Variables involved in the test
            z_data (np.ndarray | pd.DataFrame | pd.Series | None): Variables involved in the test


        Returns:
            tuple[float, float]: Test statistic and corresponding pvalue (Test decision).
        """


class KernelCI(CItest):
    """Kernel HSIC wrapper around causal-learn."""

    def __init__(self) -> None:
        """Init of the object."""
        pass

    def test(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        z_data: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """KCI test wrapper.

        Args:
            x_data (np.ndarray): _description_
            y_data (np.ndarray): _description_
            z_data (np.ndarray | None, optional): _description_. Defaults to None.

        Returns:
            tuple[float, float]: Test statistic and p_value.
        """
        if z_data is None:
            unconditional_test = KCI.KCI_UInd()
            p_value, test_stat = unconditional_test.compute_pvalue(data_x=x_data, data_y=y_data)
        else:
            conditional_test = KCI.KCI_CInd()
            p_value, test_stat = conditional_test.compute_pvalue(
                data_x=x_data, data_y=y_data, data_z=z_data
            )

        return float(test_stat), float(p_value)


class FisherZVec(CItest):
    """Simple extension of standard Fisher-Z test for independence."""

    def __init__(self) -> None:
        """Init of the object."""
        pass

    def test(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        z_data: np.ndarray | None = None,
        corr_threshold: float = 0.999,
    ) -> tuple[float, float]:
        """Retrieve (composite) p_value using Fisher z-transformation.

        Appropriate when data is jointly Gaussian.

        Args:
            x_data (np.ndarray): X_data.
            y_data (np.ndarray): Y_data.
            z_data (np.ndarray | None): Z_data. defaults to None.
            corr_threshold (float, optional): Threshold to make sure
                r in [-1,1]. Defaults to 0.999.

        Returns:
            tuple[float,float]: test_statistic, p_value
        """
        n = x_data.shape[0]

        if z_data is not None:
            sep_set_length = z_data.shape[1]
            corrdata = np.empty((n, 2 + z_data.shape[1], x_data.shape[1] * y_data.shape[1]))
            k = -1
            for i, j in product(range(x_data.shape[1]), range(y_data.shape[1])):
                k += 1
                corrdata[:, :, k] = np.concatenate(
                    [x_data[:, i][:, np.newaxis], y_data[:, j][:, np.newaxis], z_data], axis=1
                )
            precision_matrices = np.empty(
                (2 + z_data.shape[1], 2 + z_data.shape[1], x_data.shape[1] * y_data.shape[1])
            )
            for k in range(precision_matrices.shape[-1]):
                corrmat = np.corrcoef(corrdata[:, :, k].T)
                try:
                    precision_matrices[:, :, k] = np.linalg.inv(corrmat)
                except np.linalg.LinAlgError as error:
                    raise ValueError(
                        "The correlation matrix of your data is singular. \
                        Partial correlations cannot be estimated. Are there  \
                        collinearities in your data?"
                    ) from error

            r = np.empty(precision_matrices.shape[-1])
            for k in range(precision_matrices.shape[-1]):
                precision_matrix = precision_matrices[:, :, k]
                r[k] = (
                    -1
                    * precision_matrix[0, 1]
                    / np.sqrt(np.abs(precision_matrix[0, 0] * precision_matrix[1, 1]))
                )
        else:
            sep_set_length = 0
            uncond = []
            for i in range(x_data.shape[1]):
                uncond.append(
                    np.corrcoef(np.concatenate([x_data[:, i][:, np.newaxis], y_data], axis=1).T)[
                        1:, 0
                    ]
                )
            r = np.concatenate(uncond)

        r = np.minimum(
            corr_threshold, np.maximum(-1 * corr_threshold, r)
        )  # make r between -1 and 1
        # Fisher’s z-transform
        factor = np.sqrt(n - sep_set_length - 3)
        z_transform = factor * 0.5 * np.log((1 + r) / (1 - r))
        test_stat = factor * z_transform
        p_value = 2 * (1 - norm.cdf(abs(z_transform)))

        final_test_stat = test_stat[np.argmin(np.abs(test_stat))]
        final_p_value = np.max(p_value)

        return (float(final_test_stat), float(final_p_value))


class HSIC(Itest):
    """Hilbert-Schmidt Independence Criterion (HSIC) test."""

    def test(
        self,
        x_data: np.ndarray | pd.DataFrame | pd.Series,
        y_data: np.ndarray | pd.DataFrame | pd.Series,
        bw_method: str = "mdbs",
    ) -> tuple[float, float]:
        """Test for independence between two vectors.

        Args:
            x_data (np.ndarray | pd.DataFrame | pd.Series): x-data involved in the test
            y_data (np.ndarray | pd.DataFrame | pd.Series): y-data involved in the test
            bw_method (str, optional): The method used to calculate the bandwidth of the HSIC.
                * ``mdbs`` : Median distance between samples.
                * ``scott`` : Scott's Rule of Thumb.
                * ``silverman`` : Silverman's Rule of Thumb.. Defaults to "mdbs".


        Returns:
            tuple[float, float]: Test statistic and corresponding pvalue.
        """
        self._check_input(x_data, y_data)
        x_data = x_data if isinstance(x_data, np.ndarray) else x_data.values
        y_data = y_data if isinstance(y_data, np.ndarray) else y_data.values
        test_stat, p_value = self.hsic_test_gamma(X=x_data, Y=y_data, bw_method=bw_method)
        return test_stat, p_value

    def get_kernel_width(self, X: np.ndarray, sample_cut: int = 100) -> np.float64:
        """Calculate the bandwidth to median distance between points.

        Use at most 100 points (since median is only a heuristic,
        and 100 points is sufficient for a robust estimate).

        Args:
            X (np.ndarray): shape (n_samples, n_features) Training data,
            sample_cut (int, optional): Number of samples to use for bandwidth calculation.

        Returns:
            float: The bandwidth parameter.
        """
        n_samples = X.shape[0]
        if n_samples > sample_cut:
            X_med = X[:sample_cut, :]
            n_samples = sample_cut
        else:
            X_med = X

        G = np.sum(X_med * X_med, 1).reshape(n_samples, 1)
        dists = G + G.T - 2 * np.dot(X_med, X_med.T)
        dists = dists - np.tril(dists)
        dists = dists.reshape(n_samples**2, 1)

        return np.sqrt(0.5 * np.median(dists[dists > 0]))

    def _rbf_dot(self, X: np.ndarray, width: float) -> np.ndarray:
        """Calculate rbf dot, in special case with X dot X.

        Args:
            X (np.ndarray): data
            width (float): bandwidth parameter

        Returns:
            np.ndarray: Kernel matrix.
        """
        G = np.sum(X * X, axis=1)
        H = G[None, :] + G[:, None] - 2 * np.dot(X, X.T)
        return np.exp(-H / 2 / (width**2))

    def get_gram_matrix(self, X: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
        """Get the centered gram matrices.

        Args:
            X (np.ndarray): shape (n_samples, n_features)
                Training data, where ``n_samples`` is the number of samples
                and ``n_features`` is the number of features.
            width (float): The bandwidth parameter.

        Returns:
            tuple[np.ndarray, np.ndarray]: The centered gram matrices.
        """
        n = X.shape[0]

        K = self._rbf_dot(X, width)
        K_colsums = K.sum(axis=0)
        K_rowsums = K.sum(axis=1)
        K_allsum = K_rowsums.sum()
        Kc = K - (K_colsums[None, :] + K_rowsums[:, None]) / n + (K_allsum / n**2)
        return K, Kc

    def hsic_teststat(self, Kc: np.ndarray, Lc: np.ndarray, n: int) -> np.float64:
        """Get the HSIC statistic.

        Args:
            Kc (np.ndarray): Centered gram matrix.
            Lc (np.ndarray): Centered gram matrix.
            n (int): Sample size.

        Returns:
            float: HSIC statistic.
        """
        # test statistic m*HSICb under H1
        return 1 / n * np.sum(Kc.T * Lc)

    def hsic_test_gamma(
        self, X: np.ndarray, Y: np.ndarray, bw_method: str = "mdbs"
    ) -> tuple[float, float]:
        """Get the HSIC statistic and p-value.

        Args:
            X (np.ndarray): data, possibly vector-valued.
            Y (np.ndarray): data, possibly vector-valued.
            bw_method (str, optional): The method used to calculate the bandwidth of the HSIC.
                * ``mdbs`` : Median distance between samples.
                * ``scott`` : Scott's Rule of Thumb.
                * ``silverman`` : Silverman's Rule of Thumb.. Defaults to "mdbs".

        Returns:
            tuple[float, float]: HSIC test statistic and corresponding p-value
        """
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        if bw_method == "scott":
            width_x = bandwidths.bw_scott(X)
            width_y = bandwidths.bw_scott(Y)
        elif bw_method == "silverman":
            width_x = bandwidths.bw_silverman(X)
            width_y = bandwidths.bw_silverman(Y)
        # Get kernel width to median distance between points
        else:
            width_x = self.get_kernel_width(X)
            width_y = self.get_kernel_width(Y)

        # these are slightly biased estimates of centered gram matrices
        K, Kc = self.get_gram_matrix(X, width_x)
        L, Lc = self.get_gram_matrix(Y, width_y)

        # test statistic m*HSICb under H1
        n = X.shape[0]
        test_stat = self.hsic_teststat(Kc, Lc, n)

        var = (1 / 6 * Kc * Lc) ** 2
        # second subtracted term is bias correction
        var = 1 / n / (n - 1) * (np.sum(var) - np.trace(var))
        # variance under H0
        var = 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) * var

        K[np.diag_indices(n)] = 0
        L[np.diag_indices(n)] = 0
        mu_X = 1 / n / (n - 1) * K.sum()
        mu_Y = 1 / n / (n - 1) * L.sum()
        # mean under H0
        mean = 1 / n * (1 + mu_X * mu_Y - mu_X - mu_Y)

        alpha = mean**2 / var
        # threshold for hsicArr*m
        beta = var * n / mean
        p = gamma.sf(test_stat, alpha, scale=beta)

        return test_stat, p


class DISCO(Itest):
    """Simple Wrapper class around the squared distance covariance from the dcor class."""

    def __init__(self) -> None:
        """Inits the object."""
        pass

    def test(
        self,
        x_data: np.ndarray | pd.DataFrame | pd.Series,
        y_data: np.ndarray | pd.DataFrame | pd.Series,
    ) -> tuple[float, str]:
        """Test for independence between two vectors Finite joint first moments are assumed.

        Args:
            x_data (np.ndarray | pd.DataFrame | pd.Series): x-data involved in the test
            y_data (np.ndarray | pd.DataFrame | pd.Series): y-data involved in the test

        Returns:
            tuple[float, str]: Distance covariance value and some string to comply with format.
        """
        self._check_input(x_data, y_data)
        return u_distance_covariance_sqr(x=x_data, y=y_data), "Squared Distance Covariance"
