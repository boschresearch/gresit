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

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# from numba import njit
from skmisc.loess import loess

logger = logging.getLogger(__name__)


class MURGS:
    """Multi-Response Group Sparse Additive Mode (MURGS) class."""

    def __init__(self, group_names: list[str] | None = None) -> None:
        """Initializes the object."""
        self.zero_groups: list[bool]
        self.zero_group_history: dict[float, list[bool]] = {}
        self.d_g_long: np.ndarray
        self.p_g: int = 0
        self.n_tasks: int
        self.smoothing_matrices: np.ndarray | dict[str, np.ndarray]
        self.f_g_hat: np.ndarray | dict[str, np.ndarray]
        self.chosen_lambda: float = 0.0
        self.lambda_max_value: float | None = None
        self.steps_till_convergence: int = -1
        self.max_iter: int = 10000
        self.tol: float = 1e-8
        if group_names is not None:
            self.group_names = group_names
        else:
            self.group_names = []
        self.predicted_vals: np.ndarray

    def __repr__(self) -> str:
        """Print some useful information.

        Returns:
            str: _description_
        """
        return f"MT-GSpAM on {self.p_g} groups."

    def __str__(self) -> str:
        """Print some useful summary.

        Returns:
            str: _description_
        """
        display_limit = 5
        setting = {
            "Number of Groups: ": self.p_g,
            "Group sizes homogenous: ": True if isinstance(self.f_g_hat, np.ndarray) else False,
            "Steps until convergence: ": self.steps_till_convergence,
            "Final lambda: ": self.chosen_lambda,
            "Number of non-zero groups: ": self.p_g - np.asanyarray(self.zero_groups).sum(),
            "Non-zero group names: ": self.return_nonzero_groups()
            if len(self.return_nonzero_groups()) <= display_limit
            else "Too many to show",
        }
        s = ""
        for info, info_text in setting.items():
            s += f"{info}{info_text}\n"

        return s

    def return_nonzero_groups(self) -> list[str]:
        """Return the group names of the nonzero groups.

        Returns:
            list[str] | str: List of nonzero groups with their actual names if given.
        """
        if len(self.group_names) > 0:
            nonzero_groups = [
                group
                for group, zero_group in zip(self.group_names, self.zero_groups)
                if not zero_group
            ]
        else:
            nonzero_groups = ["You neither provided any group names nor ran the model."]

        return nonzero_groups

    def precalculate_smooths(
        self, X_data: np.ndarray | dict[str, np.ndarray], local_regression_method: str = "kernel"
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Precalculate smoother matrices.

        Input may be both `np.ndarray` and `dict`.

        Args:
            X_data (np.ndarray | dict[str, np.ndarray]): predictor data.
            local_regression_method (str): Method to use to calculate smoother matrix. Options
                currently are `"loess"` and `"kernel"`. When kernel is chosen, the default is set to
                Gaussian kernel regression with the standard deviation rule for bandwidth selection.

        Returns:
            np.ndarray | dict[str, np.ndarray]: smooths.
        """
        if isinstance(X_data, dict):
            X_data = self._dict_preprocessing(X_data=X_data)
            smoothing_matrices = {}
            for group, data in X_data.items():
                smoothing_matrices[group] = np.zeros((data.shape[0], data.shape[0], data.shape[1]))
                for j in range(data.shape[1]):
                    if local_regression_method == "loess":
                        smoothing_matrices[group][:, :, j] = self._make_loess_smoother_matrix(
                            data[:, j]
                        )
                    elif local_regression_method == "kernel":
                        smoothing_matrices[group][:, :, j] = (
                            self._make_gaussian_kernel_smoother_matrix(data[:, j])
                        )
                    else:
                        raise NotImplementedError("This smoothing method is not implemented yet.")
        else:
            smoothing_matrices = np.zeros(
                (X_data.shape[0], X_data.shape[0], X_data.shape[1], X_data.shape[2]),
                dtype=np.float64,
            )
            inner_smoother: np.ndarray = smoothing_matrices
            for num_groups in range(X_data.shape[2]):
                for group_members in range(X_data.shape[1]):
                    if local_regression_method == "loess":
                        inner_smoother[:, :, group_members, num_groups] = (
                            self._make_loess_smoother_matrix(X_data[:, group_members, num_groups])
                        )
                    elif local_regression_method == "kernel":
                        inner_smoother[:, :, group_members, num_groups] = (
                            self._make_gaussian_kernel_smoother_matrix(
                                X_data[:, group_members, num_groups]
                            )
                        )
                    else:
                        raise NotImplementedError("This smoothing method is not implemented yet.")
            smoothing_matrices = inner_smoother

        self.smoothing_matrices = smoothing_matrices
        return smoothing_matrices

    def _init_functions(
        self, X_data: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Initiate additive functional components.

        Args:
            X_data (np.ndarray | dict[str, np.ndarray]): predictor data.

        Returns:
            np.ndarray | dict[str, np.ndarray]: functions initiated.
        """
        if isinstance(X_data, dict):
            f_g = {}
            for group, data in X_data.items():
                f_g[group] = np.zeros(data.shape + (self.n_tasks,))
        else:
            f_g = np.zeros(X_data.shape + (self.n_tasks,))
        return f_g

    def _set_dg_pg(self, X_data: np.ndarray | dict[str, np.ndarray]) -> None:
        """Set group sizes and number of groups.

        Args:
            X_data (np.ndarray | dict[str, np.ndarray]): predictor data.
        """
        if isinstance(X_data, dict):
            self.d_g_long = np.array(
                [dat.shape[1] if len(dat.shape) > 1 else 1 for dat in X_data.values()]
            )
            # Number of predictors
            self.p_g = len(self.d_g_long)
        else:
            # Number of predictors
            self.p_g = X_data.shape[-1]

            # Number of variables per group
            self.d_g_long = np.repeat(X_data.shape[1], self.p_g)
            if not len(self.group_names) == self.p_g:
                self.group_names = [str(intgr) for intgr in range(self.p_g)]

    def _dict_preprocessing(self, X_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Dict preprocessing dealing with univariate groups.

        Args:
            X_data (dict[str, np.ndarray]): Predictor dict.

        Returns:
            dict[str, np.ndarray]: Predictor dict with axis added when univariate.
        """
        for key, value in X_data.items():
            X_data[key] = self._assure_two_dim(value)
        return X_data

    def _get_m_star(self, s_g_ordered: np.ndarray, penalty: float, d_g: int) -> int:
        """Get m* so as to differentiate the cases.

        Cases involve instanecs where more than one group attains the sup-norm.

        Args:
            s_g_ordered (np.ndarray): Ordered functional norm of group `g`.
            penalty (float): current penalty parameter.
            d_g (int): Number of predictors in group.

        Returns:
            int: m* value.
        """
        m_criterion = s_g_ordered.cumsum() - penalty * np.sqrt(d_g)
        m_prefix = np.array(range(1, m_criterion.shape[0] + 1))
        return int(m_prefix[np.argmax(m_criterion / m_prefix)])

    def smoother_direct_fit(
        self,
        g: int,
        d_g: int,
        R_g: np.ndarray,
        X_data: np.ndarray | dict[str, np.ndarray],
        local_regression_method: str,
    ) -> np.ndarray:
        """Direct fit of smoothing.

        Args:
            g (int): _description_
            d_g (int): _description_
            R_g (np.ndarray): _description_
            X_data (np.ndarray): _description_
            local_regression_method (str): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if local_regression_method == "loess":
            smooth_fit = self.loess_direct_fit(g=g, d_g=d_g, R_g=R_g, X_data=X_data)
        elif local_regression_method == "kernel":
            smooth_fit = self.gaussian_kernel_direct_fit(g=g, d_g=d_g, R_g=R_g, X_data=X_data)
        else:
            raise NotImplementedError("Smoothing method not implemented.")
        return smooth_fit

    def _init_or_insert_functions(
        self,
        X_data: np.ndarray | dict[str, np.ndarray],
        warm_start_f_hat: np.ndarray | dict[str, np.ndarray] | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        if warm_start_f_hat is None:
            f_g = self._init_functions(X_data=X_data)
        else:
            f_g = warm_start_f_hat
        return f_g

    def block_coordinate_descent(
        self,
        X_data: np.ndarray | dict[str, np.ndarray],
        Y_data: np.ndarray,
        penalty: float,
        precalculate_smooths: bool = True,
        smoothers: np.ndarray | dict[str, np.ndarray] | None = None,
        local_regression_method: str = "kernel",
        warm_start_f_hat: np.ndarray | dict[str, np.ndarray] | None = None,
    ) -> None:
        """Block coordinate descent for multitask Sparse Group Lasso.

        Args:
            X_data (np.ndarray | dict[str, np.ndarray]): _description_
            Y_data (np.ndarray): _description_
            penalty (float): _description_
            precalculate_smooths (bool, optional): _description_. Defaults to True.
            smoothers (np.ndarray | dict[str, np.ndarray] | None, optional): _description_.
                Defaults to None.
            local_regression_method (str): Defaults to "loess". Other options currently: "kernel".
            warm_start_f_hat (np.ndarray | dict[str, np.ndarray] | None, optional): _description_.
                Defaults to None.
        """
        if smoothers is None:
            smoothers = self.precalculate_smooths(
                X_data=X_data, local_regression_method=local_regression_method
            )

        self.n_tasks = Y_data.shape[1]

        if isinstance(X_data, dict):
            if not self.group_names:
                self.group_names = list(X_data.keys())
            X_data = self._dict_preprocessing(X_data=X_data)

        # Number of predictors and members in groups
        self._set_dg_pg(X_data=X_data)

        f_g = self._init_or_insert_functions(X_data, warm_start_f_hat)

        divergent_max_runs = 10
        divergent_runs = 0
        max_inc_old = 0.0
        new_functional_norm = np.zeros((self.p_g, self.n_tasks))
        # Update loop
        for t in range(self.max_iter):
            old_functional_norm = new_functional_norm
            zero_groups = [False] * self.p_g
            for g in range(self.p_g):
                d_g = self.d_g_long[g]
                # R_g_hat has shape (#samples, #tasks)
                R_g_hat = self.R_g_hat_update(f_g=f_g, Y_data=Y_data, g=g)

                if precalculate_smooths:
                    smooth_fit = self.predict_from_linear_smoother(
                        g=g, smoothing_matrices=smoothers, R_g=R_g_hat
                    )
                else:
                    smooth_fit = self.smoother_direct_fit(
                        g=g,
                        d_g=d_g,
                        X_data=X_data,
                        R_g=R_g_hat,
                        local_regression_method=local_regression_method,
                    )

                # # estimate of || Q^(k)R_g^(k) ||
                # # shape (1, #tasks)
                omega_g = self.functional_norm(smooth_fit)

                # sort in descending order for each task
                s_g_ordered = np.sort(omega_g)[::-1]

                # get m* for the case that more than one group attains the sup-norm.
                m_opt = self._get_m_star(s_g_ordered=s_g_ordered, penalty=penalty, d_g=d_g)

                # calculate nonzero groups
                zero_groups[g] = omega_g.sum() <= penalty * np.sqrt(d_g)
                f_g = self.soft_thresholding_update(
                    g=g,
                    f_g=f_g,
                    zero_groups=zero_groups,
                    smooth_fit=smooth_fit,
                    s_g_ordered=s_g_ordered,
                    penalty=penalty,
                    m_opt=m_opt,
                )

            new_functional_norm = self.omega_hat(f_g)
            max_inc: float = np.sqrt(
                np.square(new_functional_norm - old_functional_norm).sum(axis=(0, 1))
            )
            if max_inc_old < max_inc:
                divergent_runs += 1

            if divergent_runs > divergent_max_runs:
                raise ConvergenceError(
                    f"Model did not converge in group {g}. Penalty: {penalty} too low?"
                )
            max_inc_old = max_inc

            if np.all(max_inc < self.tol / 2):
                break

        self.f_g_hat = f_g
        self.zero_groups = zero_groups
        self.steps_till_convergence = t

    def R_g_hat_update(
        self, f_g: np.ndarray | dict[str, np.ndarray], Y_data: np.ndarray, g: int
    ) -> np.ndarray:
        """Update partial residuals.

        Args:
            f_g (np.ndarray): Current additive components
                of shape (#samples, #groups, #predictors, #tasks)
            Y_data (np.ndarray): Response data of shape (#samples, #tasks)
            g (int): group in question.

        Returns:
            np.ndarray: Partial Residuals.
        """
        if isinstance(f_g, np.ndarray):
            mask = np.ones(self.p_g, bool)
            mask[g] = False
            R_g = Y_data - f_g[:, :, mask, :].sum(axis=(1, 2))
        else:
            groups = list(f_g.keys())
            f_interim_sum = np.asanyarray(
                [value.sum(axis=1) for key, value in f_g.items() if key != groups[g]]
            )
            R_g = Y_data - f_interim_sum.sum(axis=0)

        return R_g

    def predict_from_linear_smoother(
        self,
        g: int,
        smoothing_matrices: np.ndarray | dict[str, np.ndarray],
        R_g: np.ndarray,
    ) -> np.ndarray:
        """Local regression fit.

        Args:
            g (int): group in question.
            R_g (np.ndarray): Partial residuals should be of
                shape (#n_samples. #n_predictors, #n_tasks)
            smoothing_matrices (np.ndarray): Precalculated smoothing matrices
                of shape (#n_samples, #n_samples #n_groups, #n_predictors)

        Returns:
            torch.Tensor: Fitted local regressions.
        """
        if isinstance(smoothing_matrices, np.ndarray):
            smoothing_matrix = smoothing_matrices[:, :, :, g]
        else:
            smoothing_matrix = list(smoothing_matrices.values())[g]

        return torch.einsum(
            "ijk, jl -> ikl", torch.from_numpy(smoothing_matrix), torch.from_numpy(R_g)
        ).numpy()

    def loess_direct_fit(
        self, g: int, d_g: int, R_g: np.ndarray, X_data: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray:
        """Local regression fit.

        Args:
            g (int): group in question.
            d_g (int): Number of predictors in group.
            R_g (np.ndarray): Partial residuals should be of
                shape (#n_samples. #n_predictors, #n_tasks)
            X_data (np.ndarray): Training data of shape (#n_samples. #n_groups, #n_predictors)

        Returns:
            np.ndarray: Fitted local regressions.
        """
        if isinstance(X_data, np.ndarray):
            data = X_data[:, :, g]
        else:
            data = list(X_data.values())[g]

        loess_g = np.zeros((R_g.shape[0], d_g, self.n_tasks))
        for k in range(self.n_tasks):
            for j in range(d_g):
                try:
                    loess_obj = loess(data[:, j], R_g[:, k])
                except ValueError:
                    loess_obj = loess(data[:, j], R_g[:, k], surface="direct")
                loess_obj.fit()
                loess_g[:, j, k] = loess_obj.outputs.fitted_values
        return loess_g

    def gaussian_kernel_direct_fit(
        self, g: int, d_g: int, R_g: np.ndarray, X_data: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray:
        """Local regression fit.

        Args:
            g (int): group in question.
            d_g (int): Number of predictors in group.
            R_g (np.ndarray): Partial residuals should be of
                shape (#n_samples. #n_predictors, #n_tasks)
            X_data (np.ndarray): Training data of shape (#n_samples. #n_groups, #n_predictors)

        Returns:
            np.ndarray: Fitted local regressions.
        """
        if isinstance(X_data, np.ndarray):
            data = X_data[:, :, g]
        else:
            data = list(X_data.values())[g]

        kernel_reg_g = np.zeros((R_g.shape[0], d_g, self.n_tasks))
        for k in range(self.n_tasks):
            for j in range(d_g):
                kernel_reg_g[:, j, k] = self._torch_gaussian_kernel_regression(
                    x_data=data[:, j],
                    y_data=R_g[:, k],
                    bandwidth=self.plugin_bandwidth(x_j=data[:, j]),
                )

        return kernel_reg_g

    def _torch_gaussian_kernel_regression(
        self, x_data: np.ndarray, y_data: np.ndarray, bandwidth: float
    ) -> np.ndarray:
        """Torch implementation of Gaussian kernel regression.

        Args:
            x_data (np.ndarray): X data
            y_data (np.ndarray): Y data
            bandwidth (float): bandwidth parameter

        Returns:
            np.ndarray: Predicted y values.
        """
        x_data = torch.from_numpy(x_data)[:, None]  # Reshape for broadcasting
        weights = torch.exp(
            -0.5 * ((x_data - x_data.T) / bandwidth) ** 2
        )  # Pairwise Gaussian weights
        weights /= weights.sum(dim=1, keepdim=True)  # Normalize weights along each row

        # Weighted sum to get predictions
        y_pred = weights @ torch.from_numpy(y_data)
        return y_pred.numpy()

    def _norm_2(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(np.square(x).sum(axis=0))

    def functional_norm(self, array_data: np.ndarray) -> np.ndarray:
        """Sample estimate of functional norm for arrays of shape.

            (#samples, #n_group_entries, #tasks).

        Args:
            array_data (np.ndarray): Array input data.

        Returns:
            np.ndarray: Array of functional norm estimates.
        """
        return np.sqrt(np.square(self._norm_2(array_data)).sum(axis=0) / array_data.shape[0])

    def soft_thresholding_update(
        self,
        g: int,
        f_g: np.ndarray | dict[str, np.ndarray],
        zero_groups: list[bool],
        smooth_fit: np.ndarray,
        s_g_ordered: np.ndarray,
        penalty: float,
        m_opt: np.ndarray,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Soft-thresholding update.

        Args:
            g (int): _description_
            f_g (np.ndarray | dict[str, np.ndarray]): _description_
            zero_groups (list[bool]): _description_
            smooth_fit (np.ndarray): _description_
            s_g_ordered (np.ndarray): _description_
            penalty (float): _description_
            m_opt (np.ndarray): _description_

        Returns:
            np.ndarray | dict[str, np.ndarray]: _description_
        """
        if isinstance(f_g, np.ndarray):
            f_g_hat = f_g[:, :, g, :]
        else:
            f_g_hat = list(f_g.values())[g]

        if zero_groups[g]:
            f_g_hat = np.zeros(f_g_hat.shape)
        else:
            f_g_hat = self.update_loop(
                f_g_hat=f_g_hat,
                d_g=self.d_g_long[g],
                smooth_fit=smooth_fit,
                m_opt=m_opt,
                s_g_ordered=s_g_ordered,
                penalty=penalty,
            )
        if isinstance(f_g, np.ndarray):
            f_g[:, :, g, :] = f_g_hat
        else:
            f_g[list(f_g.keys())[g]] = f_g_hat
        return f_g

    def update_loop(
        self,
        f_g_hat: np.ndarray,
        d_g: int,
        smooth_fit: np.ndarray,
        s_g_ordered: np.ndarray,
        penalty: float,
        m_opt: np.ndarray,
    ) -> np.ndarray:
        """Inner update loop.

        Args:
            f_g_hat (np.ndarray): _description_
            d_g (int): _description_
            smooth_fit (np.ndarray): _description_
            s_g_ordered (np.ndarray): _description_
            penalty (float): _description_
            m_opt (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        for k in range(self.n_tasks):
            for j in range(d_g):
                # for each task check whether larger or smaller m*
                if (k + 1) > m_opt:
                    f_g_hat[:, j, k] = smooth_fit[:, j, k]
                else:
                    s_g_sum = s_g_ordered[:m_opt].sum()
                    f_g_hat[:, j, k] = (
                        1
                        / m_opt
                        * (s_g_sum - np.sqrt(d_g) * penalty)
                        * smooth_fit[:, j, k]
                        / s_g_ordered[k]
                    )
                f_g_hat[:, j, k] -= f_g_hat[:, j, k].mean()
        return f_g_hat

    def omega_hat(self, f_g: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Omega hat.

        Args:
            f_g (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        if isinstance(f_g, np.ndarray):
            p_g = f_g.shape[2]
            return np.array([self.functional_norm(f_g[:, :, g, :]) for g in range(p_g)])
        else:
            return np.array([self.functional_norm(f_g_hat) for f_g_hat in f_g.values()])

    def _standardize_data(
        self, X_data: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Standardize X data.

        Args:
            X_data (np.ndarray | dict[str, np.ndarray]): Predictor array or dict.

        Returns:
            np.ndarray | dict[str, np.ndarray]: Predictor array or dict standardized.
        """
        if isinstance(X_data, dict):
            for key, value in X_data.items():
                X_data[key] = self._assure_two_dim(value)
                X_data[key] = (value - value.mean(axis=0)) / value.std(axis=0)
        else:
            X_data = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)

        return X_data

    def _assert_group_sizes_match(self, X_data: np.ndarray | dict[str, np.ndarray]) -> bool:
        """Assert whether all groups have same size.

        Args:
            X_data (np.ndarray | dict[str, np.ndarray]): predictor data.

        Returns:
            bool: True if all groups have same size
        """
        if isinstance(X_data, np.ndarray):
            return False
        else:
            return len({dat.shape[1] if len(dat.shape) > 1 else 1 for dat in X_data.values()}) == 1

    def _assure_two_dim(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr[:, np.newaxis]
        else:
            return arr

    def fit(
        self,
        X_data: np.ndarray | dict[str, np.ndarray],
        Y_data: np.ndarray,
        nlambda: int = 30,
        lambda_min_ratio: float = 0.005,
        precalculate_smooths: bool = True,
        local_regression_method: str = "kernel",
    ) -> None:
        """Fit the multitask group SpAM.

        Args:
            X_data (np.ndarray): _description_
            Y_data (np.ndarray): _description_
            nlambda (int, optional): _description_. Defaults to 30.
            lambda_min_ratio (float, optional): _description_. Defaults to 0.005.
            precalculate_smooths (bool, optional): _description_. Defaults to True.
            local_regression_method (str): Defaults to "loess". Other options currently: "kernel".
        """
        Y_data = self._assure_two_dim(Y_data)
        Y_data -= Y_data.mean(axis=0)
        if self._assert_group_sizes_match(X_data=X_data):
            self.group_names = list(X_data.keys())
            X_data = np.concatenate([data[..., np.newaxis] for data in X_data.values()], axis=2)
        X_data = self._standardize_data(X_data=X_data)

        smoothers = self.precalculate_smooths(
            X_data=X_data, local_regression_method=local_regression_method
        )

        self.select_penalty(
            X_data=X_data,
            Y_data=Y_data,
            nlambda=nlambda,
            lambda_min_ratio=lambda_min_ratio,
            precalculate_smooths=precalculate_smooths,
            smoothers=smoothers,
            local_regression_method=local_regression_method,
        )

    def predict(self) -> np.ndarray:
        """Predict the model.

        Returns:
            np.ndarray: Predicted values of shape (#n_samples, #n_tasks)
        """
        if isinstance(self.f_g_hat, np.ndarray):
            self.predicted_vals = self.f_g_hat.sum(axis=(1, 2))
        else:
            self.predicted_vals = np.asanyarray(
                [f_g.sum(axis=1) for f_g in self.f_g_hat.values()]
            ).sum(axis=0)
        return self.predicted_vals

    def plugin_bandwidth(self, x_j: np.ndarray) -> float:
        """Plugin bandwidth for Gaussian Kernel.

        Args:
            x_j (np.ndarray): data.

        Returns:
            float: Selected bandwidth.
        """
        return float(0.6 * np.std(x_j) * x_j.shape[0] ** (-1 / 5))

    def _make_gaussian_kernel_smoother_matrix(self, data: np.ndarray) -> np.ndarray:
        torch_data = torch.from_numpy(data)[:, None]  # Reshape for broadcasting
        weights = torch.exp(
            -0.5 * ((torch_data - torch_data.T) / self.plugin_bandwidth(data)) ** 2
        )  # Pairwise Gaussian weights
        weights /= weights.sum(dim=1, keepdim=True)  # Normalize weights along each row

        return weights.numpy()

    def _make_loess_smoother_matrix(self, x: np.ndarray) -> np.ndarray:
        """Make smoothing matrix for one instance based on loess.

        Args:
            x (np.ndarray): data.

        Returns:
            np.ndarray: smoothing matrix
        """
        n = x.shape[0]
        smoother = np.zeros((n, n))

        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1
            loess_obj = loess(x, e_i)
            loess_obj.fit()
            smoother[:, i] = loess_obj.outputs.fitted_values

        return smoother

    def gcv(self, Y_data: np.ndarray) -> float:
        """Generalized cross-validation.

        Args:
            Y_data (np.ndarray): Response data of shape (#n_samples. #n_tasks).

        Returns:
            float: Value of GCV.
        """
        n = Y_data.shape[0]
        numerator: float = np.einsum("ij->", self._quadratic_loss(Y_data))
        denominator: float = np.square(np.square(n * self.n_tasks) - n * self.n_tasks * self._df())
        gcv = numerator / denominator
        return gcv

    def _quadratic_loss(self, Y_data: np.ndarray) -> np.ndarray:
        """Current residual fit in terms of LS criterion.

        Args:
            Y_data (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        if isinstance(self.f_g_hat, np.ndarray):
            Q_g = np.square(Y_data - self.f_g_hat.sum(axis=(1, 2)))
        else:
            Q_g = np.square(
                Y_data
                - np.array([entry.sum(axis=1) for entry in self.f_g_hat.values()]).sum(axis=0)
            )

        return Q_g

    def _df(self) -> float:
        """Effective degrees of freedom in terms of trace of smoother matrix.

        K multiplication is due to the single-task nature of the predictors.
        In case of multitask this will need to be adapted in the future.

        Returns:
            float: K * sum_g^p sum_j^d_g trace(S_j^k) I(||f_j^k|| not 0)
        """
        v_jk = {}
        for g, d_g in enumerate(self.d_g_long):
            v_jk[g] = np.zeros(d_g)
            for j in range(d_g):
                if self.zero_groups[g]:
                    continue
                elif isinstance(self.smoothing_matrices, np.ndarray):
                    v_jk[g][j] = np.einsum(
                        "ii",
                        self.smoothing_matrices[:, :, j, g],
                    )
                else:
                    v_jk[g][j] = np.einsum(
                        "ii",
                        self.smoothing_matrices[list(self.smoothing_matrices.keys())[g]][:, :, j],
                    )

        v_jk_sum: float = np.asanyarray([df.sum() for df in v_jk.values()]).sum()
        v_jk_sum_sum = self.n_tasks * v_jk_sum
        return v_jk_sum_sum

    def select_penalty(
        self,
        X_data: np.ndarray | dict[str, np.ndarray],
        Y_data: np.ndarray,
        nlambda: int = 30,
        lambda_min_ratio: float = 5e-3,
        smoothers: np.ndarray | dict[str, np.ndarray] | None = None,
        precalculate_smooths: bool = True,
        local_regression_method: str = "kernel",
    ) -> None:
        """GCV model selection procedure.

        Args:
            X_data (np.ndarray): _description_
            Y_data (np.ndarray): _description_
            nlambda (int, optional): _description_. Defaults to 30.
            lambda_min_ratio (float, optional): _description_. Defaults to 5e-3.
            precalculate_smooths (bool, optional): _description_. Defaults to True.
            smoothers (np.ndarray | dict[str, np.ndarray] | None): _description_. Defaults to None.
            local_regression_method (str): Defaults to "loess". Other options currently: "kernel".
        """
        # This is only ever be necessary if the function is called outside fit()
        if smoothers is None:
            smoothers = self.precalculate_smooths(
                X_data=X_data, local_regression_method=local_regression_method
            )

        self._find_lambda_max_value(
            X_data=X_data,
            Y_data=Y_data,
            smoothers=smoothers,
            precalculate_smooths=precalculate_smooths,
            local_regression_method=local_regression_method,
        )
        lambda_scale = np.exp(
            np.linspace(start=np.log(1), stop=np.log(lambda_min_ratio), num=nlambda)
        )
        self.lambda_values = lambda_scale * self.lambda_max_value

        gcv_values = np.empty(len(self.lambda_values))
        f_hat_from_previous_lambda = self._init_functions(X_data=X_data)
        current_min = np.inf
        for i, lambda_value in enumerate(self.lambda_values):
            self._progressBar(i, len(self.lambda_values) - 1, suffix="Finding optimal lambda")
            try:
                self.block_coordinate_descent(
                    X_data=X_data,
                    Y_data=Y_data,
                    penalty=lambda_value,
                    precalculate_smooths=precalculate_smooths,
                    smoothers=smoothers,
                    warm_start_f_hat=f_hat_from_previous_lambda,
                    local_regression_method=local_regression_method,
                )
                gcv_values[i] = self.gcv(Y_data=Y_data)
                f_hat_from_previous_lambda = self.f_g_hat
                self.zero_group_history[lambda_value] = self.zero_groups

                if gcv_values[i] < current_min:
                    final_f_hat = self.f_g_hat  # Save for later
                    final_zero_groups = self.zero_groups  # Save for later
                    final_steps_till_convergence = self.steps_till_convergence
                    current_min = gcv_values[i]

            except ConvergenceError:  # Stop if no convergence
                if i == 0:  # If this is negative even max_lambda didn't converge.
                    raise ValueError(
                        "No lambda value converged. Something wrong with causal order?"
                    )
                break

        self.chosen_lambda = self.lambda_values[np.argmin(gcv_values)]
        self.f_g_hat = final_f_hat
        self.zero_groups = final_zero_groups
        self.steps_till_convergence = final_steps_till_convergence
        self.gcv_values = gcv_values

    def _progressBar(self, count_value: float, total: float, suffix: str = "") -> None:
        bar_length = 100
        filled_up_Length = int(round(bar_length * count_value / float(total)))
        percentage = round(100.0 * count_value / float(total), 1)
        bar = "=" * filled_up_Length + "-" * (bar_length - filled_up_Length)
        sys.stdout.write(f"[{bar}] {percentage}% ...{suffix}\r")
        sys.stdout.flush()

    def _find_lambda_max_value(
        self,
        X_data: np.ndarray | dict[str, np.ndarray],
        Y_data: np.ndarray,
        smoothers: np.ndarray | dict[str, np.ndarray] | None = None,
        precalculate_smooths: bool = True,
        local_regression_method: str = "kernel",
    ) -> None:
        """Identifies largest smallest penalty that just renders all groups \

            to be shrunken to zero.

        Args:
            X_data (np.ndarray | dict[str, np.ndarray]): Predictors.
            Y_data (np.ndarray): Targets.
            smoothers (np.ndarray | dict[str, np.ndarray] | None, optional): Pre-calculated
                Smoothers. Defaults to None.
            precalculate_smooths (bool): Whether to precalculate smooths. Is just for completeness.
                Typically the smooths will have been calculated and provided externally when calling
                this function.
            local_regression_method (str): Defaults to "loess". Other options currently: "kernel".
        """
        self.n_tasks = Y_data.shape[1]

        if isinstance(X_data, dict):
            X_data = self._dict_preprocessing(X_data=X_data)

        # Number of predictors and members in groups
        self._set_dg_pg(X_data=X_data)

        f_g = self._init_functions(X_data=X_data)

        if smoothers is None:
            smoothers = self.precalculate_smooths(
                X_data=X_data, local_regression_method=local_regression_method
            )

        omega_g_sum = np.zeros(self.p_g)
        for g in range(self.p_g):
            d_g = self.d_g_long[g]
            # R_g_hat has shape (#samples, #tasks)
            R_g_hat = self.R_g_hat_update(f_g=f_g, Y_data=Y_data, g=g)

            if precalculate_smooths:
                smooth_fit = self.predict_from_linear_smoother(
                    g=g, smoothing_matrices=smoothers, R_g=R_g_hat
                )
            elif local_regression_method == "loess":
                smooth_fit = self.loess_direct_fit(g=g, d_g=d_g, R_g=R_g_hat, X_data=X_data)
            elif local_regression_method == "kernel":
                smooth_fit = self.gaussian_kernel_direct_fit(
                    g=g, d_g=d_g, R_g=R_g_hat, X_data=X_data
                )
            else:
                raise NotImplementedError("Smoothing method not implemented.")

            # estimate of || Q^(k)R_g^(k) ||
            # shape (#predictors, #tasks)
            omega_g = self.functional_norm(smooth_fit)
            # lambda_max_value must be at least omega_g.sum()/np.sqrt(d_g)
            omega_g_sum[g] = omega_g.sum()

        self.lambda_max_value = np.ceil(np.max(omega_g_sum / np.sqrt(self.d_g_long)))

    def plot_gcv_path(self) -> None:
        """Plot GCV path."""
        if self.gcv_values is None:
            raise ValueError("No GCV values available. Run model selection first.")
        _, ax = plt.subplots()
        ax.plot(self.lambda_values, self.gcv_values)
        plt.axvline(
            x=self.chosen_lambda,
            color="red",
            linestyle="--",
            alpha=0.35,
        )
        plt.xlabel("lambda")
        plt.ylabel("gcv")
        plt.show()


class ConvergenceError(Exception):
    """Convenience class for convergence error."""

    def __init__(self, message: str = "Convergence not reached"):
        """Returns error message."""
        super().__init__(message)
