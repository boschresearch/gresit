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

import numpy as np
import pytest

from gresit.model_selection import MURGS


class TestMURGS:
    """Testing MURGS."""

    @pytest.fixture(scope="class")
    def generate_toy_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate multivariate toy data.

        Returns:
            tuple[np.ndarray, np.ndarray]: Returns toy data where only the
                first two columns are relevant in forming Y_data.
        """
        X_data = np.zeros((1000, 2, 6))
        for i in range(X_data.shape[-1]):
            corr = np.random.uniform(-0.5, 0.5)
            X_data[:, :, i] = np.random.multivariate_normal(
                mean=[0, 0], cov=[[1, corr], [corr, 1]], size=1000
            )

        Y_data = np.power(X_data[:, :, 0], 3) + 0.1 * np.random.multivariate_normal(
            mean=[0, 0], cov=[[1, 0], [0, 1]], size=1000
        )

        return Y_data, X_data

    def test_murgs_initialization(self):
        """Test if the class is initialized correctly."""
        murgs = MURGS()
        assert isinstance(murgs, MURGS)

    def test_murgs_run_works(self, generate_toy_data: tuple[np.ndarray, np.ndarray]):
        """Testing fit.

        Args:
            generate_toy_data (np.ndarray): Toy data.
        """
        Y_data, X_data = generate_toy_data
        p_g = X_data.shape[-1]
        n_tasks = Y_data.shape[-1]
        murgs = MURGS()
        murgs.fit(X_data=X_data, Y_data=Y_data)

        assert murgs.f_g_hat is not None
        assert murgs.zero_groups is not None
        assert np.all(murgs.d_g_long == np.repeat(2, repeats=p_g))
        assert murgs.p_g == p_g
        assert murgs.n_tasks == n_tasks
        assert murgs.chosen_lambda >= 0.0
        assert murgs.lambda_max_value >= 0
        assert murgs.lambda_max_value >= murgs.chosen_lambda
        assert murgs.steps_till_convergence >= 0

    def test_precalculation_makes_no_difference(self, generate_toy_data: np.ndarray):
        """Testing whether precalculating smoothing matrix makes a difference.

        Args:
            generate_toy_data (np.ndarray): Toy data.
        """
        Y_data, X_data = generate_toy_data

        murgs = MURGS()
        murgs.block_coordinate_descent(
            X_data=X_data,
            Y_data=Y_data,
            penalty=2,
            precalculate_smooths=True,
        )

        f_g_hat_pre = murgs.f_g_hat
        zero_groups_pre = murgs.zero_groups

        murgs.block_coordinate_descent(
            X_data=X_data,
            Y_data=Y_data,
            penalty=2,
            precalculate_smooths=False,
        )
        zero_groups_nopre = murgs.zero_groups
        f_g_hat_nopre = murgs.f_g_hat

        assert zero_groups_pre == zero_groups_nopre
        assert np.allclose(f_g_hat_pre, f_g_hat_nopre)

    def test_all_zero_at_lambda_max(self, generate_toy_data: np.ndarray):
        """Test whether all values for lambda larger `lambda_max_value` result in only zero groups.

        Args:
            generate_toy_data (np.ndarray): toy data.
        """
        # Initiate
        Y_data, X_data = generate_toy_data
        # Act
        murgs = MURGS()
        smooths = murgs.precalculate_smooths(X_data=X_data)
        murgs._find_lambda_max_value(X_data=X_data, Y_data=Y_data, smoothers=smooths)
        lambda_max = murgs.lambda_max_value
        murgs.block_coordinate_descent(X_data=X_data, Y_data=Y_data, penalty=lambda_max)

        assert np.all(murgs.zero_groups)
