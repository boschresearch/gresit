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

from gresit.independence_tests import DISCO, HSIC, CItest, FisherZVec, Itest, KernelCI
from gresit.synthetic_data import GenERData


class TestIndependence:
    """Testing unconditional independence."""

    @pytest.fixture(scope="class")
    def generate_toy_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate multivariate toy data.

        Returns:
            tuple[np.ndarray, np.ndarray]: Returns data Xdata, Ydata

        """
        _, dat = GenERData(number_of_nodes=5).generate_data(num_samples=30)
        Ydat = dat[:, :, 0]
        Xdat_interim = dat[:, :, 1:]
        Xdat = Xdat_interim.reshape(
            Xdat_interim.shape[0], Xdat_interim.shape[1] * Xdat_interim.shape[2]
        )

        return Xdat, Ydat

    def test_instance_created(self):
        """Test whether instance of HSIC is created."""
        # Set up and act
        my_test = HSIC()

        # Assert
        assert isinstance(my_test, HSIC)
        assert isinstance(my_test, Itest)

    def test_get_kernel_width(self, generate_toy_data: tuple[np.ndarray, np.ndarray]):
        """Test whether width of kernel behaves as expected.

        Args:
            generate_toy_data (tuple[np.ndarray, np.ndarray]): data vector to test.
        """
        # Set up
        _, my_array = generate_toy_data
        my_test = HSIC()
        kernel_width = my_test.get_kernel_width(my_array)
        # Act
        assert isinstance(kernel_width, np.float64)
        assert kernel_width > 0

    def test_rbf_dot(self, generate_toy_data: tuple[np.ndarray, np.ndarray]):
        """Test whether _rbf behaves as expected.

        Args:
            generate_toy_data (tuple[np.ndarray, np.ndarray]): data vector to test.
        """
        # Set up
        _, my_array = generate_toy_data
        my_test = HSIC()
        kernel_width = my_test.get_kernel_width(my_array)
        rbf = my_test._rbf_dot(X=my_array, width=kernel_width)
        # Act
        assert isinstance(rbf, np.ndarray)
        assert len(rbf) > 0
        assert rbf.shape == (30, 30)

    def test_get_gram_matrix(self, generate_toy_data: tuple[np.ndarray, np.ndarray]):
        """Test whether width of gram matrices behaves as expected.

        Args:
            generate_toy_data (tuple[np.ndarray, np.ndarray]): data vector to test.
        """
        # Set up
        _, my_array = generate_toy_data
        my_test = HSIC()
        kernel_width = my_test.get_kernel_width(my_array)
        gram_mat_a, gram_mat_b = my_test.get_gram_matrix(X=my_array, width=kernel_width)
        # Act
        assert isinstance(gram_mat_a, np.ndarray)
        assert isinstance(gram_mat_b, np.ndarray)
        assert gram_mat_a.shape == gram_mat_b.shape

    def test_hsic_teststat(self, generate_toy_data: tuple[np.ndarray, np.ndarray]):
        """Test whether test statistic behaves as expected.

        Args:
            generate_toy_data (tuple[np.ndarray, np.ndarray]): data vector to test.
        """
        # Set up
        _, my_array = generate_toy_data
        my_test = HSIC()
        kernel_width = my_test.get_kernel_width(my_array)
        gram_mat_a, gram_mat_b = my_test.get_gram_matrix(X=my_array, width=kernel_width)
        statistic = my_test.hsic_teststat(Kc=gram_mat_a, Lc=gram_mat_b, n=20)

        assert isinstance(statistic, np.float64)
        assert statistic > 0

    def test_hsic_test_gamma(self, generate_toy_data: tuple[np.ndarray, np.ndarray]):
        """Test whether test statistic and pvalue behave as expected.

        Args:
            generate_toy_data (tuple[np.ndarray, np.ndarray]): data vector to test.
        """
        # Set up
        x_dat, y_dat = generate_toy_data
        my_test = HSIC()
        statistic, pvalue = my_test.test(x_data=x_dat, y_data=y_dat)

        assert isinstance(statistic, float)
        assert statistic > 0
        assert isinstance(pvalue, float)
        assert pvalue >= 0

    def test_DISCO_instance_is_created(self):
        """Test disco instance."""
        # set up
        disco = DISCO()

        # assert
        assert isinstance(disco, DISCO)
        assert isinstance(disco, Itest)

    def test_DISCO_test(self):
        """Test the test method."""
        # Set up
        my_cov = np.array(
            [
                [1, 0.73, 0.23, -0.42],
                [0.73, 1, 0.1, 0.3],
                [0.23, 0.1, 1, -0.3],
                [-0.42, 0.3, -0.3, 1],
            ]
        )

        data = np.random.multivariate_normal(mean=[0, 0, 0, 0], cov=my_cov, size=100)
        X_data = data[:, :2]
        Y_data = data[:, 2:]

        # Act
        my_test = DISCO()
        test_stat, test_decision = my_test.test(x_data=X_data, y_data=Y_data)

        # Assert
        assert isinstance(test_stat, float)
        assert isinstance(test_decision, str)

    def test_fisher_z_instance(self):
        """Test whether instance is createrd."""
        fisher = FisherZVec()

        assert isinstance(fisher, FisherZVec)
        assert isinstance(fisher, CItest)

    def test_KCI_instance(self):
        """Test whether instance is createrd."""
        fisher = KernelCI()

        assert isinstance(fisher, KernelCI)
        assert isinstance(fisher, CItest)

    def test_fisher_test_works(self):
        """Test whether fisher z test works."""
        fisher = FisherZVec()
        data = np.random.multivariate_normal(mean=np.zeros(6), cov=np.eye(6), size=100)
        X_data = data[:, :2]
        Y_data = data[:, 2:4]
        Z_data = data[:, 4:]

        uncond_test, uncond_p_value = fisher.test(x_data=X_data, y_data=Y_data)
        cond_test, cond_p_value = fisher.test(x_data=X_data, y_data=Y_data, z_data=Z_data)

        assert isinstance(uncond_test, float)
        assert isinstance(cond_test, float)
        assert uncond_p_value >= 0.0
        assert cond_p_value >= 0.0

    def test_KCI_test_works(self):
        """Test whether fisher z test works."""
        kernel_ci = KernelCI()
        data = np.random.multivariate_normal(mean=np.zeros(6), cov=np.eye(6), size=100)
        X_data = data[:, :2]
        Y_data = data[:, 2:4]
        Z_data = data[:, 4:]

        uncond_test, uncond_p_value = kernel_ci.test(x_data=X_data, y_data=Y_data)
        cond_test, cond_p_value = kernel_ci.test(x_data=X_data, y_data=Y_data, z_data=Z_data)

        assert isinstance(uncond_test, float)
        assert isinstance(cond_test, float)
        assert uncond_p_value >= 0.0
        assert cond_p_value >= 0.0
