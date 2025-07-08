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

import unittest

import networkx as nx
import numpy as np

from gresit.synthetic_data import (
    FCNN,
    GaussianProcesses,
    GenChainedData,
    GenData,
    GenERData,
    GenLayeredData,
    MultiOutputANM,
)


class TestNonLinearLayerDataWithNN(unittest.TestCase):
    """Tests the NN equations."""

    def setUp(self):
        """Sets up the tests."""
        self.parent_data = np.random.random((10, 2, 3))
        self.noise_data = np.random.random((10, 2))

    def test_with_gaussian_process(self):
        """Test generation with GPs."""
        manm = MultiOutputANM(input_dim=3, group_size=2, equation_cls=GaussianProcesses)
        rhs = manm.apply_rhs(
            parent_data=self.parent_data,
            noise_data=self.noise_data,
        )

        self.assertTupleEqual(rhs.shape, (10, 2))

    def test_with_fully_connected_neural_network(self):
        """Test generation with FCN."""
        manm = MultiOutputANM(input_dim=3, group_size=2, equation_cls=FCNN)
        rhs = manm.apply_rhs(
            parent_data=self.parent_data,
            noise_data=self.noise_data,
        )
        self.assertTupleEqual(rhs.shape, (10, 2))


class TestNonLinearLayeredData:
    """Testing nonlinear data generation."""

    def test_instance_created(self):
        """Test whether instances are created."""
        # Set up and act
        data_gen = GenLayeredData()
        m_anm = MultiOutputANM(input_dim=1)
        mean_vec = np.random.random((2, 1, 3)) * 2.0 - 0.5
        w1 = np.random.random((2, 1, 3))
        w2 = np.random.random(
            (
                2,
                1,
            )
        )
        sigma = 0.4
        gp = GaussianProcesses.from_params(mean_vec=mean_vec, w1=w1, w2=w2, sigma=sigma)

        # Assert
        assert isinstance(data_gen, GenLayeredData)
        assert isinstance(m_anm, MultiOutputANM)
        assert isinstance(gp, GaussianProcesses)

    def test_gp_callable(self):
        """Test whether GP is callable."""
        # Set up and act
        mean_vec = np.random.random((2, 1, 3)) * 2.0 - 0.5
        w1 = np.random.random((2, 1, 3))
        w2 = np.random.random(
            (
                2,
                1,
            )
        )
        sigma = 0.4
        gp = GaussianProcesses.from_params(mean_vec=mean_vec, w1=w1, w2=w2, sigma=sigma)
        data = np.random.multivariate_normal(
            mean=np.array([0, 0]), cov=np.array([[1, 0.2], [0.2, 1]]), size=10
        )

        assert gp(data[:, :, np.newaxis]).shape == data.shape
        assert not np.allclose(gp(data[:, :, np.newaxis])[:, 0], data[:, 0])

    def test_multivariate_gaussian_kernel_aggregation(self):
        """Test whether the Gaussian kernel does what it is supposed to."""
        # Set up
        rng = np.random.default_rng(seed=2025)
        input_dim = 1
        n_gp = 5
        group_size = 2

        gp_mean = rng.random((group_size, input_dim, n_gp)) * 2.0 - 0.5
        # Weights in GPs.
        w1 = rng.random((group_size, input_dim, n_gp))
        w1 /= w1.sum(axis=-1, keepdims=True)

        # Weights for a linear aggregation of GPs.
        w2 = rng.random(
            (
                group_size,
                input_dim,
            )
        )
        w2 /= w2.sum(axis=-1, keepdims=True)
        # Standard deviation of GPs
        sigma = 0.3

        num_samples = 100
        data = np.random.multivariate_normal(
            mean=np.array([0, 0]), cov=np.array([[1, 0.2], [0.2, 1]]), size=num_samples
        )

        data = data[:, :, np.newaxis]

        def gaussian_kernel_function(
            x_1: np.ndarray, x_2: np.ndarray, sigma: np.ndarray | float
        ) -> np.ndarray:
            """Gaussian kernel."""
            return np.exp(-0.5 * np.square(x_1 - x_2) / np.square(sigma))

        # Act:

        kernel_list = []
        for i in range(gp_mean.shape[-1]):
            kernel_list.append(
                gaussian_kernel_function(x_1=data, x_2=gp_mean[np.newaxis, :, :, i], sigma=sigma)
            )

        kernel = np.stack(kernel_list, axis=data.ndim)

        gps: np.ndarray = (kernel * w1[np.newaxis]).sum(axis=-1)
        # Weighted sum of GPs (doesn't do anythin if there is only one parent)
        gp: np.ndarray = (gps * w2[np.newaxis]).sum(axis=-1)

        # Assert
        assert kernel.shape == (*data.shape, gp_mean.shape[-1])
        assert np.allclose(
            kernel[:, :, 0, 0],
            gaussian_kernel_function(data[:, :, 0], gp_mean[np.newaxis, :, 0, 0], sigma),
        )
        assert np.allclose(gps[:, :, 0], gp)

    def test_m_anm_input_dim_works(self):
        """Test the MultiOutputANM."""
        # Set up
        m_anm_3 = MultiOutputANM(input_dim=3)
        rhs_3 = m_anm_3.apply_rhs(
            parent_data=np.random.random((10, 2, 3)), noise_data=np.random.random((10, 2))
        )

        m_anm_5 = MultiOutputANM(input_dim=5)
        rhs_5 = m_anm_5.apply_rhs(
            parent_data=np.random.random((10, 2, 5)), noise_data=np.random.random((10, 2))
        )

        # Assert
        assert rhs_3.shape == rhs_5.shape == (10, 2)

    def test_m_anm_group_size_works(self):
        """Test the MultiOutputANM."""
        # Set up
        m_anm_3 = MultiOutputANM(input_dim=3, group_size=3)
        rhs_g3 = m_anm_3.apply_rhs(
            parent_data=np.random.random((10, 3, 3)), noise_data=np.random.random((10, 3))
        )

        # Assert
        assert rhs_g3.shape == (10, 3)

    def test_m_anm_change_number_of_gps(self):
        """Test the MultiOutputANM."""
        # Set up
        m_anm_3 = MultiOutputANM(input_dim=3, group_size=3, equation_kwargs={"n_gp": 6})
        rhs_g3 = m_anm_3.apply_rhs(
            parent_data=np.random.random((10, 3, 3)), noise_data=np.random.random((10, 3))
        )

        # Assert
        assert rhs_g3.shape == (10, 3)

    def test_m_anm_null_dimension(self):
        """Test the MultiOutputANM."""
        # Set up
        m_anm_3 = MultiOutputANM(input_dim=0)
        rhs_g3 = m_anm_3.apply_rhs(parent_data=None, noise_data=np.random.random((10, 3)))

        # Assert
        assert rhs_g3.shape == (10, 3)
        assert m_anm_3.input_dim == 0

    def test_gen_layers_works(self):
        """Test the GenLayeredData."""
        # Set up
        number_of_nodes = 20
        data_gen = GenLayeredData(
            number_of_layers=4, number_of_nodes=number_of_nodes, edge_density=0.2
        )
        data_dict, num_data = data_gen.generate_data(num_samples=100)

        # Assert
        assert len(data_dict) == number_of_nodes
        assert all([data.shape == (100, 2) for data in data_dict.values()])
        assert num_data.shape == (100, 2, number_of_nodes)

    def test_causal_orderings(self):
        """Test the GenLayeredData."""
        # Set up
        data_gen = GenLayeredData(number_of_layers=4, number_of_nodes=16, edge_density=0.3)
        all_orderings = data_gen.all_causal_orderings()
        one_order = data_gen.causal_order

        all_nx_orderings = list(nx.all_topological_sorts(data_gen.dag.to_networkx()))

        assert one_order in all_orderings
        assert one_order in all_nx_orderings
        assert all([order in all_nx_orderings for order in all_orderings])
        assert len(all_orderings) <= len(list(all_nx_orderings))

    def test_multiple_sample_calls_work(self):
        """Test the GenLayeredData."""
        # Set up
        number_of_nodes = 20
        data_gen = GenLayeredData(
            number_of_layers=4, number_of_nodes=number_of_nodes, edge_density=0.2
        )
        _, np_data_1 = data_gen.generate_data(num_samples=100)

        _, np_data_2 = data_gen.generate_data(num_samples=100)

        assert not np.allclose(np_data_1, np_data_2)

    def test_er_data_works(self):
        """Test ER data gen works."""
        data_gen = GenERData()

        assert isinstance(data_gen, GenERData)
        assert isinstance(data_gen, GenData)
        assert issubclass(GenERData, GenData)

    def test_gen_er_data_gets_generated(self):
        """Test the ER Data."""
        # Set up
        number_of_nodes = 20
        data_gen = GenERData(number_of_nodes=number_of_nodes)
        data_dict, num_data = data_gen.generate_data(num_samples=100)

        # Assert
        assert len(data_dict) == number_of_nodes
        assert all([data.shape == (100, 2) for data in data_dict.values()])
        assert num_data.shape == (100, 2, number_of_nodes)

    def test_multiple_ER_calls_work(self):
        """Test the ER data."""
        # Set up
        number_of_nodes = 20
        data_gen = GenERData(number_of_nodes=number_of_nodes, group_size=4)
        _, np_data_1 = data_gen.generate_data(num_samples=100)

        _, np_data_2 = data_gen.generate_data(num_samples=100)

        assert not np.allclose(np_data_1, np_data_2)

    def test_chain_data_works(self):
        """Test chain data gen works."""
        data_gen = GenChainedData()

        assert isinstance(data_gen, GenData)
        assert isinstance(data_gen, GenChainedData)
        assert issubclass(GenChainedData, GenData)

    def test_chain_data_gets_generated(self):
        """Test the chain Data."""
        # Set up
        number_of_nodes = 20
        data_gen = GenChainedData(number_of_nodes=number_of_nodes)
        data_dict, num_data = data_gen.generate_data(num_samples=100)

        # Assert
        assert len(data_dict) == number_of_nodes
        assert all([data.shape == (100, 2) for data in data_dict.values()])
        assert num_data.shape == (100, 2, number_of_nodes)

    def test_chain_data_gives_unique_causal_order(self):
        """Test for unique causal order."""
        # Set up
        number_of_nodes = 20
        data_gen = GenChainedData(number_of_nodes=number_of_nodes)
        nx_dag = data_gen.dag.to_networkx()
        all_sorts = list(nx.all_topological_sorts(nx_dag))

        assert len(all_sorts) == 1
