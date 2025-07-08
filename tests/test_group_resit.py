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
import pandas as pd
import pytest

from gresit.graphs import DAG
from gresit.group_resit import GroupResit
from gresit.independence_tests import HSIC
from gresit.regression_techniques import SimultaneousLinearModel
from gresit.synthetic_data import GenLayeredData


class TestGroupResit:
    """Testing GroupResit."""

    @pytest.fixture(scope="class")
    def generate_toy_data(self) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
        """Generate toy data.

        Returns:
            tuple[dict[str, np.ndarray], dict[str, list[str]]]: data dict and layering dict.
        """
        ldat = GenLayeredData(number_of_nodes=8, number_of_layers=2)
        data_dict, _ = ldat.generate_data(num_samples=500)
        layering = ldat.layering

        return data_dict, layering

    def test_instance_created(self):
        """Test whether instance of HSIC is created."""
        # Set up and act
        my_alg = GroupResit(regressor=SimultaneousLinearModel(), test=HSIC)

        # Assert
        assert isinstance(my_alg, GroupResit)
        assert my_alg.regressor is not None
        assert my_alg.independence_test is not None

    def test_get_causal_order(
        self, generate_toy_data: tuple[dict[str, np.ndarray], dict[str, list[str]]]
    ):
        """Test whether _get_causal_order behaves as expected.

        Args:
            generate_toy_data (tuple[dict[str, np.ndarray], dict[str, list[str]]]): data dict.
        """
        # Set up
        data_dict, layering = generate_toy_data
        my_alg = GroupResit(regressor=SimultaneousLinearModel(), test=HSIC)

        # Act
        my_alg._get_causal_order(data_dict=data_dict, layering=layering)

        # Assert
        assert my_alg.causal_order is not None
        assert isinstance(my_alg.causal_order[0], str)
        assert my_alg._pa is not None
        assert len(my_alg.causal_order) == len(data_dict)
        assert len(my_alg._pa) == len(data_dict)

    def test_independence_pruning(
        self, generate_toy_data: tuple[dict[str, np.ndarray], dict[str, list[str]]]
    ):
        """Test whether pruning works.

        Args:
            generate_toy_data (tuple[dict[str, np.ndarray], dict[str, list[str]]): data dict.
        """
        data_dict, layering = generate_toy_data
        my_alg = GroupResit(regressor=SimultaneousLinearModel(), test=HSIC)

        my_alg._get_causal_order(data_dict=data_dict, layering=layering)

        # Act
        alpha = 0.01
        pre_prune_pa = my_alg._pa
        my_alg._independence_prune(data_dict=data_dict, alpha=alpha)

        # Assert

        assert my_alg.alpha_level == alpha
        assert sum([len(parents) for parents in my_alg._pa.values()]) <= sum(
            [len(parents) for parents in pre_prune_pa.values()]
        )

    def test_murgs_pruning(self):
        """Test whether pruning works.

        Args:
            generate_toy_data (tuple[dict[str, np.ndarray], dict[str, list[str]]): data dict.
        """
        ldat = GenLayeredData(number_of_nodes=8, number_of_layers=2)
        data_dict, _ = ldat.generate_data(num_samples=500)
        pi = ldat.causal_order
        my_alg = GroupResit(regressor=SimultaneousLinearModel(), test=HSIC)

        # Act
        my_alg._insert_known_causal_order(pi=pi)
        pre_prune_pa = my_alg._pa
        my_alg.model_selection_with_known_causal_order(pi=pi, data_dict=data_dict)

        # Assert

        assert sum([len(parents) for parents in my_alg._pa.values()]) <= sum(
            [len(parents) for parents in pre_prune_pa.values()]
        )

    def test_learn_graph(
        self, generate_toy_data: tuple[dict[str, np.ndarray], dict[str, list[str]]]
    ):
        """Test whether the graph is learned.

        Args:
            generate_toy_data (pd.DataFrame): data vector to test.
        """
        # Setup
        data_dict, layering = generate_toy_data
        my_alg = GroupResit(regressor=SimultaneousLinearModel(), test=HSIC)

        # Act
        my_alg.learn_graph(data_dict, layering=layering)

        assert my_alg.DAG is not None
        assert isinstance(my_alg.DAG, DAG)
        assert my_alg.adjacency_matrix is not None
        assert isinstance(my_alg.adjacency_matrix, pd.DataFrame)
        assert my_alg.adjacency_matrix.shape == (len(data_dict), len(data_dict))

    def test_plotting_before_fitting_throws_Error(self):
        """Test whether plotting before fitting throws error."""
        # Setup
        my_alg = GroupResit(regressor=SimultaneousLinearModel(), test=HSIC)
        with pytest.raises(AssertionError) as e:
            my_alg.show()
        assert str(e.value) == "No graph to plot. Learn the graph first."
