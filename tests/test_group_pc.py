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

from gresit.graphs import GRAPH, PDAG
from gresit.group_pc import GroupPC, MicroPC
from gresit.synthetic_data import GenLayeredData


class TestVectorGroup:
    """Testing VectorGroup."""

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
        """Test whether instance of VectorPC is created."""
        # Set up and act
        my_alg = GroupPC()

        # Assert
        assert isinstance(my_alg, GroupPC)
        assert my_alg.pdag is not None

    def test_get_skeleton_with_layering(
        self, generate_toy_data: tuple[dict[str, np.ndarray], dict[str, list[str]]]
    ):
        """Test whether get_skeleton behaves as expected.

        Args:
            generate_toy_data (tuple[dict[str, np.ndarray], dict[str, list[str]]]): data dict.
        """
        # Set up
        data_dict, layering = generate_toy_data
        my_alg = GroupPC()

        # Act
        my_alg._find_skeleton(data=data_dict, layering=layering)

        # Assert
        assert my_alg.skel is not None
        assert isinstance(my_alg.skel, pd.DataFrame)

    def test_get_skeleton_without_layering(
        self, generate_toy_data: tuple[dict[str, np.ndarray], dict[str, list[str]]]
    ):
        """Test whether get_skeleton behaves as expected.

        Args:
            generate_toy_data (tuple[dict[str, np.ndarray], dict[str, list[str]]]): data dict.
        """
        # Set up
        data_dict, _ = generate_toy_data
        my_alg = GroupPC()

        # Act
        my_alg._find_skeleton(data=data_dict)

        # Assert
        assert my_alg.skel is not None
        assert isinstance(my_alg.skel, pd.DataFrame)

    def test_learn_graph_with_layering(
        self, generate_toy_data: tuple[dict[str, np.ndarray], dict[str, list[str]]]
    ):
        """Test whether the graph is learned.

        Args:
            generate_toy_data (pd.DataFrame): data vector to test.
        """
        # Setup
        data_dict, layering = generate_toy_data
        my_alg = GroupPC()

        # Act
        my_alg.learn_graph(data_dict=data_dict, layering=layering)

        assert my_alg.pdag is not None
        assert isinstance(my_alg.pdag, PDAG)
        assert my_alg.layering is not None
        assert isinstance(my_alg.pdag.adjacency_matrix, pd.DataFrame)
        assert my_alg.pdag.adjacency_matrix.shape == (len(data_dict), len(data_dict))

    def test_learn_graph_without_layering(
        self, generate_toy_data: tuple[dict[str, np.ndarray], dict[str, list[str]]]
    ):
        """Test whether the graph is learned.

        Args:
            generate_toy_data (pd.DataFrame): data vector to test.
        """
        # Setup
        data_dict, _ = generate_toy_data
        my_alg = GroupPC()

        # Act
        my_alg.learn_graph(data_dict=data_dict)

        assert my_alg.pdag is not None
        assert isinstance(my_alg.pdag, PDAG)
        assert my_alg.ci_test is not None
        assert isinstance(my_alg.pdag.adjacency_matrix, pd.DataFrame)
        assert my_alg.pdag.adjacency_matrix.shape == (len(data_dict), len(data_dict))

    def test_microPC_instance_created(self):
        """Test whether instance of VectorPC is created."""
        # Set up and act
        alpha = 0.1
        my_alg = MicroPC(alpha=alpha)

        # Assert
        assert isinstance(my_alg, MicroPC)
        assert my_alg.alpha == alpha

    def test_microPC_learngraph_works(
        self, generate_toy_data: tuple[dict[str, np.ndarray], dict[str, list[str]]]
    ):
        """MicroPC learn graph."""
        # Set up and act
        alpha = 0.1
        data_dict, _ = generate_toy_data
        my_alg = MicroPC(alpha=alpha)

        learned_graph = my_alg.learn_graph(data_dict)

        # Assert

        assert isinstance(learned_graph, GRAPH)
        assert my_alg.adjacency_matrix.shape == (len(data_dict), len(data_dict))
