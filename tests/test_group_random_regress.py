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

from gresit.group_random_regress import GroupRandomRegress
from gresit.synthetic_data import GenLayeredData


class TestGroupRandomRegress:
    """Testing GroupRandomRegress."""

    @pytest.fixture(scope="class")
    def generate_toy_data(self) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
        """Generate toy data.

        Returns:
            dict[str, np.ndarray]: data dict .
        """
        ldat = GenLayeredData(number_of_nodes=4, group_size=4, number_of_layers=2)
        data_dict, _ = ldat.generate_data(num_samples=500)
        return data_dict, ldat.layering

    def test_learn_graph(
        self, generate_toy_data: tuple[dict[str, np.ndarray], dict[str, list[str]]]
    ):
        """Test whether the graph is learned.

        Args:
            generate_toy_data (pd.DataFrame): data vector to test.
        """
        # Setup
        data_dict, _ = generate_toy_data
        alg = GroupRandomRegress()

        # Act
        alg.learn_graph(data_dict)
        assert isinstance(alg.causal_order, list)

    def test_learn_layered_graph(
        self, generate_toy_data: tuple[dict[str, np.ndarray], dict[str, list[str]]]
    ):
        """Test whether the graph is learned.

        Args:
            generate_toy_data (pd.DataFrame): data vector to test.
        """
        # Setup
        data_dict, layering = generate_toy_data
        alg = GroupRandomRegress()

        # Act

        alg.learn_graph(data_dict, layering=layering)
        assert isinstance(alg.causal_order, list)
