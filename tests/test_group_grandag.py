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

from gresit.group_grandag import (
    GroupGraNDAG,
    _make_group_mapping,
)
from gresit.synthetic_data import GenLayeredData


class TestGroupGraNDAG:
    """Testing GroupGraNDAG."""

    @pytest.fixture(scope="class")
    def generate_toy_data(self) -> dict[str, np.ndarray]:
        """Generate toy data.

        Returns:
            dict[str, np.ndarray]: data dict .
        """
        ldat = GenLayeredData(number_of_nodes=2, group_size=8, number_of_layers=2)
        data_dict, _ = ldat.generate_data(num_samples=500)
        return data_dict

    def test_learn_graph(self, generate_toy_data: dict[str, np.ndarray]):
        """Test whether the graph is learned.

        Args:
            generate_toy_data (pd.DataFrame): data vector to test.
        """
        # Setup
        data_dict = generate_toy_data
        alg = GroupGraNDAG(n_iterations=10, with_group_constraint=True)

        # Act
        alg.learn_graph(data_dict)

        # assert alg.causal_order is not None
        # print(alg.causal_order)

    def test_make_group_mapping(
        self, generate_toy_data: tuple[dict[str, np.ndarray], dict[str, list[str]]]
    ):
        """Test whether group mapping matrix has right shape.

        Args:
            generate_toy_data (pd.DataFrame): data vector to test.
        """
        data_dict = generate_toy_data

        Q = _make_group_mapping(data_dict)

        np.testing.assert_array_equal(
            Q,
            np.concatenate([np.ones(8) / 8, np.zeros(8), np.zeros(8), np.ones(8) / 8]).reshape(
                2, 16
            ),
        )
