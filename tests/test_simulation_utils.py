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

from gresit.params import DataParams, ExperimentParams, PCParams
from gresit.simulation_utils import (
    BenchMarker,
    _make_super_dag_adjacency,
)
from gresit.synthetic_data import GenLayeredData


class TestSimulationUtils(unittest.TestCase):
    """Test whether simulation does what it's supposed to do."""

    def test_class_instance_works(self):
        """Test whether instance is created correctly."""
        bm = BenchMarker()
        assert isinstance(bm, BenchMarker)

    def test_empty_graph_gets_created(self):
        """Test empty graph generation."""
        # Setup
        bm = BenchMarker()
        # Act
        empty_graph = bm._emptygraph(size=10)

        # Assert
        assert empty_graph.shape == (10, 10)
        assert np.allclose(empty_graph, np.zeros(empty_graph.shape))

    def test_super_dag_adjacency(self):
        """Test whether super-dag gets correctly generated."""
        order = ["X_1", "X_2", "X_3", "X_4"]
        super_dag = _make_super_dag_adjacency(order)
        nx_super_dag = nx.from_pandas_adjacency(df=super_dag, create_using=nx.DiGraph)

        assert order == list(nx.all_topological_sorts(nx_super_dag))[0]

    def test_run_benchmark_works(self):
        """Test benchmark run."""
        bm = BenchMarker()
        num_runs = 2

        params = ExperimentParams(
            algos=[
                PCParams(
                    alpha=0.1,
                ),
                PCParams(
                    alpha=0.001,
                ),
                PCParams(
                    alpha=0.05,
                ),
            ],
            data=DataParams(
                generator=GenLayeredData,
                number_of_nodes=5,
            ),
            number_of_samples=100,
        )
        # Act
        metrics = ["shd", "sid", "ancestor_aid", "ancester_ordering_aid"]
        results = bm.run_benchmark(
            params=params,
            num_runs=num_runs,
            metrics=metrics,
        )

        self.assertTrue(
            all(
                [
                    len(alg_runs) == num_runs
                    for alg_runs in results["GroupPC(alpha=0.1, test=FisherZVec)"].values()
                ]
            )
        )
        self.assertEqual(len(results), len(params.algos))
        self.assertEqual(len(results["GroupPC(alpha=0.1, test=FisherZVec)"]), len(metrics))
