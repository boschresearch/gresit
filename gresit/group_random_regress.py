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

from gresit.graphs import DAG
from gresit.learn_algorithms import LearnAlgo
from gresit.model_selection import MURGS


class GroupRandomRegress(LearnAlgo):
    """Select random order and perform MURGS pruning."""

    def __init__(self, rng: np.random.Generator = np.random.default_rng(seed=2024)) -> None:
        """Inits the class object."""
        self.rng = rng

    def _get_causal_order(
        self,
        data_dict: dict[str, np.ndarray],
        layering: dict[str, list[str]] | None = None,
    ) -> None:
        if layering is None:
            layering = {"L": list(data_dict.keys())}

        self.layering = layering
        groups = list(data_dict.keys())

        if len(layering) == 1:
            self.rng.shuffle(groups)
        else:
            shuffled_within_layer_structure = {
                layer: list(self.rng.choice(subgroup, len(subgroup), replace=False))
                for layer, subgroup in self.layering.items()
            }

            groups = [x for xs in [k for k in shuffled_within_layer_structure.values()] for x in xs]

        self._causal_order = groups
        pa = {}
        for node in groups:
            index = groups.index(node)
            pa[node] = groups[0:index]
        self._pa = pa

    def _sparse_regression_pruning(
        self, data_dict: dict[str, np.ndarray], nlambda: int = 15
    ) -> None:
        # pruning
        dat = data_dict.copy()

        pa = self._pa
        pi = self._causal_order
        for k in pi:  # for each but the first node in the causal order
            # Take every parent and check for independence
            potential_parents = pa[k].copy()
            if not potential_parents:
                continue
            # get Y_data and X_data
            Y_data = dat[k]
            # Create dict
            X_data = {key: dat[key] for key in potential_parents if key in dat}
            # initiate MURGS object
            murgs = MURGS()
            murgs.fit(
                X_data=X_data,
                Y_data=Y_data,
                nlambda=nlambda,
                precalculate_smooths=False,
                lambda_min_ratio=0.05,
            )
            # extract zero groups
            zero_groups = murgs.zero_groups
            pa[k] = [parent for i, parent in enumerate(potential_parents) if not zero_groups[i]]

        self._pa = pa

    def learn_graph(
        self,
        data_dict: dict[str, np.ndarray],
        layering: dict[str, list[str]] | None = None,
    ) -> DAG:
        """Learn the causal graph.

        Args:
            data_dict (dict[str, np.ndarray]): A dictionary of np.ndarrays. Key corresponds to
                group name and values to the corresponding data.
            layering (dict[str, list[str]]): A dictionary of layering information. Keys correspond
                to the layer and values to the variable names within each layer.

        Raises:
            NotImplementedError: _description_

        Returns:
            DAG: DAG estimate.
        """
        self._get_causal_order(data_dict=data_dict, layering=layering)
        self._sparse_regression_pruning(data_dict=data_dict)

        edge_list = [(parent, child) for child in self._pa for parent in self._pa[child]]

        self.DAG = DAG(nodes=self._causal_order)
        self.DAG.add_edges_from(edge_list)
        self._adjacency_matrix = self.DAG.adjacency_matrix

        return self.DAG

    @property
    def causal_order(self) -> list[str] | None:
        """Causal order."""
        return self._causal_order

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        """Adjacency matrix."""
        return self._adjacency_matrix
