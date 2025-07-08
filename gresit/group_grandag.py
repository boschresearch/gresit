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

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from castle.algorithms import GraNDAG
from castle.algorithms.gradient.gran_dag.torch.gran_dag import (
    compute_constraint as compute_micro_constraint,
)

from gresit.graphs import DAG, GRAPH, PDAG
from gresit.learn_algorithms import LearnAlgo


def _make_group_mapping(data_dict: dict[str, np.ndarray]) -> np.ndarray:
    group_sizes = [arr.shape[1] for arr in data_dict.values()]
    n_groups = len(group_sizes)

    Q = []

    for i in range(n_groups):
        Q.append(
            np.concatenate(
                [np.zeros(s) if i != j else 1 / s * np.ones(s) for j, s in enumerate(group_sizes)]
            )
        )

    return np.array(Q)


def _make_compute_constraint(Q: torch.Tensor) -> Callable[[GraNDAG, torch.Tensor], torch.Tensor]:
    n_groups = Q.shape[0]

    def compute_constraint(model: GraNDAG, w_adj: torch.Tensor) -> torch.Tensor:
        assert (w_adj >= 0).detach().cpu().numpy().all()

        # Compute mapping to groups
        H = torch.matmul(Q, torch.matmul(w_adj, Q.T))

        # Ignore diagonal in group matrix
        H = H - H * torch.eye(n_groups)

        return torch.trace(torch.matrix_exp(H)) - n_groups

    return compute_constraint


class GroupGraNDAG(LearnAlgo):
    """GroupGraNDAG causal discovery."""

    def __init__(
        self,
        n_iterations: int = 1000,
        with_group_constraint: bool = False,
        h_threshold: float = 1e-7,
    ) -> None:
        """Init the class."""
        self.graph: GRAPH

        self.n_iterations = n_iterations
        self.with_group_constraint = with_group_constraint
        self.h_threshold = h_threshold

    def learn_graph(
        self, data_dict: dict[str, np.ndarray], *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Learn graph.

        data_dict (dict[str, np.ndarray]): Data dict.
            *args (Any): additional args.
            **kwargs (Any): additional kwargs.

        Returns:
            GRAPH: Object of type GRAPH.
        """
        data = np.concatenate([d_data for d_data in data_dict.values()], axis=1)
        data -= data.mean(axis=0)
        data /= data.std(axis=0)

        Q = _make_group_mapping(data_dict)
        Q = torch.Tensor(Q).double()

        if "device" in kwargs:
            Q = Q.to(kwargs["device"])

        if self.with_group_constraint:
            compute_constraint = _make_compute_constraint(Q)
        else:
            compute_constraint = compute_micro_constraint

        with patch(
            "castle.algorithms.gradient.gran_dag.torch.gran_dag.compute_constraint",
            compute_constraint,
        ):
            grandag = GraNDAG(
                input_dim=data.shape[1],
                iterations=self.n_iterations,
                h_threshold=self.h_threshold,
                normalize=False,
                use_pns=False,
                **kwargs,
            )

            grandag.learn(data=data)

            micro_adjacency_matrix = grandag.causal_matrix

        interim_group_adjacency_matrix: np.ndarray = (
            (Q @ micro_adjacency_matrix @ Q.T).detach().cpu().numpy()
        )
        np.fill_diagonal(interim_group_adjacency_matrix, 0)
        grouped_w = interim_group_adjacency_matrix
        if not self._is_acyclic(grouped_w):
            w_interim = grouped_w.copy()
            intial_threshold = np.min(w_interim[w_interim > 0])
            W_clipped = np.where(w_interim <= intial_threshold, 0, w_interim)
            while not self._is_acyclic(W_clipped):
                new_threshold = np.min(W_clipped[W_clipped > 0])
                W_clipped = np.where(W_clipped <= new_threshold, 0, W_clipped)
            grouped_w = W_clipped

        group_adjacency_matrix = (grouped_w > 0).astype(int)

        group_graph: DAG | PDAG
        if not np.any((group_adjacency_matrix == 1) & (group_adjacency_matrix.T == 1)):
            group_graph = DAG.from_pandas_adjacency(
                pd.DataFrame(
                    group_adjacency_matrix, columns=data_dict.keys(), index=data_dict.keys()
                )
            )
        else:
            group_graph = PDAG.from_pandas_adjacency(
                pd.DataFrame(
                    group_adjacency_matrix, columns=data_dict.keys(), index=data_dict.keys()
                )
            )
        self.graph = group_graph
        return group_graph

    def _is_acyclic(self, w: np.ndarray) -> bool:
        d = w.shape[0]
        h: bool = np.trace(np.exp(w * w)) - d == 0
        return h

    @property
    def causal_order(self) -> list[str] | None:
        """Causal order."""
        if isinstance(self.graph, DAG):
            return self.graph.causal_order
        else:
            return None

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        """Adjacency matrix."""
        return self.graph.adjacency_matrix
