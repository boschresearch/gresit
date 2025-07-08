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

from __future__ import annotations

import logging
from abc import (
    ABCMeta,
    abstractmethod,
)
from typing import Any

import numpy as np
import pandas as pd

from gresit.graphs import GRAPH

logger = logging.getLogger(__name__)


class LearnAlgo(metaclass=ABCMeta):
    """Abstract base class for all Learn Algorithms."""

    @property
    @abstractmethod
    def causal_order(self) -> list[str] | None:
        """Causal order."""
        pass

    @property
    @abstractmethod
    def adjacency_matrix(self) -> pd.DataFrame:
        """Adjacency matrix."""
        pass

    @abstractmethod
    def learn_graph(self, data_dict: dict[str, np.ndarray], *args: Any, **kwargs: Any) -> GRAPH:
        """Learn the graph from the data.

        Args:
            data_dict (dict[str, np.ndarray]): Data dict.
            *args (Any): additional args.
            **kwargs (Any): additional kwargs.

        Returns:
            GRAPH: Object of type GRAPH.
        """
