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

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import (
    Any,
)

import numpy as np

from gresit.group_grandag import GroupGraNDAG
from gresit.group_pc import GroupPC, MicroPC
from gresit.group_random_regress import GroupRandomRegress
from gresit.group_resit import GroupResit
from gresit.independence_tests import (
    HSIC,
    CItest,
    FisherZVec,
    Itest,
)
from gresit.learn_algorithms import LearnAlgo
from gresit.regression_techniques import MultiRegressor
from gresit.synthetic_data import Equation, GaussianProcesses, GenData, GenLayeredData

_short_names = {
    "learning_rate": "lr",
    "batch_size": "bs",
    "with_early_stopping": "es",
    "val_size": "vsize",
}


def _make_short(param_name: str) -> str:
    """Shortens a parameter name."""
    return _short_names.get(param_name, param_name)


@dataclass
class AlgoParams(metaclass=ABCMeta):
    """Base class for learn algos."""

    @abstractmethod
    def init(self, rng: np.random.Generator) -> LearnAlgo:
        """Initialize the algo."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algo."""
        raise NotImplementedError


@dataclass
class ResitParams(AlgoParams):
    """Params for resit."""

    regressor: type[MultiRegressor]
    kwargs: dict[str, Any]
    test: type[Itest] = HSIC
    pruning_method: str = "murgs"
    alpha: float = 0.01
    test_size: float = 0.2
    local_regression_method = "kernel"

    def init(self, rng: np.random.Generator) -> GroupResit:
        """Inits the group resit."""
        return GroupResit(
            regressor=self.regressor(rng=rng, **self.kwargs),
            test=self.test,
            pruning_method=self.pruning_method,
            alpha=self.alpha,
            test_size=self.test_size,
            local_regression_method=self.local_regression_method,
        )

    @property
    def name(self) -> str:
        """Generates a generic name."""
        return "{method}({hparams})_{test}_{pruning}".format(
            method=GroupResit.__name__,
            hparams=", ".join([f"{_make_short(k)}={v}" for k, v in self.kwargs.items()]),
            test=self.test.__name__,
            pruning=self.pruning_method,
        )


@dataclass
class PCParams(AlgoParams):
    """Params for PC Alg."""

    alpha: float = 0.05
    test: type[CItest] = FisherZVec

    def init(self, rng: np.random.Generator) -> GroupPC:
        """Inits the group PC."""
        return GroupPC(alpha=self.alpha, test=self.test)

    @property
    def name(self) -> str:
        """Generates a generic name."""
        return f"GroupPC(alpha={self.alpha}, test={self.test.__name__})"


@dataclass
class MicroPCParams(AlgoParams):
    """Params for PC Alg."""

    alpha: float = 0.05

    def init(self, rng: np.random.Generator) -> MicroPC:
        """Inits the group PC."""
        return MicroPC(alpha=self.alpha)

    @property
    def name(self) -> str:
        """Generates a generic name."""
        return f"MicroPC(alpha={self.alpha})"


@dataclass
class GraNDAGParams(AlgoParams):
    """Params for GroupMicro."""

    n_iterations: int = 1000
    with_group_constraint: bool = True
    h_threshold: float = 1e-7

    def init(self, rng: np.random.Generator) -> GroupGraNDAG:
        """Inits the group micro method."""
        return GroupGraNDAG(
            n_iterations=self.n_iterations,
            with_group_constraint=self.with_group_constraint,
            h_threshold=self.h_threshold,
        )

    @property
    def name(self) -> str:
        """Generates a generic name."""
        return f"GroupGraN-DAG(iter={self.n_iterations}, h_thresh={self.h_threshold})"


@dataclass
class RandomRegParams(AlgoParams):
    """Params for GroupMicro."""

    def init(self, rng: np.random.Generator) -> GroupRandomRegress:
        """Inits the group micro method."""
        return GroupRandomRegress(rng=rng)

    @property
    def name(self) -> str:
        """Generates a generic name."""
        return "GroupRandomRegress()"


@dataclass
class DataParams:
    """Params for data."""

    generator: type[GenData] = GenLayeredData
    number_of_nodes: int = 15
    number_of_layers: int = 3
    equation_cls: type[Equation] = GaussianProcesses
    equation_kwargs: dict[str, Any] | None = None
    group_size: int = 2
    edge_density: float = 0.2
    snr: float = 1.0
    noise_distribution: str = "gaussian"

    def init(self, rng: np.random.Generator) -> GenData:
        """Inits the dataset."""
        common_kwargs = {
            "number_of_nodes": self.number_of_nodes,
            "equation_cls": self.equation_cls,
            "equation_kwargs": self.equation_kwargs,
            "group_size": self.group_size,
            "edge_density": self.edge_density,
            "rng": rng,
            "snr": self.snr,
            "noise_distribution": self.noise_distribution,
        }

        if issubclass(self.generator, GenLayeredData):
            return self.generator(
                number_of_layers=self.number_of_layers,
                **common_kwargs,
            )
        else:
            return self.generator(**common_kwargs)


@dataclass
class ExperimentParams:
    """Params for an experiment."""

    algos: list[AlgoParams]
    data: DataParams
    number_of_samples: int = 100
    rng: np.random.Generator = np.random.default_rng(seed=2024)

    def __iter__(self) -> Iterator[tuple[LearnAlgo, str]]:
        """Yields the gresits to benchmark."""
        for algo in self.algos:
            yield algo.init(rng=self.rng), algo.name

    def make_data(self) -> GenData:
        """Builds the data."""
        return self.data.init(self.rng)
