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

from abc import (
    ABCMeta,
    abstractmethod,
)
from collections import defaultdict
from itertools import combinations, product
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import nn

from gresit.graphs import DAG, LayeredDAG


class Equation(metaclass=ABCMeta):
    """Abstract class for non-linear equations."""

    def __init__(
        self,
        group_size: int,
        input_dim: int,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
    ):
        """Abstract class for non-linear equations.

        Args:
            input_dim (int): parent dimension
            group_size (int, optional): _description_. Defaults to 2.
            rng (np.random.Generator, optional): Random number generator. Defaults to default rng.
        """
        self.group_size = group_size
        self.input_dim = input_dim
        self.rng = rng

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Abstract call method for equation."""


class FCNN(Equation):
    """A randomly initialized fully connected neural net."""

    def __init__(
        self,
        group_size: int,
        input_dim: int,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
        hidden_layer: int = 10,
    ) -> None:
        """A randomly initialized fully connected neural net.

        Args:
            input_dim (int): parent dimension
            group_size (int, optional): _description_. Defaults to 2.
            rng (np.random.Generator, optional): Random number generator. Defaults to None.
            hidden_layer (int, optional): Size of hidden dimension.
        """
        super().__init__(group_size, input_dim, rng=rng)

        layers = [
            nn.Linear(self.group_size * self.input_dim, hidden_layer, dtype=torch.float32),
            nn.Sigmoid(),
            nn.Linear(hidden_layer, self.group_size, dtype=torch.float32),
        ]
        self.fcn = nn.Sequential(*layers)
        self.fcn.apply(self.init_weights)

    def init_weights(self, m: nn.Module) -> None:
        """Initializes the weights of the neural net.

        Args:
            m (nn.Module): The layer to be initialized
        """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=1)
            nn.init.normal_(m.bias, mean=0, std=1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Computes the right hand side of the equation.

        Args:
            x (np.ndarray): Input data

        Returns:
            np.ndarray: output data of size n x multioutput_dim
        """
        X = torch.from_numpy(x).flatten(1, -1).float()

        with torch.no_grad():
            return self.fcn(X).detach().cpu().numpy()


class GaussianProcesses(Equation):
    """A weighted sum of Gaussian processes."""

    def __init__(
        self,
        group_size: int,
        input_dim: int,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
        n_gp: int = 5,
        sigma: float = 0.3,
    ) -> None:
        """A weighted sum of Gaussian processes.

        Args:
            input_dim (int): parent dimension
            group_size (int, optional): _description_. Defaults to 2.
            rng (np.random.Generator, optional): Random number generator. Defaults to None.
            n_gp (int): Number of processes
            sigma (float): The standard deviation of GPs. Note that all GPS have the same std.

        Raises:
            ValueError: Gets thrown when input dimensions don't match.
        """
        super().__init__(group_size, input_dim, rng=rng)

        self.n_gp = n_gp

        mean_vec = self.rng.random((self.group_size, self.input_dim, self.n_gp)) * 2.0 - 0.5
        # Weights in GPs.
        w1 = self.rng.random((self.group_size, self.input_dim, self.n_gp))
        w1 /= w1.sum(axis=-1, keepdims=True)
        # Weights for a linear aggregation of GPs.
        w2 = self.rng.random((self.group_size, self.input_dim))
        w2 /= w2.sum(axis=-1, keepdims=True)

        (_, n1, _) = mean_vec.shape
        (_, n2, _) = w1.shape
        (_, n3) = w2.shape

        if n1 != n2 or n1 != n3:
            raise ValueError("Input variable dimensions need to match.")

        # add first axis corresponding to the number of samples.
        self.mean_vec = mean_vec[np.newaxis]
        self.w1 = w1[np.newaxis]
        self.w2 = w2[np.newaxis]
        self.sigma = sigma

    @staticmethod
    def from_params(
        mean_vec: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
        sigma: float = 0.3,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
    ) -> "GaussianProcesses":
        """Creates a gaussian process from parameters.

        Args:
            mean_vec (np.ndarray): Mean vectors of Gaussian processes with
                shape=(group_size, num_parents, num_kernels), where num_parents is
                the dimension of the input variable and num_kernels is
                the number of Gaussian kernels in one GP.
            w1 (np.ndarray): Weights of GPs with shape=(num_parents, num_kernels).
            w2 (np.ndarray): Weights for aggregating GPs with shape=(n,)
            sigma (float): The standard deviation of GPs. Note that all GPS have the same std.
            rng (np.random.Generator, optional): The random number generator to use.
        """
        gp = GaussianProcesses(group_size=mean_vec.shape[0], input_dim=mean_vec.shape[1], rng=rng)

        gp.mean_vec = mean_vec[np.newaxis]
        gp.w1 = w1[np.newaxis]
        gp.w2 = w2[np.newaxis]
        gp.sigma = sigma
        return gp

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Depending on the shape of x dimension will be adjusted.

        Args:
            x (float | np.ndarray): Input data

        Raises:
            ValueError: Gets thrown when input dimensions don't match.

        Returns:
            float | np.ndarray: output data of size n x multioutput_dim
        """
        kernel_list = []
        for i in range(self.mean_vec.shape[-1]):
            kernel_list.append(
                self._gaussian_kernel_function(
                    x_1=x, x_2=self.mean_vec[:, :, :, i], sigma=self.sigma
                )
            )

        kernel = np.stack(kernel_list, axis=x.ndim)
        if not kernel.shape == (*x.shape, self.mean_vec.shape[-1]):
            raise ValueError(
                f"""something went wrong, shape of kernel function should be \
                {(*x.shape, self.mean_vec.shape[-1])}, \
                but in fact it is {kernel.shape}."""
            )
        # kernel should have shape #samples, #group_size, #parents #kernels
        # Weighted sum of q Gaussian kernels.
        gps: np.ndarray = (kernel * self.w1).sum(axis=-1)
        # Weighted sum of GPs (doesn't do anythin if there is only one parent)
        gp: np.ndarray = (gps * self.w2).sum(axis=-1)

        if not gp.shape == (x.shape[0], x.shape[1]):
            raise ValueError(
                f"""something went wrong, shape of function output should be \
                {x.shape}, but in fact it is {gp.shape}."""
            )

        return gp

    def _gaussian_kernel_function(
        self, x_1: np.ndarray, x_2: np.ndarray, sigma: np.ndarray | float
    ) -> np.ndarray:
        """Gaussian kernel."""
        return np.exp(-0.5 * np.square(x_1 - x_2) / np.square(sigma))


class MultiOutputANM:
    """Class to construct nonlinear multi-outcome data that follows an ANM."""

    def __init__(
        self,
        input_dim: int,
        equation_cls: type[Equation] = GaussianProcesses,
        equation_kwargs: dict[str, Any] | None = None,
        group_size: int = 2,
        snr: float = 1.0,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
    ) -> None:
        """Initiates the ANM object.

        Args:
            input_dim (int): parent dimension
            group_size (int, optional): _description_. Defaults to 2.
            rng (np.random.Generator, optional): Random number generator. Defaults to None.
            equation_cls (Equation): The type of the equation that should be used.
            equation_kwargs (dict): Arguments for equation.
            snr (float, optional): Signal-to-noise ratio. Defaults to 1.0.
        """
        self.input_dim = input_dim
        self.group_size = group_size
        self.snr = snr
        self.rng = rng
        self._lambda = self.snr / (1.0 + self.snr)
        self.equation_cls = equation_cls
        if not equation_kwargs:
            equation_kwargs = {}
        self.equation_kwargs = equation_kwargs
        if self.input_dim > 0:
            self.f_nonlinear = self.generate_nonlinear_map()

    def generate_nonlinear_map(self) -> Equation:
        """Nonlinear function."""
        return self.equation_cls(
            group_size=self.group_size,
            input_dim=self.input_dim,
            rng=self.rng,
            **self.equation_kwargs,
        )

    def apply_rhs(self, parent_data: np.ndarray | None, noise_data: np.ndarray) -> np.ndarray:
        """See whether this works."""
        noise_standard_dev = 1 / np.std(noise_data, axis=0)
        noise_std = noise_standard_dev * noise_data

        if self.input_dim > 0:
            y_interim = self.f_nonlinear(parent_data)
            y_standard_dev = 1 / np.std(y_interim, axis=0)
            y_std = y_interim * y_standard_dev
            y = np.sqrt(self._lambda) * y_std + np.sqrt(1.0 - self._lambda) * noise_std
            y = y - np.mean(y, axis=0)
        else:
            y = noise_std
        return y


class GenData:
    """Parent class to GenDate classes."""

    def __init__(
        self,
        number_of_nodes: int = 15,
        equation_cls: type[Equation] = GaussianProcesses,
        equation_kwargs: dict[str, Any] | None = None,
        group_size: int = 2,
        edge_density: float = 0.2,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
        snr: float = 1.0,
        noise_distribution: str = "gaussian",
    ) -> None:
        """Initiate parent class DAG.

        Args:
            rng (np.random.Generator, optional): Random number generator. Defaults to None.
            number_of_nodes (int, optional): _description_. Defaults to 15.
            equation_cls (Equation): The type of the equation that should be used.
            equation_kwargs (dict): Arguments for equation.
            group_size (int, optional): Number of entries in groups. Defaults to 2.
            edge_density (float, optional): _description_. Defaults to 0.2.
            snr (float, optional): Signal to noise ratio. Defaults to 1.0.
            noise_distribution (str, optional): Which distribution to choose for the noise.
                defaults to Gaussian noise. Options include: `lognormal`
        """
        self.number_of_nodes = number_of_nodes
        self.group_size = group_size
        self.edge_density = edge_density
        self.rng = rng
        self.equation_cls = equation_cls
        self.equation_kwargs = equation_kwargs
        self.dag = DAG()
        self.causal_order: list[str] = []
        self.snr = snr
        self.noise_distribution = noise_distribution

    def _initiate(self) -> None:
        """Initiate DAG and random functions."""
        self._make_dag()
        self._initiate_ANM()

    def _make_dag(self) -> None:
        """Make DAG.

        Raises:
            NotImplementedError: Needs to be overwritten.
        """
        raise NotImplementedError

    def _initiate_ANM(self) -> None:
        """Initiate the ANM equations."""
        equations: dict[str, MultiOutputANM] = {}
        for node in self.dag.nodes:
            parents = self.dag.parents(of_node=node)
            equations[node] = MultiOutputANM(
                input_dim=len(parents),
                equation_cls=self.equation_cls,
                equation_kwargs=self.equation_kwargs,
                group_size=self.group_size,
                rng=self.rng,
                snr=self.snr,
            )

        self.equations = equations

    def _generate_random_correlation_matrix(self, group_size: int) -> np.ndarray:
        """Generate random pd correlation matrix.

        Args:
            group_size (int): dimension of matrix.

        Returns:
            np.ndarray: correlation matrix.
        """
        L = np.tril(self.rng.uniform(-0.8, 0.8, (group_size, group_size)), k=-1)
        np.fill_diagonal(L, 1)  # Set diagonal to 1

        # Compute symmetric positive definite matrix
        Sigma = L @ L.T

        # Normalize diagonal to 1
        D = np.sqrt(np.diag(Sigma))
        Sigma = Sigma / np.outer(D, D)

        return Sigma

    def generate_data(self, num_samples: int = 1000) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Sample data from the layered DAG."""
        noise_data = []
        for _ in self.dag.nodes:
            mean = self.rng.uniform(-0.8, 0.8, self.group_size)
            corr = self._generate_random_correlation_matrix(self.group_size)
            if self.noise_distribution == "gaussian":
                noise_data.append(
                    self.rng.multivariate_normal(mean=mean, cov=corr, size=num_samples)
                )
            else:
                noise_data.append(
                    np.exp(self.rng.multivariate_normal(mean=mean, cov=corr, size=num_samples))
                )
        reshaped_noise = np.moveaxis(np.asanyarray(noise_data), 0, -1)

        data = np.zeros(reshaped_noise.shape)
        data_dict = {}
        for i, node in enumerate(self.causal_order):
            parents = self.dag.parents(of_node=node)
            parent_indices = [self.dag.causal_order.index(parent) for parent in parents]
            data[:, :, i] = self.equations[node].apply_rhs(
                parent_data=data[:, :, parent_indices], noise_data=reshaped_noise[:, :, i]
            )
            data[:, :, i] -= data[:, :, i].mean(axis=0)
            data[:, :, i] /= data[:, :, i].std(axis=0)
            data_dict[node] = data[:, :, i]
        return data_dict, data


class GenERData(GenData):
    """Class to generate general nonlinear data following Erdos-Renyi (ER) graph."""

    def __init__(
        self,
        number_of_nodes: int = 15,
        equation_cls: type[Equation] = GaussianProcesses,
        equation_kwargs: dict[str, Any] | None = None,
        group_size: int = 2,
        edge_density: float = 0.2,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
        snr: float = 1.0,
        noise_distribution: str = "gaussian",
    ) -> None:
        """Initiate the ER DAG.

        Args:
            rng (np.random.Generator, optional): Random number generator. Defaults to None.
            number_of_nodes (int, optional): _description_. Defaults to 15.
            equation_cls (Equation): The type of the equation that should be used.
            equation_kwargs (dict): Arguments for equation.
            group_size (int, optional): Number of entries in groups. Defaults to 2.
            edge_density (float, optional): _description_. Defaults to 0.2.
            snr (float, optional): Signal to noise ratio. Defaults to 1.0.
            noise_distribution (str, optional): Which distribution to choose for the noise.
                defaults to Gaussian noise. Options include: `lognormal`
        """
        super().__init__(
            number_of_nodes=number_of_nodes,
            equation_cls=equation_cls,
            equation_kwargs=equation_kwargs,
            group_size=group_size,
            edge_density=edge_density,
            rng=rng,
            snr=snr,
            noise_distribution=noise_distribution,
        )
        self.dag = DAG()
        self._initiate()
        self.causal_order = self.dag.causal_order

    def _make_dag(self) -> None:
        """Generate an Erdős-Rényi Directed Acyclic Graph (DAG)."""
        # Assign a random topological ordering
        nodes = [f"X_{i}" for i in range(self.number_of_nodes)]
        self.rng.shuffle(nodes)

        G = DAG(nodes=nodes)

        # Add edges based on Erdős-Rényi model
        for i, j in combinations(range(self.number_of_nodes), 2):
            # Ensure edges go from lower to higher index
            if self.rng.random() < self.edge_density:
                G.add_edge(edge=(nodes[i], nodes[j]))

        self.dag = G


class GenLayeredData(GenData):
    """Class to generate general nonlinear data following multivariate ANM (mANM)."""

    def __init__(
        self,
        number_of_nodes: int = 15,
        number_of_layers: int = 3,
        equation_cls: type[Equation] = GaussianProcesses,
        equation_kwargs: dict[str, Any] | None = None,
        group_size: int = 2,
        edge_density: float = 0.2,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
        snr: float = 1.0,
        noise_distribution: str = "gaussian",
    ) -> None:
        """Initiate the layered DAG.

        Args:
            rng (np.random.Generator, optional): Random number generator. Defaults to None.
            number_of_nodes (int, optional): _description_. Defaults to 15.
            number_of_layers (int, optional): _description_. Defaults to 3.
            equation_cls (Equation): The type of the equation that should be used.
            equation_kwargs (dict): Arguments for equation.
            group_size (int, optional): Number of entries in groups. Defaults to 2.
            edge_density (float, optional): _description_. Defaults to 0.2.
            snr (float, optional): Signal to noise ratio. Defaults to 1.0.
            noise_distribution (str, optional): Which distribution to choose for the noise.
                defaults to Gaussian noise. Options include: `lognormal`
        """
        super().__init__(
            number_of_nodes=number_of_nodes,
            equation_cls=equation_cls,
            equation_kwargs=equation_kwargs,
            group_size=group_size,
            edge_density=edge_density,
            rng=rng,
            snr=snr,
            noise_distribution=noise_distribution,
        )
        self.number_of_layers = number_of_layers
        self.dag: LayeredDAG = LayeredDAG()
        self._initiate()
        self.layering = self.dag.layering

    def _divide_equal(self, number: int, pieces: int) -> list[int]:
        """Divides an integer into a specified number of pieces, distributing the remainder.

        Args:
            number: The integer to divide.
            pieces: The desired number of pieces.

        Returns:
            A list containing the values of each piece.
        """
        base_size = number // pieces
        remainder = number % pieces
        return [base_size + 1 if i < remainder else base_size for i in range(pieces)]

    def all_causal_orderings(self) -> list[list[str]]:
        """Valid causal ordering of the layered DAG.

        Raises:
            ValueError: If there are no nodes or edges.

        Returns:
            list[list[str]]: list of causal orders.
        """
        if not self.dag.nodes:
            raise ValueError("There are no nodes in the graph")

        if not self.layering:
            raise ValueError("Layering must be provided.")

        all_within_orders = [
            list(nx.all_topological_sorts(self.dag.layer_induced_subgraph(layer).to_networkx()))
            for layer in self.layering.values()
        ]

        all_orderings_tupled = list(product(*all_within_orders))
        all_orderings = []
        for order_tuple in all_orderings_tupled:
            all_orderings.append([x for xs in [portion for portion in order_tuple] for x in xs])
        return all_orderings

    def _make_dag(self) -> None:
        """Create a layered DAG."""
        nodes_per_layer = self._divide_equal(self.number_of_nodes, self.number_of_layers)

        layering = defaultdict(list)
        for i, layer_i in enumerate(nodes_per_layer):
            for j in range(layer_i):
                layering[f"L_{i}"].append(f"X_{i}_{j}")

        for list_of_nodes in layering.values():
            self.rng.shuffle(list_of_nodes)

        self.dag.layering = layering

        causal_ordering = [x for xs in [k for k in layering.values()] for x in xs]
        self.causal_order = causal_ordering
        # Add edges based on Erdős-Rényi model

        self.dag.add_nodes_from(causal_ordering)

        for i in range(len(causal_ordering)):
            for j in range(i + 1, len(causal_ordering)):
                if self.rng.random() < self.edge_density:
                    self.dag.add_edge(edge=(causal_ordering[i], causal_ordering[j]))


class GenChainedData(GenData):
    """Class to generate chain DAG with nonlinear data following multivariate ANM (mANM)."""

    def __init__(
        self,
        number_of_nodes: int = 15,
        equation_cls: type[Equation] = GaussianProcesses,
        equation_kwargs: dict[str, Any] | None = None,
        group_size: int = 2,
        edge_density: float = 0.2,
        rng: np.random.Generator = np.random.default_rng(seed=2024),
        snr: float = 1.0,
        noise_distribution: str = "gaussian",
    ) -> None:
        """Initiate the layered DAG.

        Args:
            rng (np.random.Generator, optional): Random number generator. Defaults to None.
            number_of_nodes (int, optional): _description_. Defaults to 15.
            equation_cls (Equation): The type of the equation that should be used.
            equation_kwargs (dict): Arguments for equation.
            group_size (int, optional): Number of entries in groups. Defaults to 2.
            edge_density (float, optional): _description_. Defaults to 0.2.
            snr (float, optional): Signal to noise ratio. Defaults to 1.0.
            noise_distribution (str, optional): Which distribution to choose for the noise.
                defaults to Gaussian noise. Options include: `lognormal`
        """
        super().__init__(
            number_of_nodes=number_of_nodes,
            equation_cls=equation_cls,
            equation_kwargs=equation_kwargs,
            group_size=group_size,
            edge_density=edge_density,
            rng=rng,
            snr=snr,
            noise_distribution=noise_distribution,
        )
        self.dag = DAG()
        self._initiate()
        self.causal_order = self.dag.causal_order

    def _make_dag(self) -> None:
        """Generate an chain Directed Acyclic Graph (DAG)."""
        # Assign a random topological ordering
        nodes = [f"X_{i}" for i in range(self.number_of_nodes)]
        self.rng.shuffle(nodes)
        self.causal_order = nodes
        chain_edges = [(nodes[i], nodes[i + 1]) for i in range(self.number_of_nodes - 1)]
        G = DAG(nodes=nodes, edges=chain_edges)
        for i in range(self.number_of_nodes):
            for j in range(i + 2, self.number_of_nodes):
                if self.rng.random() < self.edge_density:
                    G.add_edge(edge=(nodes[i], nodes[j]))
        self.dag = G
