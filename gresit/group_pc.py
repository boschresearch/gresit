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

from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc

from gresit.graphs import DAG, GRAPH, PDAG, rule_1, rule_2, rule_3, rule_4
from gresit.group_grandag import _make_group_mapping
from gresit.independence_tests import CItest, FisherZVec
from gresit.learn_algorithms import LearnAlgo


class GroupPC(LearnAlgo):
    """This class provides tools for causal discovery.

    Particularly, in the context where data is known to follow
    a layered structure.
    """

    def __init__(self, alpha: float = 0.05, test: type[CItest] = FisherZVec) -> None:
        """Initiates VectorPC.

        Args:
            alpha (float, optional): Acts as a tuning parameter. The significance
                threshold for the conditional independence test. The smaller,
                the sparser the resulting graph. Defaults to 0.05.
            test (CItest, optional): Which CI test to use.
        """
        self.layering: dict[str, list[str]] | None = None
        self.alpha: float = alpha
        self.pdag: PDAG = PDAG()
        self.ambiguities: list[tuple[str, str, str]] = []
        self.skel: pd.DataFrame
        self.ci_test = test()

    def learn_graph(
        self,
        data_dict: dict[str, np.ndarray],
        threshold: float = 0.5,
        layering: dict[str, list[str]] | None = None,
    ) -> PDAG:
        """Learns the graph from the given data.

        If layering is provided it is taken to be unambiguous.
        If layering is not Null, then the separation sets may never
        contain variables that appear in future layers to the pair of
        variables considered.

        Args:
            data_dict (dict | ndarray): relevant data.
            threshold (float, optional): The majority vote threshold for deciding
                on ambiguous collider structures. Defaults to 0.5.
            layering (dict[str, list[str]], optional): The layering of the nodes.

        Returns:
            PDAG: Graph estimate.

        """
        self.layering = layering

        self._find_skeleton(data=data_dict, alpha=self.alpha, layering=layering)
        self.maximally_orient(data=data_dict, alpha=self.alpha, threshold=threshold)
        return self.pdag

    def _find_skeleton(
        self,
        data: dict[str, np.ndarray],
        alpha: float = 0.05,
        layering: dict[str, list[str]] | None = None,
    ) -> None:
        """First Phase of layered PC (stable) algorithm.

        If layering is not Null, then the separation sets may never
        contain variables that appear in future layers to the pair of
        variables considered.

        Args:
            data (dict[str, np.ndarray]): _description_
            alpha (float, optional): _description_. Defaults to 0.05.
            layering (dict[str, list[str]], optional): _description_. Defaults to None.
        """
        n_features = len(data)
        node_names = list(data.keys())
        skeleton = pd.DataFrame(
            (np.ones((n_features, n_features)) - np.eye(n_features)),
            columns=node_names,
            index=node_names,
        )

        nodes = sorted(node_names)
        sep_set: dict[int, dict[tuple[str, str], list[str]]] = defaultdict(dict)
        d = -1
        node_sets = list(combinations(nodes, 2))
        node_sets = sorted(node_sets, key=lambda x: x[0])

        while self._adj_size_criterion(skeleton, d):  # until for each adj(C,i)\{j} < l
            d += 1
            c_stable = deepcopy(skeleton)  # needed for stable
            if d not in sep_set:
                sep_set[d] = {}
            for node_i, node_j in node_sets:
                if skeleton.loc[node_i, node_j] == 0:
                    continue
                adj_i = set(c_stable.index[c_stable[node_i] == 1].to_list())
                z = adj_i - {node_j}  # adj(C, i)\{j}
                if len(z) >= d:
                    # |adj(C, i)\{j}| >= l
                    z_list = sorted([*z])
                    k = sorted([*combinations(z_list, d)], reverse=True)
                    for subset in k:
                        sub_z = list(subset)
                        if layering is not None and sub_z:
                            max_layer = self._return_max_layer(
                                layering=layering, node_i=node_i, node_j=node_j
                            )
                            allowed_dict = self._remove_pairs_after_key(
                                layering=layering, max_layer=max_layer
                            )
                            sub_z = self._remove_future_nodes(
                                sep_set=sub_z,
                                allowed_dict=allowed_dict,
                            )
                        if not sub_z:
                            _, p_value = self.ci_test.test(x_data=data[node_i], y_data=data[node_j])
                        else:
                            # find highest layer of node_{i,j} and restrict sep
                            # set such that no nodes can be in layers further
                            # than said highest layer.
                            _, p_value = self.ci_test.test(
                                x_data=data[node_i],
                                y_data=data[node_j],
                                z_data=np.concatenate(
                                    [dat for node, dat in data.items() if node in sub_z], axis=1
                                ),
                            )

                        if p_value >= alpha:
                            skeleton.loc[node_i, node_j] = skeleton.loc[node_j, node_i] = 0
                            sep_set[d][(node_i, node_j)] = sub_z
                            break

        self.skel = skeleton
        self.pdag = PDAG.from_pandas_adjacency(skeleton)

    def _return_max_layer(self, layering: dict[str, list[str]], node_i: str, node_j: str) -> str:
        """Highest layer of the pair of nodes considered.

        Args:
            layering (dict[str, list[str]]): given layering
            node_i (str): first node in the pair
            node_j (str): second node in the pair

        Returns:
            str: key corresponding to highest layer.
        """
        for key, value in reversed(layering.items()):
            if node_i in value or node_j in value:
                max_layer = key
                break
        return max_layer

    def _remove_pairs_after_key(
        self, layering: dict[str, list[str]], max_layer: str
    ) -> dict[str, list[str]]:
        items = list(layering.items())
        # Find the index of the target key
        try:
            target_index = next(i for i, (key, _) in enumerate(items) if key == max_layer)
        except StopIteration:
            # If the target key is not found, return the original dictionary
            return layering

        # Recreate the dictionary up to and including the target key
        return dict(items[: target_index + 1])

    def _remove_future_nodes(
        self, sep_set: list[str], allowed_dict: dict[str, list[str]]
    ) -> list[str]:
        """Remove selected nodes in allowed_dict from sep_set.

        Args:
            sep_set (list[str]): sep set
            allowed_dict (dict[str, list[str]]): allowed dict

        Returns:
            list[str]: list with future nodes removed.
        """
        dict_values = {item for value in allowed_dict.values() for item in value}
        return [item for item in sep_set if item in dict_values]

    def _unshielded_triples(self, pdag: PDAG) -> list[tuple[str, str, str]]:
        """Unshielded triples of a given PDAG.

        Args:
            pdag (PDAG): Skeleton after first phase

        Returns:
            list[tuple[str]]: List of unshielded triples of the form (i,j,k).
        """
        unshielded_triples: list[tuple[str, str, str]] = []
        for node in set(pdag.nodes):
            neighbors = pdag.neighbors(node=node)
            for neighbor in neighbors:
                distant_neighbors = list(pdag.neighbors(neighbor) - {node})
                if distant_neighbors:
                    unshielded_triples.extend(
                        [
                            (node, neighbor, dist_neigh)
                            for dist_neigh in distant_neighbors
                            if (node, dist_neigh) not in pdag.undir_edges
                            and (dist_neigh, node) not in pdag.undir_edges
                        ]
                    )
        unique_triples: list[tuple[str, str, str]] = []
        for tri in unshielded_triples:
            equivalent_tri = tri[::-1]
            if equivalent_tri not in unique_triples:
                unique_triples.append(tri)
        return unique_triples

    def _power_set(self, a_set: set[str]) -> set[frozenset[str]]:
        """Return power set.

        Args:
            a_set (set): Input set.

        Returns:
            frozenset[str]: all combinations of set members.
        """
        length = len(a_set)
        power_set: set[frozenset[str]] = {
            frozenset({e for e, b in zip(a_set, f"{i:{length}b}") if b == "1"})
            for i in range(2**length)
        }
        return power_set

    def _get_conditioning_sets(
        self, triples: tuple[str, str, str], pdag: PDAG
    ) -> set[frozenset[str]]:
        """Return conditioning set.

        Args:
            triples (tuple): triples
            pdag (PDAG): current pdag

        Returns:
            set[tuple[str]]: conditioning set from tuples and pdag.
        """
        set_a = self._power_set(pdag.neighbors(node=triples[0]))
        set_b = self._power_set(pdag.neighbors(node=triples[2]))

        unique_cond_set: set[frozenset[str]] = set_a.union(set_b)
        return unique_cond_set

    def _orient_vstructs_and_flag_amgiguities(
        self,
        data: dict[str, np.ndarray],
        alpha: float = 0.05,
        threshold: float = 0.5,
    ) -> None:
        """Given data orient all unshielded triples to vstrutures if possible.

        Args:
            data (dict[str, np.ndarray]): data
            alpha (float, optional): Significance level of test. Defaults to 0.05.
            threshold (float, optional): Threshold for ambiguity condition. Defaults to 0.5.

        Raises:
            AssertionError: If skeleton changes due to operation error is thrown.
        """
        pdag = self.pdag.copy()
        all_unshielded_triples = self._unshielded_triples(pdag=pdag)
        flag = []
        for triple in all_unshielded_triples:
            node_i, node_j, node_k = triple
            cond_sets = self._get_conditioning_sets(
                triples=triple, pdag=pdag
            )  # getting all adj(X_i) and adj(X_k)

            conditioning_subsetter = []  # initiate candidate conditioning sets

            for cond_set in list(cond_sets):
                sep_set = list(cond_set)
                if not sep_set:
                    _, p_value = self.ci_test.test(x_data=data[node_i], y_data=data[node_k])
                else:
                    _, p_value = self.ci_test.test(
                        x_data=data[node_i],
                        y_data=data[node_k],
                        z_data=np.concatenate(
                            [dat for node, dat in data.items() if node in sep_set], axis=1
                        ),
                    )
                if p_value >= alpha:  # if p_value is large then test statistic is small i.e.
                    # the Null of (cond) independence cannot be rejected.
                    conditioning_subsetter.append(
                        sep_set
                    )  # all subsets that give us cond. independence

            if sum([node_j in sep for sep in conditioning_subsetter]) == threshold * len(
                conditioning_subsetter
            ):
                flag.append(triple)
            elif sum([node_j in sep for sep in conditioning_subsetter]) < threshold * len(
                conditioning_subsetter
            ):
                pdag.undir_to_dir_edge(tail=node_i, head=node_j)
                pdag.undir_to_dir_edge(tail=node_k, head=node_j)

        original_skeleton = nx.from_pandas_adjacency(pdag.adjacency_matrix, create_using=nx.Graph)

        if not self.skeleton.equals(nx.to_pandas_adjacency(original_skeleton)):
            raise AssertionError(
                "Skeleton has changed. This shouldn't be possible. Check your inputs!"
            )
        self.pdag = pdag
        self.ambiguities = flag

    def maximally_orient(
        self, data: dict[str, np.ndarray], alpha: float = 0.05, threshold: float = 0.5
    ) -> None:
        """Given a skeleton, the following orientation steps are taken.

            1. All undirected edges between layers are immediately oriented
                according to the given layering.
            2. Potential v-structures are ordiented.
            3. The remaining undirected edges are oriented according to the four
                Meek rules.

        Args:
            data (dict[str, np.ndarray]): The data.
            alpha (float, optional): The significance
                threshold for the conditional independence test. Defaults to 0.05.
            threshold (float, optional): The majority vote threshold for deciding
                on ambiguous collder structures. Defaults to 0.5.
        """
        # orient immediately according to layering if present
        if self.layering is not None:
            self._orient_between_layers()
        # Orient v-structures
        self._orient_vstructs_and_flag_amgiguities(data=data, alpha=alpha, threshold=threshold)
        # Apply Meek Rules
        self._orient_according_to_meek_rules()

    def _adj_size_criterion(self, skel: pd.DataFrame, ell: int) -> bool:
        r"""Check if |adj(C, X_i) \\ {X_j}| >= l for every pair of adjacent vertices in C.

        Args:
            skel (pd.DataFrame): Skeleton C
            ell (int): size of separating sets

        Returns:
            bool: True if size of adjacency set is larger or equal l and False else.
        """
        assert skel.shape[0] == skel.shape[1]
        columns = skel.columns
        columns = sorted(columns)
        k = list(combinations(columns, 2))
        sorted(k, reverse=True)
        node_pairs = [(x, y) for x, y in k]
        less_l = 0
        for node_i, node_j in node_pairs:
            adj_i = set(skel.index[skel[node_i] != 0].tolist())
            adj_ij = adj_i - {node_j}
            if len(adj_ij) < ell:
                less_l += 1
            else:
                break
        if less_l == len(node_pairs):
            return False
        else:
            return True

    def _between_edges(self) -> list[tuple[str, str]]:
        """Return between edges when layers are present.

        Returns:
            list[tuple]: list of edges between layers.
        """
        if self.layering is None:
            raise ValueError("Layering is not provided. Cannot retrieve between edges.")
        no_orient = []
        undir_edges = self.pdag.undir_edges
        for edge in undir_edges:
            i, j = edge
            for cell_nodes in self.layering.values():
                if set([i, j]).issubset(set(cell_nodes)):
                    no_orient.append(edge)
                    break
        return list(set(undir_edges).difference(set(no_orient)))

    def _orient_between_layers(self) -> None:
        """Orients edges between layers."""
        if self.layering is None:
            raise ValueError("Layering is not provided. Cannot orient between edges.")
        to_orient = self._between_edges()

        flat_mapper: list[str] = []
        for nodelist in self.layering.values():
            flat_mapper.extend(nodelist)

        final_edge_list = []
        for edge_to_orient in to_orient:
            sorted_edge = sorted(edge_to_orient, key=flat_mapper.index)
            final_edge_list.append(tuple(sorted_edge))

        for tail, head in final_edge_list:
            self.pdag.undir_to_dir_edge(tail=tail, head=head)

    def _orient_according_to_meek_rules(self) -> None:
        """Orient edges according to Meek rules."""
        cpdag = self.pdag.copy()
        cpdag = rule_1(pdag=cpdag)
        cpdag = rule_2(pdag=cpdag)
        cpdag = rule_3(pdag=cpdag)
        cpdag = rule_4(pdag=cpdag)
        self.pdag = cpdag

    @property
    def skeleton(self) -> pd.DataFrame:
        """Represent the underlying skeleton as adjacency matrix.

        Returns:
            pd.DataFrame: Adjacency matrix of the skeleton.
        """
        undir_graph = nx.from_pandas_adjacency(self.pdag.adjacency_matrix, create_using=nx.Graph)
        return nx.to_pandas_adjacency(undir_graph)

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        """Represent the underlying learned PDAG as adjacency matrix.

        Returns:
            pd.DataFrame: Adjacency matrix of the PDAG.
        """
        amat = self.pdag.adjacency_matrix.values
        cpdag_amat = amat.copy()
        upper_triangle_indices = np.triu_indices_from(amat, k=1)
        mask = (amat[upper_triangle_indices] == 1) & (amat[upper_triangle_indices[::-1]] == 1)

        # Set these entries to 2 in both (i, j) and (j, i) locations
        cpdag_amat[upper_triangle_indices[0][mask], upper_triangle_indices[1][mask]] = 2
        cpdag_amat[upper_triangle_indices[1][mask], upper_triangle_indices[0][mask]] = 2

        amat_names = self.pdag.adjacency_matrix.columns

        return pd.DataFrame(cpdag_amat, columns=amat_names, index=amat_names)

    @property
    def causal_order(self) -> list[str] | None:
        """Returns causal order if PDAG is in fact a DAG.

        Else it will return None.

        Returns:
            list[str] | None: causal order if appropriate.
        """
        ordering = None
        if self.pdag.dir_edges and not self.pdag.undir_edges:
            dag = DAG(nodes=self.pdag.nodes, edges=self.pdag.dir_edges)
            ordering = dag.causal_order
        return ordering


class MicroPC(LearnAlgo):
    """Standard PC stable on micro nodes aggregated after the fact."""

    def __init__(self, alpha: float = 0.05) -> None:
        """Inits the object.

        Args:
            alpha (float, optional): Significance level of the test. Defaults to 0.05.
        """
        self.alpha = alpha
        self.graph: DAG | PDAG

    def _causallearn2amat(self, causal_learn_graph: np.ndarray) -> np.ndarray:
        amat = np.zeros(causal_learn_graph.shape)
        for col in range(causal_learn_graph.shape[1]):
            for row in range(causal_learn_graph.shape[0]):
                if causal_learn_graph[row, col] == -1 and causal_learn_graph[col, row] == 1:
                    amat[row, col] = 1
                if causal_learn_graph[row, col] == -1 and causal_learn_graph[col, row] == -1:
                    amat[row, col] = amat[col, row] = 1
                if causal_learn_graph[row, col] == 1 and causal_learn_graph[col, row] == 1:
                    amat[row, col] = amat[col, row] = 1
        return amat

    def learn_graph(self, data_dict: dict[str, np.ndarray], *args: Any, **kwargs: Any) -> GRAPH:
        """Learn graph.

        Args:
            data_dict (_type_): _description_
            *args (Any): additional args.
            **kwargs (Any): additional kwargs.

        Returns:
            PDAG: _description_
        """
        micro_data = np.concatenate([d_data for d_data in data_dict.values()], axis=1)
        micro_pc = pc(
            data=micro_data, alpha=self.alpha, indep_test="fisherz", uc_rule=1, show_progress=True
        )
        micro_amat = self._causallearn2amat(causal_learn_graph=micro_pc.G.graph)
        Q = _make_group_mapping(data_dict=data_dict)
        interim_group_adjacency_matrix = Q @ micro_amat @ Q.T
        np.fill_diagonal(interim_group_adjacency_matrix, 0)
        group_adjacency_matrix = (interim_group_adjacency_matrix > 0).astype(int)

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
