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
from collections import defaultdict
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GRAPH(metaclass=ABCMeta):
    """Abstract base class for all Graphs in current project."""

    def __init__(self) -> None:
        """Init ABC."""
        pass

    @property
    @abstractmethod
    def adjacency_matrix(self) -> pd.DataFrame:
        """Return adjacency matrix.

        Raises:
            AssertionError: _description_
            AssertionError: _description_
            ValueError: _description_
            AssertionError: _description_
            AssertionError: _description_
            TypeError: _description_

        Returns:
            pd.DataFrame: Adjacency matrix of underlying graph.
        """

    @property
    @abstractmethod
    def causal_order(self) -> list[str] | None:
        """Return causal order.

        Raises:
            AssertionError: _description_
            AssertionError: _description_
            ValueError: _description_
            AssertionError: _description_
            AssertionError: _description_
            TypeError: _description_

        Returns:
            list[str] | None: Causal order of underlying graph.
                None if not a DAG.
        """


class PDAG(GRAPH):
    """Class for dealing with partially directed graph i.e.

    graphs that contain both directed and undirected edges.
    """

    def __init__(
        self,
        nodes: list[str] | None = None,
        dir_edges: list[tuple[str, str]] | None = None,
        undir_edges: list[tuple[str, str]] | None = None,
    ) -> None:
        """PDAG constructor.

        Args:
            nodes (list[str] | None, optional): Nodes in the PDAG. Defaults to None.
            dir_edges (list[tuple[str,str]] | None, optional): directed edges. Defaults to None.
            undir_edges (list[tuple[str,str]] | None, optional): undirected edges. Defaults to None.
        """
        if nodes is None:
            nodes = []
        if dir_edges is None:
            dir_edges = []
        if undir_edges is None:
            undir_edges = []

        self._nodes = set(nodes)
        self._undir_edges: set[tuple[str, str]] = set()
        self._dir_edges: set[tuple[str, str]] = set()
        self._parents: defaultdict[str, set[str]] = defaultdict(set)
        self._children: defaultdict[str, set[str]] = defaultdict(set)
        self._neighbors: defaultdict[str, set[str]] = defaultdict(set)
        self._undirected_neighbors: defaultdict[str, set[str]] = defaultdict(set)

        for dir_edge in dir_edges:
            self._add_dir_edge(*dir_edge)
        for unir_edge in undir_edges:
            self._add_undir_edge(*unir_edge)

    def _add_dir_edge(self, i: str, j: str) -> None:
        self._nodes.add(i)
        self._nodes.add(j)
        self._dir_edges.add((i, j))

        self._neighbors[i].add(j)
        self._neighbors[j].add(i)

        self._children[i].add(j)
        self._parents[j].add(i)

    def _add_undir_edge(self, i: str, j: str) -> None:
        self._nodes.add(i)
        self._nodes.add(j)
        self._undir_edges.add((i, j))

        self._neighbors[i].add(j)
        self._neighbors[j].add(i)

        self._undirected_neighbors[i].add(j)
        self._undirected_neighbors[j].add(i)

    def children(self, node: str) -> set[str]:
        """Gives all children of node `node`.

        Args:
            node (str): node in current PDAG.

        Returns:
            set: set of children.
        """
        if node in self._children.keys():
            return self._children[node]
        else:
            return set()

    def parents(self, node: str) -> set[str]:
        """Gives all parents of node `node`.

        Args:
            node (str): node in current PDAG.

        Returns:
            set: set of parents.
        """
        if node in self._parents.keys():
            return self._parents[node]
        else:
            return set()

    def neighbors(self, node: str) -> set[str]:
        """Gives all neighbors of node `node`.

        Args:
            node (str): node in current PDAG.

        Returns:
            set: set of neighbors.
        """
        if node in self._neighbors.keys():
            return self._neighbors[node]
        else:
            return set()

    def undir_neighbors(self, node: str) -> set[str]:
        """Gives all undirected neighbors of node `node`.

        Args:
            node (str): node in current PDAG.

        Returns:
            set: set of undirected neighbors.
        """
        if node in self._undirected_neighbors.keys():
            return self._undirected_neighbors[node]
        else:
            return set()

    def is_adjacent(self, i: str, j: str) -> bool:
        """Return True if the graph contains an directed or undirected edge between i and j.

        Args:
            i (str): node i.
            j (str): node j.

        Returns:
            bool: True if i-j or i->j or i<-j
        """
        return any(
            (
                (j, i) in self.dir_edges or (j, i) in self.undir_edges,
                (i, j) in self.dir_edges or (i, j) in self.undir_edges,
            )
        )

    def is_clique(self, potential_clique: set[str]) -> bool:
        """Check every pair of node X potential_clique is adjacent."""
        return all(self.is_adjacent(i, j) for i, j in combinations(potential_clique, 2))

    @classmethod
    def from_pandas_adjacency(cls, pd_amat: pd.DataFrame) -> PDAG:
        """Build PDAG from a Pandas adjacency matrix.

        Args:
            pd_amat (pd.DataFrame): input adjacency matrix.

        Returns:
            PDAG
        """
        assert pd_amat.shape[0] == pd_amat.shape[1]
        nodes = pd_amat.columns

        all_connections = []
        start, end = np.where(pd_amat != 0)
        for idx, _ in enumerate(start):
            all_connections.append((pd_amat.columns[start[idx]], pd_amat.columns[end[idx]]))

        temp = [set(i) for i in all_connections]
        temp2 = [arc for arc in all_connections if temp.count(set(arc)) > 1]
        undir_edges = [tuple(item) for item in set(frozenset(item) for item in temp2)]

        dir_edges = [edge for edge in all_connections if edge not in temp2]

        return PDAG(nodes=nodes, dir_edges=dir_edges, undir_edges=undir_edges)

    def remove_edge(self, i: str, j: str) -> None:
        """Removes edge in question.

        Args:
            i (str): tail
            j (str): head

        Raises:
            AssertionError: if edge does not exist
        """
        if (i, j) not in self.dir_edges and (i, j) not in self.undir_edges:
            raise AssertionError("Edge does not exist in current PDAG")

        self._undir_edges.discard((i, j))
        self._dir_edges.discard((i, j))
        self._children[i].discard(j)
        self._parents[j].discard(i)
        self._neighbors[i].discard(j)
        self._neighbors[j].discard(i)
        self._undirected_neighbors[i].discard(j)
        self._undirected_neighbors[j].discard(i)

    def undir_to_dir_edge(self, tail: str, head: str) -> None:
        """Takes a undirected edge and turns it into a directed one.

        tail indicates the starting node of the edge and head the end node, i.e.
        tail -> head.

        Args:
            tail (str): starting node
            head (str): end node

        Raises:
            AssertionError: if edge does not exist or is not undirected.
        """
        if (tail, head) not in self.undir_edges and (
            head,
            tail,
        ) not in self.undir_edges:
            raise AssertionError("Edge seems not to be undirected or even there at all.")
        self._undir_edges.discard((tail, head))
        self._undir_edges.discard((head, tail))
        self._neighbors[tail].discard(head)
        self._neighbors[head].discard(tail)
        self._undirected_neighbors[tail].discard(head)
        self._undirected_neighbors[head].discard(tail)

        self._add_dir_edge(i=tail, j=head)

    def remove_node(self, node: str) -> None:
        """Remove a node from the graph.

        Args:
            node (str): node to remove
        """
        self._nodes.remove(node)

        self._dir_edges = {(i, j) for i, j in self._dir_edges if node not in (i, j)}

        self._undir_edges = {(i, j) for i, j in self._undir_edges if node not in (i, j)}

        for child in self._children[node]:
            self._parents[child].remove(node)
            self._neighbors[child].remove(node)

        for parent in self._parents[node]:
            self._children[parent].remove(node)
            self._neighbors[parent].remove(node)

        for u_nbr in self._undirected_neighbors[node]:
            self._undirected_neighbors[u_nbr].remove(node)
            self._neighbors[u_nbr].remove(node)

        self._parents.pop(node, "I was never here")
        self._children.pop(node, "I was never here")
        self._neighbors.pop(node, "I was never here")
        self._undirected_neighbors.pop(node, "I was never here")

    def to_dag(self) -> nx.DiGraph:
        r"""Algorithm as described in Chickering (2002).

            1. From PDAG P create DAG G containing all directed edges from P
            2. Repeat the following: Select node v in P s.t.
                i. v has no outgoing edges (children) i.e. \\(ch(v) = \\emptyset \\)

                ii. \\(neigh(v) \\neq \\emptyset\\)
                    Then \\( (pa(v) \\cup (neigh(v) \\) form a clique.
                    For each v that is in a clique and is part of an undirected edge in P
                    i.e. w - v, insert a directed edge w -> v in G.
                    Remove v and all incident edges from P and continue with next node.
                    Until all nodes have been deleted from P.

        Returns:
            nx.DiGraph: DAG that belongs to the MEC implied by the PDAG
        """
        pdag = self.copy()

        dag = nx.DiGraph()
        dag.add_nodes_from(pdag.nodes)
        dag.add_edges_from(pdag.dir_edges)

        if pdag.num_undir_edges == 0:
            return dag
        else:
            while pdag.num_nodes > 0:
                # find node with (1) no directed outgoing edges and
                #                (2) the set of undirected neighbors is either empty or
                #                    undirected neighbors + parents of X are a clique
                found = False
                for node in pdag.nodes:
                    children = pdag.children(node)
                    neighbors = pdag.neighbors(node)
                    # pdag._undirected_neighbors[node]
                    parents = pdag.parents(node)
                    potential_clique_members = neighbors.union(parents)

                    is_clique = pdag.is_clique(potential_clique_members)

                    if not children and (not neighbors or is_clique):
                        found = True
                        # add all edges of node as outgoing edges to dag
                        for edge in pdag.undir_edges:
                            if node in edge:
                                incident_node = set(edge) - {node}
                                dag.add_edge(*incident_node, node)

                        pdag.remove_node(node)
                        break

                if not found:
                    logger.warning("PDAG not extendible: Random DAG on skeleton drawn.")

                    dag = nx.from_pandas_adjacency(self._amat_to_dag(), create_using=nx.DiGraph)

                    break

            return dag

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        """Returns adjacency matrix.

        The i,jth entry being one indicates that there is an edge
        from i to j. A zero indicates that there is no edge.

        Returns:
            pd.DataFrame: adjacency matrix
        """
        amat = pd.DataFrame(
            np.zeros([self.num_nodes, self.num_nodes]),
            index=self.nodes,
            columns=self.nodes,
        )
        for edge in self.dir_edges:
            amat.loc[edge] = 1
        for edge in self.undir_edges:
            amat.loc[edge] = amat.loc[edge[::-1]] = 1
        return amat

    @property
    def causal_order(self) -> None:
        """Causal order is None.

        This is because PDAGs only allow for a partial causal order.

        Returns:
            None: None
        """
        return None

    def _amat_to_dag(self) -> pd.DataFrame:
        """Transform the adjacency matrix of an PDAG to the adjacency matrix.

            of SOME DAG in the Markov equivalence class.

        Returns:
            pd.DataFrame: DAG, a member of the MEC.
        """
        pdag_amat = self.adjacency_matrix.to_numpy()

        p = pdag_amat.shape[0]
        ## amat to skel
        skel = pdag_amat + pdag_amat.T
        skel[np.where(skel > 1)] = 1
        ## permute skel
        permute_ord = np.random.choice(a=p, size=p, replace=False)
        skel = skel[:, permute_ord][permute_ord]

        ## skel to dag
        for i in range(1, p):
            for j in range(0, i + 1):
                if skel[i, j] == 1:
                    skel[i, j] = 0

        ## inverse permutation
        i_ord = np.sort(permute_ord)
        skel = skel[:, i_ord][i_ord]
        return pd.DataFrame(
            skel,
            index=self.adjacency_matrix.index,
            columns=self.adjacency_matrix.columns,
        )

    def vstructs(self) -> set[tuple[str, str]]:
        """Retrieve v-structures.

        Returns:
            set: set of all v-structures
        """
        vstructures = set()
        for node in self._nodes:
            for p1, p2 in combinations(self._parents[node], 2):
                if p1 not in self._parents[p2] and p2 not in self._parents[p1]:
                    vstructures.add((p1, node))
                    vstructures.add((p2, node))
        return vstructures

    def copy(self) -> PDAG:
        """Return a copy of the graph."""
        return PDAG(
            nodes=list(self._nodes),
            dir_edges=list(self._dir_edges),
            undir_edges=list(self._undir_edges),
        )

    def show(self) -> None:
        """Plot PDAG."""
        graph = self.to_networkx()
        pos = nx.circular_layout(graph)
        nx.draw(graph, pos=pos, with_labels=True)

    def to_networkx(self) -> nx.MultiDiGraph:
        """Convert to networkx graph.

        Returns:
            nx.MultiDiGraph: Graph with directed and undirected edges.
        """
        nx_pdag = nx.MultiDiGraph()
        nx_pdag.add_nodes_from(self.nodes)
        nx_pdag.add_edges_from(self.dir_edges)
        for edge in self.undir_edges:
            nx_pdag.add_edge(*edge)
            nx_pdag.add_edge(*edge[::-1])

        return nx_pdag

    def _meek_mec_enumeration(self, pdag: PDAG, dag_list: list[DAG]) -> None:
        """Apply Meek's MEC enumeration algorithm.

        Args:
            pdag (PDAG): partially directed graph in question.
            dag_list (list): list of currently found DAGs.

        References:
            Wienöbst, Marcel, et al. "Efficient enumeration of Markov equivalent DAGs."
            Proceedings of the AAAI Conference on Artificial Intelligence.
            Vol. 37. No. 10. 2023.
        """
        g_copy = pdag.copy()
        g_copy = self._apply_meek_rules(g_copy)  # Apply Meek rules

        undir_edges = g_copy.undir_edges
        if undir_edges:
            i, j = undir_edges[0]  # Take first undirected edge

        if not g_copy.undir_edges:
            # makes sure that flaoting nodes are preserved
            new_member = DAG()
            new_member.add_nodes_from(g_copy.nodes)
            new_member.add_edges_from(g_copy.dir_edges)
            dag_list.append(new_member)
            return  # Add DAG to current list

        # Recursion first orientation:
        g_copy.undir_to_dir_edge(i, j)
        self._meek_mec_enumeration(pdag=g_copy, dag_list=dag_list)
        g_copy.remove_edge(i, j)

        # Recursion second orientation
        g_copy._add_dir_edge(j, i)
        self._meek_mec_enumeration(pdag=g_copy, dag_list=dag_list)

    def to_allDAGs(self) -> list[DAG]:
        """Recursion algorithm which recursively applies the following steps.

            1. Orient the first undirected edge found.
            2. Apply Meek rules.
            3. Recurse with each direction of the oriented edge.
        This corresponds to Algorithm 2 in Wienöbst et al. (2023).

        References:
            Wienöbst, Marcel, et al. "Efficient enumeration of Markov equivalent DAGs."
            Proceedings of the AAAI Conference on Artificial Intelligence.
            Vol. 37. No. 10. 2023.
        """
        all_dags: list[DAG] = []
        self._meek_mec_enumeration(pdag=self, dag_list=all_dags)
        return all_dags

    # use Meek's cpdag2alldag
    def _apply_meek_rules(self, G: PDAG) -> PDAG:
        """Apply all four Meek rules to a PDAG turning it into a CPDAG.

        Args:
            G (PDAG): PDAG to complete

        Returns:
            PDAG: completed PDAG.
        """
        # Apply Meek Rules
        cpdag = G.copy()
        cpdag = rule_1(pdag=cpdag)
        cpdag = rule_2(pdag=cpdag)
        cpdag = rule_3(pdag=cpdag)
        cpdag = rule_4(pdag=cpdag)
        return cpdag

    def to_random_dag(self) -> DAG:
        """Provides a random DAG residing in the MEC.

        Returns:
            nx.DiGraph: random DAG living in MEC
        """
        to_dag_candidate = self.copy()

        while to_dag_candidate.num_undir_edges > 0:
            chosen_edge = to_dag_candidate.undir_edges[
                np.random.choice(to_dag_candidate.num_undir_edges)
            ]
            choose_orientation = [chosen_edge, chosen_edge[::-1]]
            node_i, node_j = choose_orientation[np.random.choice(len(choose_orientation))]

            to_dag_candidate.undir_to_dir_edge(tail=node_i, head=node_j)
            to_dag_candidate = to_dag_candidate._apply_meek_rules(G=to_dag_candidate)

        return DAG.from_pandas_adjacency(to_dag_candidate.adjacency_matrix)

    @property
    def nodes(self) -> list[str]:
        """Get all nods in current PDAG.

        Returns:
            list: list of nodes.
        """
        return sorted(list(self._nodes))

    @property
    def num_nodes(self) -> int:
        """Number of nodes in current PDAG.

        Returns:
            int: Number of nodes
        """
        return len(self._nodes)

    @property
    def num_undir_edges(self) -> int:
        """Number of undirected edges in current PDAG.

        Returns:
            int: Number of undirected edges
        """
        return len(self._undir_edges)

    @property
    def num_dir_edges(self) -> int:
        """Number of directed edges in current PDAG.

        Returns:
            int: Number of directed edges
        """
        return len(self._dir_edges)

    @property
    def num_adjacencies(self) -> int:
        """Number of adjacent nodes in current PDAG.

        Returns:
            int: Number of adjacent nodes
        """
        return self.num_undir_edges + self.num_dir_edges

    @property
    def undir_edges(self) -> list[tuple[str, str]]:
        """Gives all undirected edges in current PDAG.

        Returns:
            list[tuple[str,str]]: List of undirected edges.
        """
        return list(self._undir_edges)

    @property
    def dir_edges(self) -> list[tuple[str, str]]:
        """Gives all directed edges in current PDAG.

        Returns:
            list[tuple[str,str]]: List of directed edges.
        """
        return list(self._dir_edges)


def vstructs(dag: nx.DiGraph) -> set[tuple[str, str]]:
    """Retrieve all v-structures in a DAG.

    Args:
        dag (nx.DiGraph): DAG in question

    Returns:
        set: Set of all v-structures.
    """
    vstructures = set()
    for node in dag.nodes():
        for p1, p2 in combinations(list(dag.predecessors(node)), 2):  # get all parents of node
            if not dag.has_edge(p1, p2) and not dag.has_edge(p2, p1):
                vstructures.add((p1, node))
                vstructures.add((p2, node))
    return vstructures


def rule_1(pdag: PDAG) -> PDAG:
    """Applies first Meek rule.

    Given the following pattern X -> Y - Z. Orient Y - Z to Y -> Z
    if X and Z are non-adjacent (otherwise a new v-structure arises).

    Args:
        pdag (PDAG): PDAG before application of rule.

    Returns:
        PDAG: PDAG after application of rule.
    """
    copy_pdag = pdag.copy()
    for edge in copy_pdag.undir_edges:
        reverse_edge = edge[::-1]
        test_edges = [edge, reverse_edge]
        for tail, head in test_edges:
            orient = False
            undir_parents = copy_pdag.parents(tail)
            if undir_parents:
                for parent in undir_parents:
                    if not copy_pdag.is_adjacent(parent, head):
                        orient = True
            if orient:
                copy_pdag.undir_to_dir_edge(tail=tail, head=head)
                break
    return copy_pdag


def rule_2(pdag: PDAG) -> PDAG:
    """Applies the second Meek rule.

    Given the following directed triple
    X -> Y -> Z where X - Z are indeed adjacent.
    Orient X - Z to X -> Z otherwise a cycle arises.

    Args:
        pdag (PDAG): PDAG before application of rule.

    Returns:
        PDAG: PDAG after application of rule.
    """
    copy_pdag = pdag.copy()
    for edge in copy_pdag.undir_edges:
        reverse_edge = edge[::-1]
        test_edges = [edge, reverse_edge]
        for tail, head in test_edges:
            orient = False
            undir_children = copy_pdag.children(tail)
            if undir_children:
                for child in undir_children:
                    if head in copy_pdag.children(child):
                        orient = True
            if orient:
                copy_pdag.undir_to_dir_edge(tail=tail, head=head)
                break
    return copy_pdag


def rule_3(pdag: PDAG) -> PDAG:
    """Apply 3rd Meek rule.

    Orient X - Z to X -> Z, whenever there are two triples
    X - Y1 -> Z and X - Y2 -> Z such that Y1 and Y2 are non-adjacent.

    Args:
        pdag (PDAG): PDAG before application of rule.

    Returns:
        PDAG: PDAG after application of rule.
    """
    copy_pdag = pdag.copy()
    for edge in copy_pdag.undir_edges:
        reverse_edge = edge[::-1]
        test_edges = [edge, reverse_edge]
        for tail, head in test_edges:
            # if true that tail - node1 -> head and tail - node2 -> head
            # while {node1 U node2} = 0 then orient tail -> head
            orient = False
            num_neighbors = 2
            if len(copy_pdag.undir_neighbors(tail)) >= num_neighbors:
                undir_n = copy_pdag.undir_neighbors(tail)
                selection = [
                    (node1, node2)
                    for node1, node2 in combinations(undir_n, 2)
                    if not copy_pdag.is_adjacent(node1, node2)
                ]
                if selection:
                    for node1, node2 in selection:
                        if head in copy_pdag.parents(node1).intersection(copy_pdag.parents(node2)):
                            orient = True
            if orient:
                copy_pdag.undir_to_dir_edge(tail=tail, head=head)
                break
    return pdag


def rule_4(pdag: PDAG) -> PDAG:
    """Apply 4th Meek rule.

    Orient X - Y1 to X -> Y1, whenever there are
    two triples with X - Z and X - Y1 <- Z and X - Y2 -> Z
    such that Y1 and Y2 are non-adjacent.

    Args:
        pdag (PDAG): PDAG before application of rule.

    Returns:
        PDAG: PDAG after application of rule.
    """
    copy_pdag = pdag.copy()
    for edge in copy_pdag.undir_edges:
        reverse_edge = edge[::-1]
        test_edges = [edge, reverse_edge]
        for tail, head in test_edges:
            orient = False
            if len(copy_pdag.undir_neighbors(tail)) > 0:
                undirected_n = copy_pdag.undir_neighbors(tail)
                for undir_n in undirected_n:
                    if tail in copy_pdag.children(undir_n):
                        children_select = list(copy_pdag.children(undir_n))
                        if children_select:
                            for parent in children_select:
                                if head in copy_pdag.children(parent):
                                    orient = True
            if orient:
                copy_pdag.undir_to_dir_edge(tail=tail, head=head)
                break
    return pdag


def dag2cpdag(dag: nx.DiGraph) -> PDAG:
    """Convertes a DAG into its unique CPDAG.

    Args:
        dag (nx.DiGraph): DAG the CPDAG corresponds to.

    Returns:
        PDAG: unique CPDAG
    """
    copy_dag = dag.copy()
    # Skeleton
    skeleton = nx.to_pandas_adjacency(copy_dag.to_undirected())
    # v-Structures
    vstructures = vstructs(dag=copy_dag)

    for edge in vstructures:  # orient v-structures
        skeleton.loc[edge[::-1]] = 0

    pdag_init = PDAG.from_pandas_adjacency(skeleton)

    # Apply Meek Rules
    cpdag = rule_1(pdag=pdag_init)
    cpdag = rule_2(pdag=cpdag)
    cpdag = rule_3(pdag=cpdag)
    cpdag = rule_4(pdag=cpdag)

    return cpdag


class DAG(GRAPH):
    """General class for dealing with directed acyclic graph i.e.

    graphs that are directed and must not contain any cycles.
    """

    def __init__(
        self,
        nodes: list[str] | None = None,
        edges: list[tuple[str, str]] | None = None,
    ) -> None:
        """DAG constructor.

        Args:
            nodes (list[str] | None, optional): Nodes. Defaults to None.
            edges (list[tuple[str,str]] | None, optional): Edges. Defaults to None.
        """
        if nodes is None:
            nodes = []
        if edges is None:
            edges = []

        self._nodes: set[str] = set(nodes)
        self._edges: set[tuple[str, str]] = set()
        self._parents: defaultdict[str, set[str]] = defaultdict(set)
        self._children: defaultdict[str, set[str]] = defaultdict(set)
        self._random_state: np.random.Generator = np.random.default_rng(seed=2023)

        for edge in edges:
            self._add_edge(*edge)

    def _add_node(self, node: str) -> None:
        self._nodes.add(node)

    def _add_edge(self, i: str, j: str) -> None:
        self._nodes.add(i)
        self._nodes.add(j)
        self._edges.add((i, j))

        # Check if graph is acyclic
        if not self.is_acyclic():
            raise ValueError(
                "The edge set you provided \
                induces one or more cycles.\
                Check your input!"
            )

        self._children[i].add(j)
        self._parents[j].add(i)

    @property
    def random_state(self) -> np.random.Generator:
        """Current random state.

        Returns:
            np.random.Generator: Generator object.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, r: np.random.Generator) -> None:
        if not isinstance(r, np.random.Generator):
            raise AssertionError("Specify numpy random number generator object!")
        self._random_state = r

    def add_edge(self, edge: tuple[str, str]) -> None:
        """Add edge to DAG.

        Args:
            edge (tuple[str, str]): Edge to add
        """
        self._add_edge(*edge)

    def add_node(self, node: str) -> None:
        """Add node to DAG.

        Args:
            node (str): node to add
        """
        self._add_node(node)

    def add_edges_from(self, edges: list[tuple[str, str]]) -> None:
        """Add multiple edges to DAG.

        Args:
            edges (list[tuple[str, str]]): Edges to add
        """
        for edge in edges:
            self.add_edge(edge=edge)

    def add_nodes_from(self, nodes: list[str]) -> None:
        """Add multiple nodes to DAG.

        Args:
            nodes (list[str]): nodes to add
        """
        for node in nodes:
            self.add_node(node)

    def children(self, of_node: str) -> list[str]:
        """Gives all children of node `node`.

        Args:
            of_node (str): node in current DAG.

        Returns:
            list: of children.
        """
        if of_node in self._children.keys():
            return list(self._children[of_node])
        else:
            return []

    def parents(self, of_node: str) -> list[str]:
        """Gives all parents of node `node`.

        Args:
            of_node (str): node in current DAG.

        Returns:
            list: of parents.
        """
        if of_node in self._parents.keys():
            return list(self._parents[of_node])
        else:
            return []

    def induced_subgraph(self, nodes: list[str]) -> DAG:
        """Returns the induced subgraph on the nodes in `nodes`.

        Args:
            nodes (list[str]): List of nodes.

        Returns:
            DAG: Induced subgraph.
        """
        edges = [(i, j) for i, j in self.edges if i in nodes and j in nodes]
        return DAG(nodes=nodes, edges=edges)

    def is_adjacent(self, i: str, j: str) -> bool:
        """Return True if the graph contains an directed edge between i and j.

        Args:
            i (str): node i.
            j (str): node j.

        Returns:
            bool: True if i->j or i<-j
        """
        return (j, i) in self.edges or (i, j) in self.edges

    def is_clique(self, potential_clique: set[str]) -> bool:
        """Check every pair of node X potential_clique is adjacent."""
        return all(self.is_adjacent(i, j) for i, j in combinations(potential_clique, 2))

    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic.

        Returns:
            bool: True if graph is acyclic.
        """
        nx_dag = self.to_networkx()
        acyclic: bool = nx.is_directed_acyclic_graph(nx_dag)
        return acyclic

    @classmethod
    def from_pandas_adjacency(cls, pd_amat: pd.DataFrame, *args: Any, **kwargs: Any) -> DAG:
        """Build DAG from a Pandas adjacency matrix.

        Args:
            pd_amat (pd.DataFrame): input adjacency matrix.
            args (Any): Additional arguments.
            kwargs (Any): Additional arguments.

        Returns:
            DAG
        """
        assert pd_amat.shape[0] == pd_amat.shape[1]
        nodes = pd_amat.columns

        all_connections = []
        start, end = np.where(pd_amat != 0)
        for idx, _ in enumerate(start):
            all_connections.append((pd_amat.columns[start[idx]], pd_amat.columns[end[idx]]))

        temp = [set(i) for i in all_connections]
        temp2 = [arc for arc in all_connections if temp.count(set(arc)) > 1]

        dir_edges = [edge for edge in all_connections if edge not in temp2]

        return DAG(nodes=nodes, edges=dir_edges)

    def remove_edge(self, i: str, j: str) -> None:
        """Removes edge in question.

        Args:
            i (str): tail
            j (str): head

        Raises:
            AssertionError: if edge does not exist
        """
        if (i, j) not in self.edges:
            raise AssertionError("Edge does not exist in current DAG")

        self._edges.discard((i, j))
        self._children[i].discard(j)
        self._parents[j].discard(i)

    def remove_node(self, node: str) -> None:
        """Remove a node from the graph."""
        self._nodes.remove(node)

        self._edges = {(i, j) for i, j in self._edges if node not in (i, j)}

        for child in self._children[node]:
            self._parents[child].remove(node)

        for parent in self._parents[node]:
            self._children[parent].remove(node)

        self._parents.pop(node, "I was never here")
        self._children.pop(node, "I was never here")

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        """Returns adjacency matrix.

        The i,jth entry being one indicates that there is an edge
        from i to j. A zero indicates that there is no edge.

        Returns:
            pd.DataFrame: adjacency matrix
        """
        amat = pd.DataFrame(
            np.zeros([self.num_nodes, self.num_nodes]),
            index=self.nodes,
            columns=self.nodes,
        )
        for edge in self.edges:
            amat.loc[edge] = 1
        return amat

    def vstructs(self) -> set[tuple[str, str]]:
        """Retrieve v-structures.

        Returns:
            set: set of all v-structures
        """
        vstructures = set()
        for node in self._nodes:
            for p1, p2 in combinations(self._parents[node], 2):
                if p1 not in self._parents[p2] and p2 not in self._parents[p1]:
                    vstructures.add((p1, node))
                    vstructures.add((p2, node))
        return vstructures

    def copy(self) -> DAG:
        """Return a copy of the graph."""
        return DAG(nodes=list(self._nodes), edges=list(self._edges))

    def show(self) -> None:
        """Plot DAG."""
        graph = self.to_networkx()
        pos = nx.circular_layout(graph)
        nx.draw(graph, pos=pos, with_labels=True)

    def to_networkx(self) -> nx.DiGraph:
        """Convert to networkx graph.

        Returns:
            nx.MultiDiGraph: Graph with directed and undirected edges.
        """
        nx_dag = nx.DiGraph()
        nx_dag.add_nodes_from(self.nodes)
        nx_dag.add_edges_from(self.edges)

        return nx_dag

    @property
    def nodes(self) -> list[str]:
        """Get all nods in current DAG.

        Returns:
            list: list of nodes.
        """
        return sorted(list(self._nodes))

    @property
    def num_nodes(self) -> int:
        """Number of nodes in current DAG.

        Returns:
            int: Number of nodes
        """
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Number of directed edges in current DAG.

        Returns:
            int: Number of directed edges
        """
        return len(self._edges)

    @property
    def sparsity(self) -> float:
        """Sparsity of the graph.

        Returns:
            float: in [0,1]
        """
        s = self.num_nodes
        return self.num_edges / s / (s - 1) * 2

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Gives all directed edges in current DAG.

        Returns:
            list[tuple[str,str]]: List of directed edges.
        """
        return list(self._edges)

    @property
    def causal_order(self) -> list[str]:
        """Returns the causal order of the current graph.

        Note that this order is in general not unique.

        Returns:
            list[str]: Causal order
        """
        return list(nx.lexicographical_topological_sort(self.to_networkx()))

    @property
    def sink_nodes(self) -> list[str]:
        """Returns all sink nodes, i.e.

        nodes with no descendents in particular no children.

        Returns:
            list[str]: list of sink nodes.
        """
        return [
            s
            for b, s in zip([self.children(of_node=node) == [] for node in self.nodes], self.nodes)
            if b
        ]

    @property
    def source_nodes(self) -> list[str]:
        """Returns all source nodes, i.e.

        nodes with no ancesters in particular no parents.

        Returns:
            list[str]: list of sink nodes.
        """
        return [
            s
            for b, s in zip([self.parents(of_node=node) == [] for node in self.nodes], self.nodes)
            if b
        ]

    @property
    def max_in_degree(self) -> int:
        """Maximum in-degree of the graph.

        Returns:
            int: Maximum in-degree
        """
        return max(len(self._parents[node]) for node in self._nodes)

    @property
    def max_out_degree(self) -> int:
        """Maximum out-degree of the graph.

        Returns:
            int: Maximum out-degree
        """
        return max(len(self._children[node]) for node in self._nodes)

    @classmethod
    def from_nx(cls, nx_dag: nx.DiGraph, *args: Any, **kwargs: Any) -> DAG:
        """Convert to DAG from nx.DiGraph.

        Args:
            nx_dag (nx.DiGraph): DAG in question.
            args (Any): additional arguments
            kwargs (Any): additional arguments

        Raises:
            TypeError: If DAG is not nx.DiGraph

        Returns:
            DAG
        """
        if not isinstance(nx_dag, nx.DiGraph):
            raise TypeError("DAG must be of type nx.DiGraph")
        return DAG(nodes=list(nx_dag.nodes), edges=list(nx_dag.edges))

    def to_cpdag(self) -> PDAG:
        """Convert DAG to CPDAG.

        Returns:
            PDAG: CPDAG representing the MEC.
        """
        return dag2cpdag(dag=self.to_networkx())


class LayeredDAG(DAG):
    """Class to construct Layered DAGs.

    Layered DAGs `L` are DAGs where the Nodes `V` follow some natural layering.
    In other words, no edge can ever point into any of the earlier layers.
    """

    def __init__(
        self,
        nodes: list[str] | None = None,
        edges: list[tuple[str, str]] | None = None,
        layering: dict[str, list[str]] | None = None,
    ) -> None:
        """Layered DAG constructor.

        Args:
            nodes (list[str] | None, optional): Nodes of LDAG. Defaults to None.
            edges (list[tuple[str, str]] | None, optional): Edges of LDAG. Defaults to None.
            layering (dict[str, list[str]] | None, optional): Layering. Defaults to None.
        """
        self._layering = layering
        super().__init__(nodes=nodes, edges=edges)

    def _add_edge(self, i: str, j: str) -> None:
        if not self.layering:
            raise ValueError("Layering must be provided before adding edges.")

        self._nodes.add(i)
        self._nodes.add(j)
        self._edges.add((i, j))

        # Check if graph is acyclic
        if not self.is_acyclic():
            raise ValueError(
                "The edge set you provided \
                induces one or more cycles.\
                Check your input!"
            )

        # Check if edge is allowed due to layering
        if not self._is_allowed(edge=(i, j)):
            raise ValueError(
                "The edge set you provided \
                does not agree with the layering.\
                Check your input!"
            )

        self._children[i].add(j)
        self._parents[j].add(i)

    @property
    def layering(self) -> dict[str, list[str]] | None:
        """Current layering dict.

        Returns:
            dict[str, list[str]]: Layering
        """
        return self._layering

    @layering.setter
    def layering(self, la: np.random.Generator) -> None:
        if not isinstance(la, dict):
            raise AssertionError("Layering must be a dictionary!")
        self._layering = la

    def _is_allowed(self, edge: tuple[str, str]) -> bool:
        if not self.layering:
            raise ValueError("Layering must be provided before adding edges.")
        i, j = edge
        layers = list(self.layering.keys())
        for layer, nodes in self.layering.items():
            if i in nodes:
                i_layer = layers.index(layer)
            if j in nodes:
                j_layer = layers.index(layer)
        return i_layer <= j_layer

    def layer_induced_subgraph(self, nodes: list[str]) -> DAG:
        """Returns the induced subgraph on the nodes in `nodes`.

        Args:
            nodes (list[str]): List of nodes.

        Returns:
            DAG: Induced subgraph.
        """
        if self.layering is not None and not any(
            [nodes == layer for layer in self.layering.values()]
        ):
            raise ValueError("Nodes you provide must correspond to a layer.")
        edges = [(i, j) for i, j in self.edges if i in nodes and j in nodes]
        return DAG(nodes=nodes, edges=edges)

    @classmethod
    def from_pandas_adjacency(
        cls, pd_amat: pd.DataFrame, layering: dict[str, list[str]]
    ) -> LayeredDAG:
        """Build LayeredDAG from a Pandas adjacency matrix.

        Args:
            pd_amat (pd.DataFrame): input adjacency matrix.
            layering (dict[str, list[str]]): layering of nodes.

        Returns:
            LayeredDAG
        """
        assert pd_amat.shape[0] == pd_amat.shape[1]
        nodes = pd_amat.columns

        all_connections = []
        start, end = np.where(pd_amat != 0)
        for idx, _ in enumerate(start):
            all_connections.append((pd_amat.columns[start[idx]], pd_amat.columns[end[idx]]))

        temp = [set(i) for i in all_connections]
        temp2 = [arc for arc in all_connections if temp.count(set(arc)) > 1]

        dir_edges = [edge for edge in all_connections if edge not in temp2]

        return LayeredDAG(nodes=nodes, edges=dir_edges, layering=layering)

    def copy(self) -> LayeredDAG:
        """Return a copy of the graph."""
        return LayeredDAG(nodes=list(self._nodes), edges=list(self._edges), layering=self.layering)

    @classmethod
    def from_nx(cls, nx_dag: nx.DiGraph, layering: dict[str, list[str]]) -> LayeredDAG:
        """Convert to DAG from nx.DiGraph.

        Args:
            nx_dag (nx.DiGraph): DAG in question.
            layering (dict[str, list[str]]): layering of nodes.

        Raises:
            TypeError: If DAG is not nx.DiGraph

        Returns:
            LayeredDAG
        """
        if not isinstance(nx_dag, nx.DiGraph):
            raise TypeError("DAG must be of type nx.DiGraph")
        return LayeredDAG(nodes=list(nx_dag.nodes), edges=list(nx_dag.edges), layering=layering)
