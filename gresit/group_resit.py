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

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from gresit.graphs import DAG
from gresit.independence_tests import Itest
from gresit.learn_algorithms import LearnAlgo
from gresit.model_selection import MURGS
from gresit.regression_techniques import MultiRegressor
from gresit.torch_models import Multioutcome_MLP


class GroupResit(LearnAlgo):
    """A class representing the groupResit algorithm.

    This algorithm is used to learn a DAG based on vector/group valued ANMs.

    """

    def __init__(
        self,
        regressor: MultiRegressor,
        test: type[Itest],
        alpha: float = 0.01,
        pruning_method: str = "murgs",
        test_size: float = 0.2,
        local_regression_method: str = "kernel",
    ) -> None:
        """Initialize the GroupResit object.

        Args:
            regressor (MultivariateRegressor): A regressor object.
            test (IndependenceTest): An independence test object.
            alpha (float): Alpha
            pruning_method (str): The pruning method
            test_size (float): Relative size of test-dataset, 0 means no test data
            local_regression_method (str): Type of local linear smoother to use. Options are
                `loess`, `kernel`, soon to be implemented `spline`. Defaults to `kernel`.
        """
        self.regressor = regressor
        self.independence_test = test()
        self.local_regression_method = local_regression_method
        self._pa: dict[str, list[str]] = {}
        self._pa_history: dict[str, dict[float, list[str]]] = {}
        self.alpha_level: float
        self.alpha = alpha
        self.pruning_method = pruning_method
        self.test_size = test_size
        self.DAG: DAG
        self.layering: dict[str, list[str]] = {}
        super().__init__()

    def __repr__(self) -> str:
        """Repr method.

        Returns:
            str: Description of the object.
        """
        return f"GroupResit(regressor={self.regressor}, independence_test={self.independence_test})"

    def __str__(self) -> str:
        """Str method.

        Returns:
            str: Human-readable description of the object.
        """
        method_description = {
            "Regression method: ": self.regressor.__class__.__name__,
            "Independece test:": self.independence_test.__class__.__name__,
            "Inferred causal order: ": "Yes" if self._causal_order else "Not yet",
        }
        s = ""
        for info, info_text in method_description.items():
            s += f"{info:<14}{info_text:>5}\n"

        return s

    def _get_causal_order(
        self,
        data_dict: dict[str, np.ndarray],
        layering: dict[str, list[str]] | None = None,
    ) -> None:
        """Get causal order of the groups respecting the current layering.

        Args:
            data_dict (dict[str, np.ndarray]): A dictionary of np.ndarrays. Key corresponds to
                group name and values to the corresponding data.
            layering (dict[str, list[str]]): A dictionary of layering information. Keys correspond
                to the layer and values to the variable names within each layer.
        """
        if layering is None:
            layering = {"L": list(data_dict.keys())}

        self.layering = layering
        pa: dict[str, list[str]] = {}
        pi: list[str] = []
        data_to_delete_from = data_dict.copy()

        indices = np.arange(data_dict[list(data_dict.keys())[0]].shape[0])
        if self.test_size > 0:
            idx_train, idx_test = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=2024,
            )
        else:
            idx_test = indices
            idx_train = indices

        self._idx_test = idx_test

        for _, vars in reversed(layering.items()):
            if pa:  # add previous layer nodes to current layer parents
                for _, pre_layer_parents in pa.items():
                    pre_layer_parents.extend(vars)

            within_layer_order = vars.copy()
            for _ in vars:
                if len(within_layer_order) == 1:  # in each layer if there's only one node left,
                    # this must be the first in the causal ordering
                    pa[within_layer_order[0]] = []
                    pi.insert(0, within_layer_order[0])
                    del data_to_delete_from[within_layer_order[0]]
                    continue

                test_stats: list[np.float64] = []
                for var in within_layer_order:
                    Y = data_to_delete_from[var].copy()  # remove columns from data!
                    X = np.concatenate(
                        [_d for _group, _d in data_to_delete_from.items() if _group != var],
                        axis=1,
                    )

                    X_train = self._standardize(X[idx_train])
                    Y_train = self._standardize(Y[idx_train])

                    X_test = (X[idx_test] - X[idx_train].mean(axis=0)) / (X[idx_train].std(axis=0))
                    Y_test = (Y[idx_test] - Y[idx_train].mean(axis=0)) / (Y[idx_train].std(axis=0))

                    self.regressor.fit(X=X_train, Y=Y_train)
                    Y_pred = self.regressor.predict(X_test)

                    residuals = Y_test - np.squeeze(np.asarray(Y_pred))
                    residuals = self._standardize(residuals)
                    X_test = self._standardize(X_test)
                    test_stat, _ = self.independence_test.test(x_data=residuals, y_data=X_test)
                    test_stats.append(test_stat)

                k = within_layer_order[np.argmin(test_stats)]  # get most independent group
                within_layer_order.remove(k)  # remove from remaining order
                pa[k] = within_layer_order.copy()  # add all within layer potential parents
                pi.insert(0, k)  # prepend to causal ordering

                del data_to_delete_from[k]

        self._causal_order = pi
        self._pa = pa

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - x.mean(axis=0)) / x.std(axis=0)

    def _get_causal_order_via_dependence_loss(
        self,
        data_dict: dict[str, np.ndarray],
        layering: dict[str, list[str]] | None = None,
    ) -> None:
        """Get causal order of the groups respecting the current layering.

        Args:
            data_dict (dict[str, np.ndarray]): A dictionary of np.ndarrays. Key corresponds to
                group name and values to the corresponding data.
            layering (dict[str, list[str]]): A dictionary of layering information. Keys correspond
                to the layer and values to the variable names within each layer.
        """
        if layering is None:
            layering = {"L": list(data_dict.keys())}

        self.layering = layering
        pa: dict[str, list[str]] = {}
        pi: list[str] = []
        data_to_delete_from = data_dict.copy()

        indices = np.arange(data_dict[list(data_dict.keys())[0]].shape[0])
        if self.test_size > 0:
            idx_train, idx_test = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=2024,
            )
        else:
            idx_test = indices
            idx_train = indices

        self._idx_test = idx_test

        for _, vars in reversed(layering.items()):
            if pa:  # add previous layer nodes to current layer parents
                for _, pre_layer_parents in pa.items():
                    pre_layer_parents.extend(vars)

            within_layer_order = vars.copy()
            for _ in vars:
                if len(within_layer_order) == 1:  # in each layer if there's only one node left,
                    # this must be the first in the causal ordering
                    pa[within_layer_order[0]] = []
                    pi.insert(0, within_layer_order[0])
                    del data_to_delete_from[within_layer_order[0]]
                    continue

                final_loss_values: list[np.float64] = []
                for var in within_layer_order:
                    Y = data_to_delete_from[var].copy()  # remove columns from data!
                    X = np.concatenate(
                        [_d for _group, _d in data_to_delete_from.items() if _group != var],
                        axis=1,
                    )
                    if isinstance(self.regressor, Multioutcome_MLP):
                        self.regressor.fit(X=X, Y=Y, idx_train=idx_train, idx_test=idx_test)
                        final_loss_values.append(self.regressor.final_loss)
                    else:
                        raise ValueError("Independence loss only implemented in MLP.")

                k = within_layer_order[np.argmin(final_loss_values)]  # get most independent group
                within_layer_order.remove(k)  # remove from remaining order
                pa[k] = within_layer_order.copy()  # add all within layer potential parents
                pi.insert(0, k)  # prepend to causal ordering
                del data_to_delete_from[k]

        self._causal_order = pi
        self._pa = pa

    def _independence_prune(
        self,
        data_dict: dict[str, np.ndarray],
        alpha: float = 0.01,
    ) -> None:
        """Prune according to Peters et al. (2014).

        Args:
            data_dict (dict[str, np.ndarray]): Data in form of dict with groups as keys
                and data as values.
            alpha (float, optional): Significance level. Defaults to 0.01.
        """
        # pruning
        dat = data_dict.copy()
        pa = self._pa
        pi = self._causal_order
        idx_test = self._idx_test

        for k in pi:  # for each but the first node in the causal order
            # Take every parent and check for independence
            parents = pa[k].copy()
            if not parents:
                continue

            for parent in parents:
                Y = dat[k].copy()
                # see whether parent may be removed
                remainders = [var for var in parents if var != parent]
                if not remainders:
                    continue

                X = np.concatenate(
                    [dat[x] for x in remainders],
                    axis=1,
                )

                self.regressor.fit(
                    X=X[idx_test],
                    Y=Y[idx_test],
                )

                y_pred = self.regressor.predict(X[idx_test])
                residual = Y[idx_test] - np.squeeze(np.asarray(y_pred))
                _, decision = self.independence_test.test(x_data=residual, y_data=X[idx_test])
                if isinstance(decision, float):
                    if decision > alpha:  # if indepedence is not rejected, remove the edge
                        pa[k].remove(parent)
                elif isinstance(decision, str):
                    if decision == "Reject Null of vector independence: False":
                        pa[k].remove(parent)
                else:
                    raise ValueError("Test decision is neither float nor string")

        self.alpha_level = alpha
        self._pa = pa

    def _sparse_regression_pruning(
        self, data_dict: dict[str, np.ndarray], nlambda: int = 30
    ) -> None:
        # pruning
        dat = data_dict.copy()
        if self.test_size > 0:
            dat = {nodes: data for nodes, data in data_dict.items()}
            # data[self._idx_test]
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
                precalculate_smooths=True,
                local_regression_method=self.local_regression_method,
            )
            # extract zero groups
            zero_groups = murgs.zero_groups
            pa[k] = [parent for i, parent in enumerate(potential_parents) if not zero_groups[i]]
            self._pa_history[k] = {
                penalty: [
                    parent for i, parent in enumerate(potential_parents) if not zero_groups[i]
                ]
                for penalty, zero_groups in murgs.zero_group_history.items()
            }

        self._pa = pa

    def _dict_preprocessing(self, data_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Dict preprocessing dealing with univariate groups.

        Args:
            data_dict (dict[str, np.ndarray]): data dict.

        Returns:
            dict[str, np.ndarray]: data dict with axis added when univariate.
        """
        for key, value in data_dict.items():
            if value.ndim == 1:
                data_dict[key] = value[:, np.newaxis]
        return data_dict

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
        clean_data_dict = self._dict_preprocessing(data_dict)
        self._get_causal_order(data_dict=clean_data_dict, layering=layering)
        if self.pruning_method == "murgs":
            self._sparse_regression_pruning(data_dict=clean_data_dict)
        elif self.pruning_method == "independence":
            self._independence_prune(data_dict=clean_data_dict, alpha=self.alpha)
        else:
            raise NotImplementedError()

        edge_list = [(parent, child) for child in self._pa for parent in self._pa[child]]

        learned_DAG = DAG(nodes=self._causal_order)
        learned_DAG.add_edges_from(edge_list)
        self._adjacency_matrix = learned_DAG.adjacency_matrix

        self.DAG = learned_DAG
        return learned_DAG

    def _insert_known_causal_order(self, pi: list[str]) -> None:
        """Insert a known causal order.

        Args:
            pi (list[str]): A list of group names in the causal order.
        """
        self._causal_order = pi
        pa = {}
        for node in pi:
            index = pi.index(node)
            pa[node] = pi[0:index]
        self._pa = pa

    def model_selection_with_known_causal_order(
        self,
        pi: list[str],
        data_dict: dict[str, np.ndarray],
        alpha: float = 0.01,
        pruning_method: str = "murgs",
    ) -> None:
        """Given a known causal order perform model selection.

        Args:
            pi (list[str]): Causal ordering. Entries in `pi` need
                to coincide with the keys in `data_dict`.
            data_dict (dict[str, np.ndarray]): A dictionary of np.ndarrays. Key corresponds to
                group name and values to the corresponding data.
            alpha (float, optional): The significance level for the independence test.
                Defaults to 0.1.
            pruning_method (str, optional): The pruning method to use. Defaults to "murgs".
                other options include `"independence"`

        """
        self._insert_known_causal_order(pi=pi)
        if not hasattr(self, "_idx_test"):
            self._idx_test = np.arange(data_dict[list(data_dict.keys())[0]].shape[0])
        if pruning_method == "murgs":
            self._sparse_regression_pruning(data_dict=data_dict)
        elif pruning_method == "independence":
            self._independence_prune(data_dict=data_dict, alpha=alpha)
        else:
            raise NotImplementedError()

        edge_list = [(parent, child) for child in self._pa for parent in self._pa[child]]

        learned_DAG = DAG(nodes=self._causal_order)
        learned_DAG.add_edges_from(edge_list)
        self._adjacency_matrix = learned_DAG.adjacency_matrix
        self.DAG = learned_DAG

    @property
    def causal_order(self) -> list[str] | None:
        """Causal order."""
        return self._causal_order

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        """Adjacency matrix."""
        return self._adjacency_matrix

    def _add_layered_layout_to_graph(self, nx_graph: nx.DiGraph) -> nx.DiGraph:
        """Add coordinate pos to nx.DiGraph.

        Args:
            nx_graph (nx.DiGraph): DAG in question.

        Returns:
            nx_graph (nx.DiGraph): DAG with attributes set.
        """
        if len(self.layering) > 1:
            for layer, nodes in enumerate(self.layering.values()):
                for node in nodes:
                    nx_graph.nodes[node]["layer"] = layer
        else:
            for layer, nodes in enumerate(nx.topological_generations(nx_graph)):
                for node in nodes:
                    nx_graph.nodes[node]["layer"] = layer

        # Plot multipartite_layout using the "layer" node attribute
        pos = nx.multipartite_layout(nx_graph, subset_key="layer")
        nx.set_node_attributes(G=nx_graph, name="pos", values=pos)
        return nx_graph

    def show(self, title: str = "Group RESIT DAG") -> None:
        """Plot the learned DAG.

        The plot is interactive,
        hovering over the nodes reveals the node labels.
        Colors get brighter the higher the node degree.

        Args:
            title (str, optional): Plot title. Defaults to "Group RESIT DAG".

        Raises:
            AssertionError: Throws error if DAG not yet learned.
        """
        if not hasattr(self, "DAG"):
            raise AssertionError("No graph to plot. Learn the graph first.")

        nx_dag = self.DAG.to_networkx()

        nx_dag = self._add_layered_layout_to_graph(nx_graph=nx_dag)

        edge_x = []
        edge_y = []
        for edge in nx_dag.edges():
            x0, y0 = nx_dag.nodes[edge[0]]["pos"]
            x1, y1 = nx_dag.nodes[edge[1]]["pos"]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines+markers",
            marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"),
        )

        node_x = []
        node_y = []
        for node in nx_dag.nodes():
            x, y = nx_dag.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale="YlGnBu",
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(thickness=15, title="node degree", xanchor="left", titleside="right"),
                line_width=2,
            ),
        )

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(nx_dag.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(list(nx_dag.nodes)[node])

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                # titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
        fig.show()

    def _layer_pos(
        self, G: nx.DiGraph, layers: dict[str, list[str]], layer_gap: float = 8.0
    ) -> dict[str | int, np.ndarray]:
        pos = {}
        for i, nodes in enumerate(layers.values()):
            pos.update(
                nx.spring_layout(
                    G.subgraph(nodes), center=np.array([layer_gap * i, 0]), seed=42, k=50
                )
            )
        return pos

    def _create_edge_trace(
        self, G: nx.DiGraph, pos: dict[str | int, np.ndarray], highlight: bool = False
    ) -> go.Scatter:
        edge_x, edge_y = [], []

        if G.edges():
            for edge in G.edges():
                x0, y0 = pos[edge[0]]  # Source node
                x1, y1 = pos[edge[1]]  # Target node
                # Store edge coordinates
                edge_x.extend([x0, x1, None])  # Keep source unchanged
                edge_y.extend([y0, y1, None])

        # Define edge trace with customized width and color
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(
                width=2 if highlight else 1,  # Thicker for first graph
                color="dimgray" if highlight else "#888",  # Darker first graph edges
            ),
            hoverinfo="none",
            mode="lines+markers",
            marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"),
        )

        return edge_trace

    def _make_graph_list(self) -> list[nx.DiGraph]:
        unique_parent_history_sparser = {}
        unique_parent_history = {}

        for node, possible_parents in self._pa_history.items():
            unique_parent_history[node] = sorted(
                [
                    list(x)
                    for x in {
                        frozenset(sublist)
                        for sublist in [parents for parents in possible_parents.values()]
                    }
                    if len(x) >= len(self._pa[node])
                ],
                key=len,
            )
            unique_parent_history_sparser[node] = sorted(
                [
                    list(x)
                    for x in {
                        frozenset(sublist)
                        for sublist in [parents for parents in possible_parents.values()]
                    }
                    if len(x) < len(self._pa[node])
                ],
                key=len,
            )

        max_length = max([len(penalties) for penalties in unique_parent_history.values()])
        # Pad each list of lists
        padded_dict = unique_parent_history.copy()
        for node, _ in unique_parent_history.items():
            last_element = (
                unique_parent_history[node][-1] if unique_parent_history[node] else []
            )  # Last element of the list

            while len(unique_parent_history[node]) < max_length:
                padded_dict[node].append(last_element)

        max_length_sparser = max(
            [len(penalties) for penalties in unique_parent_history_sparser.values()]
        )
        # Pad each list of lists
        padded_dict_sparser = unique_parent_history_sparser.copy()
        for node, _ in unique_parent_history_sparser.items():
            last_element = (
                unique_parent_history_sparser[node][-1]
                if unique_parent_history_sparser[node]
                else []
            )  # Last element of the list

            while len(unique_parent_history_sparser[node]) < max_length_sparser:
                padded_dict_sparser[node].append(last_element)

        combined_dict = {}
        for key in padded_dict.keys():
            combined_dict[key] = padded_dict_sparser[key] + padded_dict[key]

        graph_list = []
        for i in range(max_length_sparser + max_length):
            pa = {node: parents[i] for node, parents in combined_dict.items()}
            edge_list = [(parent, child) for child in pa for parent in pa[child]]
            learned_DAG = nx.DiGraph()
            learned_DAG.add_nodes_from(list(pa.keys()))
            learned_DAG.add_edges_from(edge_list)
            graph_list.append(learned_DAG)

        return graph_list

    def show_interactive(self, layer_gap: float = 8.0) -> go.Figure:
        """Show interactive plot with slider to select sparsity level.

        Args:
            layer_gap (float, optional): gap between layers when displaying. Defaults to 8.0.
        """
        # Fixed layout for consistency
        graph_list = self._make_graph_list()
        pos = self._layer_pos(G=graph_list[-1], layers=self.layering, layer_gap=layer_gap)
        nodes = list(graph_list[-1].nodes)

        # Extract node coordinates
        node_x, node_y = zip(*[pos[n] for n in nodes])

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            text=nodes,
            hoverinfo="text",
            marker=dict(size=15, color="lightblue", line=dict(color="black", width=1)),
        )

        # Identify the "optimal" graph
        optimal_graph_id = next(
            (
                i
                for i, graph in enumerate(graph_list)
                if nx.utils.graphs_equal(self.DAG.to_networkx(), graph)
            ),
            None,
        )
        if optimal_graph_id is not None:
            # Create the optimal graph's edge trace
            optimal_edge_trace = self._create_edge_trace(
                G=graph_list[optimal_graph_id], pos=pos, highlight=True
            )
        else:
            raise ValueError("Optimal graph not found in graph list.")

        # Create frames for each graph
        frames = []
        num_frames = len(graph_list)

        for i, G in enumerate(graph_list):
            # highlight = i == optimal_graph_id  # Highlight only the optimal graph
            edge_trace = self._create_edge_trace(G, pos, highlight=False)

            # If we have reached or surpassed the optimal graph, add its highlighted edges
            if i >= optimal_graph_id:
                frames.append(
                    go.Frame(
                        data=[node_trace, edge_trace, optimal_edge_trace],  # Include optimal edges
                        name=f"Graph {i + 1}",
                    )
                )
            else:
                frames.append(
                    go.Frame(
                        data=[node_trace, edge_trace],  # Normal edges only
                        name=f"Graph {i + 1}",
                    )
                )

        # Custom slider labels (first, optimal, and last only)
        slider_labels = [
            "high"
            if i == 0
            else "optimal"
            if i == optimal_graph_id
            else "low"
            if i == num_frames - 1
            else ""  # Empty label for intermediate frames
            for i in range(num_frames)
        ]

        # **Ensure first graph's edges are displayed initially**
        first_edge_trace = self._create_edge_trace(
            graph_list[0], pos, highlight=(0 == optimal_graph_id)
        )

        fig = go.Figure(
            data=[
                node_trace,
                first_edge_trace,  # Now displaying edges initially
            ],
            layout=go.Layout(
                title="",
                showlegend=False,
                hovermode="closest",
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                sliders=[
                    {
                        "steps": [
                            {
                                "args": [
                                    [frame.name],
                                    {"frame": {"duration": 0}, "mode": "immediate", "redraw": True},
                                ],
                                "label": slider_labels[i],  # Apply custom labels
                                "method": "animate",
                            }
                            for i, frame in enumerate(frames)
                        ],
                        "active": 0,
                        "currentvalue": {"prefix": "Penalty level: ", "font": {"size": 16}},
                        "pad": {"b": 10, "t": 50},
                    }
                ],
            ),
            frames=frames,  # Ensure frames update correctly
        )

        # Show figure
        return fig
