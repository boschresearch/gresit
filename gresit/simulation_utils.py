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

import json
import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gadjid import (
    ancestor_aid,
    parent_aid,
    shd,
)

from gresit.graphs import DAG, GRAPH, PDAG
from gresit.params import ExperimentParams

logger = logging.getLogger(__name__)

_metric_names = {
    "precision": r"Precision$^{\rightarrow}$",
    "recall": r"Recall$^{\rightarrow}$",
    "f1": r"F1$^{\rightarrow}$",
    "shd": r"SHD$^{\leftarrow}$",
    "sid": r"SID$^{\leftarrow}$",
    "ancestor_aid": r"AAID$^{\leftarrow}$",
    "ancester_ordering_aid": r"OAID$^{\leftarrow}$",
}


def _make_super_dag_adjacency(order: list[str]) -> pd.DataFrame:
    """Return adjacency matrix of super-DAG given a causal ordering.

    Args:
        order (list[str]): Causal ordering

    Returns:
        pd.DataFrame: Adjacency matrix with row and col names.
    """
    return pd.DataFrame(
        np.triu(np.ones((len(order), len(order))), 1),
        columns=np.array(order),
        index=np.array(order),
    )


def _metric_mapping_fn(
    metric_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    edge_direction: str | None = "from row to column",
    which_metric: str | None = None,
) -> Callable[[pd.DataFrame, pd.DataFrame, list[str] | None], tuple[np.ndarray, np.ndarray]]:
    if edge_direction is not None:
        kwargs = {"edge_direction": edge_direction}
    elif which_metric is not None:
        kwargs = {"which_metric": which_metric}
    else:
        kwargs = {}

    def metric(
        gt: pd.DataFrame, est: pd.DataFrame, causal_order: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        return metric_fn(
            gt.to_numpy(dtype=np.int8),
            est.reindex(index=gt.index, columns=gt.columns).to_numpy(dtype=np.int8),
            **kwargs,
        )

    return metric


def _ancester_ordering_aid(
    gt: pd.DataFrame, est: pd.DataFrame, causal_order: list[str] | None
) -> tuple[np.ndarray, np.ndarray]:
    _ = est
    if causal_order is None:
        return np.empty(0, dtype=np.float64), np.empty(0, np.int8)

    est_super_dag = _make_super_dag_adjacency(order=causal_order)
    metric_fn = _metric_mapping_fn(ancestor_aid)
    return metric_fn(gt, est_super_dag, causal_order)


def _revert_cpdag_amat(cpdag: np.ndarray) -> np.ndarray:
    """Revert adjacency matrix with `2` at undirected entries to `0-1` adjacency matrix.

    Args:
        cpdag (np.ndarray): CPDAG

    Returns:
        np.ndarray: Adjacency matrix
    """
    constant_two = 2
    cpdag_amat = cpdag.copy()
    upper_triangle_indices = np.triu_indices_from(cpdag, k=1)
    mask = (cpdag[upper_triangle_indices] == constant_two) & (
        cpdag[upper_triangle_indices[::-1]] == constant_two
    )

    # Set these entries back to 1 in both (i, j) and (j, i) locations
    cpdag_amat[upper_triangle_indices[0][mask], upper_triangle_indices[1][mask]] = 1
    cpdag_amat[upper_triangle_indices[1][mask], upper_triangle_indices[0][mask]] = 1

    return cpdag_amat


def _standard_metrics(
    gt: np.ndarray, est: np.ndarray, which_metric: str = "F1"
) -> tuple[str, float]:
    """Standard metrics for binary adjacency matrices.

    Args:
        est (np.ndarray): Estimated adjacency matrix
        gt (np.ndarray): Ground truth adjacency matrix
        which_metric (str): Which metric to return

    Returns:
        tuple[str, float]: Dict of metrics
    """
    constant_two = 2
    if np.isin(est, constant_two).any():
        est = _revert_cpdag_amat(est)
    true_positives = np.where((est + gt) == constant_two, 1, 0).sum(axis=1).sum()
    tp_fp = est.sum(axis=1).sum()
    tp_fn = gt.sum(axis=1).sum()

    precision = true_positives / max(tp_fp, 1)
    recall = true_positives / max(tp_fn, 1)
    F1 = 2 * (recall * precision) / max((recall + precision), 1)
    for metric_name, metric_value in [("Precision", precision), ("Recall", recall), ("F1", F1)]:
        if not (0 <= metric_value <= 1):
            raise ValueError(f"{metric_name} {metric_value} is outside the [0, 1] interval")
    if which_metric == "precision":
        return "precision", precision
    elif which_metric == "recall":
        return "recall", recall
    else:
        return "F1", F1


_metric_mapping = {
    "shd": _metric_mapping_fn(shd, edge_direction=None),
    "ancestor_aid": _metric_mapping_fn(ancestor_aid, edge_direction="from row to column"),
    "sid": _metric_mapping_fn(parent_aid, edge_direction="from row to column"),
    "ancester_ordering_aid": _ancester_ordering_aid,
    "precision": _metric_mapping_fn(
        _standard_metrics, which_metric="precision", edge_direction=None
    ),
    "recall": _metric_mapping_fn(_standard_metrics, which_metric="recall", edge_direction=None),
    "f1": _metric_mapping_fn(_standard_metrics, which_metric="F1", edge_direction=None),
}


class BenchMarker:
    """Class to instantiate and run simulations."""

    @staticmethod
    def _emptygraph(size: int) -> np.ndarray:
        return np.zeros((size, size))

    def _cpdag_dag_processing(
        self,
        gt: pd.DataFrame,
        learned_graph: GRAPH,
        cpdag_strategy: str = "random_dag",
    ) -> tuple[pd.DataFrame, list[str] | None]:
        """Converts CPDAGs to DAGs according to CPDAG strategy.

        If PDAG is not a proper CPDAG, and orientations lead to cycles,
        we don't do anything.

        Args:
            gt (pd.DataFrame): ground truth adjacency matrix
            learned_graph (GRAPH): learned graph
            cpdag_strategy (str, optional): Conversion strategy. Defaults to "random_dag".

        Raises:
            NotImplementedError: _description_
            ValueError: _description_

        Returns:
            tuple[pd.DataFrame, list[str] | None]: Tuple containing amat and causal order (or None).
        """
        learned_dag: PDAG | DAG
        if isinstance(learned_graph, PDAG):
            if cpdag_strategy == "random_dag":
                try:
                    learned_dag = learned_graph.to_random_dag()
                except ValueError:
                    learned_dag = learned_graph
            elif cpdag_strategy == "best_dag":
                try:
                    all_dags = learned_graph.to_allDAGs()
                    original_order = gt.columns
                    shd_array = np.full(len(all_dags), np.inf)
                    for i, each_dag in enumerate(all_dags):
                        shd_array[i] = shd(
                            gt.to_numpy(dtype=np.int8),
                            each_dag.adjacency_matrix.reindex(
                                index=original_order, columns=original_order
                            ).to_numpy(dtype=np.int8),
                        )[1]
                    learned_dag = all_dags[shd_array.argmin()]
                except ValueError:
                    learned_dag = learned_graph
            else:
                raise NotImplementedError()

            learned_adjacency_matrix = learned_dag.adjacency_matrix
            learned_causal_order = learned_dag.causal_order

        elif isinstance(learned_graph, DAG):
            learned_adjacency_matrix = learned_graph.adjacency_matrix
            learned_causal_order = learned_graph.causal_order

        else:
            raise ValueError("Something went wrong, result is neither `DAG` nor `CPDAG`.")

        return learned_adjacency_matrix, learned_causal_order

    def run_benchmark(
        self,
        params: ExperimentParams,
        num_runs: int = 30,
        metrics: list[str] = ["shd", "sid", "ancestor_aid", "ancester_ordering_aid"],
        cpdag_strategy: str = "random_dag",
    ) -> defaultdict[str, defaultdict[str, list[int | float | None]]]:
        """Benchmark run.

        Args:
            num_runs (int, optional): _description_. Defaults to 30.
            params (ExperimenParams): Params to benchmark
            metrics (list[str], optional): _description_. Defaults to
                ["shd", "sid", "ancestor_aid", "ancester_ordering_aid"].
            cpdag_strategy (str): How to convert CPDAG to DAG to enable fair comparison.
                Options are `random_dag` (default) and `best_dag`. In the former, a random
                DAG is chosen from the MEC, in the latter the SHD is calculated for all DAGs
                in the MEC and the best one is chosen.
        """
        results: defaultdict[str, defaultdict[str, list[int | float | None]]] = defaultdict(
            partial(defaultdict, list)
        )
        for run in range(num_runs):
            # Generate data to also generate a new DAG
            try:
                ldat = params.make_data()
                data, _ = ldat.generate_data(num_samples=params.number_of_samples)

                layering = getattr(ldat.dag, "layering", None)
                gt = ldat.dag.adjacency_matrix

                for algo, name in params:
                    learned_graph = algo.learn_graph(
                        data_dict=data,
                        layering=layering,
                    )

                    learned_adjacency_matrix, learned_causal_order = self._cpdag_dag_processing(
                        gt=gt, learned_graph=learned_graph, cpdag_strategy=cpdag_strategy
                    )

                    for metric in metrics:
                        _, resulting_metric = _metric_mapping[metric](
                            gt,
                            learned_adjacency_matrix,
                            learned_causal_order,
                        )

                        if isinstance(resulting_metric, np.ndarray):
                            resulting_metric = None

                        results[name][metric].append(resulting_metric)
            except Exception as e:
                print(f"Error in run {run}: {e}. Skipping this run.")
                continue
        return results

    def write_results(
        self, results: defaultdict[str, defaultdict[str, list[int | float | None]]], path: str
    ) -> None:
        """Write results to JSON files with time signature.

        Args:
            results (defaultdict[str, defaultdict[str, list[int | float | None]]]): Result
                dict from a benchmark run.
            path (str): Destination path where to save files.
        """
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M")
        with open(path + f"/results_{current_time}.json", "w", encoding="utf-8") as outfile:
            json.dump(results, outfile)


def _draw_full_boxplot(
    result_dict: defaultdict[str, defaultdict[str, list[int | float | None]]],
    position: plt.Axes | None = None,
    title: str = "",
    x_label: str = "",
    metric: str = "",
    dashed_line_for_emptygraph: int | None = None,
) -> plt.Axes:
    """Draw boxplot.

    Args:
        result_dict (dict): _description_
        position (_type_): _description_
        title (str): _description_
        x_label (str): _description_
        metric (str): _description_
        dashed_line_for_emptygraph (int | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    interim_dict = {
        algo_name: metric_results[metric] for algo_name, metric_results in result_dict.items()
    }
    interim_melt = pd.melt(
        pd.DataFrame(interim_dict),
        value_name=metric,
    )
    interim_melt[metric] = interim_melt[metric].astype("float")
    bp = sns.boxplot(
        y="variable",
        x=metric,
        data=interim_melt,
        orient="h",
        palette="colorblind",
        hue="variable",
        ax=position,
    )

    bp.set(ylabel=None)
    bp.set(xlabel=x_label)

    accuracy_metrics = ["precision", "recall", "f1"]
    if position is not None:
        if metric in accuracy_metrics:
            position.set(xlim=(0, 1.1))
        else:
            position.set(xlim=(0, None))

    if dashed_line_for_emptygraph is not None:
        bp.axvline(x=dashed_line_for_emptygraph, linewidth=2, color="grey", ls=":")
    bp.set(title=title)
    sns.despine(right=True)
    return bp


def draw_result_boxplots(
    result_dict: defaultdict[str, defaultdict[str, list[float | None]]],
    title: str = "",
    file_path: str | None = None,
    file_name: str | None = "example_name",
) -> None:
    """Draw full set of result boxplots.

    Args:
        result_dict (dict): _description_
        title (str, optional): title of the plot.
        file_path (str | None, optional): _description_. Defaults to None.
        file_name (str | None, optional): _description_. Defaults to None.
    """
    num_plots = len(result_dict[list(result_dict.keys())[0]].keys())
    MAX_PLOT_PER_ROW = 3
    if num_plots > MAX_PLOT_PER_ROW:
        num_rows = 2
        num_cols = int(np.ceil(num_plots / num_rows))

    fig, ax = plt.subplots(
        figsize=(18, 4.2),
        nrows=num_rows,
        ncols=num_cols,
        sharex=False,
        sharey=True,
    )

    valid_axes = [axi for axi in ax.flat[:] if axi in fig.axes]
    if (num_plots % 2) != 0:
        fig.delaxes(ax[0, -1])
        valid_axes = [axi for axi in ax.flat[:] if axi in fig.axes]

    for axi, metric in zip(
        valid_axes,
        result_dict[list(result_dict.keys())[0]].keys(),
    ):
        _draw_full_boxplot(
            result_dict=result_dict,
            position=axi,
            title="",
            x_label=_metric_names[metric],
            metric=metric,
        )

    plt.figtext(
        0.16,
        1.01,
        r"$\leftarrow$ lower is better",
        horizontalalignment="right",
        fontsize="x-small",
    )
    plt.figtext(
        0.41,
        1.01,
        r"$\rightarrow$ higher is better",
        horizontalalignment="right",
        fontsize="x-small",
    )

    fig.tight_layout()
    fig.suptitle(title)
    fig.subplots_adjust(top=0.88)
    if file_path is not None:
        plt.savefig(f"{file_path}/{file_name}.pdf", bbox_inches="tight")
    plt.show()
