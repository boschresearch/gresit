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
from sklearn.linear_model import LinearRegression

from gresit.graphs import DAG
from gresit.independence_tests import HSIC
from gresit.learn_algorithms import LearnAlgo
from gresit.model_selection import MURGS


class GroupLiNGAM(LearnAlgo):
    """A class representing the GroupDirectLiNGAM algorithm.

    This algorithm is used to learn a DAG based on vector/group valued ANMs.

    """

    def __init__(
        self,
        local_regression_method: str = "kernel",
    ) -> None:
        """Initialize the GroupDirectLiNGAM object.

        Args:
            regressor (MultivariateRegressor): A regressor object.
            test (IndependenceTest): An independence test object.
            alpha (float): Alpha
            pruning_method (str): The pruning method
            test_size (float): Relative size of test-dataset, 0 means no test data
            local_regression_method (str): Type of local linear smoother to use. Options are
                `loess`, `kernel`, soon to be implemented `spline`. Defaults to `kernel`.
        """
        self.local_regression_method = local_regression_method
        self._pa: dict[str, list[str]] = {}
        self._pa_history: dict[str, dict[float, list[str]]] = {}
        self.DAG: DAG
        self.layering: dict[str, list[str]] = {}
        super().__init__()

    def __repr__(self) -> str:
        """Repr method.

        Returns:
            str: Description of the object.
        """
        return "GDLiNGAM()"

    def __str__(self) -> str:
        """Str method.

        Returns:
            str: Human-readable description of the object.
        """
        method_description = {
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
        pi: list[str] = []

        for _, vars in layering.items():
            within_layer_order = vars.copy()
            if len(within_layer_order) == 1:  # in each layer if there's only one node,
                # this must be the first in the causal ordering
                pi.append(within_layer_order[0])
                continue

            groups = {name: data_dict[name] for name in within_layer_order}
            ordering = []
            while groups:
                scores, residuals = self._compute_scores_and_residuals(groups)

                # pick the most exogenous group
                exog = max(scores, key=scores.__getitem__)
                ordering.append(exog)

                # prune: build the next `groups` as those residuals not involving `exog`
                groups = {
                    exog_resid: res
                    for (exog_resid, pred_name), res in residuals.items()
                    if exog not in (exog_resid, pred_name)
                }

            # if there’s one final remaining original variable, append it
            remaining = set(within_layer_order) - set(ordering)
            if remaining:
                ordering.append(remaining.pop())

            pi.extend(ordering)

        pa = {}
        for node in pi:
            index = pi.index(node)
            pa[node] = pi[0:index]
        self._pa = pa
        self._causal_order = pi

    def _estimate_residual_dependence(
        self, X_pred: np.ndarray, X_resp: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Estimate residuals and test independence.

        Args:
            X_pred (np.ndarray): _description_
            X_resp (np.ndarray): _description_

        Returns:
            tuple[float, np.ndarray]: _description_
        """
        model = LinearRegression().fit(X_pred, X_resp)
        residuals = X_resp - model.predict(X_pred)
        stat = HSIC().test(X_pred, residuals)[0]
        return stat, residuals

    def _compute_scores_and_residuals(
        self, groups: dict[str, np.ndarray]
    ) -> tuple[dict[str, float], dict[tuple[str, str], np.ndarray]]:
        """Compute scores and residuals.

        Args:
            groups (dict[str, np.ndarray]): _description_

        Returns:
            tuple[dict[str, float], dict[tuple[str, str], np.ndarray]]: _description_
        """
        scores = {}
        residuals = {}
        for pred_name, X_pred in groups.items():
            stats = []
            for resp_name, X_resp in groups.items():
                if pred_name == resp_name:
                    continue
                stat, res = self._estimate_residual_dependence(X_pred, X_resp)
                stats.append(stat)
                residuals[(pred_name, resp_name)] = res
            # higher score = more “exogenous”
            scores[pred_name] = -np.log(stats).sum()
        return scores, residuals

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - x.mean(axis=0)) / x.std(axis=0)

    def _sparse_regression_pruning(
        self, data_dict: dict[str, np.ndarray], nlambda: int = 30
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
        self._sparse_regression_pruning(data_dict=clean_data_dict)

        edge_list = [(parent, child) for child in self._pa for parent in self._pa[child]]

        learned_DAG = DAG(nodes=self._causal_order)
        learned_DAG.add_edges_from(edge_list)
        self._adjacency_matrix = learned_DAG.adjacency_matrix

        self.DAG = learned_DAG
        return learned_DAG

    @property
    def causal_order(self) -> list[str] | None:
        """Causal order."""
        return self._causal_order

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        """Adjacency matrix."""
        return self._adjacency_matrix
