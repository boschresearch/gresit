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

import unittest

import numpy as np

from gresit.group_grandag import GroupGraNDAG
from gresit.group_pc import GroupPC
from gresit.group_random_regress import GroupRandomRegress
from gresit.group_resit import GroupResit
from gresit.independence_tests import HSIC
from gresit.params import (
    DataParams,
    ExperimentParams,
    GraNDAGParams,
    PCParams,
    RandomRegParams,
    ResitParams,
)
from gresit.synthetic_data import GenERData, GenLayeredData
from gresit.torch_models import Multioutcome_MLP


class TestParams(unittest.TestCase):
    """Tests the parameter class."""

    def setUp(self):
        """Sets up."""
        self.rng = np.random.default_rng(seed=2024)

    def test_init(self):
        """Tests init."""
        params = ResitParams(
            regressor=Multioutcome_MLP,
            kwargs={
                "val_size": 0.1,
                "learning_rate": 0.0001,
            },
        )

        self.assertEqual(params.kwargs["val_size"], 0.1)
        self.assertEqual(params.kwargs["learning_rate"], 0.0001)

    def test_get_name(self):
        """Tests name."""
        params = ResitParams(
            regressor=Multioutcome_MLP,
            kwargs={"val_size": 0.2, "learning_rate": 0.1},
        )

        self.assertEqual(
            params.name,
            "GroupResit(vsize=0.2, lr=0.1)_HSIC_murgs",
        )

    def test_make_regressor(self):
        """Tests init."""
        params = ResitParams(
            regressor=Multioutcome_MLP,
            kwargs={
                "val_size": 0.1,
            },
        )
        self.assertIsInstance(params.init(self.rng), GroupResit)

    def test_data_params(self):
        """Tests init."""
        data_params_er = DataParams(
            generator=GenERData,
            number_of_nodes=6,
            number_of_layers=2,  # should be immaterial
            group_size=4,
        )

        data_params_ld = DataParams(
            generator=GenLayeredData,
            number_of_nodes=6,
            number_of_layers=2,
            group_size=4,
        )

        members_er = list(vars(data_params_er).keys())
        members_ld = list(vars(data_params_ld).keys())

        layered_data_generator = data_params_ld.init(rng=np.random.default_rng(seed=2024))

        self.assertIsNotNone(layered_data_generator.layering)
        with self.assertRaises(AttributeError) as ctx:
            data_params_er.init(rng=np.random.default_rng(seed=2024)).layering
        self.assertEqual("'GenERData' object has no attribute 'layering'", str(ctx.exception))
        self.assertEqual(len(layered_data_generator.layering), 2)
        self.assertEqual(members_er, members_ld)

    def test_experiment_params(self):
        """Tests init."""
        regressor_kwargs = {
            "val_size": 0.1,
        }

        params = ExperimentParams(
            algos=[
                ResitParams(
                    regressor=Multioutcome_MLP,
                    kwargs=regressor_kwargs,
                    pruning_method="murgs",
                    test=HSIC,
                ),
                ResitParams(
                    regressor=Multioutcome_MLP,
                    kwargs=regressor_kwargs,
                    pruning_method="murgs",
                    test=HSIC,
                ),
            ],
            data=DataParams(
                equation_kwargs={},
            ),
        )

        self.assertEqual(len([a for a in params]), 2)

    def test_pc_params(self):
        """Tests GroupPC params."""
        # Set up
        pcparams = PCParams(alpha=0.1)

        assert pcparams.name == "GroupPC(alpha=0.1, test=FisherZVec)"
        assert isinstance(pcparams.init(rng=np.random.default_rng()), GroupPC)

    def test_grandag_params(self):
        """Tests GraNDAGParams."""
        # Set up
        grandagparams = GraNDAGParams(n_iterations=100, h_threshold=0.01)

        assert grandagparams.name == "GroupGraN-DAG(iter=100, h_thresh=0.01)"
        assert isinstance(grandagparams.init(rng=np.random.default_rng()), GroupGraNDAG)

    def test_randomregress_params(self):
        """Tests randomregress params."""
        # Set up
        rr_params = RandomRegParams()

        assert rr_params.name == "GroupRandomRegress()"
        assert isinstance(rr_params.init(rng=np.random.default_rng()), GroupRandomRegress)
