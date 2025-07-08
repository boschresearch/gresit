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
import pytest

from gresit.regression_techniques import (
    CurdsWhey,
    MultiRegressor,
    ReducedRankRegressor,
    SimultaneousLinearModel,
)
from gresit.synthetic_data import GenERData


class TestRegression:
    """Testing regression techniques."""

    @pytest.fixture(scope="class")
    def generate_toy_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate multivariate toy data.

        Returns:
            tuple[np.ndarray, np.ndarray]: Returns data Xdata, Ydata

        """
        _, dat = GenERData(number_of_nodes=5).generate_data(num_samples=30)
        Ydat = dat[:, :, 0]
        Xdat_interim = dat[:, :, 1:]
        Xdat = Xdat_interim.reshape(
            Xdat_interim.shape[0], Xdat_interim.shape[1] * Xdat_interim.shape[2]
        )

        return Xdat, Ydat

    def test_class_instance(self):
        """Initiate MultivariateRegressor."""
        my_reg = SimultaneousLinearModel()
        my_cw = CurdsWhey()
        my_rrr = ReducedRankRegressor(rank=1)
        assert isinstance(my_reg, MultiRegressor)
        assert isinstance(my_cw, MultiRegressor)
        assert isinstance(my_rrr, MultiRegressor)

        assert isinstance(my_reg, SimultaneousLinearModel)
        assert isinstance(my_cw, CurdsWhey)
        assert isinstance(my_rrr, ReducedRankRegressor)

    def test_fitting_works(self, generate_toy_data: tuple[np.ndarray, np.ndarray]):
        """Test regression fit.

        Args:
            generate_toy_data (pd.DataFrame): example data.
        """
        # Initiate
        Xdat, Ydat = generate_toy_data

        my_reg = SimultaneousLinearModel()
        my_reg.fit(X=Xdat, Y=Ydat)

        assert isinstance(my_reg.reg.coef_, np.ndarray)

    def test_predicting_works(self, generate_toy_data: tuple[np.ndarray, np.ndarray]):
        """Test prediction.

        Args:
            generate_toy_data (pd.DataFrame): example data.
        """
        # Initiate
        Xdat, Ydat = generate_toy_data
        my_reg = SimultaneousLinearModel()
        my_reg.fit(X=Xdat, Y=Ydat)
        reg_prediction = my_reg.predict(X_test=Xdat)

        my_cw = CurdsWhey()
        my_cw.fit(X=Xdat, Y=Ydat)
        cw_prediction = my_cw.predict()

        my_rrr = ReducedRankRegressor(rank=3)
        my_rrr.fit(X=Xdat, Y=Ydat)
        rrr_prediction = my_rrr.predict()

        assert reg_prediction.shape[1] == Ydat.shape[1]
        assert len(reg_prediction) > 0

        assert cw_prediction.shape[1] == Ydat.shape[1]
        assert len(cw_prediction) > 0

        assert rrr_prediction.shape[1] == Ydat.shape[1]
        assert len(rrr_prediction) > 0

        assert my_reg.mse(Y_test=Ydat, X_test=Xdat) >= 0
        assert my_cw.mse(Y_test=Ydat, X_test=Xdat) >= 0
        assert my_rrr.mse(Y_test=Ydat, X_test=Xdat) >= 0
