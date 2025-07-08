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

from gresit.torch_models import MLP, EarlyStopping, Multioutcome_MLP, MultiRegressor


class TorchTesting:
    """Testing Torch models."""

    def test_instance_is_created(self):
        """Test MLP instance."""
        # Set up
        mlp_multi_model = Multioutcome_MLP()
        mlp = MLP(input_dim=10, output_dim=2)
        es = EarlyStopping()

        # Assert
        assert isinstance(mlp_multi_model, Multioutcome_MLP)
        assert isinstance(mlp_multi_model, MultiRegressor)
        assert isinstance(mlp, MLP)
        assert isinstance(es, EarlyStopping)

    def test_forward_method_defined(self):
        """Test whether forward is defined."""
        mlp = MLP(input_dim=10, output_dim=2)

        assert hasattr(mlp, "forward")

    def test_mlp_fit_and_predictions(self):
        """Test mlp fit and predictions."""
        # Set up
        mlp_model = Multioutcome_MLP()
        x_dat = np.random.random((20, 2, 3)) * 2.0 - 0.5
        y_dat = np.random.random((20, 2))
        # Act
        mlp_model.fit(X=x_dat, Y=y_dat)
        x_dat_test = x_dat[0:10, :, :]
        y_dat_test = y_dat[0:10, :]

        # Assert
        assert mlp_model.predict().shape[1] == y_dat.shape[1]
        assert mlp_model.mse() >= 0
        assert mlp_model.input_dim == x_dat.shape[1]
        assert mlp_model.output_dim == y_dat.shape[1]
        assert mlp_model.predict().shape == mlp_model._Y_test.shape
        assert mlp_model.predict(x_dat_test).shape == y_dat_test.shape

    def test_early_stopping(self):
        """Test early stopping."""
        # Set up
        patience = 5
        es = EarlyStopping(patience=patience)
        # Assert
        assert es.min_delta == 0
        assert es.patience == patience
        assert callable(es)
