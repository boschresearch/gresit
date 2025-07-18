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

from gresit.background_knowledge import BackgroundKnowledge


class TestBackgroundKnowledge:
    """Test Background Knowledge."""

    def test_class_gets_initiated(self):
        """Does class get initiated?"""
        full_data_dict = {"example_a": np.array([1, 2, 3, 4])}
        bk = BackgroundKnowledge(full_data_dict=full_data_dict)
        assert isinstance(bk, BackgroundKnowledge)
