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


class BackgroundKnowledge:
    """Class to store background knowledge."""

    def __init__(self, full_data_dict: dict[str, np.ndarray]) -> None:
        """Initializes the BK object.

        Args:
            full_data_dict (dict[str, np.ndarray]): input data without BK.
        """
        self.final_data = full_data_dict

    def target_overwrite(self, target_dict: dict[str, np.ndarray]) -> None:
        """Restrictions on target variable."""
        for target_name, target_values in target_dict.items():
            self.final_data[target_name] = target_values

    def target_remove_by_index(self, target_dict: dict[str, np.ndarray]) -> None:
        """Takes array columns in target key, value pair and removes them.

        Removal is done by (multiple) index array.

        Args:
            target_dict (dict[str, np.ndarray]): Dict with key equal to groups
                and values equal to indices.
        """
        for target_name, target_indices in target_dict.items():
            self.final_data[target_name] = np.delete(
                arr=self.final_data[target_name], obj=target_indices, axis=1
            )
