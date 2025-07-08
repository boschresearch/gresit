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

Loss functions for training the model.

`loss_hsic()` taken from https://github.com/danielgreenfeld3/XIC/blob/master/hsic.py as
used in paper https://arxiv.org/abs/1910.00270
"""

import torch


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Get pairwise distance.

    Args:
        x (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Get gaussian kernel matrix.

    Args:
        x (torch.Tensor): _description_
        sigma (float, optional): _description_. Defaults the median heuristic when None.

    Returns:
        _type_: _description_
    """
    pairwise_distances_ = pairwise_distances(x)
    if sigma is None:
        sigma = _median_heuristic(pairwise_distances_)
    return torch.exp(-pairwise_distances_ / sigma)


def _median_heuristic(x: torch.Tensor) -> torch.Tensor:
    return torch.median(x)


def HSIC(
    x: torch.Tensor,
    y: torch.Tensor,
    s_x: float | None = None,
    s_y: float | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Get test statistic.

    Args:
        x (torch.Tensor): _description_
        y (torch.Tensor): _description_
        s_x (float, optional): _description_. Defaults to None.
        s_y (float, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        torch.Tensor: _description_
    """
    m = x.shape[0]
    K = GaussianKernelMatrix(x, s_x)
    L = GaussianKernelMatrix(y, s_y)

    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().to(device)
    return torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)


def loss_hsic(
    x: torch.Tensor,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Get HSIC loss.

    Args:
        x (torch.Tensor): _description_
        y_pred (torch.Tensor): _description_
        y_true (torch.Tensor): _description_
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        torch.Tensor: _description_
    """
    r = y_pred - y_true
    return HSIC(x=x, y=r, device=device)


def loss_mse(
    x: torch.Tensor,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """MSE loss.

    Returns:
        torch.Tensor: _description_
    """
    _ = x  # Ignored on purpose, kept for compatibility
    return ((y_pred - y_true) ** 2).mean()


def loss_disco(
    x: torch.Tensor,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Get UCRT loss.

    Args:
        x (torch.Tensor): _description_
        y_pred (torch.Tensor): _description_
        y_true (torch.Tensor): _description_
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        torch.Tensor: _description_
    """
    r = y_pred - y_true
    return u_distance_cov_squared(a=x, b=r, device=device)


def _pairwise_distance_matrix(a: torch.Tensor) -> torch.Tensor:
    diff = a.unsqueeze(0) - a.unsqueeze(1)
    distances = torch.linalg.norm(diff, dim=2)
    return distances


def _u_centered_matrix(x: torch.Tensor) -> torch.Tensor:
    x_ij = x.clone()
    n_rows = x_ij.shape[0]
    total_mean = x_ij.sum(dtype=torch.float32) / ((n_rows - 1) * (n_rows - 2))
    axis_mean = x_ij.sum(dim=1, dtype=torch.float32) / (n_rows - 2)

    x_ij -= axis_mean[None, :]
    x_ij -= axis_mean[:, None]
    x_ij += total_mean

    x_ij.fill_diagonal_(0)

    return x_ij


def u_distance_cov_squared(
    a: torch.Tensor, b: torch.Tensor, device: str = "cpu", robust: bool = False
) -> torch.Tensor:
    """Unbiased squared distance covariance.

    Args:
        a (torch.Tensor): _description_
        b (torch.Tensor): _description_
        device (str): device to run on.
        robust (bool): if you want to make sure that the joint finite first moment
            condition for the distance covariance holds, setting `robust` to `True`
            transforms data to univariate ranks.

    Returns:
        torch.Tensor: _description_
    """
    if robust:
        a = _univariate_ranks(a)
        b = _univariate_ranks(b)
    n = a.shape[0]
    a_distance = _pairwise_distance_matrix(a)
    b_distance = _pairwise_distance_matrix(b)

    a_ij = _u_centered_matrix(a_distance)
    b_ij = _u_centered_matrix(b_distance)

    return (a_ij * b_ij).sum(dtype=torch.float32) / (n * (n - 3))


def _univariate_ranks(tensor: torch.Tensor) -> torch.Tensor:
    TARGET_DIM = 2
    assert tensor.ndim == TARGET_DIM, "Input tensor must be 2D"

    sorted_indices = torch.argsort(tensor, dim=0)
    ranks = torch.argsort(sorted_indices, dim=0)

    ranks += 1

    return ranks
