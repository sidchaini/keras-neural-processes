"""Keras Neural Processes

Copyright (C) 2026  Siddharth Chaini
-----
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from .models import ANP, NP, CNP
from .data import (
    get_ragged_tensor,
    compute_peak_indices,
    create_stratified_np_dataset,
    get_context_set_dense,
    get_context_set_dense_forecast,
    get_gplike_valset,
    unscale_values,
)
from .metrics import (
    msse_1d,
    mase_1d,
    msle_1d,
    mape_1d,
    mrmse_1d,
    chi2_1d,
    nrmseo_1d,
    nrmse_po_1d,
    picp_mpiw_1d,
    rmse_1d,
    mae_1d,
    rse_1d,
    rae_1d,
    nlpd_1d,
    nrmsep_1d,
    picp_1d,
    mpiw_1d,
)

__version__ = "0.1.0"

__all__ = [
    # models
    "ANP",
    "NP",
    "CNP",
    # data utilities
    "get_ragged_tensor",
    "compute_peak_indices",
    "create_stratified_np_dataset",
    "get_context_set_dense",
    "get_context_set_dense_forecast",
    "get_gplike_valset",
    "unscale_values",
    # 1D metrics
    "msse_1d",
    "mase_1d",
    "msle_1d",
    "mape_1d",
    "mrmse_1d",
    "chi2_1d",
    "nrmseo_1d",
    "nrmse_po_1d",
    "picp_mpiw_1d",
    "rmse_1d",
    "mae_1d",
    "rse_1d",
    "rae_1d",
    "nlpd_1d",
    "nrmsep_1d",
    "picp_1d",
    "mpiw_1d",
]
