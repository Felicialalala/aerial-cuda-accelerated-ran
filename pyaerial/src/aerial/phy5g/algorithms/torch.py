# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pyAerial library - PyTorch-based algorithm implementations."""
# PyTorch type hints are buggy...
# mypy: disable-error-code=call-overload
import math
from typing import Generic
from typing import List

import cupy as cp  # type: ignore
import numpy as np
import torch  # type: ignore

from aerial.phy5g.api import Array


class MmseChannelEstimator(Generic[Array]):
    """MMSE-based frequency interpolation of the least squares channel estimates.

    This class implements interpolation of LS channel estimates in DMRS REs to full
    band channel estimates. The interpolation is done using an MMSE filter.
    """

    def __init__(self,
                 *,
                 max_num_prbs: int,
                 subcarrier_spacing: float,
                 tau_rms: float,
                 noise_var: float,
                 pdp_type: str,
                 device: int) -> None:
        """Initialize MmseChannelEstimator.

        Args:
            max_num_prbs (int): Maximum number of PRBs.
            subcarrier_spacing (float): Subcarrier spacing in Hz.
            tau_rms (float): RMS delay spread (in seconds) for which the filters
                get optimized.
            noise_var (float): Noise variance for which the filters get optimized.
            pdp_type (str): Power delay profile type assumed for filter computation.
                Either "exponential" or "uniform".
            device (int): Index of the GPU device to be used.
        """
        dmrs_energy = 2

        self.device = device

        if pdp_type not in ["exponential", "uniform"]:
            raise ValueError(f"Invalid PDP type: {pdp_type}! Options: exponential, uniform.")

        freq_k, freq_l = torch.meshgrid(
            torch.arange(0, max_num_prbs * 12, 2, device=device),
            torch.arange(0, max_num_prbs * 12, 2, device=device),
            indexing='ij')
        delta_f = (freq_k - freq_l) * subcarrier_spacing
        if pdp_type == "uniform":
            freq_corr = torch.sinc(tau_rms * delta_f)
        elif pdp_type == "exponential":
            freq_corr = torch.div(1, 1 + 1j * 2 * math.pi * delta_f * tau_rms)

        f_occ = torch.ones(max_num_prbs * 6, device=device)
        f_occ[1::2] = -1

        f_occ = torch.outer(f_occ, f_occ) + \
            torch.ones(max_num_prbs * 6, max_num_prbs * 6, device=device)
        freq_corr = torch.mul(f_occ, freq_corr)  # pylint: disable=used-before-assignment
        freq_corr *= dmrs_energy
        auto_corr = freq_corr + noise_var * torch.eye(max_num_prbs * 6, device=device)

        freq_k, freq_l = torch.meshgrid(torch.arange(0, max_num_prbs * 12, device=device),
                                        torch.arange(0, max_num_prbs * 12, 2, device=device),
                                        indexing='ij')
        delta_f = (freq_k - freq_l) * subcarrier_spacing
        if pdp_type == "uniform":
            cross_corr = torch.sinc(tau_rms * delta_f)
        elif pdp_type == "exponential":
            cross_corr = torch.div(1, 1 + 1j * 2 * math.pi * delta_f * tau_rms)

        cross_corr *= math.sqrt(dmrs_energy)  # pylint: disable=undefined-variable

        self.ch_est_filter = torch.mm(
            cross_corr,
            torch.linalg.inv(auto_corr)  # pylint: disable=not-callable
        ).to(torch.complex64)

    def estimate(self, ls_ch_ests: List[Array]) -> List[Array]:
        """Run MMSE estimation.

        Args:
            ls_ch_ests (List[Array]): Least squares channel estimates for DMRS symbols, e.g. from
                pyAerial channel estimator.

        Returns:
            List[Array]: Interpolated channel estimates.
        """
        is_numpy = isinstance(ls_ch_ests, np.ndarray)
        ch_est = []
        for ls_ch_est in ls_ch_ests:
            if is_numpy:
                ls_tensor = torch.from_numpy(ls_ch_est).to(self.device)
            else:
                ls_tensor = torch.as_tensor(ls_ch_est, device=self.device)
            num_layers, num_rx_ant, num_dmrs = ls_tensor.shape[1:]
            ls_tensor = torch.reshape(ls_tensor, (ls_tensor.shape[0], -1))
            mmse_est = torch.mm(self.ch_est_filter, ls_tensor)

            # Reshape back
            mmse_est = torch.reshape(mmse_est,
                                     (mmse_est.shape[0], num_layers, num_rx_ant, num_dmrs))
            mmse_est = torch.permute(mmse_est, (2, 1, 0, 3))

            if is_numpy:
                ch_est.append(mmse_est.numpy())
            else:
                ch_est.append(cp.array(mmse_est))

        return ch_est
