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

"""pyAerial - CSI-RS pipeline and configuration classes."""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Generic
from typing import List

import numpy as np

from aerial.phy5g.api import SlotConfig
from aerial.phy5g.api import Array
from aerial.phy5g.api import Pipeline
from aerial.phy5g.api import _SlotConfigT


class CsiRsTxPipeline(Pipeline, Generic[_SlotConfigT, Array]):
    """A base class for CSI-RS transmitter pipeline implementations."""

    @abstractmethod
    def __call__(self,
                 config: _SlotConfigT,
                 tx_buffers: List[Array],
                 **kwargs: Any) -> List[Array]:
        """Abstract method that runs the CSI-RS transmitter pipeline.

        This method gives the signature that the CSI-RS transmitter pipelines should
        implement. It generates the CSI-RS transmission for multiple cells with a single
        API call.

        Args:
            config (_SlotConfigT): Dynamic CSI-RS slot configuration in this slot. Note that the
                type of this configuration is derived from `SlotConfig`.
            tx_buffers (List[Array]): A list of transmit slot buffers, one per cell. These
                represent the slot buffers prior to inserting the CSI-RS.

        Returns:
            List[Array]: Transmit buffers for the slot for each cell after inserting CSI-RS.
        """
        raise NotImplementedError


class CsiRsRxPipeline(Pipeline, Generic[_SlotConfigT, Array]):
    """A base class for CSI-RS receiver pipeline implementations."""

    @abstractmethod
    def __call__(self,
                 rx_data: List[Array],
                 config: _SlotConfigT,
                 **kwargs: Any) -> List[List[Array]]:
        """Abstract method that runs the CSI-RS receiver pipeline.

        This method gives the signature that the CSI-RS transmitter pipelines should
        implement. It generates the CSI-RS transmission for multiple UEs with a single
        API call.

        Args:
            rx_slot (List[Array]): Received slot per cell.
            config (_SlotConfigT): Dynamic CSI-RS slot configuration in this slot. Note that the
                type of this configuration is derived from `SlotConfig`.

        Returns:
            List[List[Array]]: Channel estimation buffers for the slot for each UE.
        """
        raise NotImplementedError


@dataclass
class CsiRsConfig:
    """CSI-RS parameters.

    The RRC parameters for CSI-RS. Used together with PDSCH Tx and
    CSI-RS Tx and Rx. See TS 38.211 section 7.4.1.5.3 and in particular table 7.4.1.5.3-1
    for exact definitions of the fields.

    Args:
        start_prb (int): PRB where this CSI resource starts. Expected value < 273.
        num_prb (int): Number of PRBs across which this CSI resource spans.
            Expected value <= 273 - start_prb.
        freq_alloc (List[int]): Bitmap defining frequency domain allocation. Counting is started
            from least significant bit (first element of the list). This corresponds to the
            `frequencyDomainAllocation` field in CSI-RS RRC parameters.
        row (int): Row entry into the CSI-RS resource location table. Valid values 1-18.
        symb_L0 (int): Time domain location L0. This corresponds to the
            `firstOFDMSymbolInTimeDomain` field in CSI-RS RRC parameters.
        symb_L1 (int): Time domain location L1. This corresponds to the
            `firstOFDMSymbolInTimeDomain2` field in CSI-RS RRC parameters.
        freq_density (int): The density field, p and comb offset (for dot5),

            - 0: dot5 (even RB)
            - 1: dot5 (odd RB)
            - 2: one
            - 3: three.

        scramb_id (int): Scrambling ID of CSI-RS.
        idx_slot_in_frame (int): Slot index in frame.
        cdm_type (int): CDM Type.

            - 0: noCDM
            - 1: fd-CDM2
            - 2: cdm4-FD2-TD2
            - 3: cdm8-FD2-TD4

        beta (float): Power scaling factor
        enable_precoding (bool): Enable/disable precoding.
        precoding_matrix_index (int): Index of the precoding matrix to use. The list of precoding
            matrices needs to be given separately.
    """
    start_prb: int
    num_prb: int
    freq_alloc: List[int]
    row: int
    symb_L0: int  # pylint: disable=invalid-name
    symb_L1: int  # pylint: disable=invalid-name
    freq_density: int
    scramb_id: int
    idx_slot_in_frame: int
    cdm_type: int
    beta: float = 1.0
    enable_precoding: bool = False
    precoding_matrix_index: int = 0


@dataclass
class CsiRsTxConfig(SlotConfig):
    """CSI-RS transmission configuration.

    Args:
        csirs_configs (List[List[CsiRsConfig]]): A list of CSI-RS RRC parameters,
            one list per cell. See `CsiRsConfig`.
        precoding_matrices (List[np.ndarray]): A list of precoding matrices. This list
            gets indexed by the `precoding_matrix_index` field in `CsiRsConfig`.
    """
    csirs_configs: List[List[CsiRsConfig]]
    precoding_matrices: List[np.ndarray]


@dataclass
class CsiRsRxConfig(SlotConfig):
    """CSI-RS reception configuration.

    Args:
        csirs_configs (List[List[CsiRsConfig]]): A list of CSI-RS RRC parameters,
            one list per cell. See `CsiRsConfig`.
        ue_cell_association (List[int]): Association of UEs to cells. Index of the cell
            per UE, used to index `csirs_configs`.
    """
    csirs_configs: List[List[CsiRsConfig]]
    ue_cell_association: List[int]
