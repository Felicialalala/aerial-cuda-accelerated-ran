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

"""pyAerial - SRS pipeline and configuration classes."""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Generic
from typing import List
from typing import TypeVar

import numpy as np

from aerial.phy5g.api import SlotConfig
from aerial.phy5g.api import Array
from aerial.phy5g.api import Pipeline
from aerial.phy5g.api import _SlotConfigT


_SrsOutputT = TypeVar("_SrsOutputT", bound="SrsOutput")


class SrsTxPipeline(Pipeline, Generic[Array, _SlotConfigT]):
    """A base class for SRS transmitter pipeline implementations."""

    @abstractmethod
    def __call__(self,
                 config: _SlotConfigT,
                 **kwargs: Any) -> List[Array]:
        """Abstract method that runs the SRS transmitter pipeline.

        This method gives the signature that the SRS transmitter pipelines should
        implement. It generates the SRS transmission for multiple UEs with a single
        API call.

        Args:
            config (_SlotConfigT): Dynamic SRS slot configuration in this slot. Note that the
                type of this configuration is derived from `SlotConfig`.

        Returns:
            List[Array]: The SRS transmit buffers per UE.
        """
        raise NotImplementedError


class SrsRxPipeline(Pipeline, Generic[_SlotConfigT, _SrsOutputT, Array]):
    """A base class for SRS receiver pipeline implementations."""

    @abstractmethod
    def __call__(self,
                 rx_data: List[Array],
                 config: _SlotConfigT,
                 **kwargs: Any) -> List[_SrsOutputT]:
        """Abstract method that runs the SRS receiver pipeline.

        This method gives the signature that the SRS receiver pipelines should
        implement. It generates runs SRS reception for multiple cells with a single
        API call.

        Args:
            rx_slot (List[Array]): Received slot per cell.
            config (_SlotConfigT): Dynamic SRS slot configuration in this slot. Note that the
                type of this configuration is derived from `SlotConfig`.

        Returns:
            List[_SrsOutputT]: SRS output per UE. The type of this output is derived from
                `SrsOutput`.
        """
        raise NotImplementedError


@dataclass
class SrsOutput:
    """An empty base class for all SRS output data classes."""
    pass


@dataclass
class SrsConfig:
    """SRS transmission configuration.

    Args:
        num_ant_ports (int): Number of SRS antenna ports. 1,2, or 4.
        num_syms (int): Number of SRS symbols. 1,2, or 4.
        num_repetitions (int): Number of repetitions. 1,2, or 4.
        comb_size (int): SRS comb size. 2 or 4.
        start_sym (int): Starting SRS symbol. 0 - 13.
        sequence_id (int): SRS sequence ID. 0 - 1023.
        config_idx (int): SRS bandwidth configuration index. 0 - 63.
        bandwidth_idx (int): SRS bandwidth index. 0 - 3.
        comb_offset (int): SRS comb offset. 0 - 3.
        cyclic_shift (int): Cyclic shift. 0 - 11.
        frequency_position (int): Frequency domain position. 0 - 67.
        frequency_shift (int): Frequency domain shift. 0 - 268.
        frequency_hopping (int): Frequency hopping options. 0 - 3.
        resource_type (int): Type of SRS allocation.

            - 0: Aperiodic.
            - 1: Semi-persistent.
            - 2: Periodic.

        periodicity (int): SRS periodicity in slots.
            0, 2, 3, 5, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320, 640, 1280, 2560.
        offset (int): Slot offset value. 0 - 2569.
        group_or_sequence_hopping (int): Hopping configuration.

            - 0: No hopping.
            - 1: Group hopping.
            - 2: Sequence hopping.
    """
    num_ant_ports: int
    num_syms: int
    num_repetitions: int
    comb_size: int
    start_sym: int
    sequence_id: int
    config_idx: int
    bandwidth_idx: int
    comb_offset: int
    cyclic_shift: int
    frequency_position: int
    frequency_shift: int
    frequency_hopping: int
    resource_type: int
    periodicity: int
    offset: int
    group_or_sequence_hopping: int


@dataclass
class SrsTxConfig(SlotConfig):
    """SRS transmitter pipeline configuration for a slot.

    Args:
        slot (int): Slot number.
        frame (int): Frame number.
        srs_configs (List[SrsConfig]): SRS configuration for each UE.
    """
    slot: int
    frame: int
    srs_configs: List[SrsConfig]


@dataclass
class SrsRxUeConfig:
    """SRS receiver configuration corresponding to a single UE in a slot.

    Args:
        cell_idx (int): Index of the cell that this UE is attached to. This
            index indexes the `SrsRxCellConfig` list of cell configurations,
            as well as the list of Rx data slots given to the SRS receiver
            pipeline.
        srs_config (SrsConfig): SRS configuration for this UE.
        srs_ant_port_to_ue_ant_map (np.array):  Mapping between SRS antenna ports and UE
            antennas in channel estimation buffer: Store estimates for SRS antenna port i in
            `srs_ant_port_to_ue_ant_map[i]`.
        prg_size (int): Number of PRBs per PRB group.
        start_prg (int): Starting PRB group.
        num_prgs (int): Number of PRB groups.
    """
    cell_idx: int
    srs_config: SrsConfig
    srs_ant_port_to_ue_ant_map: np.ndarray
    prg_size: int
    start_prg: int
    num_prgs: int


@dataclass
class SrsRxCellConfig:
    """SRS receiver configuration for a single cell in a slot.

    Args:
        slot (int): Slot number.
        frame (int): Frame number.
        srs_start_sym (int): SRS start symbol in this slot (all UEs).
        num_srs_sym (int): Number of SRS symbols in this slot (all UEs).
    """
    slot: int
    frame: int
    srs_start_sym: int
    num_srs_sym: int


@dataclass
class SrsRxConfig(SlotConfig):
    """SRS receiver pipeline configuration for a slot.

    Args:
        srs_cell_configs (List[SrsRxCellConfig]): List of cell configurations for this slot.
        srs_ue_configs (List[SrsRxUeConfig]): List of UE SRS configurations for this slot.
    """
    srs_cell_configs: List[SrsRxCellConfig]
    srs_ue_configs: List[SrsRxUeConfig]


@dataclass
class SrsReport(SrsOutput):
    """SRS output report.

    This report is returned by the SRS receiver pipeline.

    Args:
        ch_est (np.ndarray): The channel estimates.
        ch_est_to_L2 (np.ndarray): The channel estimates as returned to L2.
        to_est_ms (np.float32): Time offset estimate in microseconds.
        wideband_snr (np.float3): Wideband SNR.
        wideband_noise_energy (np.float32): Wideband noise energy.
        wideband_signal_energy (np.float32): Wideband signal energy.
        wideband_sc_corr (np.complex64): Wideband subcarrier correlation.
        wideband_cs_corr_ratio_db (np.float32):
        wideband_cs_corr_use (np.float32):
        wideband_cs_corr_not_use (np.float32):
        high_density_ant_port_flag (bool):
    """
    ch_est: np.ndarray
    ch_est_to_L2: np.ndarray
    rb_snr: np.ndarray
    to_est_ms: np.float32
    wideband_snr: np.float32
    wideband_noise_energy: np.float32
    wideband_signal_energy: np.float32
    wideband_sc_corr: np.complex64
    wideband_cs_corr_ratio_db: np.float32
    wideband_cs_corr_use: np.float32
    wideband_cs_corr_not_use: np.float32
    high_density_ant_port_flag: bool
