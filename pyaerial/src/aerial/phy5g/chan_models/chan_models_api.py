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

"""
pyAerial library - channel models API.

DOCUMENTATION ONLY - This file is kept for reference and documentation purposes.
The actual implementation now uses the C++ binding classes directly.
"""

# pylint: disable=no-member,too-many-positional-arguments
from typing import List, TypeVar, Optional
import cupy as cp  # type: ignore
import numpy as np

# Import implementation class with error handling

# Remove the TYPE_CHECKING import of StatisChanModelImpl
# try:
#     from .chan_models_src import StatisChanModelImpl
# except ImportError:
#     StatisChanModelImpl = None  # type: ignore

# Direct C++ binding import
try:
    from aerial import pycuphy  # type: ignore
    CPP_BINDING_AVAILABLE = True
except ImportError:
    pycuphy = None  # type: ignore
    CPP_BINDING_AVAILABLE = False

Array = TypeVar("Array", np.ndarray, cp.ndarray)

# Summary of the API
#
# The `chan_models_api.py` module provides a comprehensive set of classes and
# configurations for simulating and modeling wireless communication channels,
# particularly in the context of 5G networks. The API is designed to be
# flexible and extensible, allowing users to configure various aspects of the
# channel model according to their specific needs.
#
# Key Components:
#
# 1. **Scenario Enum**:
#    - Defines different deployment scenarios such as Urban Macro (UMa), Urban
#      Micro (UMi), and Rural Macro (RMa).
#
# 2. **Coordinate Class**:
#    - Represents a 3D coordinate in the global coordinate system, used for
#      specifying locations of user terminals (UTs) and cells.
#
# 3. **AntPanelConfig Class**:
#    - Configures antenna panel parameters, including:
#      - Number of antennas and array dimensions (M_g, N_g, M, N, P)
#      - Antenna spacing in wavelengths
#      - Antenna patterns (theta and phi patterns in dB)
#      - Polarization angles
#      - Antenna model type (isotropic, directional, or direct pattern)
#
# 4. **UtParamCfg Class**:
#    - Defines parameters for user terminals, including:
#      - Unique ID and location
#      - Outdoor/indoor indicator
#      - Antenna panel configuration index
#      - Antenna panel orientation in GCS (theta, phi, slant offset)
#      - Mobility parameters
#      - Serving cell ID
#
# 5. **CellParam Class**:
#    - Specifies parameters for cells, including:
#      - Cell ID and site ID
#      - Location in GCS
#      - Antenna panel configuration index
#      - Antenna panel orientation in GCS (theta, phi, slant offset)
#      - Co-sited cells share the same site ID and LSP
#
# 6. **SimConfig Class**:
#    - Configures simulation parameters, including:
#      - Link simulation settings
#      - Frequency and bandwidth
#      - Subcarrier spacing and FFT size
#      - PRB and PRBG configurations
#      - Channel realization settings
#
# 7. **SystemLevelConfig Class**:
#    - Configures system-level parameters, including:
#      - Scenario type and inter-site distance
#      - Number of sites and sectors
#      - Number of UTs
#      - Path loss and shadowing options
#      - O2I penetration loss settings
#      - Near-field and non-stationarity effects
#
# 8. **LinkLevelConfig Class**:
#    - Configures link-level parameters, including:
#      - Fast fading type and delay profile
#      - Delay spread and mobility
#      - Number of rays and paths
#      - CFO and delay settings
#
# 9. **ExternalConfig Class**:
#    - Manages external configurations, including:
#      - Cell and UT configurations
#      - Channel buffers in sparse format:
#        - CIR coefficients and indices
#        - Number of non-zero taps
#        - CFR per subcarrier and PRB group
#
# 10. **StatisChanModel Class**:
#     - The main class that integrates all configurations and provides:
#       - Channel model simulation with active cells and UTs
#       - UT location and mobility updates
#       - LOS/NLOS and path loss/shadowing statistics
#       - Channel state reset functionality
#
# Usage:
# The API is designed to be used in simulations where detailed modeling of
# wireless channels is required. It allows for the configuration of various
# parameters to match real-world scenarios as specified by standards like
# 3GPP TR 38.901. The modular design enables easy extension and adaptation to
# different research and development needs in wireless communications.
#
# Key Features:
# - Support for both isotropic and directional antenna patterns
# - Flexible coordinate system (GCS and LCS)
# - Sparse format for efficient CIR storage
# - Configurable path loss and shadowing models
# - Support for O2I penetration losses
# - Near-field and non-stationarity effects
#
# This summary provides an overview of the key components and their roles
# within the API, offering a foundation for users to understand and utilize
# the module effectively.


# Helper stuctures
# Enum for scenario types
class Scenario:
    """Enumeration of supported channel scenarios."""
    UMa = 'UMa'
    UMi = 'UMi'
    RMa = 'RMa'
    Indoor = 'Indoor'  # TODO: Not supported yet
    InF = 'InF'  # TODO: Not supported yet
    SMa = 'SMa'  # TODO: Not supported yet, currently in CR


# Coordinate structure
# all location are in global coordinate system
class Coordinate:
    """3D coordinate representation in global coordinate system."""

    def __init__(self,
                 x: float = 0,  # x-coordinate in global coordinate system
                 y: float = 0,  # y-coordinate in global coordinate system
                 z: float = 0):  # z-coordinate in global coordinate system
        """Initialize coordinate with x, y, z values."""
        self.x = x
        self.y = y
        self.z = z


# Antenna panel parameters in LCS
# use to construct a list of antenna panel parameters and can be used for
# indexing by UE and BS
class AntPanelConfig:
    """Antenna panel configuration parameters."""

    def __init__(self,
                 n_ant: int = 4,  # Number of antennas in the array,
                                  # n_ant = M_g * N_g * P * M * N
                 ant_size: List[int] = None,  # Dimensions of the
                 # antenna array (M_g,N_g,M,N,P), TODO: only support one
                 # panel for now M_g=N_g=1;
                 ant_spacing: List[float] = None,  # Spacing between
                 # antennas in terms of wavelength (d_g_h,d_g_v,d_h,d_v)
                 ant_theta: List[float] = None,  # Antenna pattern
                 # A(theta, phi = 0) in dB, dimension: 181 * 1.
                 # for theta in [0,180]
                 ant_phi: List[float] = None,  # Antenna pattern
                 # A(theta = 90, phi) in dB, dimension: 360 * 1.
                 # for phi in [0,359]
                 ant_polar_angles: List[float] = None,  # Antenna
                 # polar angles (roll_angle_first_polz, roll_angle_second_polz),
                 ant_model: int = 2  # Antenna model type (0: isotropic,
                                     # 1: directional, 2: direct antenna pattern)
                 ):
        """Initialize antenna panel configuration."""
        if ant_size is None:
            ant_size = [1, 1, 1, 2, 2]
        if ant_spacing is None:
            ant_spacing = [0, 0, 0.5, 0.5]
        if ant_polar_angles is None:
            # default to ±45 deg for dual-pol, 0 deg for single-pol
            ant_polar_angles = [45, -45] if ant_size[4] == 2 else [0]

        # ensure attribute exists early
        self.ant_model = ant_model

        if ant_model == 2:
            if ant_theta is None or ant_phi is None:
                raise ValueError("antenna pattern is not provided if"
                                 "antenna model is direct antenna pattern,"
                                 "please provide ant_theta and ant_phi")
        else:
            # calculate the antenna pattern if it's 0 or 1
            self.calc_ant_pattern()

        self.n_ant = n_ant
        self.ant_size = ant_size
        self.ant_spacing = ant_spacing
        self.ant_theta = ant_theta
        self.ant_phi = ant_phi
        self.ant_polar_angles = ant_polar_angles

        # check number of antenna equal to the product of the antenna size
        if n_ant != np.prod(ant_size):
            raise ValueError("number of antenna does not match the product of the antenna size")
        # check if P=1 agrees with the antenna ant_polar_angles
        if ant_size[4] != ant_polar_angles.shape[0]:
            raise ValueError("number of polar angles does not match the number of antenna panels")
        # check dimesion of the antenna size prod(ant_size) == n_ant
        if ant_size[4] != 1 and ant_size[4] != 2:
            raise ValueError("Only support P=1 or P=2 for now")
        # check if the antenna polar angles are consistent with the antenna model
        if ant_model == 2 and (ant_theta is None or ant_phi is None):
            raise ValueError("antenna pattern is not provided if antenna model is 2")

    def calc_ant_pattern(self) -> None:
        """Calculate antenna patterns for isotropic and directional models."""
        if self.ant_model == 0:  # Isotropic pattern
            # Isotropic pattern has constant gain of 1 (0 dB) in all directions
            self.ant_theta = np.zeros(181).tolist()  # 0 dB gain for all theta angles
            self.ant_phi = np.zeros(360).tolist()    # 0 dB gain for all phi angles
        elif self.ant_model == 1:  # Directional pattern (3GPP)
            # Generate theta angles from 0 to 180 degrees
            theta_deg = np.linspace(0, 180, 181)
            # Generate phi angles from 0 to 359 degrees
            phi_deg = np.linspace(0, 359, 360)

            # Calculate 3GPP directional pattern according to TR 38.901
            # A(theta) = -min[12*(theta/theta_3dB)^2, A_m]
            # where theta_3dB = 65 degrees
            theta_3db = 65  # degrees
            sla_v = 30  # dB
            amplitude_theta = -np.minimum(12 * ((theta_deg - 90) / theta_3db)**2, sla_v)

            # A(phi) = -min[12*(phi/phi_3dB)^2, A_m]
            # where phi_3dB = 65 degrees
            phi_3db = 65  # degrees
            a_max = 30  # dB
            phi_wrap = np.where(phi_deg >= 180, phi_deg - 360, phi_deg)
            amplitude_phi = -np.minimum(12 * (phi_wrap / phi_3db)**2, a_max)

            # Store the patterns in dB
            self.ant_theta = amplitude_theta.tolist()
            self.ant_phi = amplitude_phi.tolist()

            # Note: The actual antenna pattern will be calculated using:
            # F_theta = sqrt(10^(A_theta/10)) * cos(slant_angle)
            # F_phi = sqrt(10^(A_phi/10)) * sin(slant_angle)
            # where slant_angle is determined by ant_polar_angles and
            # additional slant offset


# UT parameters for a single TTI
class UtParamCfg:
    """User Terminal (UE) configuration parameters.

    Note: This class corresponds to the C++ UtParamCfgCfg struct (public API).
    The internal implementation adds additional fields like d_2d_in, but these
    are not exposed to Python users.
    """

    def __init__(self,
                 uid: int = 0,  # global UE ID
                 loc: Coordinate = None,  # UE location at beginning
                 outdoor_ind: int = 0,  # outdoor indicator, 0: indoor,
                 # 1: outdoor, calculated at AODT side; Generate at the
                 # beginning of the simulation
                 ant_panel_idx: int = 0,  # antenna panel configuration index
                 ant_panel_orientation: np.ndarray = None,
                 # antenna panel orientation in GCS, dim: 3: (theta, phi,
                 # additional slant offset) for each antenna element.
                 # antenan angle calculation: (_prime is in LCS and _ is in GCS)
                 # theta_n_m_ZOA_prime = theta_n_m_ZOA(n, m) -
                 #     UtParamCfg.antPanelOrintation[0];
                 # phi_n_m_AOA_prime = phi_n_m_AOA(n, m) -
                 #     UtParamCfg.antPanelOrintation[1];
                 # theta_n_m_ZOD_prime = theta_n_m_ZOD(n, m) -
                 #     CellParam.antPanelOrientation[0];
                 # phi_n_m_AOD_prime = phi_n_m_AOD(n, m) -
                 #     CellParam.antPanelOrientation[1];
                 # these angles are used to read the antenna pattern F_theta
                 # and F_phi in LCS
                 # for single polarization, slant angle = additional slant
                 # offset + AntPanelConfig.ant_polar_angles[0]
                 # for dual polarization, slant angle 0 = additional slant
                 # offset + AntPanelConfig.ant_polar_angles[0]
                 #                        slant angle 1 = additional slant
                 # offset + AntPanelConfig.ant_polar_angles[1]
                 # these angles are used to calculate the antenna pattern by
                 # cos(slant angle *) * F_theta and sin(slant angle *) * F_phi,
                 # Eq 7.3-4 and Eq 7.3-5 in 3GPP TR 38.901
                 velocity: np.ndarray = None):  # mobility
        # parameters, (vx, vy, vz), abs(velocity_direction) = speed
        # in m/s, vz = 0 per 3GPP spec
        """Initialize User Terminal parameters."""
        if loc is None:
            loc = Coordinate(0, 0, 0)
        if ant_panel_orientation is None:
            ant_panel_orientation = np.array([0, 0, 0])
        if velocity is None:
            velocity = np.array([0, 0, 0])

        self.uid = uid
        self.loc = loc
        self.outdoor_ind = outdoor_ind
        self.ant_panel_idx = ant_panel_idx
        self.ant_panel_orientation = ant_panel_orientation
        self.velocity = velocity


# Cell parameters at the beginning of the simulation,
# as well as per-TTI update if needed
class CellParam:
    """Base station cell configuration parameters."""

    def __init__(self,
                 cid: int,  # global cell ID, 0~n_site*n_sector_per_site-1
                 site_id: int,  # site ID, 0~n_site-1, use to access LSP
                 # same number of site_id are co-sited cells, with the same LSP
                 # E.g., cid = 0,1,2; site_id = 0; cid = 3,4,5; site_id = 1;
                 # cid = 6,7,8; site_id = 2
                 loc: Coordinate,  # cell location, (x,y,z), constant during
                                   # simulation
                 ant_panel_idx: int,  # antenna parameters, index of the panel
                                   # configuration
                 ant_panel_orientation: np.ndarray = None,
                 # antenna panel orientation in GCS, dim: 3: (theta, phi,
                 # additional slant offset) for each antenna element.
                 # the co-sites cells will have antPanelOrintation[1] separated
                 # by 120, 240 degrees
                 # antenan angle calculation: (_prime is in LCS and _ is in GCS)
                 # theta_n_m_ZOA_prime = theta_n_m_ZOA(n, m) -
                 #     UtParamCfg.antPanelOrientation[0];
                 # phi_n_m_AOA_prime = phi_n_m_AOA(n, m) -
                 #     UtParamCfg.antPanelOrientation[1];
                 # theta_n_m_ZOD_prime = theta_n_m_ZOD(n, m) -
                 #     CellParam.antPanelOrientation[0];
                 # phi_n_m_AOD_prime = phi_n_m_AOD(n, m) -
                 #     CellParam.antPanelOrientation[1];
                 # these angles are used to read the antenna pattern F_theta
                 # and F_phi in LCS
                 # for single polarization, slant angle = additional slant
                 # offset + AntPanelConfig.ant_polar_angles[0]
                 # for dual polarization, slant angle 0 = additional slant
                 # offset + AntPanelConfig.ant_polar_angles[0]
                 # slant angle 1 = additional slant
                 # offset + AntPanelConfig.ant_polar_angles[1]
                 # these angles are used to calculate the antenna pattern by
                 # cos(slant angle *) * F_theta and sin(slant angle *) * F_phi,
                 # Eq 7.3-4 and Eq 7.3-5 in 3GPP TR 38.901
                 ):
        """Initialize base station cell parameters."""
        if ant_panel_orientation is None:
            ant_panel_orientation = np.array([0, 0, 0])

        self.cid = cid
        self.site_id = site_id
        self.loc = loc
        self.ant_panel_idx = ant_panel_idx
        self.ant_panel_orientation = ant_panel_orientation


# System level configuration,
# set once per channel model configuration
class SystemLevelConfig:
    """System-level simulation configuration parameters."""

    def __init__(self,  # pylint: disable=too-many-arguments
                 scenario: str = Scenario.UMa,  # scenario type
                 isd: float = 1732.0,  # inter-site distance in meters, only
                 # used for RMa scenario, 1732 or 5000; Will be ignored for
                 # UMa/UMi
                 n_site: int = 1,  # number of sites
                 n_sector_per_site: int = 3,  # number of sectors per site
                 n_ut: int = 100,  # total number of UTs

                 # need to add to AODT UI, exclusive to stochastic channel model
                 optional_pl_ind: int = 0,  # 0: standard pathloss equation,
                                            # 1: optional path loss equation
                 o2i_building_penetr_loss_ind: int = 1,  # 0: no penetration
                 # loss, 1: low-loss building penetration loss, 2: 50% low-loss,
                 # 50% high-loss building penetration loss, 3: 100% high-loss
                 # building penetration loss.
                 # UMa/UMi: 0,1,2,3;  RMa: 0,1,2; Only for indoor UT
                 o2i_car_penetr_loss_ind: int = 0,  # 0: no penetration loss,
                 # 1: basic car penetration loss, 2: 50% basic, 50% metallized
                 # car penetration loss, 3: 100% metallized car window
                 # penetration loss. Only applicable for RMa, not applicable
                 # for UMa/UMi
                 enable_near_field_effect: int = 0,  # 0: disable near field
                                                     # effect, 1: enable near
                                                     # field effect
                 enable_non_stationarity: int = 0,  # 0: disable
                                                    # non-stationarity,
                                                    # 1: enable non-stationarity
                 force_los_prob: List[float] = None,  # force LOS
                 # two elements, probability for indoor and outdoor links,
                 # each element is the probability for the corresponding link;
                 # use [0,1] for valid values, use -1 for invalid values, will
                 # be calculated using Table 7.4.1-1
                 force_ut_speed: List[float] = None,  # force UT
                 # two elements, speed for indoor and outdoor links, each
                 # element is the speed for the corresponding link; use -1 for
                 # invalid values, will be calculated using Table 7.4.1-1
                 force_indoor_ratio: float = -1,  # force indoor ratio for all
                 # links, [0,1] for all links; use -1 for invalid values,
                 # will be set based on the scenario
                 disable_pl_shadowing: int = 0,  # disable pathloss and
                 # shadowing calculation, 0: calculate PL and shadowing,
                 # 1: disable PL and shadowing calculation ()
                 disable_small_scale_fading: int = 0,  # disable small scale
                 # fading calculation, 0: calculate small scale fading,
                 # 1: disable small scale fading (fast fading = 1, only pathloss)
                 enable_per_tti_lsp: int = 0,  # enable LSP per TTI, 0: disable,
                 # 1: only update PL, O2I penetration, and shadowing, 2: update everything
                 enable_propagation_delay: int = 1  # 0: disable propagation
                 # delay in CIR generation, 1: enable propagation delay in CIR
                 # generation. Propagation delay is link-specific,
                 # distance / speed of light
                 # CIR: delay = cluster_delay + propagation_delay
                 # CFR: delay = cluster_delay + propagation_delay, FFT of CIR
                 # need to add to AODT UI
                 ):
        """Initialize system-level configuration parameters."""
        if force_los_prob is None:
            force_los_prob = [-1, -1]
        if force_ut_speed is None:
            force_ut_speed = [-1, -1]

        self.scenario = scenario
        self.isd = isd
        self.n_site = n_site
        self.n_sector_per_site = n_sector_per_site
        self.n_ut = n_ut
        self.optional_pl_ind = optional_pl_ind
        self.o2i_building_penetr_loss_ind = o2i_building_penetr_loss_ind
        self.o2i_car_penetr_loss_ind = o2i_car_penetr_loss_ind
        self.enable_near_field_effect = enable_near_field_effect
        self.enable_non_stationarity = enable_non_stationarity
        self.force_los_prob = force_los_prob
        self.force_ut_speed = force_ut_speed
        self.force_indoor_ratio = force_indoor_ratio
        self.disable_pl_shadowing = disable_pl_shadowing
        self.disable_small_scale_fading = disable_small_scale_fading
        self.enable_per_tti_lsp = enable_per_tti_lsp
        self.enable_propagation_delay = enable_propagation_delay


class LinkLevelConfig:
    """Link-level simulation configuration parameters."""

    def __init__(self,
                 fast_fading_type: int = 0,  # fast fading type, 0: AWGN,
                                             # 1: TDL, 2: CDL
                 delay_profile: str = 'A',  # delay profile, 'A' -> 'C',
                                            # TODO: add support of 'D' and 'E"
                 delay_spread: float = 30.0,  # delay spread in nanoseconds
                 mobility: np.ndarray = None,  # mobility
                 # parameters, (x, y, z), abs(mobility_direction) = speed in
                 # m/s, z = 0 per 3GPP spec
                 num_ray: int = 0,  # number of rays to add per path; default
                                    # 48 for TDL, 20 for CDL
                 cfo_hz: float = 200.0,  # carrier frequency offset in Hz
                 delay: float = 0.0):  # delay in seconds
        """Initialize link-level configuration parameters."""
        if mobility is None:
            mobility = np.array([0, 0, 0])

        self.fast_fading_type = fast_fading_type
        self.delay_profile = delay_profile
        self.delay_spread = delay_spread
        self.mobility = mobility
        self.num_ray = num_ray
        self.cfo_hz = cfo_hz
        self.delay = delay


class SimConfig:
    """Simulation configuration parameters for channel simulation."""

    def __init__(self,  # pylint: disable=too-many-arguments
                 link_sim_ind: int = 0,  # indicator for link simulation
                 center_freq_hz: float = 3e9,  # center frequency in Hz
                 bandwidth_hz: float = 100e6,  # bandwidth in Hz
                 sc_spacing_hz: float = 15e3 * 2,  # subcarrier spacing in Hz
                 fft_size: int = 4096,  # FFT size
                 n_prb: int = 273,  # number of PRBs
                 n_prbg: int = 137,  # number of PRBG
                 n_snapshot_per_slot: int = 1,  # number of channel realizations
                                                # per slot, 1 or 14
                 run_mode: int = 0,  # run mode, 0: CIR only, 1: CIR and CFR
                                     # on PRBG, 2: CIR and CFR on Sc
                                     # 3: CIR and CFR on PRBG/Sc, 4: CIR and
                                     # CFR on all N_FFT subcarriers, no PRBG
                 internal_memory_mode: int = 0,  # 0: use external memory for
                 # CIR and CFR, 1: use internal memory for CIR and external
                 # memory for CFR, 2: use internal memory for CIR/CFR
                 # if external: buffer allocated outside of channel model,
                 # read pointer to put channel
                 # if internal: buffer allocated inside of channel model,
                 # use get*() to get channel, any channel data will stil be
                 # copied to external memory if given (e.g., cir_norm_delay)
                 freq_convert_type: int = 1,  # frequency conversion type,
                 # 0: use first SC for CFR on the PRBG, 1: use center SC for
                 # CFR on the PRBG, 2: use last SC for CFR on the PRBG,
                 # 3: use average SC for CFR on the PRBG, 4: use average SC
                 # for CFR on the PRBG with removing frequency ramping; only
                 # valid if we convert CFR on SC to PRBG
                 sc_sampling: int = 1,  # whether to only calculate CFR for a
                 # subset of Scs, within a Prbg, only Scs for
                 # 0:scSampling:N_sc_Prbg-1 wil be calculated; only appliable
                 # when not using FFT and freqConvertType = 3 or 4
                 tx_sig_in: Optional[List[Array]] = None,  # input signal for transmission,
                 # TODO: not used for now
                 proc_sig_freq: int = 0,  # indicator for processing signal
                 optional_cfr_dim: int = 0,  # optional CFR dimension:
                 # 0: [nActiveUtForThisCell, n_snapshot_per_slot, nUtAnt, nBsAnt, nPrbg / nSc]
                 # 1: [nActiveUtForThisCell, n_snapshot_per_slot, nPrbg / nSc, nUtAnt, nBsAnt]
                 cpu_only_mode: int = 0  # 0: GPU mode, 1: CPU only mode
                 ):
        # frequency TODO: not used for now
        """Initialize test configuration parameters."""
        self.link_sim_ind = link_sim_ind
        self.center_freq_hz = center_freq_hz
        self.bandwidth_hz = bandwidth_hz
        self.sc_spacing_hz = sc_spacing_hz
        self.fft_size = fft_size
        self.n_prb = n_prb
        self.n_prbg = n_prbg
        self.n_snapshot_per_slot = n_snapshot_per_slot
        self.run_mode = run_mode
        self.internal_memory_mode = internal_memory_mode
        self.freq_convert_type = freq_convert_type
        self.sc_sampling = sc_sampling
        self.tx_sig_in: Optional[List[Array]] = tx_sig_in  # type: ignore
        self.proc_sig_freq = proc_sig_freq
        self.optional_cfr_dim = optional_cfr_dim
        self.cpu_only_mode = cpu_only_mode


class ExternalConfig:
    """External configuration containing cells, UTs, and antenna panels."""

    def __init__(self,
                 # CPU
                 cell_config: List[CellParam] = None,  # cell configuration per
                 # sector at start of simulation, dimension: n_site *
                 # n_sector_per_site
                 ut_config: List[UtParamCfg] = None,  # UT configuration at start
                 # of simulation, dimension: n_ut
                 ant_panel_config: List[AntPanelConfig] = None,  # pool of
                 # antenna panel configurations on GPU (cuda_array_t) or CPU
                 # (List[AntPanelConfig]), dimension: n_antenna_panel_types
                 ):
        """Initialize external configuration."""
        self.cell_config = cell_config
        self.ut_config = ut_config
        self.ant_panel_config = ant_panel_config


# Python wrapper for C++ channel model
class StatisChanModel:
    """Statistical channel model Python wrapper for C++ implementation."""

    def __init__(self,
                 sim_config: SimConfig,  # simulation configuration
                 system_level_config: SystemLevelConfig,  # system-level
                                                          # configuration
                 link_level_config: LinkLevelConfig,  # link-level configuration
                 external_config: ExternalConfig):  # external configuration
        """Initialize the statistical channel model."""
        self.sim_config = sim_config
        self.system_level_config = system_level_config
        self.link_level_config = link_level_config
        self.external_config = external_config

        # This is a Python wrapper. The actual implementation will be in the
        # C++ binding.
        if pycuphy is not None:
            self.impl = pycuphy.StatisChanModel(sim_config, system_level_config,
                                                link_level_config, external_config)
        else:
            raise ImportError("C++ binding not available")

    def run(self,  # pylint: disable=too-many-arguments
            # CPU pass
            ref_time: float = 0.0,  # reference time for CIR generation,
                                    # ttiIdx * 5e-4
            continuous_fading: int = 1,  # 0: discontinuous mode, regenerate
                                         # every TTI, 1: continuous mode, time
                                         # correlation is maintained
            active_cell: List[int] = None,  # Array of active cell IDs,
            # dimension: n_active_sector
            active_ut: Optional[List[List[int]]] = None,  # Array of active Ut,
            # List contains nSector elements, each element is a List of
            # active UTs for each sector, can be different for each
            # sector,
            # List 0: ndarray([0, 1, 2, 3]) List 1: ndarray([0, 1, 3, 5])
            ut_new_loc: np.ndarray = None,  # Array of new UT locations,
            # dimension: [n_ut, 3], update the location of all UTs,
            # indexing 0 ~ n_ut-1
            ut_new_velocity: np.ndarray = None,  # Array of new UT velocity,
            # dimension: [n_ut, 3], update the velocity of all UTs,
            # indexing 0 ~ n_ut-1, [vx, vy, vz]
            # channel, no UE/BS reconfiguration; 1: continuous mode, no
            # channel regeneration, UE/BS reconfiguration
            # Channel Impulse Response (CIR) parameters

            # GPU pass, memory is allocated outside of channel model
            # 24 is the maximum number of non-zero taps for each CIR
            cir_coe: Optional[List[np.ndarray]] = None,  # CIR coefficients in
            # sparse format, List contains nSector elements, each element is
            # a np.ndarray of dimension: [nActiveUtForThisCell,
            # n_snapshot_per_slot, nUtAnt, nBsAnt, 24]
            cir_norm_delay: Optional[List[np.ndarray]] = None,  # Normalized delay
            # of each tap, round(delay/sampling period), List contains nSector
            # elements, each element is a np.ndarray of dimension:
            # [nActiveUtForThisCell, 24]. delay is from
            # (arrival time - ref_time)
            cir_n_taps: Optional[List[np.ndarray]] = None,  # Number of non-zero
            # taps per CIR, List contains nSector elements, each element is a
            # np.ndarray of dimension: [nActiveUtForThisCell]
            # Channel Frequency Response (CFR) parameters
            cfr_sc: Optional[List[np.ndarray]] = None,  # CFR per subcarrier, List
            # contains nSector elements, each element is a np.ndarray of
            # dimension: [nActiveUtForThisCell, n_snapshot_per_slot, nUtAnt,
            # nBsAnt, fft_size]
            cfr_prbg: Optional[List[np.ndarray]] = None) -> None:  # CFR per PRB group,
            # List contains nSector elements, each element is a np.ndarray of  # noqa: E117
            # dimension: [nActiveUtForThisCell, n_snapshot_per_slot, nUtAnt,  # noqa: E117
            # nBsAnt, nPrbg]  # noqa: E117
        """
        Run channel simulation for current TTI.

        Parameters
        ----------
        activeCell : List[int], optional
            Active cell IDs
        activeUt : List[np.ndarray], optional
            Active UE IDs per cell
        ut_new_loc : np.ndarray, optional
            New UE locations
        ut_new_velocity : np.ndarray, optional
            New UE velocities
        ref_time : float, optional
            Reference time
        continuous_fading : int, optional
            Fading mode
        cir_coe : List[np.ndarray], optional
            CIR coefficients output
        cir_norm_delay : List[np.ndarray], optional
            CIR delays output
        cir_n_taps : List[np.ndarray], optional
            Number of CIR taps output
        cfr_sc : List[np.ndarray], optional
            CFR per subcarrier output
        cfr_prbg : List[np.ndarray], optional
            CFR per PRB group output
        """
        return self.impl.run(ref_time=ref_time,
                             continuous_fading=continuous_fading,
                             active_cell=active_cell,
                             active_ut=active_ut,
                             ut_new_loc=ut_new_loc,
                             ut_new_velocity=ut_new_velocity,
                             cir_coe=cir_coe,
                             cir_norm_delay=cir_norm_delay,
                             cir_n_taps=cir_n_taps,
                             cfr_sc=cfr_sc,
                             cfr_prbg=cfr_prbg)

    def reset(self) -> None:
        """Reset channel model state."""
        return self.impl.reset()

    def dump_los_nlos_stats(self,
                            lost_nlos_stats: Optional[np.ndarray] = None,  # Array of
                            # lost and nlos stats, dimension: [n_sector, n_ut]
                            ) -> None:
        """
        Dump LOS/NLOS statistics.

        Parameters
        ----------
        lost_nlos_stats : np.ndarray, optional
            Output array for LOS/NLOS statistics
        """
        return self.impl.dump_los_nlos_stats(lost_nlos_stats)

    def dump_pathloss_shadowing_stats(self,
                                      pathloss_shadowing: np.ndarray,  # Array of
                                      # pathloss and shadowing, required
                                      # If activeCell and activeUt are provided:
                                      #   dimension [activeCell.size(), activeUt.size()]
                                      # If activeCell or activeUt are empty:
                                      #   use dimension n_sector*n_site or n_ut for the empty one
                                      active_cell: Optional[np.ndarray] = None,  # Array of
                                      # active cell IDs, dimension:
                                      # n_active_sector (optional)
                                      active_ut: Optional[np.ndarray] = None,  # Array of
                                      # active UT IDs, dimension: n_active_ut (optional)
                                      ) -> None:
        """
        Dump pathloss and shadowing statistics. (negative value in dB)

        Parameters
        ----------
        pathloss_shadowing : np.ndarray
            Output array for pathloss and shadowing (required).
            Values are total loss = - (pathloss - shadow_fading) (negative value in dB)
            The sign of the shadow fading is defined so that positive SF means more
            received power at UT than predicted by the path loss model
        active_cell : np.ndarray, optional
            Active cell IDs. If not provided, dumps all cells.
        active_ut : np.ndarray, optional
            Active UE IDs. If not provided, dumps all UEs.
        """
        return self.impl.dump_pathloss_shadowing_stats(pathloss_shadowing, active_cell, active_ut)

    def get_cir(self,
                cir_coe: Optional[List[np.ndarray]] = None,  # CIR coefficients in
                # sparse format, List contains nSector elements, each element
                # is a np.ndarray of dimension: [nActiveUtForThisCell,
                # n_snapshot_per_slot, nUtAnt, nBsAnt, 24]
                cir_norm_delay: Optional[List[np.ndarray]] = None,  # Normalized
                # delay of each tap, round(delay/sampling period), List
                # contains nSector elements, each element is a np.ndarray of
                # dimension: [nActiveUtForThisCell, 24]
                cir_n_taps: Optional[List[np.ndarray]] = None,  # Number of
                # non-zero taps per CIR, List contains nSector elements, each
                # element is a np.ndarray of dimension: [nActiveUtForThisCell]
                ) -> None:
        """
        Get CIR data.

        Parameters
        ----------
        cir_coe : List[np.ndarray], optional
            CIR coefficients output
        cir_norm_delay : List[np.ndarray], optional
            CIR delays output
        cir_n_taps : List[np.ndarray], optional
            Number of CIR taps output
        """
        return self.impl.get_cir(cir_coe, cir_norm_delay, cir_n_taps)

    def get_cfr(self,
                cfr_sc: Optional[List[np.ndarray]] = None,  # CFR per subcarrier,
                # List contains nSector elements, each element is a np.ndarray
                # of dimension: [nActiveUtForThisCell, n_snapshot_per_slot,
                # nUtAnt, nBsAnt, fft_size]
                cfr_prbg: Optional[List[np.ndarray]] = None,  # CFR per PRB group,
                # List contains nSector elements, each element is a np.ndarray
                # of dimension: [nActiveUtForThisCell, n_snapshot_per_slot,
                # nUtAnt, nBsAnt, nPrbg]
                ) -> None:
        """
        Get CFR data.

        Parameters
        ----------
        cfr_sc : List[np.ndarray], optional
            CFR per subcarrier output
        cfr_prbg : List[np.ndarray], optional
            CFR per PRB group output
        """
        return self.impl.get_cfr(cfr_sc, cfr_prbg)
