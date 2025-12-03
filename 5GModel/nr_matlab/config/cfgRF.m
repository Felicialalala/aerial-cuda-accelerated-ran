% SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
% SPDX-License-Identifier: Apache-2.0
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function RF = cfgRF

    RF.ref_impedance_ohm = 50;
    RF.noise_floor_dBmPerHz = -174;

    %% UL Rx
    RF.UL_Rx_tot_NF_dB = 5; % total NF of the entire RF chain, including FE insertion loss, LNA NF, etc.

    % refer to the LNA used by ArrayComm
    RF.UL_Rx_gain_dB = 33.5;
    RF.UL_Rx_IIP3_dBm = -3.5;

    RF.UL_Rx_IQ_imblance_gain_dB = 0.5;
    RF.UL_Rx_IQ_imblance_phase_degree = 5;

    RF.UL_Rx_PN_spectral_mask_freqOffset_Hz = [1e3, 1e4, 1e5, 1e6];
    RF.UL_Rx_PN_spectral_mask_power_dBcPerHz = [-90, -90, -90, -140]; % https://www.analog.com/en/technical-articles/transceiver-phase-noise-teardown-informs-performance-capability.html
    RF.UL_Rx_PN_level_offset_dB = 0;

    RF.UL_Rx_DC_offset_real_volt = 0;
    RF.UL_Rx_DC_offset_imag_volt = 0;

    RF.UL_Rx_baseband_gain_dB = 15.18; % LNA_gain + baseband_gain = 48.68dB = ul_configured_gain used by O-RU

    % Note: CFO config is in cfgChan

    %% DL Tx
    RF.DL_Tx_signal_distortion_var_dB = -30;

    RF.DL_Tx_gain_dB = 25;
    RF.DL_Tx_IIP3_dBm = 20;

    RF.DL_Tx_IQ_imblance_gain_dB = 0.5;
    RF.DL_Tx_IQ_imblance_phase_degree = 5;

    RF.DL_Tx_PN_spectral_mask_freqOffset_Hz = [1e3, 1e4, 1e5, 1e6];
    RF.DL_Tx_PN_spectral_mask_power_dBcPerHz = [-90, -90, -90, -140]; % https://www.analog.com/en/technical-articles/transceiver-phase-noise-teardown-informs-performance-capability.html
    RF.DL_Tx_PN_level_offset_dB = 0;

    RF.DL_Tx_DC_offset_real_volt = 0;
    RF.DL_Tx_DC_offset_imag_volt = 0;
end