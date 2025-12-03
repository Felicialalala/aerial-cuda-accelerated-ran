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

function RF = initRF_DL_Tx(SysPar)
    RF = SysPar.RF;
    
    % PA
    RF.DL_Tx_a1 = 10^(SysPar.RF.DL_Tx_gain_dB/20);
    RF.DL_Tx_a3 = -500*RF.DL_Tx_a1/RF.ref_impedance_ohm/10^(RF.DL_Tx_IIP3_dBm/10);
    RF.DL_Tx_Th = sqrt(-RF.DL_Tx_a1/3/RF.DL_Tx_a3);

    % IQ imbalance
    gain_error      = 10^(RF.DL_Tx_IQ_imblance_gain_dB/20);
    epsilon         = (1-gain_error)/(1+gain_error); % note that gain error is calculated as 20log((1+epsilon)/(1-epsilon)), where (1+epsilon) and (1-epsilon) are the gains for I/Q branches 
    theta           = RF.DL_Tx_IQ_imblance_phase_degree*pi/180;    
    RF.DL_Tx_Gain1 = cos(theta/2) + 1i*epsilon*sin(theta/2);
    RF.DL_Tx_Gain2 = epsilon*cos(theta/2) - 1i*sin(theta/2);

    % Phase noise
    PN_level = RF.DL_Tx_PN_level_offset_dB + RF.DL_Tx_PN_spectral_mask_power_dBcPerHz;
    RF.DL_Tx_obj_phase_noise = comm.PhaseNoise(PN_level,RF.DL_Tx_PN_spectral_mask_freqOffset_Hz,SysPar.carrier.f_samp);
