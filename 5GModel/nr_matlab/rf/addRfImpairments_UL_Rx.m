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

% function to add RF impairments including NF + nonlinearity + CFO + PN + IQ imbalance + DC offset
function out = addRfImpairments_UL_Rx(SysPar, Chan, chSamp)
    RF = SysPar.RF;
    [lenSamp, numRxAnt] = size(chSamp);
    SamplingRate        = SysPar.carrier.f_samp;

    % scale the original noise-free Rx signal based on noise floor and SNR at Rx antenna
    input_SNR_dB = Chan.SNR;
    ref_BW_Hz_dB = pow2db(SamplingRate);%pow2db(SysPar.carrier.N_grid_size_mu*2^SysPar.carrier.mu*15e3*12);
    scale = sqrt(10^(((RF.noise_floor_dBmPerHz-30+ref_BW_Hz_dB+input_SNR_dB)/10))*2*RF.ref_impedance_ohm); % -30 is for dBm to dB conversion
    chSamp_scaled = scale*chSamp;
    
    % add thermal noise
    noise_std = sqrt(10^(((RF.noise_floor_dBmPerHz-30+ref_BW_Hz_dB+RF.UL_Rx_tot_NF_dB)/10))*2*RF.ref_impedance_ohm);
    thermal_noise = noise_std*sqrt(2)/2*(randn(size(chSamp))+1i*randn(size(chSamp)));
    chSamp_noisy = chSamp_scaled + thermal_noise;

    % LNA nonlinearity
    chSamp_nl = zeros(size(chSamp));
    for idxAnt = 1:numRxAnt
        idxGeTh = logical(abs(chSamp_noisy(:,idxAnt))>=RF.UL_Rx_Th);
        chSamp_nl(idxGeTh,idxAnt) = RF.UL_Rx_Th./abs(chSamp_noisy(idxGeTh,idxAnt)).*chSamp_noisy(idxGeTh,idxAnt);
        idxLtTh = ~idxGeTh;
        tmp = chSamp_noisy(idxLtTh,idxAnt);
        chSamp_nl(idxLtTh,idxAnt) = RF.UL_Rx_a1*tmp + RF.UL_Rx_a3*tmp.*abs(tmp).^2;
    end

    % CFO is considered in Channel.m
%     % CFO
%     CFO = Chan.CFO;
%     T_samp = Chan.T_samp;
%     CFOseq = exp(1j*2*pi*([0:lenSamp-1]*T_samp*CFO));
%     chSamp_cfo = chSamp_nl.*repmat(CFOseq(:), [1, Chan.Nout]);

    % phase noise
    chSamp_pn = SysPar.RF.UL_Rx_obj_phase_noise(chSamp_nl);
    
    % IQ imbalance
    chSamp_ib = RF.UL_Rx_Gain1*chSamp_pn + RF.UL_Rx_Gain2*conj(chSamp_pn);

    % DC offset
    chSamp_dc = chSamp_ib + (RF.UL_Rx_DC_offset_real_volt + 1i*RF.UL_Rx_DC_offset_imag_volt);

    % DVGA
%     mean_rx_power_this_slot = mean(abs(chSamp_dc).^2,'all');
%     gain_dvga = sqrt(1/mean_rx_power_this_slot);
%     out = gain_dvga*chSamp_dc;

%     out = chSamp_dc/scale/RF.UL_Rx_a1;

    baseband_scale = 10^(SysPar.RF.UL_Rx_baseband_gain_dB/20.0);
    out = chSamp_dc*baseband_scale;

end