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

% function to add RF impairments
function out = addRfImpairments_DL_Tx(SysPar, txSamp)
    RF = SysPar.RF;
    [lenSamp, numTxAnt] = size(txSamp);
    SamplingRate        = SysPar.carrier.f_samp;
    % add signal distortion
    noise_std = sqrt(db2pow(RF.DL_Tx_signal_distortion_var_dB));
    distortion_realization = noise_std*sqrt(2)/2*(randn(size(txSamp))+1i*randn(size(txSamp)));
    txSamp_distorted = txSamp + distortion_realization;

    % scale the tx sample to the level of 20dB lower than PA IIP3 (20dB: 10dB PAPR + 10 difference between 1dB compression point and IIP3 )
    txSampPowerBeforePA_dB = RF.DL_Tx_IIP3_dBm - 20 -30; % -30 is for dBm to dB conversion
    scale = sqrt(db2pow(txSampPowerBeforePA_dB)*2*RF.ref_impedance_ohm);
    txSamp_scaled = scale*txSamp_distorted;
    
    % IQ imbalance
    txSamp_ib = RF.DL_Tx_Gain1*txSamp_scaled + RF.DL_Tx_Gain2*conj(txSamp_scaled);

    % Phase noise
    txSamp_pn = SysPar.RF.DL_Tx_obj_phase_noise(txSamp_ib);

    % CFO is considered in Channel.m
%     % CFO
%     CFO = Chan.CFO;
%     T_samp = 1/SamplingRate;
%     CFOseq = exp(1j*2*pi*([0:lenSamp-1]*T_samp*CFO));
%     txSamp_cfo = txSamp_pn.*repmat(CFOseq(:), [1, numTxAnt]);

    % nonlinearity    
    txSamp_nl = zeros(size(txSamp));
    for idxAnt = 1:numTxAnt
        idxGeTh = logical(abs(txSamp_pn(:,idxAnt))>=RF.DL_Tx_Th);
        txSamp_nl(idxGeTh,idxAnt) = RF.DL_Tx_Th./abs(txSamp_pn(idxGeTh,idxAnt)).*txSamp_pn(idxGeTh,idxAnt);
        idxLtTh = ~idxGeTh;
        tmp = txSamp_pn(idxLtTh,idxAnt);
        txSamp_nl(idxLtTh,idxAnt) = RF.DL_Tx_a1*tmp + RF.DL_Tx_a3*tmp.*abs(tmp).^2;
    end

    % scale back to digital domain
    out = 1/scale/RF.DL_Tx_a1*txSamp_nl;

end