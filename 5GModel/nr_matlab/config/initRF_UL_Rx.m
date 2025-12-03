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

function RF = initRF_UL_Rx(SysPar)

    % LNA
    SysPar.RF.UL_Rx_a1 = 10^(SysPar.RF.UL_Rx_gain_dB/20);
    SysPar.RF.UL_Rx_a3 = -500*SysPar.RF.UL_Rx_a1/SysPar.RF.ref_impedance_ohm/10^(SysPar.RF.UL_Rx_IIP3_dBm/10);
    SysPar.RF.UL_Rx_Th = sqrt(-SysPar.RF.UL_Rx_a1/3/SysPar.RF.UL_Rx_a3);

    % IQ imbalance
    gain_error      = 10^(SysPar.RF.UL_Rx_IQ_imblance_gain_dB/20);
    epsilon         = (1-gain_error)/(1+gain_error); % note that gain error is calculated as 20log((1+epsilon)/(1-epsilon)), where (1+epsilon) and (1-epsilon) are the gains for I/Q branches 
    theta           = SysPar.RF.UL_Rx_IQ_imblance_phase_degree*pi/180;    
    SysPar.RF.UL_Rx_Gain1 = cos(theta/2) + 1i*epsilon*sin(theta/2);
    SysPar.RF.UL_Rx_Gain2 = epsilon*cos(theta/2) - 1i*sin(theta/2);

    % Phase noise
    PN_level = SysPar.RF.UL_Rx_PN_level_offset_dB + SysPar.RF.UL_Rx_PN_spectral_mask_power_dBcPerHz;
    SysPar.RF.UL_Rx_obj_phase_noise = comm.PhaseNoise(PN_level,SysPar.RF.UL_Rx_PN_spectral_mask_freqOffset_Hz,SysPar.carrier.f_samp);
%     SamplingRate        = SysPar.carrier.f_samp;
%     PSD_f_Hz            = SysPar.RF.PN_spectral_mask_freqOffset_Hz; %[  0  1e3  1e4  1e5  1e6  10e6];
%     PSD_Val_dBcPerHz    = SysPar.RF.PN_spectral_mask_power_dBcPerHz; %[-65  -65  -95 -115 -125  -125];
%     NumBins             = SamplingRate/min(PSD_f_Hz(PSD_f_Hz>0)); % min(PSD_f_Hz(PSD_f_Hz>0)): find the first nonzero freq. offset    
%     if PSD_f_Hz(1)~= 0
%       PSD_f_Hz = [0, PSD_f_Hz];
%       PSD_Val_dBcPerHz = [max(PSD_Val_dBcPerHz),PSD_Val_dBcPerHz];
%     end
%     if max(PSD_f_Hz>SamplingRate/2)
%         error('Sampling rate should be larger than the maximun PN frequency offset!')
%     end
%     PSD_f_Hz            = [PSD_f_Hz,SamplingRate/2];
%     PSD_Val_dBcPerHz    = [PSD_Val_dBcPerHz, min(PSD_Val_dBcPerHz)];
%     % defining the frequency vector
%     FreqVecHz           = [0:NumBins/2]/NumBins*SamplingRate;
%     DELTA_F             = SamplingRate/NumBins;
%     slope               = (PSD_Val_dBcPerHz(2:end) - PSD_Val_dBcPerHz(1:end-1))./...
%                             (log10(PSD_f_Hz(2:end)) - log10(PSD_f_Hz(1:end-1)+eps));      
%     % interpolating the phase noise psd values
%     psd_ssb_dB          = -Inf*ones(1,NumBins/2+1); % from [0:NumBins/2]
%     psd_ssb_f_Hz        = [0:NumBins/2]*DELTA_F;
%     for ii=1:length(PSD_f_Hz)-1
%         if PSD_f_Hz(ii) <  DELTA_F
%             fl_idx = 1;
%         else
%             fl_idx  = floor(PSD_f_Hz(ii)/DELTA_F)+2;  
%         end
%         fr_idx  = floor(PSD_f_Hz(ii+1)/DELTA_F)+1;
%         fvec = [];
%         pvec = [];
%         fvec    = FreqVecHz(fl_idx):DELTA_F:FreqVecHz(fr_idx);
%         pvec    = slope(ii)*log10(fvec+eps) + PSD_Val_dBcPerHz(ii) - slope(ii)*log10(PSD_f_Hz(ii)+eps);
%         psd_ssb_dB(fl_idx:fr_idx) = pvec;
%     end
%     psd_ssb_dB(1)=PSD_Val_dBcPerHz(1);
%     % form the full vector [-NumBins/2:NumBins/2-1 ]/NumBins*SamplingRate
%     psd_dB                  = -Inf*ones(1,NumBins);
%     psd_dB([-NumBins/2:-1]+NumBins/2+1) = psd_ssb_dB([NumBins/2+1:-1:2]);
%     psd_dB([0:NumBins/2-1]+NumBins/2+1) = psd_ssb_dB(1:NumBins/2);
%     SysPar.RF.PN_psd_linear = 10.^(psd_dB/20).';

    RF = SysPar.RF;
end