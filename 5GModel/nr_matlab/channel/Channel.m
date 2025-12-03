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

function [rxSampSum, Chan] = Channel(txSamp, Chan, SimCtrl, carrier)
%
% function [rxSamp, Chan] = Channel(txSamp, Chan)
%
% This function simulates mobile channel including multipath, doppler, ...
% antenna correlation, path delay, noise, CFO 
%
% Input:    txSamp: Input time domain samples to channel
%           Chan: structure for channel
%
% Output:   rxSamp: Output time domain samples from channel
%           Chan: structure for channel
%

gen_ChanMatrix_FD_oneSlot_flag = SimCtrl.alg.enable_get_genie_meas || SimCtrl.enable_get_genie_channel_matrix; % flag to generate genie channel. Note this will increase memory usage
[lenSamp, ~] = size(txSamp);
currentTime = (SimCtrl.global_idxSlot - 1) * 0.001 / (2^carrier.mu);  % time stamp of the first sample
rxSampSum = zeros(lenSamp, Chan.Nout);                
switch Chan.type
    case {'AWGN', 'P2P'}
        if SimCtrl.timeDomainSim
            rxSampSum = txSamp*Chan.chanMatrix; % Chan.chanMatrix dim: [num Tx, num Rx]
            rxSampSum = rxSampSum * Chan.gain;
            if gen_ChanMatrix_FD_oneSlot_flag
                Chan.chanMatrix_FD_oneSlot = permute(repmat(Chan.gain * Chan.chanMatrix,[1,1,carrier.N_symb_slot, carrier.N_sc]),[4,3,2,1]); % collect genie channel in FD
            end
        else
            dim = size(txSamp);
            if Chan.Nin > 1 % multiple input antennas
                if length(dim) == 3 % data
                    txSamp1 = reshape(txSamp, dim(1)*dim(2), dim(3));
                    rxSampSum1 = txSamp1*Chan.chanMatrix;
                    rxSampSum = reshape(rxSampSum1, dim(1), dim(2), Chan.Nout);
                else % preamble
                    rxSampSum = txSamp*Chan.chanMatrix;
                end
            else % single input antenna
                if length(dim) == 2 && dim(2) > 1% data
                    txSamp1 = reshape(txSamp, dim(1)*dim(2), 1);
                    rxSampSum1 = txSamp1*Chan.chanMatrix;
                    rxSampSum = reshape(rxSampSum1, dim(1), dim(2), Chan.Nout);
                else % preamble
                    rxSampSum = txSamp*Chan.chanMatrix;
                end                
            end
            rxSampSum = rxSampSum * Chan.gain;
            if abs(Chan.delay) > 0 || abs(Chan.CFO) > 0
                error('To simulate non-zero Chan.CFO or Chan.delay, SimCtrl.timeDomainSim must be set to 1.\n')
            end
            if gen_ChanMatrix_FD_oneSlot_flag
                Chan.chanMatrix_FD_oneSlot = permute(repmat(Chan.gain * Chan.chanMatrix,[1,1,carrier.N_symb_slot, carrier.N_sc]),[4,3,2,1]); % collect genie channel in FD
            end
            return
        end        
    case {'TDL', 'TDLA30-5-Low', 'TDLA30-10-Low', 'TDLB100-400-Low', 'TDLC300-100-Low','TDLA30-10-UplinkMedium', 'TDLB100-400-UplinkMedium', 'TDLC300-100-UplinkMedium'}
        if SimCtrl.timeDomainSim
            if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
                if SimCtrl.enable_get_genie_channel_matrix
                    [rxSampSum, pathGains] = Chan.tdl(txSamp); % use MATLAB 5G toolbox
                    pathFilters = getPathFilters(Chan.tdl);
                    kappa = 64; 
                    lenCP1 = (144*2^(-carrier.mu))*kappa*carrier.T_c/carrier.T_samp;
                    delay_samp = round(Chan.delay/Chan.T_samp);
                    pathFilters((lenCP1-delay_samp+1):end,:) = 0;
                    % refer to https://www.mathworks.com/help/5g/ref/nrperfectchannelestimate.html
                    % Note that pathGains are time-variant on sample basis instead of on OFDM symbol basis
                    if gen_ChanMatrix_FD_oneSlot_flag
                        Chan.chanMatrix_FD_oneSlot = nrPerfectChannelEstimate(pathGains,pathFilters,carrier.N_BWP_size,carrier.delta_f/1e3,0,0,'CyclicPrefixFraction',1, 'Nfft',carrier.Nfft); % Chan.chanMatrix dim: [num subcarriers, num of sym, num Rx ant., num Tx ant]  
                        nSc = carrier.N_BWP_size*12;
                        phase_ramping_vector = exp(-1j*2*pi*delay_samp*[-nSc/2:nSc/2-1]/carrier.Nfft).'; % [-nSc/2:nSc/2-1] [-nSc/2:-1, 1:nSc/2]
                        Chan.chanMatrix_FD_oneSlot = Chan.gain * Chan.chanMatrix_FD_oneSlot .*repmat(phase_ramping_vector,[1,size(Chan.chanMatrix_FD_oneSlot,[2,3])]);
                    end
                else
                    rxSampSum = Chan.tdl(txSamp); % use MATLAB 5G toolbox
                end
            elseif strcmp(Chan.model_source, 'sionna')
                error('To simulate fading channel with sionna, SimCtrl.timeDomainSim must be set to 0.\n')
            elseif strcmp(Chan.model_source, 'custom')
                rxSampSum = fadingChan(txSamp, Chan, SimCtrl, currentTime); % Chan.chanMatrix dim:
            else
                error("Undefined!")
            end
            rxSampSum = rxSampSum * Chan.gain;
        else % in the freq domain
            if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
                error('To simulate fading channel with MATLAB 5G toolbox, SimCtrl.timeDomainSim must be set to 1.\n')
            elseif strcmp(Chan.model_source, 'sionna') 
                txSamp_5d = permute(txSamp, [5,4,3,2,1]); % dim of txSamp_5d: num_batches x num_transmitters x num_tx_ant x num_OFDM_sym x num_REs_carrier
                eval("txSamp_tensor = py.tensorflow.convert_to_tensor(py.numpy.array(txSamp_5d).astype('complex64'));");
                out = Chan.sionna_chan(txSamp_tensor);
                eval("rxSampSum = permute(double(py.numpy.array(out{1})),[5,4,3,2,1]);");
                rxSampSum = rxSampSum * Chan.gain;
                if abs(Chan.delay) > 0 || abs(Chan.CFO) > 0
                    error('To simulate non-zero Chan.CFO or Chan.delay, SimCtrl.timeDomainSim must be set to 1.\n')
                end
                if gen_ChanMatrix_FD_oneSlot_flag
                    eval("Chan.chanMatrix_FD_oneSlot = Chan.gain*permute(double(py.numpy.array(out{2})),[7,6,3,5,2,4,1]);"); % collect genie channel in FD, dim: num_REs_carrier x num_OFDM_sym x num_rx_ant x num_tx_ant x num_receivers x num_transmitters x num_batches            
            
                end
            end
        end
    case {'CDL'}
        if SimCtrl.timeDomainSim
            if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
                if SimCtrl.enable_get_genie_channel_matrix
                    [rxSampSum, pathGains] = Chan.cdl(txSamp); % use MATLAB 5G toolbox
                    pathFilters = getPathFilters(Chan.cdl);
                    kappa = 64; 
                    lenCP1 = (144*2^(-carrier.mu))*kappa*carrier.T_c/carrier.T_samp;
                    delay_samp = round(Chan.delay/Chan.T_samp);
                    pathFilters((lenCP1-delay_samp+1):end,:) = 0;
                    if strcmp(Chan.CDL_updateRateType, 'perSymbol')
                        NsamplesPersym = size(txSamp,1)/size(pathGains,1);
                        
                    end
                    % refer to https://www.mathworks.com/help/5g/ref/nrperfectchannelestimate.html
                    % Note that pathGains are time-variant on sample basis instead of on OFDM symbol basis
                    if gen_ChanMatrix_FD_oneSlot_flag
                        Chan.chanMatrix_FD_oneSlot = nrPerfectChannelEstimate(pathGains,pathFilters,carrier.N_BWP_size,carrier.delta_f/1e3,0,0,'CyclicPrefixFraction',1, 'Nfft',carrier.Nfft); % Chan.chanMatrix dim: [num subcarriers, num of sym, num Rx ant., num Tx ant]  
                        nSc = carrier.N_BWP_size*12;
                        phase_ramping_vector = exp(-1j*2*pi*delay_samp*[-nSc/2:nSc/2-1]/carrier.Nfft).'; % [-nSc/2:nSc/2-1] [-nSc/2:-1, 1:nSc/2]
                        Chan.chanMatrix_FD_oneSlot = Chan.gain * Chan.chanMatrix_FD_oneSlot .*repmat(phase_ramping_vector,[1,size(Chan.chanMatrix_FD_oneSlot,[2,3])]);
                    end
                else
                    rxSampSum = Chan.cdl(txSamp); % use MATLAB 5G toolbox
                end
            elseif strcmp(Chan.model_source, 'custom')
                rxSampSum = fadingChan(txSamp, Chan, SimCtrl, currentTime);
            else
                error("To simulate CDL fading channel in the time domain, you need to set Chan.model_source to 'MATLAB5Gtoolbox' or 'custom'! ")
            end
        else % in the freq domain
            if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
                error('To simulate CDL fading channel with MATLAB 5G toolbox or custom models, SimCtrl.timeDomainSim must be set to 1.\n')
            elseif strcmp(Chan.model_source, 'sionna') 
                txSamp_5d = permute(txSamp, [5,4,3,2,1]); % dim of txSamp_5d: num_batches x num_transmitters x num_tx_ant x num_OFDM_sym x num_REs_carrier
                eval("txSamp_tensor = py.tensorflow.convert_to_tensor(py.numpy.array(txSamp_5d).astype('complex64'));");
                out = Chan.sionna_chan(txSamp_tensor);
                eval("rxSampSum = permute(double(py.numpy.array(out{1})),[5,4,3,2,1]);");
                rxSampSum = rxSampSum * Chan.gain;
                if abs(Chan.delay) > 0 || abs(Chan.CFO) > 0
                    error('To simulate non-zero Chan.CFO or Chan.delay, SimCtrl.timeDomainSim must be set to 1.\n')
                end
                if gen_ChanMatrix_FD_oneSlot_flag
                    eval("Chan.chanMatrix_FD_oneSlot = Chan.gain*permute(double(py.numpy.array(out{2})),[7,6,3,5,2,4,1]);"); % collect genie channel in FD, dim: num_REs_carrier x num_OFDM_sym x num_rx_ant x num_tx_ant x num_receivers x num_transmitters x num_batches
                end
            end
        end 
    case {'UMi', 'UMa', 'RMa'}
        if (~SimCtrl.timeDomainSim) && strcmp(Chan.model_source, 'sionna') 
            txSamp_5d = permute(txSamp, [5,4,3,2,1]); % dim of txSamp_5d: num_batches x num_transmitters x num_tx_ant x num_OFDM_sym x num_REs_carrier
            eval("txSamp_tensor = py.tensorflow.convert_to_tensor(py.numpy.array(txSamp_5d).astype('complex64'));");
            out = Chan.sionna_chan(txSamp_tensor);
            eval("rxSampSum = permute(double(py.numpy.array(out{1})),[5,4,3,2,1]);");
            rxSampSum = rxSampSum * Chan.gain;
            if abs(Chan.delay) > 0 || abs(Chan.CFO) > 0
                error('To simulate non-zero Chan.CFO or Chan.delay, SimCtrl.timeDomainSim must be set to 1.\n')
            end
            eval("Chan.chanMatrix_FD_oneSlot = Chan.gain*permute(double(py.numpy.array(out{2})),[7,6,3,5,2,4,1]);"); % collect genie channel in FD, dim: num_REs_carrier x num_OFDM_sym x num_rx_ant x num_tx_ant x num_receivers x num_transmitters x num_batches            
        else
            error("To use UMi/UMa/RMa channel models, you need to disable SimCtrl.timeDomainSim and set Chan.model_source to 'sionna'.")
        end
    otherwise
        error('channel type is not supported ...\n');
end

if SimCtrl.timeDomainSim
    % add delay
    delay_samp = round(Chan.delay/Chan.T_samp);
    rxSampSum = circshift(rxSampSum, delay_samp, 1);
    
    % add CFO
    CFO = Chan.CFO;
    T_samp = Chan.T_samp;
    CFOseq = exp(1j*2*pi*([0:lenSamp-1]*T_samp*CFO));
    rxSampSum = rxSampSum.*repmat(CFOseq(:), [1, Chan.Nout]);
end


return
