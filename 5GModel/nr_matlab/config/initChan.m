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

function Chan = initChan(SysPar, idxUE, DLUL, isInterferenceChan)
% function UE = initChan(SysPar)
%
% This function initializes configuations for a single channel. 
%
% Input:    SysPar: structure with all configurations
%
% Output:   Chan: structure for a single Channel
%
if nargin<3
    error('Not enough arguments!')
elseif nargin<4
    isInterferenceChan = 0;
end

if isInterferenceChan
    SysPar.Chan = SysPar.Interf_Chan;
end

if length(SysPar.Chan) < idxUE
    Chan = SysPar.Chan{1};
else
    Chan = SysPar.Chan{idxUE};
end

Chan.T_samp = SysPar.carrier.T_samp;
Chan.f_samp = SysPar.carrier.f_samp;

if strcmp(DLUL, 'UL')
    Chan.link_direction = 'Uplink';
    Chan.Nin = SysPar.SimCtrl.UE{idxUE}.Nant;
    Chan.Nout = SysPar.SimCtrl.gNB.Nant;
else
    Chan.link_direction = 'Downlink';
    Chan.Nin = SysPar.SimCtrl.gNB.Nant;
    Chan.Nout = SysPar.SimCtrl.UE{idxUE}.Nant;
end

% create channel matrix or channel model object from 5G toolbox
% for 5G toolbox, create the channnel object using Chan.link_direction
% otherwise, the channel matrix is swapped in the generation process. ChanMatrix dimension is [nTxAnt, nRxAnt, lenFir, Nbatch]
switch Chan.type
    case 'P2P' % direct port to port connection
        Chan.chanMatrix = zeros(Chan.Nin, Chan.Nout);
        minDim = min([Chan.Nin, Chan.Nout]);
        Chan.chanMatrix(1:minDim, 1:minDim) = diag(ones(1, minDim));
    case 'AWGN' % with random phase
        Chan.chanMatrix = exp(1j*2*pi*rand(Chan.Nin, Chan.Nout));
    case 'TDLA30-5-Low' 
        if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
            tdl = nrTDLChannel; % use MATLAB 5G toolbox
            if Chan.simplifiedDelayProfile
                tdl.DelayProfile = 'TDLA30'; % we should use the simplified TDL channel model defined in 38.141 instead of the original one defined in 38.901
            else
                tdl.DelayProfile = 'TDL-A';
                tdl.DelaySpread = 30e-9;
            end
            tdl.MaximumDopplerShift = 5;
            tdl.MIMOCorrelation = 'Low';
            tdl.Polarization = "Co-Polar";
            tdl.TransmissionDirection = Chan.link_direction;
            tdl.NumTransmitAntennas = Chan.Nin;
            tdl.NumReceiveAntennas = Chan.Nout;
            tdl.SampleRate = Chan.f_samp;
            tdl.NormalizeChannelOutputs = false;
            tdl.RandomStream = 'Global Stream';
            Chan.tdl = tdl;
        elseif strcmp(Chan.model_source, 'custom')
            Chan = genTDL(Chan, SysPar);
        else
            error('Undefined channel model given the model_source!')
        end  
    case 'TDLA30-10-Low' 
        if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
            tdl = nrTDLChannel; % use MATLAB 5G toolbox
            if Chan.simplifiedDelayProfile
                tdl.DelayProfile = 'TDLA30';
            else
                tdl.DelayProfile = 'TDL-A';
                tdl.DelaySpread = 30e-9;
            end
            tdl.MaximumDopplerShift = 10;
            tdl.MIMOCorrelation = 'Low';
            tdl.Polarization = "Co-Polar";
            tdl.TransmissionDirection = Chan.link_direction;
            tdl.NumTransmitAntennas = Chan.Nin;
            tdl.NumReceiveAntennas = Chan.Nout;
            tdl.SampleRate = Chan.f_samp;
            tdl.NormalizeChannelOutputs = false;
            tdl.RandomStream = 'Global Stream';
            Chan.tdl = tdl;
        elseif strcmp(Chan.model_source, 'custom')
            Chan = genTDL(Chan, SysPar);
        else
            error('Undefined channel model given the model_source!')
        end   
    case 'TDLB100-400-Low' 
        if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
            tdl = nrTDLChannel; % use MATLAB 5G toolbox
            if Chan.simplifiedDelayProfile
                tdl.DelayProfile = 'TDLB100';
            else
                tdl.DelayProfile = 'TDL-B';
                tdl.DelaySpread = 100e-9;
            end
            tdl.MaximumDopplerShift = 400;
            tdl.MIMOCorrelation = 'Low';
            tdl.Polarization = "Co-Polar";
            tdl.TransmissionDirection = Chan.link_direction;
            tdl.NumTransmitAntennas = Chan.Nin;
            tdl.NumReceiveAntennas = Chan.Nout;
            tdl.SampleRate = Chan.f_samp;
            tdl.NormalizeChannelOutputs = false;
            tdl.RandomStream = 'Global Stream';
            Chan.tdl = tdl;
        elseif strcmp(Chan.model_source, 'custom')
            Chan = genTDL(Chan, SysPar);
        else
            error('Undefined channel model given the model_source!')
        end          
    case 'TDLC300-100-Low' 
        if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
            tdl = nrTDLChannel; % use MATLAB 5G toolbox
            if Chan.simplifiedDelayProfile
            tdl.DelayProfile = 'TDLC300';
            else
                tdl.DelayProfile = 'TDL-C';
                tdl.DelaySpread = 300e-9;
            end
            tdl.MaximumDopplerShift = 100;
            tdl.MIMOCorrelation = 'Low';
            tdl.Polarization = "Co-Polar";
            tdl.TransmissionDirection = Chan.link_direction;
            tdl.NumTransmitAntennas = Chan.Nin;
            tdl.NumReceiveAntennas = Chan.Nout;
            tdl.SampleRate = Chan.f_samp;
            tdl.NormalizeChannelOutputs = false;
            tdl.RandomStream = 'Global Stream';
            Chan.tdl = tdl;
        elseif strcmp(Chan.model_source, 'custom')
            Chan = genTDL(Chan, SysPar);
        else
            error('Undefined channel model given the model_source!')
        end
    case 'TDLA30-10-UplinkMedium' 
        if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
            tdl = nrTDLChannel; % use MATLAB 5G toolbox
            if Chan.simplifiedDelayProfile
                tdl.DelayProfile = 'TDLA30';
            else
                tdl.DelayProfile = 'TDL-A';
                tdl.DelaySpread = 30e-9;
            end
            tdl.MaximumDopplerShift = 10;
            tdl.MIMOCorrelation = 'UplinkMedium';
            tdl.Polarization = "Co-Polar";
            tdl.TransmissionDirection = Chan.link_direction;
            tdl.NumTransmitAntennas = Chan.Nin;
            tdl.NumReceiveAntennas = Chan.Nout;
            tdl.SampleRate = Chan.f_samp;
            tdl.NormalizeChannelOutputs = false;
            tdl.RandomStream = 'Global Stream';
            Chan.tdl = tdl;
        elseif strcmp(Chan.model_source, 'custom')
            Chan = genTDL(Chan, SysPar);
        else
            error('Undefined channel model given the model_source!')
        end   
    case 'TDLB100-400-UplinkMedium' 
        if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
            tdl = nrTDLChannel; % use MATLAB 5G toolbox
            if Chan.simplifiedDelayProfile
                tdl.DelayProfile = 'TDLB100';
            else
                tdl.DelayProfile = 'TDL-B';
                tdl.DelaySpread = 100e-9;
            end
            tdl.MaximumDopplerShift = 400;
            tdl.MIMOCorrelation = 'UplinkMedium';
            tdl.Polarization = "Co-Polar";
            tdl.TransmissionDirection = Chan.link_direction;
            tdl.NumTransmitAntennas = Chan.Nin;
            tdl.NumReceiveAntennas = Chan.Nout;
            tdl.SampleRate = Chan.f_samp;
            tdl.NormalizeChannelOutputs = false;
            tdl.RandomStream = 'Global Stream';
            Chan.tdl = tdl;
        elseif strcmp(Chan.model_source, 'custom')
            Chan = genTDL(Chan, SysPar);
        else
            error('Undefined channel model given the model_source!')
        end              
    case 'TDLC300-100-UplinkMedium' 
        if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
            tdl = nrTDLChannel; % use MATLAB 5G toolbox
            if Chan.simplifiedDelayProfile
            tdl.DelayProfile = 'TDLC300';
            else
                tdl.DelayProfile = 'TDL-C';
                tdl.DelaySpread = 300e-9;
            end
            tdl.MaximumDopplerShift = 100;
            tdl.MIMOCorrelation = 'UplinkMedium';
            tdl.Polarization = "Co-Polar";
            tdl.TransmissionDirection = Chan.link_direction;
            tdl.NumTransmitAntennas = Chan.Nin;
            tdl.NumReceiveAntennas = Chan.Nout;
            tdl.SampleRate = Chan.f_samp;
            tdl.NormalizeChannelOutputs = false;
            tdl.RandomStream = 'Global Stream';
            Chan.tdl = tdl;
        elseif strcmp(Chan.model_source, 'custom')
            Chan = genTDL(Chan, SysPar);
        else
            error('Undefined channel model given the model_source!')
        end 
    case 'TDL'
        if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
            tdl = nrTDLChannel;
            tdl.DelayProfile = Chan.DelayProfile;
            tdl.DelaySpread = Chan.DelaySpread;
            tdl.MaximumDopplerShift = Chan.MaximumDopplerShift;
            tdl.MIMOCorrelation = Chan.MIMOCorrelation;
            tdl.Polarization = "Co-Polar";
            tdl.TransmissionDirection = Chan.link_direction;
            tdl.NumTransmitAntennas = Chan.Nin;
            tdl.NumReceiveAntennas = Chan.Nout;
            tdl.SampleRate = Chan.f_samp;
            tdl.NormalizeChannelOutputs = false;
            tdl.RandomStream = 'Global Stream';
            Chan.tdl = tdl;
        elseif strcmp(Chan.model_source, 'sionna')  
            eval("Chan = initSionnaChan(SysPar, Chan);")
        elseif strcmp(Chan.model_source, 'custom')
            Chan = genTDL(Chan, SysPar);
        else
            error('Not defined!');
        end
    case 'CDL'
        if strcmp(Chan.model_source, 'MATLAB5Gtoolbox')
            % Note the following configurations assume downlink, which
            % aligns with the cluster angle assumptions. The Tx and Rx will
            % be swapped at the end if UL is configurated.
            cdl = nrCDLChannel;
            cdl.DelayProfile = Chan.DelayProfile;
            cdl.DelaySpread = Chan.DelaySpread;
            cdl.CarrierFrequency = SysPar.carrier.carrierFreq*1e9; % carrier freq in Hz
            cdl.MaximumDopplerShift = Chan.MaximumDopplerShift;
            cdl.SampleRate = Chan.f_samp;
            if strcmp(Chan.CDL_updateRateType, 'perSample')
                cdl.SampleDensity = inf;
            elseif strcmp(Chan.CDL_updateRateType, 'perSymbol')
                cdl.SampleDensity = (100*SysPar.carrier.N_slot_frame_mu*(SysPar.carrier.N_symb_slot-1))/2/Chan.MaximumDopplerShift;
            end
            % config antenna array
            cdl.TransmitAntennaArray.PolarizationAngles = Chan.gNB_AntPolarizationAngles;
            cdl.ReceiveAntennaArray.PolarizationAngles = Chan.UE_AntPolarizationAngles;
            cdl.TransmitAntennaArray.Size = [Chan.gNB_AntArraySize, 1, 1];
            cdl.ReceiveAntennaArray.Size = [Chan.UE_AntArraySize, 1, 1];
            assert(prod(Chan.gNB_AntArraySize)==SysPar.carrier.Nant_gNB);
            assert(prod(Chan.UE_AntArraySize)==SysPar.carrier.Nant_UE);
            cdl.TransmitAntennaArray.Element = Chan.gNB_AntPattern;
            cdl.ReceiveAntennaArray.Element = Chan.UE_AntPattern;
            cdl.TransmitAntennaArray.ElementSpacing = [Chan.gNB_AntSpacing 1.0 1.0];
            cdl.ReceiveAntennaArray.ElementSpacing = [Chan.UE_AntSpacing 1.0 1.0];
            % swap Tx and Rx for to get UL channel
            if strcmp(DLUL, 'UL')
                swapTransmitAndReceive(cdl);
            end
            cdl.NormalizePathGains = true;
            cdl.NormalizeChannelOutputs = false;
            cdl.RandomStream = 'Global Stream';
            
            Chan.cdl = cdl;
        elseif strcmp(Chan.model_source, 'sionna')  
            eval("Chan = initSionnaChan(SysPar, Chan);")
        elseif strcmp(Chan.model_source, 'custom')
            Chan = genCDL(Chan, SysPar);
        else
            error('To use CDL channel model, you need to set Chan.model_source to MATLAB5Gtoolbox or sionna.')
        end
    case {'UMi','UMa','RMa'}
        if strcmp(Chan.model_source, 'sionna')  
            eval("Chan = initSionnaChan(SysPar, Chan);")
        else
            error('To use UMi/UMa/RMa channel model, you need to set Chan.model_source to sionna.')
        end
    otherwise
        error('Channel type is not supported ...\n');
end

return
end

function [a, b] = findTwoIntegerFactorsCloseToSquareRoot(x)
   sqrt_x = sqrt(x);
   a = floor(sqrt_x);
   while a > 0
        if mod(x, a) == 0
            b = x/a;
            break;
        else
            a = a - 1;
        end
   end
end
