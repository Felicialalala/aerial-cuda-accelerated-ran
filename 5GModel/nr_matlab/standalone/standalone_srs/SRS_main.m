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

close all;
if (~isdeployed)
    addpath(genpath('.'))
end
rng('default');

%%
%PARAMATERS
fp16 = 1;

% gnb paramaters:
gnb.Nf = 12*272;      % number of subcarrier
gnb.Nt = 14;          % number of OFDM symbols
gnb.L = 64;           % number of gNB digital antennas
gnb.df = 30*10^3;     % subcarrier spacing (Hz)

% simulation paramaters:
sim.N0 = 10^(-2.0);               % input noise variance
sim.channelModel = 'randomDelay'; % options: 'awgn' or 'randomDelay' 
sim.delaySpread = 0.5*10^(-6);    % channel delay spread
sim.nTaps = 30;                   % number of channel taps

% uplink slot srs paramaters:
srs.nRes = 8;                 % number of SRS resourses
srs.nSym = 2;                 % number of srs symbols in slot
srs.symIdx = [12 13];         % indicies of srs symbols in slot
srs.pdu_cell = [];            % cell containing fapi srs pdu of each resourse

% SRS ChEst paramaters:
srsChEst.delaySpread = 1.2*10^(-6);   % delay spread assumed by channel estimation filters
srsChEst.N0 = 10^(-2.0);              % noise variance assumed by estimation filters
srsChEst.df = gnb.df;                 % subcarrier spacing (Hz)
srsChEst.nPrbPerThreadBlock = 4; %16;      % number of prb per threadblock
if mod(272,srsChEst.nPrbPerThreadBlock)~=0
    error('nPrbPerThreadBlock must divide 272');
end

% assign fapi pdu to each srs resourse:
srs.pdu_cell = assign_pdu;

% derive SRS channel estimation objects (filters and sequences):
W = derive_ChEst_objects(srsChEst);

%%
%SETUP

%channel:
H = generate_rnd_channel_srs(gnb,srs,sim);

%%
%START

%generate srs tx signal:
s_tx = srs_tx_main(gnb,srs);

%apply channel:
y = apply_channel(H,s_tx,gnb,srs,sim);

%set cuphy srs paramaters:
cuphySRSpar = set_cuphy_srs_paramaters(gnb,srs,srsChEst);

ySlot = sqrt(sim.N0/2)*(randn(gnb.Nf,14,gnb.L) + 1i*randn(gnb.Nf,14,gnb.L));
ySlot(:,srs.symIdx,:) = y;
%FP16 conversion
if fp16
    ySlot = reshape(fp16nv(real(double(ySlot)), 2) + 1i*fp16nv(imag(double(ySlot)), 2), [size(ySlot)]);
    y = ySlot(:,srs.symIdx,:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%CUPHY START%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%estimate channel:
[H_est, dbg] = srs_ChEst_main(y,W,cuphySRSpar);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%CUPHY END%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%evaluate:
compute_ChEst_snr(H,H_est);

genSrsChEstTv(cuphySRSpar, W, ySlot, H_est, dbg, fp16);
