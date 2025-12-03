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

%clear all; 
function get_BFC_TV(nLayers,L_gNB,nWeights, tvName)
close all;
% reference script for beamforming weight computation

wrkspaceDir = pwd;
tvDirName = 'GPU_test_input'; [status,msg] = mkdir(tvDirName);

rng('default');

%%
%PARAMATERS

%beamforming paramaters:
bfc.nLayers = nLayers;   % number of downlink layers. Value: 1,2,4,8,16
bfc.L_gNB = L_gNB;       % number of gNB antennas. Value: 2,4,8,16,32,64
bfc.nWeights = nWeights; % 3200;%136; % number of beamforming weights. Value: 1-136
bfc.lambda = 10^(-2);    % beamforming regularization coefficent


%simulation paramaters:
sim.channelModel = 'tdl';     % options: 'tdl' or 'randomDelay' 
sim.df_est = 24*30*10^3;      % Hz between channel estimates
sim.N0 = 10^(-2);             % channel noise variance (linear)


%following paramaters used in uniform reflector channel model:
if strcmp(sim.channelModel,'randomDelay')
    sim.nTaps = 10;                 % Number of reflectors
    sim.delaySpread = 1.0*10^(-6);  % Delay spread (seconds)
end

%following paramaters used in tdl channel model:
if strcmp(sim.channelModel,'tdl')
    sim.mode = 'c';   % choice of tdl mode, options: (a,b,c,d,e,f)
    sim.ds = 100;     % desired delay spread (ns).d
end

%%
%START

H = generate_rnd_channel(bfc,sim);

[W, Wgpu]= compute_rZF_BFW(H,bfc);

compare(W, Wgpu, 'BFC Filter Coefficients', 0);

%%
%EVALUATE

compute_bf_snr(H,W,bfc,sim);
compute_bf_snr(H,Wgpu,bfc,sim);

%%
genBfcCoefTv(bfc,H,Wgpu, tvName);
