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

function H = generate_rnd_channel_srs(gnb,srs,sim)

% function generates channel for each user (i.e. srs resourse)

%outputs:
%H --> dimension: Nf x L_gNB x 4 x nRes

%%
%PARAMATERS

% gnb paramaters:
Nf = gnb.Nf;        % number of subcarrier
L_gNB = gnb.L;      % number of gNB digital antennas
df = gnb.df;        % subcarrier spacing (Hz)

% simulation paramaters:
channelModel = sim.channelModel; % awgn channel model

% uplink slot srs paramaters:
nRes = srs.nRes;         % number of SRS resourses

%%
%SETUP

f = 0 : (Nf - 1);
f = df*f';

switch channelModel
    
    case('awgn')
        nTaps = 1;
        tau = zeros(nTaps,nRes);
        a = ones(nTaps,nRes);
        Hs = ones(L_gNB,4,nRes);
        
    case('randomDelay')
        nTaps = sim.nTaps;
        delaySpread = sim.delaySpread;
        
        tau = delaySpread*(rand(nTaps,nRes));
        a = sqrt(1/2)*(randn(nTaps,nRes) + 1i*randn(nTaps,nRes));
        Hs = sqrt(1/2)*(randn(L_gNB,4,nRes) + 1i*randn(L_gNB,4,nRes));
end
        

%%
%START

H = zeros(L_gNB,4,Nf,nRes);

for ue = 1 : nRes
    
    %random frequency signature:
    Hf = zeros(Nf,1);
    for t = 1 : nTaps
        Hf = Hf + a(t)*exp(-2*pi*1i*tau(t)*f);
    end
    
    %combine frequency/space:
    Hfs = zeros(L_gNB,4,Nf);
    for i = 1 : Nf
        Hfs(:,:,i) = Hs(:,:,ue)*Hf(i);
    end
    
    %normalize to unit energy:
    E = abs(Hfs).^2;
    Hfs = Hfs / sqrt(mean(E(:)));
    
    %store:
    H(:,:,:,ue) = Hfs;
    
end

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
