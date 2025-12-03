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

function H = generate_rnd_channel(bfc,sim)

% function generates a random channel

% outputs:
% H --> generated channel. Dim: nWeights x nLayers x L_gNB

%%
%PARAMATERS

%beamforming paramaters:
nLayers = bfc.nLayers;    % number of downlink layers. Value: 1,2,4,8,16
L_gNB = bfc.L_gNB;        % number of gNB antennas. Value: 2,4,8,16,32,64
nWeights = bfc.nWeights;  % number of beamforming weights

%simulation paramaters:
channelModel = sim.channelModel;  % options: 'TDL' or 'randomDelay' 
df_est = sim.df_est;              % Hz between channel estimates

%%
%FREQUENCY
% build frequency grid for sampled channel

f = 0 : (nWeights - 1);
f = df_est*f;

%%
%TAPS
% here we generate or load channel taps

switch channelModel
    
    case('randomDelay')
            %paramaters:
            nTaps = sim.nTaps;              % Number of reflectors
            delaySpread  = sim.delaySpread; % Delay spread (seconds)
            
            %taps
            tau = delaySpread*rand(nTaps,1); % tap delays (seconds)
            E = randn(nTaps,1).^2;           % tap energy (linear)
            
    case('tdl')
        %paramaters:
        switch(sim.mode)
        	case('a')
                load('tdl_a.mat');
            case('b')
                load('tdl_b.mat');
            case('c')
                load('tdl_c.mat');
            case('d')
                load('tdl_d.mat');
            case('e')
                load('tdl_e.mat');
        end
        
        % taps:
        tau = T(:,2) * sim.ds * 10^(-9);  % tap delays
        E = 10.^(T(:,3) / 10);        % tap energy   
        nTaps = length(E); 
        
end


%%
%COMPUTE
% here we generate frequency/space channel


H = zeros(nWeights,nLayers,L_gNB);

for t = 1 : nTaps
    %random spatial response:
    Hs = sqrt(1/2)*(randn(nLayers*L_gNB,1) + 1i*randn(nLayers*L_gNB,1));
    
    %frequency response:
    Hf = exp(-2*pi*1i*f*tau(t));
    
    %freq/space response:
    Hsf = Hs*Hf;
    
    %reshape:
    Hsf = reshape(Hsf,L_gNB,nLayers,nWeights);
    Hsf = permute(Hsf,[3 2 1]);
    
    %add:
    H = H + sqrt(E(t))*Hsf;
    
end


%normalize:
E = abs(H).^2;
H = H / sqrt(mean(E(:)));


    
    




