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

function compute_bf_snr(H,W_gNB,bfc,sim)

%function computes snr experienced by users when gNB uses beamforming
%weights

%inputs:
% H     --> channel. Dim: nWeights x nLayers x L_gNB
% W_gNB --> beamforming weights. Dim: L_gNB x nLayers x nWeights

%%
%PARAMATERS

%downlink channel:
H = permute(H,[2 3 1]);

%beamforming paramaters:
nWeights = bfc.nWeights;    % number of beamforming weights
nLayers = bfc.nLayers;      % number of downlink layers. Value: 1,2,4,8,16


%simulation paramaters:
N0 = sim.N0;              % channel noise variance (linear)

%%
%POWER
% here we compute the power normalization coefficent

E = abs(W_gNB).^2;
E_tot = sum(E(:));

mu = sqrt(nLayers*nWeights /E_tot);


%%
%USER EQULIZER
% here we compute the user equalizer coefficents:

W_ue = zeros(nLayers,nLayers,nWeights);

for f = 1 : nWeights
    H_bf = mu*H(:,:,f)*W_gNB(:,:,f);
    H_bf = diag(diag(H_bf));
    
    W_ue(:,:,f) = H_bf^(-1);
end


%%
%SINR
% here we compute SINR experiences by users:

I = zeros(nLayers,nWeights);
N = zeros(nLayers,nWeights);

for f = 1 : nWeights
    H_tot = mu*W_ue(:,:,f)*H(:,:,f)*W_gNB(:,:,f);
    
    I(:,f) = diag((H_tot - eye(nLayers))*(H_tot - eye(nLayers))');
    N(:,f) = abs(diag(W_ue(:,:,f))).^2*N0;
end

SINR = -10*log10(I + N);

%%
%PLOT  

% figure
% plot(SINR');
% grid on
% xlabel('Prb/2');
% ylabel('SINR (dB)');
% title('BF SINR experienced by users');


    
