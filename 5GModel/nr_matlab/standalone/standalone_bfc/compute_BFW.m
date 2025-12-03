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

function W = compute_rZF_BFW(H,bfc)

%function compute regularized-zero-forincing beamforming weights

%inputs:
% H --> estimated channels. Dim: nWeights x nLayers x L_gNB

%outputs:
% W --> beamforming weights. Dim: L_gNB x nLayers x nWeights

%%
%PARAMATERS

%beamforming paramaters:
nLayers = bfc.nLayers;       % number of downlink layers. Value: 1,2,4,8,16
L_gNB = bfc.L_gNB ;          % number of gNB antennas. Value: 2,4,8,16,32,64
nWeights = bfc.nWeights;     % number of beamforming weights. Value: 1-136
lambda = bfc.lambda;         % beamforming regularization coefficent

%%
%START

W = zeros(L_gNB,nLayers,nWeights);


%loop over subcarriers
for f = 1 : nWeights
    
    %pull frequency estimate:
    Hf = squeeze(H(f,:,:));
    
    %compute regularized Gram matrix:
    Gr = Hf*Hf' + lambda*eye(nLayers);
    
    %beamforming weight:
    Wf = Hf'*Gr^(-1);
    
    %store:
    W(:,:,f) = Wf;
    
end
