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

function [W, Wgpu] = compute_rZF_BFW(H,bfc)

%function computes regularized-zero-forincing beamforming weights

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
Wgpu = zeros(L_gNB,nLayers,nWeights);

% Inermediates for debug
A0 = zeros(nLayers,2*nLayers+L_gNB,nWeights);
A1 = zeros(nLayers,2*nLayers+L_gNB,nWeights);
Dinv = zeros(nLayers,nWeights);
Linv = zeros(nLayers,nLayers,nWeights);
frobNorm = zeros(nWeights);

%loop over subcarriers
for f = 1 : nWeights
    
    %pull frequency estimate:
    Hf = squeeze(H(f,:,:));
    Hf = reshape(Hf,nLayers,L_gNB);
    
    %compute regularized Gram matrix:
    Gr = Hf*Hf' + lambda*eye(nLayers);
    
    %beamforming weight:
    Wf = Hf'*Gr^(-1);
    Wf = Wf./norm(Wf,'fro');
    
    %store:
    W(:,:,f) = Wf;
    
    
    % G = LU 
    % Since G is Hermitian symmetric, G = LDL'
    % => U = DL', where D = diag(diag(U))
    % Since G = LDL', inv(G) = inv(L')*inv(D)*inv(L)
    
    % W = H'*inv(G) = H'*inv(L')*inv(D)*inv(L)
    % W = (inv(L)*H)' * inv(D) * inv(L)
    
    % A = [G | H | I]
    % Apply LU factorization to A
    % A = [U | Linv*H | Linv] = [U | M | Linv], D = diag(diag(U))
    
    % Scale columns of M' or rows of inv(L) by inv(D), since inv(L) is
    % lower triangular it may be less compute (but may not matter for GPU)
    
    % Multiply M'*(inv(D)*inv(L))
    % W = M'*(inv(D)*inv(L))        
    [~,D_f,Linv_f,A0_f,A1_f] = lu_solve2(Gr, Hf);
    
    Mf = A1_f(:,2*nLayers+1:end); % M = (Linv*H)./diag(U);
    
    Wf_gpu = Mf'*inv(D_f)*Linv_f;
    frobNorm(f) = norm(Wf_gpu,'fro');
    
    Wf_gpu = Wf_gpu./frobNorm(f);
    
    Wgpu(:,:,f) = Wf_gpu;
    
    % Inermediates for debug
    A0(:,:,f) = A0_f;
    A1(:,:,f) = A1_f;
    Dinv(:,f) = diag(inv(D_f));
end

saveBfcProbes(bfc,H,A0,A1,Dinv,Wgpu,frobNorm);
