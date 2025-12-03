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

function [W,Ree] = derive_equalizer(H_est, N0_ref, nl, Nf, posDmrs, subslot_processing_option)

%function build mmse equalization filters, along with the mmse error
%variance

%inputs:
%H_est --> channel estimate. Dim: L_BS x L_UE x Nf

%outputs:
%W     --> mmse equalization filters. Dim: L_UE x L_BS x Nf
%Ree   --> mmse error variance. Dim: L_UE x Nf

global SimCtrl;

biasCorrection = 1; % option to enable bias correction
applyReeLimit = 1;

L_UE = nl;      % max number of spatially multiplexed layers
[L_BS, ~, ~] = size(H_est); % total number of bs antennas

N0_ref_tmp = N0_ref(posDmrs, 1); % note that for full-slot proc (i.e., SimCtrl.subslot_processing_option == 0), all entries of N0_ref should be the same, which should be the avg value across all DMRS addPos.

N0 = N0_ref_tmp*ones(Nf,1);

W = zeros(L_UE,L_BS,Nf);
Ree = zeros(L_UE,Nf);

if applyReeLimit == 1
    maxReeInvVal = 10000; % this constant should align with cuPHY
    minReeVal = 1/maxReeInvVal; % this constant should align with cuPHY
    minRee = minReeVal.*ones(L_UE,1);
end

for f = 1 : Nf

    %compute Gram matrix:
    G_f = H_est(:,:,f)'*H_est(:,:,f) / N0(f) + eye(L_UE);

    %compute error covariance:
    Ree_f = pinv(G_f);

    %compute filter:
    W_f = Ree_f * H_est(:,:,f)' / N0(f);

    %wrap:
    W(:,:,f) = W_f;
    Ree(:,f) = diag(Ree_f);

    if biasCorrection == 1
        lambda = 1 ./ (1 - Ree(:,f));
        lambda = min(lambda, SimCtrl.alg.pusch_equalizer_bias_corr_limit);
        W(:,:,f) = diag(lambda) * W(:,:,f);
        Ree(:,f) = lambda .* Ree(:,f);
    end

    if applyReeLimit == 1
        Ree(:,f) = max(minRee, Ree(:,f));
    end

end


return
