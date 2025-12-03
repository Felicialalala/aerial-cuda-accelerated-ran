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

function H_est = apply_ChEst_MMSE_filter_kernel(H_est, H_LS_est, shiftSeq, unShiftSeq, ChEst_par)

% function applies ChEst for a small prb cluster. 

%inputs:
% H_est      --> current channel estimate. Dim: Nf x L_UE
% shiftSeq   --> shift sequence. Dim: 6*4 x 1
% unShiftSeq --> unshift sequence. Dim: (12*4 + 1) x 1 


%outputs:
% H_est    --> updated channel estimate. Dim: Nf x L_UE

%%
%PARAMTERS

% mimo paramaters:
nl = ChEst_par.nl;                     % number of layers assigned to cluster

% dmrs paramaters:
grid_cfg = ChEst_par.grid_cfg;         % for each port indicates which grid used. Dim: nl x 1


% Estimation filter:
W = ChEst_par.W;                         % Estimation filter. Dim: Nf_out x Nf_in
remainderFlag = ChEst_par.remainderFlag;

% Indicies:
startPrb_dmrs = ChEst_par.startPrb_dmrs;  % starting dmrs prb
nPrb_dmrs = ChEst_par.nPrb_dmrs;          % number of dmrs prbs
startPrb_est = ChEst_par.startPrb_est;    % starting prb (within cluster) to estimate
nPrb_est = ChEst_par.nPrb_est;            % number of prbs to estimate

%%
%SETUP

% number of dmrs tones in cluster:
Nf_dmrs = nPrb_dmrs * 6;    

% Global indicies of dmrs subcarriers (no grid offset):
global_dmrs_idx = 12*(startPrb_dmrs - 1) + 2*(0 : (Nf_dmrs - 1)) + 1;

% Global indicies of estimation subcarriers:
global_est_idx = 12*(startPrb_dmrs + startPrb_est - 2) + 1 : 12*(startPrb_dmrs + startPrb_est + nPrb_est - 2);

% local indicies of data subcarriers:
local_data_idx = (12*(startPrb_est - 1) + 1) : (12*(startPrb_est + nPrb_est - 1) + 1) ;


% delay shift and power normalize the dmrs signal
for idx_layer = 1:nl
    tmp_H_LS_rot = H_LS_est(global_dmrs_idx,idx_layer).*shiftSeq(1 : Nf_dmrs);
    % apply filter
    H_est_local = W * tmp_H_LS_rot;
    if remainderFlag == 1
        H_est_local = H_est_local(13 : end,:);
    end
    % undo shift
    H_est_local = H_est_local .* unShiftSeq(local_data_idx);

    % save results
    if grid_cfg(idx_layer) == 0
        H_est(global_est_idx, idx_layer) = H_est_local(2 : end);
    else
        H_est(global_est_idx, idx_layer) = H_est_local(1 : (end - 1));
    end

end
