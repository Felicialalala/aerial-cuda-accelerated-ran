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

function H_est = apply_ChEst_kernel(H_est, Y, shiftSeq, unShiftSeq, r_dmrs, ChEst_par)

% function applies ChEst for a small prb cluster. 

%inputs:
% Y          --> signal recieved by a BS antenna. Dim: Nf x Nt
% H_est      --> current channel estimate. Dim: Nf x L_UE
% shiftSeq   --> shift sequence. Dim: 6*4 x 1
% unShiftSeq --> unshift sequence. Dim: (12*4 + 1) x 1 
% r_dmrs     --> dmrs scrambling sequence. Dim: Nf x Nt x 2


%outputs:
% H_est    --> updated channel estimate. Dim: Nf x L_UE

%%
%PARAMTERS

% mimo paramaters:
nl = ChEst_par.nl;                     % number of layers assigned to cluster
portIdx = ChEst_par.portIdx;           % dmrs port index used by each layer. Dim: nl x 1

% dmrs paramaters:
fOCC_cfg = ChEst_par.fOCC_cfg;         % 0 or 1. For each layer indicates if fOCC used. Dim: nl x 1
tOCC_cfg = ChEst_par.tOCC_cfg;         % 0 or 1. For each layer indicates if tOCC used. Dim: nl x 1
grid_cfg = ChEst_par.grid_cfg;         % for each port indicates which grid used. Dim: nl x 1
maxLength = ChEst_par.maxLength;       % 1 for single dmrs, 2 for double dmrs
symIdx_dmrs = ChEst_par.symIdx_dmrs;   % indicies of dmrs symbols
nGrids = ChEst_par.nGrids;             % number of unique dmrs grids used
gridIdx = ChEst_par.gridIdx;           % indicies of grids used. Dim: num_grids x 1
n_scid = ChEst_par.n_scid;

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

% Global indicies of scrambling sequence:
global_scram_idx = 6*(startPrb_dmrs - 1) + 1 : 6*(startPrb_dmrs + nPrb_dmrs - 1);

% local indicies of dmrs subcarriers:
local_dmrs_idx = 0 : (Nf_dmrs - 1);
local_dmrs_idx = 2*local_dmrs_idx + 1;

% local indicies of data subcarriers:
local_data_idx = (12*(startPrb_est - 1) + 1) : (12*(startPrb_est + nPrb_est - 1) + 1) ;

% build fOCC:
fOCC = ones(Nf_dmrs,1);
fOCC(mod(1:Nf_dmrs,2) == 0) = -1;

%%
%STEP 1

% extract raw dmrs signal (different grids and symbols)

Y_local = zeros(Nf_dmrs,maxLength,nGrids);

for i = 1 : nGrids
    Y_local(:,:,i) = Y(global_dmrs_idx + gridIdx(i),symIdx_dmrs(1 : maxLength));
end

%%
%STEP 2

% delay shift and power normalize the dmrs signal

for i = 1 : nGrids
    for j = 1 : maxLength
        Y_local(:,j,i) = Y_local(:,j,i) .* shiftSeq(1 : Nf_dmrs);
%         conj(r_dmrs(global_scram_idx,symIdx_dmrs(j),n_scid + 1)) * (1 / sqrt(energy));
    end
end

%%
%STEP 3

% remove scrambling sequences

for i = 1 : nGrids    
    Y_local(:,:,i) = conj(r_dmrs(global_scram_idx,symIdx_dmrs(1 : maxLength),n_scid(i) + 1)) .* Y_local(:,:,i);
end

%%
%STEP 4

% remove fOCC, tOCC (if needed)

Y_local2 = zeros(Nf_dmrs,nl);

for i = 1 : nl
    
    grid = mod(grid_cfg(i),nGrids) + 1; %users grid
    
    % remove tOCC (if needed):
    if maxLength == 1
        Y_local2(:,i) = Y_local(:,1,grid);
    else
        if tOCC_cfg(i) == 0
            Y_local2(:,i) = (Y_local(:,1,grid) + Y_local(:,2,grid)) / 2;
        else
            Y_local2(:,i) = (Y_local(:,1,grid) - Y_local(:,2,grid)) / 2;
        end
    end
    
    % permute fOCC (if needed):
    if fOCC_cfg(i) == 1
        Y_local2(:,i) = fOCC .* Y_local2(:,i);
    end

end

%%
%STEP 4.1

Y_local1 = Y_local2(:, 1);
Y_local1 = Y_local1(1:2:end) + Y_local1(2:2:end);
rot = angle(sum(Y_local1(2:end).*conj(Y_local1(1:end-1))));
Y_derot = Y_local1.*exp(-1j*rot*[0:length(Y_local1)-1]');
np = mean(abs(Y_derot(2:end) - Y_derot(1:end-1)).^2);
snp = mean(abs(Y_derot).^2);
snr = 10*log10((snp - np)/np);
np_db = 10*log10(np);

%%
%Step 4.2 add vPRBs
global SimCtrl
if SimCtrl.alg.ChEst_use_vPRBs
    tmp_Y_local2 = zeros(Nf_dmrs + 2*6,nl);
    tmp_Y_local2(6+1:6+Nf_dmrs,:) = Y_local2; % middle part
    tmp_mean_vprb1 = mean(Y_local2(1:6,:).*repmat(conj(shiftSeq(1:6)),[1,nl]),1);
    tmp_Y_local2(1:6,:) = flipud(conj(shiftSeq(1:6)))*tmp_mean_vprb1;
    tmp_mean_vprb2 = mean(Y_local2(end-6+1:end).*repmat(conj(shiftSeq(Nf_dmrs-6+1:Nf_dmrs)),[1,nl]),1);
    tmp_Y_local2(Nf_dmrs+6+1:Nf_dmrs+12,:) = (shiftSeq(Nf_dmrs+1:Nf_dmrs+6))*tmp_mean_vprb2;
    Y_local2 = tmp_Y_local2;
end

%STEP 5

% apply filter:
H_est_local = W * Y_local2;

if remainderFlag == 1
    H_est_local = H_est_local(13 : end,:);
end

%%
%STEP 6

% undo shift:
H_est_local = H_est_local .* (repmat(unShiftSeq(local_data_idx),[1 nl]));

%%
%STEP 7 

% save results:

% for i = 1 : nl
%     if grid_cfg(i) == 0
%         H_est(global_est_idx, portIdx(i) + 1) = H_est_local(2 : end, i);
%     else
%         H_est(global_est_idx, portIdx(i) + 1) = H_est_local(1 : (end - 1), i);
%     end
% end    

for i = 1 : nl
    if grid_cfg(i) == 0
        H_est(global_est_idx, i) = H_est_local(2 : end, i);
    else
        H_est(global_est_idx, i) = H_est_local(1 : (end - 1), i);
    end
end 
