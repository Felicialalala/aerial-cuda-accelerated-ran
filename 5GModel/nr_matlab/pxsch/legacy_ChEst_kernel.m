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

function H_est = legacy_ChEst_kernel(H_est,Y, pdschTable, r_dmrs, ChEst_par, ...
    nl, portIdx, n_scid, symIdx_dmrs, Nt_dmrs, nPrb, startPrb, s,s_grid)

% function applies ChEst for a small prb cluster. 

%inputs:
% Y         --> signal recieved by a BS antenna. Dim: Nf x Nt
% H_est     --> current channel estimate. Dim: Nf x L_UE
% s         --> 2nd shift sequence, applied on filter output, un-centers delay. Dim: (Nf + 1) x 1
% s_grid    --> 1st shift sequence. Applied on dmrs grid, centers delay to zero. Dim: Nf / 2 x 1
% d         --> dmrs descrambling sequence. Dim: Nf / 2 x Nt_dmrs


%outputs:
% H_est    --> updated channel estimate. Dim: Nf x L_UE

% Estimation filter:
W = ChEst_par.W;                          % Estimation filter. Dim: Nf_out x Nf_in

% Indicies:
startPrb_dmrs = ChEst_par.startPrb_dmrs;  % starting dmrs prb
nPrb_dmrs = ChEst_par.nPrb_dmrs;          % number of dmrs prbs
startPrb_est = ChEst_par.startPrb_est;    % starting prb to estimate
nPrb_est = ChEst_par.nPrb_est;            % number of prbs to estimate

fOCC_table = pdschTable.fOCC_table;
grid_table = pdschTable.grid_table;
tOCC_table = pdschTable.tOCC_table;

%extract dmrs scrambling sequence:
d = r_dmrs(:,symIdx_dmrs,n_scid+1);

%build fOCC:
fOCC = ones(6*nPrb_dmrs,1);
fOCC(mod(1 : 6*nPrb_dmrs,2) == 0) = -1;

% dmrs paramaters:
fOCC_cfg = fOCC_table(portIdx); % FEGpar.dmrs.fOCC_cfg;          % 0 or 1. For each layer indicates if fOCC used. Dim: nl x 1
tOCC_cfg = tOCC_table(portIdx); % FEGpar.dmrs.tOCC_cfg;          % 0 or 1. For each layer indicates if tOCC used. Dim: nl x 1
grid_cfg = grid_table(portIdx); % FEGpar.dmrs.grid_cfg;          % for each port indicates which grid used. Dim: nl x 1

maxLength = length(symIdx_dmrs);

nGrids = nl;              % number of unique dmrs grids used
gridIdx = (0:nGrids-1);            % indicies of grids used. Dim: num_grids x 1

%%
%SETUP

% number of dmrs tones in cluster:
Nf_dmrs = nPrb_dmrs * 6;    

% Global indicies of dmrs subcarriers (no grid offset):
global_dmrs_idx = 12*(startPrb_dmrs - 1) + 2*(0 : (Nf_dmrs - 1)) + 1;

% Global indicies of estimation subcarriers:
global_est_idx = 12*(startPrb_est - 1) + 1 : 12*(startPrb_est + nPrb_est - 1);

% Global indicies of scrambling sequence:
global_scram_idx = 6*(startPrb_dmrs - 1) + 1 : 6*(startPrb_dmrs + nPrb_dmrs - 1);

% global shift index:
global_shift_idx = (12*(startPrb_est - 1) + 1) : ( 12*(startPrb_est + nPrb_est - 1) + 1) ;

%%
%STEP 1

% extract raw dmrs signal (different grids and symbols)

Y_local = zeros(Nf_dmrs,maxLength,nGrids);

for i = 1 : nGrids
    Y_local(:,:,i) = Y(global_dmrs_idx + gridIdx(i),symIdx_dmrs(1 : maxLength));
end

%%
%STEP 2

% shift + descramble dmrs signal:

for i = 1 : nGrids
    for j = 1 : maxLength
%         Y_local(:,j,i) = Y_local(:,j,i) .* sd(global_scram_idx,j);OLD, COMBINED SEQUENCES 
         d = r_dmrs(:,symIdx_dmrs,n_scid(i)+1);
         Y_local(:,j,i) = Y_local(:,j,i) .* ...
              s_grid(global_scram_idx).*d(global_scram_idx,j); %NEW, SEPERATE SEQUENCES 
    end
end

%%
%STEP 3

% remove fOCC and tOCC (if needed)

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
%STEP 3.1

Y_local1 = Y_local2(:, 1);
rot = angle(sum(Y_local1(2:end).*conj(Y_local1(1:end-1))));
Y_derot = Y_local1.*exp(-1j*rot*[0:length(Y_local1)-1]');
np = mean(abs(Y_derot(2:end) - Y_derot(1:end-1)).^2);
snp = mean(abs(Y_derot).^2);
snr = 10*log10((snp - np)/np);
np_db = 10*log10(np);

%%
%STEP 4

% apply filter:
H_est_local = W * Y_local2;

%
% STEP 5

% undo shift:
H_est_local = H_est_local .* repmat(s(global_shift_idx),[1 nl]);

%
%STEP 6 

% save results:
for i = 1 : nl
    if grid_cfg(i) == 0
%         H_est(global_est_idx,portIdx(i)) = H_est_local(2 : end, i);
        H_est(global_est_idx,i) = H_est_local(2 : end, i);
    else
%         H_est(global_est_idx,portIdx(i)) = H_est_local(1 : (end - 1), i);
        H_est(global_est_idx,i) = H_est_local(1 : (end - 1), i);
    end
end

return


