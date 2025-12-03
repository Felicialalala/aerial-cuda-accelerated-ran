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

function [H_est, R0_avg, R1_avg] = apply_ChEst_LS_main(Y,table,slotNumber,Nf,Nt,N_dmrs_id,...
    nl, portIdx, n_scid, symIdx_dmrs, Nt_dmrs, nPrb, startPrb, df, ...
    maxLength, numDmrsCdmGrpsNoData, enableTfPrcd, ...
    N_slot_frame, N_symb_slot, puschIdentity,...
    groupOrSequenceHopping, r_dmrs)

% perform channel estimation for all users.


%outputs:
% H_est       --> estimated channel. Dim: L_BS x L_UE x Nf
% table       --> constant
% slotNumber  --> current slot index
% Nf          --> number of subcarriers in slot
% Nt          --> number of symbols in slot
% N_dmrs_id   --> dmrs scrambling id
% nl          --> number of layers
% portIdx     --> dmrs port of each layer. Dim: nl x 1
% n_scid      --> 0 or 1
% symIdx_dmrs --> indicies of dmrs symbols. Dim: Nt_dmrs x 1
% Nt_dmrs     --> number of dmrs symbols
% nPrb        --> number of prbs in ue group allocation
% startPrb    --> start prb of ue group allocation
% df          --> subcarrier spacing (Hz)
% maxLength   --> 1 or 2. Number of adjacent DMRS symbols.

%%
%PARAMATERS

portIdx = uint8(portIdx - 1); % convert to integer for bitwise operations

[~, ~, L_BS] = size(Y);  % total number of bs antennas
L_UE = nl;               % max number of spatially multiplexed layers


%%
%SEQUENCES

% scrambling sequence:

% if enableTfPrcd == 0
%     r_dmrs  = gen_dmrs_sym(slotNumber, Nf, Nt, N_dmrs_id);
% else
%     [r_dmrs, ~, ~] = gen_dmrs_dftsofdm(Nf, startPrb, nPrb, N_slot_frame, N_symb_slot, symIdx_dmrs, ...
%         slotNumber, puschIdentity, groupOrSequenceHopping);
% end

%%
% DMRS CFG

ChEstPar = [];
ChEstPar.fOCC_cfg = zeros(nl,1); % 0 or 1. For each layer indicates if fOCC used. Dim: nl x 1
ChEstPar.tOCC_cfg = zeros(nl,1); % 0 or 1. For each layer indicates if tOCC used. Dim: nl x 1
ChEstPar.grid_cfg = zeros(nl,1); % for each port indicates which grid used. Dim: nl x 1

for layerIdx = 1 : nl
    ChEstPar.fOCC_cfg(layerIdx) = bitand(portIdx(layerIdx),1); 
    ChEstPar.grid_cfg(layerIdx) = bitshift(bitand(portIdx(layerIdx),2),-1);
    ChEstPar.tOCC_cfg(layerIdx) = bitshift(bitand(portIdx(layerIdx),4),-2); 
end

ChEstPar.gridIdx       = unique(ChEstPar.grid_cfg);
ChEstPar.nGrids        = length(ChEstPar.gridIdx);
ChEstPar.maxLength     = maxLength;
ChEstPar.symIdx_dmrs   = symIdx_dmrs;
ChEstPar.nl            = nl;
ChEstPar.portIdx       = portIdx;
ChEstPar.n_scid        = n_scid;
ChEstPar.remainderFlag = 0;

%%
% %START

% inititialize channel estimate:
H_est = zeros(Nf,L_UE,L_BS);
      

% check for small allocation
if nPrb <= 3
           
    ChEstPar.nPrb_dmrs     = nPrb;
    ChEstPar.startPrb_dmrs = startPrb;
    ChEstPar.nPrb_est      = nPrb;
    ChEstPar.startPrb_est  = 1;
 
    % launch a "thread block" for each BS antenna:
    R0_acc = 0;
    R1_acc = 0;
    for bs = 1 : L_BS     
        [H_est(:,:,bs),tmp_R0_acc, tmp_R1_acc] = apply_ChEst_LS_kernel(H_est(:,:,bs),Y(:,:,bs),r_dmrs,ChEstPar);
        R0_acc = R0_acc + tmp_R0_acc;
        R1_acc = R1_acc + tmp_R1_acc;
    end
    % average R0 and R1
    R0_avg = R0_acc/(6*ChEstPar.nPrb_dmrs)/ChEstPar.nl/L_BS;
    R1_avg = R1_acc/(6*ChEstPar.nPrb_dmrs-1)/ChEstPar.nl/L_BS;
   
else % including the case of ((mod(nPrb,4) == 0) && (nPrb >= 8))
    num_std_cluster =  floor(nPrb/4); % divide all PRBs into clusters of size 4 PRBs
    num_remaining_Prbs = mod(nPrb,4);
    if num_remaining_Prbs>0
        num_remaining_cluster = 1;
    else
        num_remaining_cluster = 0;
    end
    R0_acc = 0;
    R1_acc = 0;
    % for normal clusters
    for j = 1 : num_std_cluster
        ChEstPar.startPrb_dmrs = 4*(j-1) + startPrb;
        ChEstPar.nPrb_dmrs = 4;
        ChEstPar.startPrb_est = 1;
        ChEstPar.nPrb_est = 4;
        for bs = 1 : L_BS           
            [H_est(:,:,bs),tmp_R0_acc, tmp_R1_acc] = apply_ChEst_LS_kernel(H_est(:,:,bs),Y(:,:,bs),r_dmrs,ChEstPar);
            R0_acc = R0_acc + tmp_R0_acc;
            R1_acc = R1_acc + tmp_R1_acc;
        end
    end
    % for remaining cluster
    ChEstPar.startPrb_dmrs = 4*(num_std_cluster) + startPrb;
    ChEstPar.nPrb_dmrs = num_remaining_Prbs;
    ChEstPar.startPrb_est = 1;
    ChEstPar.nPrb_est = num_remaining_Prbs;
    for bs = 1 : L_BS           
        [H_est(:,:,bs),tmp_R0_acc, tmp_R1_acc] = apply_ChEst_LS_kernel(H_est(:,:,bs),Y(:,:,bs),r_dmrs,ChEstPar);
        R0_acc = R0_acc + tmp_R0_acc;
        R1_acc = R1_acc + tmp_R1_acc;
    end
    % average R0 and R1
    if ChEstPar.nl==1
        num_LS_REs_per_PRB = 6;
    else
        num_LS_REs_per_PRB = 3;
    end
    R0_avg = R0_acc/(num_LS_REs_per_PRB*4*num_std_cluster + num_LS_REs_per_PRB*num_remaining_Prbs*num_remaining_cluster)/ChEstPar.nl/L_BS;
    R1_avg = R1_acc/( (num_LS_REs_per_PRB*4-1)*num_std_cluster + (num_LS_REs_per_PRB*num_remaining_Prbs-1)*num_remaining_cluster )/ChEstPar.nl/L_BS;
    
    
end
   

return
