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

function H_est = apply_ChEst_MMSE_filter_main(H_LS_est,table,slotNumber,Nf,Nt,N_dmrs_id,...
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

[~, ~, L_BS] = size(H_LS_est);  % total number of bs antennas
L_UE = nl;               % max number of spatially multiplexed layers

if ((mod(nPrb,4) == 0) && (nPrb >= 8))
    W_lower = table.W_lower;
    W_middle = table.W_middle;
    W_upper = table.W_upper; 
else
    W_lower  = table.W4_lower;   % lower ChEst filter.     Dim: 25 x 24
    W_middle = table.W4_middle;  % middle ChEst filter.    Dim: 25 x 24
    W_upper  = table.W4_upper;   % upper ChEst filter.     Dim: 25 x 24
end
W3       = table.W3;         % three prb ChEst filter. Dim: 37 x 18
W2       = table.W2;         % two prb ChEst filter.   Dim: 25 x 12
W1       = table.W1;         % one prb ChEst filter.   Dim: 13 x 6

if ((mod(nPrb,4) == 0) && (nPrb >= 8))
    shiftSeq   = table.shiftSeq;
    unShiftSeq = table.unShiftSeq;
else
    shiftSeq   = table.shiftSeq4;
    unShiftSeq = table.unShiftSeq4;
end

%%
%SEQUENCES

% scrambling sequence:

% if enableTfPrcd == 0
%     r_dmrs  = gen_dmrs_sym(slotNumber, Nf, Nt, N_dmrs_id);
% else
%     [r_dmrs, ~, ~] = gen_dmrs_dftsofdm(Nf, startPrb, nPrb, N_slot_frame, N_symb_slot, symIdx_dmrs, ...
%         slotNumber, puschIdentity, groupOrSequenceHopping);
% end

% % compute shift sequence:
% tau        = (2.0*10^(-6) - 2*10^(-6)/10) / 2; % delay shift
% 
% f_dmrs     = 0 : 2 : (12*4 - 1);
% f_dmrs     = df * f_dmrs;
% 
% f_data     = -1 : (12*4 - 1);
% f_data     = df * f_data;
% 
% shiftSeq   = exp(2*pi*1i*tau*f_dmrs).';
% unShiftSeq = exp(-2*pi*1i*tau*f_data).';

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

    % configure "thread block" for small filter:
    switch nPrb
        case 1
          ChEstPar.W = W1;
        case 2
          ChEstPar.W = W2;
        case 3
          ChEstPar.W = W3;
    end
    global SimCtrl;
    if SimCtrl.alg.ChEst_use_vPRBs
        ChEstPar.W = W53;
    end
           
    ChEstPar.nPrb_dmrs     = nPrb;
    ChEstPar.startPrb_dmrs = startPrb;
    ChEstPar.nPrb_est      = nPrb;
    ChEstPar.startPrb_est  = 1;
 
    % launch a "thread block" for each BS antenna:
    for bs = 1 : L_BS     
        H_est(:,:,bs) = apply_ChEst_MMSE_filter_kernel(H_est(:,:,bs),H_LS_est(:,:,bs),shiftSeq,unShiftSeq,ChEstPar);
    end

elseif ((mod(nPrb,4) == 0) && (nPrb >= 8))
    % Lower band
    ChEstPar.W = W_lower;
    ChEstPar.nPrb_dmrs = 8;
    ChEstPar.startPrb_dmrs = startPrb;
    ChEstPar.nPrb_est = 4;
    ChEstPar.startPrb_est = 1;   
    % launch a "thread block" for each BS antenna:
    for bs = 1 : L_BS    
        H_est(:,:,bs) = apply_ChEst_MMSE_filter_kernel(H_est(:,:,bs),H_LS_est(:,:,bs),shiftSeq,unShiftSeq,ChEstPar);
    end
    
    % Middle band
    num_middle_cluster = floor( (nPrb - 8) / 4);    
    for i = 1 : num_middle_cluster
        % configure thread block:
        ChEstPar.W = W_middle;
        ChEstPar.nPrb_dmrs = 8;
        ChEstPar.startPrb_dmrs = startPrb + i*4 - 2;
        ChEstPar.nPrb_est = 4;
        ChEstPar.startPrb_est = 3;%startPrb + i*4;        
        % launch a "thread block" for each BS antenna:
        for bs = 1 : L_BS
            H_est(:,:,bs) = apply_ChEst_MMSE_filter_kernel(H_est(:,:,bs),H_LS_est(:,:,bs),shiftSeq,unShiftSeq,ChEstPar);
        end        
    end
    
    % Upper band
    ChEstPar.W = W_upper;
    ChEstPar.nPrb_dmrs = 8;
    ChEstPar.startPrb_dmrs = startPrb + nPrb - 8;
    ChEstPar.nPrb_est = 4;
    ChEstPar.startPrb_est = 5;%startPrb + nPrb - 4;    
    % launch a "thread block" for each BS antenna:
    for bs = 1 : L_BS
        H_est(:,:,bs) = apply_ChEst_MMSE_filter_kernel(H_est(:,:,bs),H_LS_est(:,:,bs),shiftSeq,unShiftSeq,ChEstPar);
    end

else
    % for a "large" FEG, split up FEG into
    % smaller prb clusters:

    % check for remainder:
    if mod(nPrb - 4, 2) == 1
        remainder_flag = 1;
    else
        remainder_flag = 0;
    end
    
    % compute number of standard *--* prb clusters:
    if remainder_flag == 0
        num_std_cluster = (nPrb - 4) / 2;
    else
        num_std_cluster = (nPrb - 3) / 2;
    end
        
        
    % configure "thread block" for lower edge:
    ChEstPar.W = W_lower;
    ChEstPar.startPrb_dmrs = startPrb;
    ChEstPar.nPrb_dmrs = 4;
    ChEstPar.startPrb_est = 1;
    ChEstPar.nPrb_est = 2;

    for bs = 1 : L_BS
        H_est(:,:,bs) = apply_ChEst_MMSE_filter_kernel(H_est(:,:,bs),H_LS_est(:,:,bs),shiftSeq,unShiftSeq,ChEstPar);
   end
   
   % configure "thread blocks" for middle prbs
    for j = 1 : num_std_cluster
        ChEstPar.W = W_middle;
        ChEstPar.startPrb_dmrs = 2*(j-1) + 1 + startPrb;
        ChEstPar.nPrb_dmrs = 4;
        ChEstPar.startPrb_est = 2;
        ChEstPar.nPrb_est = 2;

        for bs = 1 : L_BS           
            H_est(:,:,bs) = apply_ChEst_MMSE_filter_kernel(H_est(:,:,bs),H_LS_est(:,:,bs),shiftSeq,unShiftSeq,ChEstPar);
        end
    end

    % configure "thread blocks" for upper edge:
    if remainder_flag == 0
        ChEstPar.W = W_upper;
        ChEstPar.startPrb_dmrs = startPrb + nPrb - 4;
        ChEstPar.nPrb_dmrs = 4;
        ChEstPar.startPrb_est = 3;
        ChEstPar.nPrb_est = 2;

        for bs = 1 : L_BS
            H_est(:,:,bs) = apply_ChEst_MMSE_filter_kernel(H_est(:,:,bs),H_LS_est(:,:,bs),shiftSeq,unShiftSeq,ChEstPar);
        end
    else
        ChEstPar.W = W_upper;
        ChEstPar.startPrb_dmrs = startPrb + nPrb - 4;
        ChEstPar.nPrb_dmrs = 4;
        ChEstPar.startPrb_est = 4;
        ChEstPar.nPrb_est = 1;
        ChEstPar.remainderFlag = 1;
        
        for bs = 1 : L_BS
            H_est(:,:,bs) = apply_ChEst_MMSE_filter_kernel(H_est(:,:,bs),H_LS_est(:,:,bs),shiftSeq,unShiftSeq,ChEstPar);
        end
    end
end

H_est = permute(H_est,[3 2 1]);
        
if numDmrsCdmGrpsNoData == 1
    H_est = H_est * sqrt(2);
end    

return
