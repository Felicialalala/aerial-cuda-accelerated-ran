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

function  [cwBitTypes,cwTreeTypes] = compBitTypesKernel(polarUciSegPrms)

% function computes polar codeword bit types for a specific uci segment

%output:
% cwBitTypes(i) gives type of the "i-th" bit. Dim: N x 1
% 0: frozen
% 1: payload
% 2: parity

%%
% EXTRACT PARAMATERS

E = polarUciSegPrms.E_cw;    % number of rateMatch bits per codeword
K = polarUciSegPrms.K_cw;    % cw input size (info + CRC + possible zero insertion)
N = polarUciSegPrms.N_cw;    % codeword size
n = polarUciSegPrms.n_cw;    % n_cw = log2(N_cw)


%%
%% OUTPUT MEMORY

cwBitTypes = zeros(N,1);

%%
%DETERMINE PARITY BITS

n_pc   = 0;
wmFlag = 0;

if((K >= 18) && (K <= 25))
    n_pc = 3;
    
    if((E - K + 3) > 192)
       wmFlag = 1;
    else
        wmFlag = 0;
    end
end

if((K + n_pc) > E)
    warning('Configuration unsupported by 3gpp spec: (K + n_pc) > E');  
end

% force number information and parity bits to enable coding (WARNING! THIS IS NOT FOLLOWING 3GPP SPEC)
if((K + n_pc) > E)
    if(K > E)
        K = E;
    else
        n_pc = E - K;
    end
end

%%
% COMPUTE Q_IN
% Q_IN is the set of K + n_pc most reliable bit indicies. This section can be reused from the
% polar encoder.

% STEP 1: load the polar sequence
load('Q_N_max.mat');

% STEP 2: remove indicies >= N, while maintaining order
Q_0N = Q_N_max(Q_N_max < N);
 
% STEP 3: compute forbidden indicies 
Q_FN_temp = [];
if (E < N)
    
    block_size = N / 32;
    nBlocks = floor( (N - E ) / block_size );
    r = (N - E) - nBlocks * block_size;

    if (K/E <= 7/16)
     
        load('I_fwd.mat');
        interval_1_start = block_size * (I_fwd(nBlocks+1,1) - 1);
        interval_1_end = block_size * I_fwd(nBlocks+1,2) - 1;
        interval_2_start = block_size * (I_fwd(nBlocks+1,3) - 1);
        interval_2_end = block_size * (I_fwd(nBlocks+1,4) - 1) - 1 + r;
        
        if (E >= 3*N/4)
            interval_3_start = 0;
            interval_3_end = ceil(3*N/4 - E/2) - 1;       
        else
            interval_3_start = 0;
            interval_3_end = ceil(9*N/16 - E/4) - 1;
        end
        
        Q_FN_temp = [ (interval_1_start : interval_1_end) ...
                        (interval_2_start : interval_2_end) ...
                        (interval_3_start : interval_3_end)];
        
    else
        
        load('I_bwd.mat');
        interval_1_start = block_size * (I_bwd(nBlocks+1,1) - 1);
        interval_1_end = block_size * I_bwd(nBlocks+1,2) - 1;
        interval_2_start = block_size * I_bwd(nBlocks+1,3) - r;
        interval_2_end = block_size * I_bwd(nBlocks+1,4) - 1;
        
        Q_FN_temp = [ (interval_1_start : interval_1_end) ...
                (interval_2_start : interval_2_end)];
    end
end

% STEP 4: remove forbidden indicies, while maintaining order
Q_IN_tmp = setdiff(Q_0N,Q_FN_temp,'stable');

% STEP 5: extract K + n_pc most reliable indicies
Q_IN = Q_IN_tmp(end - (K + n_pc) + 1 : end);

%%
% SET DATA TYPES

cwBitTypes(Q_IN + 1) = 1; % some may be converted to parity in next step

%%
% SET PARITY TYPES 

% first (n_pc - wmFlag) parity bits placed in least reliable indicies
pc_Idxs = Q_IN(1 : (n_pc - wmFlag));
cwBitTypes(pc_Idxs + 1) = 2;

if(wmFlag)
    load('wmArrays.mat');
    switch N
        case 32
            wmArray = wmArray_n5;
        case 64
            wmArray = wmArray_n6;
        case 128
            wmArray = wmArray_n7;
        case 256
            wmArray = wmArray_n8;
        case 512
            wmArray = wmArray_n9;
        case 1024
            wmArray = wmArray_n10;
    end
    
    % If wmFlag active, a single parity bit is placed in remaing bit index with
    % lowest weight metric. Ties broken by taking bit index with highest
    % reliablity.
    minWm    = Inf;
    minWmIdx = 0;
    
    for relIdx = n_pc : (K - 1)
        bitIdx   = Q_IN(relIdx + 1);
        bitIdxWm = wmArray(bitIdx + 1);
        
        if(bitIdxWm <= minWm)
            minWm    = bitIdxWm;
            minWmIdx = bitIdx;
        end
    end
    
    cwBitTypes(minWmIdx + 1) = 2;
end

%%
%COMPUTE TREE TYPES

cwTreeTypes = zeros(2*N - 2,1);
          
 %initialize stage "0" types using cwBitTypes:
 s = 0;
 for subTreeIdx = 0 : (N - 1)
     cwTreeTypes = setType(cwTreeTypes, cwBitTypes(subTreeIdx + 1), subTreeIdx, s, n);
 end

 %propogte using buttefly structure:
 for s = 1 : (n - 1)
     for subTreeIdx = 0 : (2^(n - s) - 1)

         type0 = getType(cwTreeTypes, 2*subTreeIdx  , s-1, n);
         type1 = getType(cwTreeTypes, 2*subTreeIdx+1, s-1, n);

         if (type0 == 0) &&  (type1 == 0)
             cwTreeTypes = setType(cwTreeTypes, 0, subTreeIdx, s, n);
%                  
         elseif (type0 == 1) &&  (type1 == 1)
             cwTreeTypes = setType(cwTreeTypes, 1, subTreeIdx, s, n);

         else
             cwTreeTypes = setType(cwTreeTypes, 3, subTreeIdx, s, n);
         end

     end
 end
end

%%
%EXTRACT CODEWORD TYPE
% Get the coding type for a subtree

function type = getType(cwTypes, subTreeIdx, idxStage, n)          
  e = (n - 1 - idxStage);
  type = cwTypes(2^(e+1) - 2 + subTreeIdx + 1);
end

%%
%SET TYPE
% Set the coding type for a subtree

function cwTypes = setType(cwTypes, type, subTreeIdx, idxStage, n)           
    e = (n - 1 - idxStage);
    cwTypes(2^(e+1) - 2 + subTreeIdx + 1) = type;
end 











