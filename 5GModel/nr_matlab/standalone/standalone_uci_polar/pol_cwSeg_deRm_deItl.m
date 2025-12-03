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

function cwLLRs = pol_cwSeg_deRm_deItl(polarUciSegPrms, deRmDeItlDynDesc, uciSegLLRs)

% Function takes as input LLRs of a polar encoded uci segment and performs
% cw segmentation, de-rate-matching, and de-interleaving. Produces LLRs for
% the one or two polar codewords.

% inputs:
% polarUciSegPrms  --> polar uci segment paramaters
% deRmDeItlDynDesc --> constants for polar deRm + deItl kernel
% uciSegLLRs       --> LLRs of polar encoded uci segment. Dim: E_seg x 1

% outputs:
% cwLLRs --> LLRs of constituent polar codewords. Dim: N_cb x nCbs

%%
%PARAMATERS

% interleave index
load('P1.mat');

% sizes
E_cw = polarUciSegPrms.E_cw;  % number of rate-matched bits per codeword
N_cw = polarUciSegPrms.N_cw;  % codeword size
nCbs = polarUciSegPrms.nCbs;  % number of codeblock uci segment split into. 1 or 2.

subBlockSize       = deRmDeItlDynDesc.subBlockSize;       
rmMethod           = deRmDeItlDynDesc.rmMethod;          % rate-matching method. 0 -> rep, 1->punct, 2->short
T                  = deRmDeItlDynDesc.T;                 % size of channel Itl matrix buffer
nRowsRegion1       = deRmDeItlDynDesc.nRowsRegion1;      % number of rows in region 1
nColsRegion1       = deRmDeItlDynDesc.nColsRegion1;      % number of columns in region 1
nBitsRegion1       = deRmDeItlDynDesc.nBitsRegion1;      % number of bits in region 1
nRowsRegion2       = deRmDeItlDynDesc.nRowsRegion2;      % number of rows in region 2
nBitsRegion1And2   = deRmDeItlDynDesc.nBitsRegion1And2;  % number of bits in region 1 and 2

%%
%START

if (rmMethod == 2)  % Shortened codeword bits known by decoder to be 0, initialize their LLRs to large value  
    cwLLRs = 20*ones(N_cw, nCbs);
else% Puncuted bits uknown to decoder, intialize their LLRs to 0
    cwLLRs = zeros(N_cw, nCbs);
end



for cbIdx = 0 : (nCbs - 1)
        
    %%
    % deChannel deInterleave + deRateMatch
    
    for chanItlIdx = 0 : (E_cw - 1)
        % First determine region. Based on region, compute row/col index:
        if (chanItlIdx < nBitsRegion1)
            colIdx = floor(chanItlIdx / nRowsRegion1);
            rowIdx = mod(chanItlIdx, nRowsRegion1);
        elseif (chanItlIdx < nBitsRegion1And2)
            colIdx = floor((chanItlIdx - nBitsRegion1) / nRowsRegion2) + nColsRegion1;
            rowIdx = mod(chanItlIdx - nBitsRegion1, nRowsRegion2);
        else
            flippedIdx    = E_cw - 1 - chanItlIdx;
            flippedColIdx = floor((-1 + sqrt(1 + 8 * flippedIdx)) / 2);
            flippedRowIdx = flippedIdx - flippedColIdx * (flippedColIdx + 1) / 2;  
            colIdx        = T - 1 - flippedColIdx;
            rowIdx        = flippedColIdx - flippedRowIdx; 
        end
        
       % Compute rate-matched index:
        rmIdx = colIdx + T * rowIdx - rowIdx * (rowIdx - 1) / 2;
        
        % Extract rmIdx LLR (this is channel deInterleaving)
        rmLLR = uciSegLLRs(cbIdx * E_cw + chanItlIdx + 1);
        
        % Write rmIdx LLR to cwLLR buffer (this is deRateMatching)
        switch rmMethod
            case 0 % Repetition, add LLRs of repeated bits
                subBlockItlCwIdx = mod(rmIdx, N_cw);
                subBlockIdx      = floor(subBlockItlCwIdx / subBlockSize);
                cwIdx            = P(subBlockIdx + 1) * subBlockSize + mod(subBlockItlCwIdx, subBlockSize);
                
                cwLLRs(cwIdx + 1, cbIdx + 1) = cwLLRs(cwIdx + 1, cbIdx + 1) + rmLLR; % Depending on thread sizes, this will need to be atomic.
          
            case 1 % Puncturing
                subBlockItlCwIdx = N_cw - E_cw + rmIdx;
                subBlockIdx      = floor(subBlockItlCwIdx / subBlockSize);
                cwIdx            = P(subBlockIdx + 1) * subBlockSize + mod(subBlockItlCwIdx, subBlockSize);
         
                cwLLRs(cwIdx + 1, cbIdx + 1) = rmLLR;
                
            case 2 % Shortening
                subBlockItlCwIdx = rmIdx;
                subBlockIdx      = floor(subBlockItlCwIdx / subBlockSize);
                cwIdx            = P(subBlockIdx + 1) * subBlockSize + mod(subBlockItlCwIdx, subBlockSize);
               
                cwLLRs(cwIdx + 1, cbIdx + 1) = rmLLR;
        end
    end % end rmIdx loop
end % end cbIdx loop

% clip LLRs
cwLLRs(cwLLRs > 100)  = 100;
cwLLRs(cwLLRs < -100) = -100;



end

    
    
    
    
    
    
    
    
        
        
                
        
        
        
        
        
        
    
    
    
    
    

    


