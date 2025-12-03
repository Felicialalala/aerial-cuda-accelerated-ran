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

function polCwChanItl = uplinkPolarRmItl(polarCbPrms, polCw)

% function performs uplink block-segmentation + rate-matching + output
% interleaving

%inputs:
% polarCbPrms   --> polar codeblock paramaters
% polCw         --> polar codeword. Dim: N_cb x 1 

%outputs:
% polCwRmItl --> interleaved and rate-matched codeblock. Dim: E_cb x 1

%%
% PARAMATERS

K_cw = polarCbPrms.K_cw; % number of input bits per codeword (data + crc + possible zero insertion)
N_cw = polarCbPrms.N_cw; % polar codeword size (power of two)
E_cw = polarCbPrms.E_cw; % number of rate-match bits per codeword

%%
%STEP 1
% Sub-block interleaving. 38.212 section 5.4.1.1

load('P1.mat');
subBlockItlCw = zeros(N_cw,1);
subBlockSize  = N_cw / 32;

for subBlockItlCwIdx = 0 : (N_cw - 1)
    subBlockIdx  = floor(subBlockItlCwIdx / subBlockSize);
    cwIdx     = P(subBlockIdx + 1) * subBlockSize + mod(subBlockItlCwIdx, subBlockSize);
    
    subBlockItlCw(subBlockItlCwIdx + 1) = polCw(cwIdx + 1);
end


%%
%STEP 2 
% rate-matching. 38.212 section 5.4.1.2 

polCwRm = zeros(E_cw, 1);

if E_cw >= N_cw % repetition
    
    for k = 0 : (E_cw - 1)
        polCwRm(k + 1) = subBlockItlCw(mod(k,N_cw) + 1);
    end
    
%     fprintf('\n rep! \n');
    
else
    
    if K_cw/E_cw <= 7/16 % puncturing
        
        for k = 0 : (E_cw - 1)
            polCwRm(k + 1) = subBlockItlCw(k + N_cw - E_cw + 1);
        end
        
%         fprintf('\n punct! \n');
        
    else % shortening
        
        for k = 0 : (E_cw - 1)
            polCwRm(k + 1) = subBlockItlCw(k + 1);
        end
        
%          fprintf('\n short! \n');
        
    end
end


%%
% %STEP 3
% % interleaving. 38.212 section: 5.4.1.3. Implmentation follows spec closely, however not
% % implmentation friendly.
% 
% T = ceil((-1 + sqrt(1 + 8*E_cb)) / 2);
% 
% B = -ones(T);
% bitIdx = 0;
% 
% for rowIdx = 0 : (T - 1)
%     for colIdx = 0 : (T - 1 - rowIdx)
%         if (bitIdx == E_cb)
%             B(rowIdx + 1, colIdx + 1) = -1;
%         else
%             B(rowIdx + 1, colIdx + 1) = polCbRm(bitIdx + 1);
%             bitIdx = bitIdx + 1;
%         end
%     end
% end
% 
% polCbRmItl = zeros(E_cb,1);
% bitIdx     = 0;
% for colIdx = 0 : (T - 1)
%     for rowIdx = 0 : (T - 1 - colIdx)
%         if(B(rowIdx + 1, colIdx + 1) ~= -1)
%             polCbRmItl(bitIdx + 1) = B(rowIdx + 1, colIdx + 1);
%             bitIdx = bitIdx + 1;
%         end
%     end
% end

% %%
% % ALTERNATE ITL
% % harder to understand, but maybe more implmentation friendly.
% 
% polCwChanItl = zeros(E_cb,1);
% 
% % Interleaved matrix buffer size:
% T     = ceil((-1 + sqrt(1 + 8*E_cb)) / 2);
% 
% % Constants for quadradic formula:
% b     = -(1 + 2*T);
% b_sqr = b^2;
% 
% % Region 1 boundaries:
% lastRmIdx          = E_cb - 1;
% lastRowIdxRegion1  = floor((-b - sqrt(b^2 - 8*lastRmIdx)) / 2);
% lastColIdxRegion1  = lastRmIdx - lastRowIdxRegion1 * T + (lastRowIdxRegion1 - 1) * lastRowIdxRegion1 / 2;
% nBitsRegion1       = (lastRowIdxRegion1 + 1) * (lastColIdxRegion1 + 1);
% 
% % Region 2 boundaries:
% firstColIdxRegion2 = lastColIdxRegion1 + 1;
% lastColIdxRegion2  = T - (lastRowIdxRegion1 - 1);
% 
% 
% for rmIdx = 0 : (E_cb - 1)
%     % First compute buffer indicies (rowIdx,colIdx)
%     rowIdx = floor((-b - sqrt(b_sqr - 8*(rmIdx))) / 2);
%     colIdx = rmIdx - rowIdx * T + (rowIdx - 1) * rowIdx / 2;
%     
%     % Based on buffer indicies, determine which region rm-bit placed
%     % into. Then compute interleaved index.
%     if(colIdx <= lastColIdxRegion1)
%         chanItlIdx = colIdx * (lastRowIdxRegion1 + 1) + rowIdx;
%         
%     elseif (colIdx <= lastColIdxRegion2)
%         chanItlIdx = nBitsRegion1 + (colIdx - firstColIdxRegion2) * lastRowIdxRegion1 + rowIdx;
%         
%     else
%         flippedColIdx = T - 1 - colIdx;
%         chanItlIdx     = rowIdx + E_cb - (flippedColIdx + 1) * (flippedColIdx + 2) / 2;
%     end
%     
%     % Interleave the rm-bit
%     polCwChanItl(chanItlIdx + 1) = polCwRm(rmIdx + 1);
% end



%%
% Alternate channel interleaver
% harder to understand, but maybe more implmentation friendly.

polCwChanItl = zeros(E_cw,1);

% Interleaved matrix buffer size:
T     = ceil((-1 + sqrt(1 + 8*E_cw)) / 2);

% Region 1 boundaries:
b                  = -(1 + 2*T);
lastRmIdx          = E_cw - 1;
lastRowIdxRegion1  = floor((-b - sqrt(b^2 - 8*lastRmIdx)) / 2);
lastColIdxRegion1  = lastRmIdx - lastRowIdxRegion1 * T + (lastRowIdxRegion1 - 1) * lastRowIdxRegion1 / 2;

% Region 1 sizes:
nBitsRegion1       = (lastRowIdxRegion1 + 1) * (lastColIdxRegion1 + 1);
nRowsRegion1       = lastRowIdxRegion1 + 1;
nColsRegion1       = lastColIdxRegion1 + 1;

% Region 2 sizes:
nRowsRegion2       = nRowsRegion1 - 1;
nColsRegion2       = ((T - nRowsRegion2 + 1) - nColsRegion1);
nBitsRegion1And2   = nBitsRegion1 + nColsRegion2 * nRowsRegion2;     


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
    
    % interleave the bit:
    polCwChanItl(chanItlIdx + 1) = polCwRm(rmIdx + 1);
end






