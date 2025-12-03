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

matDim = 16;
halfMatDim = matDim/2;

A = zeros(matDim,matDim);

% (N_LAYERS*N_LAYERS)/2
matDim*(2 + matDim*2)/2

if 1
   nWarpsPerThrdBlk = 16; %16;
   nWarpThreads = 32;
   nLayers = 16;
   nRxBBPorts = 64;
   
   nCols = nLayers;
   nRows = nRxBBPorts;
   nWarpsPerCol = nRows/nWarpThreads;
   nColsPerThrdBlk = nWarpsPerThrdBlk/nWarpsPerCol;
   nIterToCompute = nCols/nColsPerThrdBlk;
   
   idxMetaData = zeros(nWarpsPerThrdBlk,nIterToCompute,2);  
   
   fprintf('iterIdx warpIdx warpGrpIdx colIdx rowIdx\n');
   for i = 0:nIterToCompute-1
       for k = 0:nWarpsPerThrdBlk-1
           colIdx = (fix(k/nWarpsPerCol)*(nIterToCompute/2)) + mod(i, nIterToCompute/2);
           if(i >= fix(nIterToCompute/2))
               colIdx = nCols - colIdx - 1;
           end
           rowIdx = mod(k, nWarpsPerCol);
           fprintf('  %2d       %2d        %2d     %2d     %2d\n', i, k, j, colIdx, rowIdx);
           idxMetaData(k+1,i+1,1) = colIdx;
           idxMetaData(k+1,i+1,2) = rowIdx;
       end
   end         
   sum(idxMetaData,3)
end
%sum(idxMetaData,2,2)

if 1
outerLoopCnt = max(round(nTriMatElem/nWarpsPerThrdBlk), 1);
innerLoopCnt = nRxBBPorts/nWarpThreads;
nTriMatElem = matDim*(matDim+1)/2;

iRow = 1;
iCol = iRow;
rowEndMarker = matDim-1;

for i = 0:outerLoopCnt-1
    for j = 0:nWarpsPerThrdBlk-1
        k = i*nWarpsPerThrdBlk + j;
        if k < nTriMatElem 
            if k > rowEndMarker
                rowEndMarker = rowEndMarker + matDim - iRow;
                iRow = iRow+1;
                iCol = iRow;
            end
            A(iRow,iCol) = k;
            iCol = iCol + 1;
        end
    end
end

end

