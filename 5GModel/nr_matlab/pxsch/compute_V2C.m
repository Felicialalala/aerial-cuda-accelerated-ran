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

function V2C = compute_V2C(c,APP,C2V,Zc,alpha,TannerPar)

%compute V2C for a single check node

%inputs:
% c     --> index of check node to be updated
% APP   --> current APP of variable nodes. Dim: Zc x nV
% C2V   --> current C2V messages. Dim: Zc x nC x 19
% Zc    --> lifting size
% alpha --> normalization constant

%outputs:
%V2C --> V2C messages for check node c. Dim: Zc x nNeighbors(c) x 2. (abs , sgn)

%%
%PARAMATERS

numNeighbors = TannerPar.numNeighbors(c);       % number of variable nodes connected to check node c
NeighborIdx = TannerPar.NeighborIdx{c};     % indicies of variable nodes connected to check node c. Dim: nNeighbors x 1 
NeighborShift = TannerPar.NeighborShift{c}; % lists expansion cyclic shifts used by variable nodes connected to check node c. Dim: nNeighbors x 1 

%%
%START

V2C = ones(Zc,numNeighbors,2);

for i = 1 : numNeighbors
    
    vIdx = NeighborIdx(i);     
    
    v2c = circshift(APP(:,vIdx) - alpha*C2V(:,c,i),-NeighborShift(i));
    
    V2C(:,i,1) = abs(v2c);
    V2C((v2c < 0),i,2) = -1; 
    
end


