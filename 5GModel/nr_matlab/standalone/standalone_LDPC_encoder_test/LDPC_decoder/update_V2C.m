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

function update_V2C(c,APP,C2V,TannerPar)

%update V2C for a single check node

%inputs:
%c   --> index of check node to be updated
%APP --> current APP of variable nodes. Dim: Zc x nV
%C2V --> current C2V messages. Dim: Zc x nC x 19
%V2C --> current V2C messages. Dim: Zc x nC x 19

%%
%PARAMATERS

nNeighbors = TannerPar.nNeighbors(c);    % number of variable nodes connected to check node c
NeighborIdx = TannerPar.NeighborIdx;     % indicies of variable nodes connected to check node c. Dim: nNeighbors x 1 
NeighborShift = TannerPar.NeighborShift; % lists expansion cyclic shifts used by variable nodes connected to check node c. Dim: nNeighbors x 1 

%%
%START


for i = 1 : nNeighbors
    
    vIdx = NeighborIdx(i);     
    C2V(:,c,i) = circshift(APP(:,vIdx) - C2V(:,c,i),-NeighborShift(i));
    
end


