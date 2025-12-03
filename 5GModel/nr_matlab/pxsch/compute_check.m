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

function ci = compute_check(i,Zc,cb,TannerPar)

%computes the value of the i'th check node

%inputs:
%i  --> index of check node
%Zc --> lifting size
%cb --> codeblock. Dim: Zc x nV


%outputs:
%ci --> value of i'th check node. Dim: Zc x 1

%%
%PARAMATERS

%tanner paramaters:
numNeighbors = TannerPar.numNeighbors(i);   %number of variable nodes adjacent to i'th check node
NeighborIdx = TannerPar.NeighborIdx{i};     %indicies of variable nodes adjacent to i'th check node. Dim: 1 x numNeighbors
NeighborShift = TannerPar.NeighborShift{i}; %expansion shift by variable nodes adjacent to i'th check node. Dim: 1 x numNeighbors

%%
%START

ci = zeros(Zc,1);

for j = 1 : numNeighbors
    
    v = NeighborIdx(j);   %variable node of neighbor
    p = NeighborShift(j); %permutation of neighbor
    
    ci = ci + circshift(cb(:,v),-p);
    
end


end




