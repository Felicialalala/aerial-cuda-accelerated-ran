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

function C2V = update_C2V(c,cC2V,C2V,V2C,TannerPar)

%function updates the C2V message for check node c

%inputs:
%c    --> index of check node to be updated
%cC2V --> compressed C2V message for check node c. Dim: Zc x 4 (min1,min2,sgnPrd,min1_idx)
%V2C  --> V2C message for check node c. Dim: Zc x nNeighors(c) x 2 (abs,sgn)
%Zc   --> lifting size


%%
%PARAMATERS

nNeighbors = TannerPar.nNeighbors(c);         % number of variable nodes connected to check node c
NeighborShift = TannerPar.NeighborShift{c}; % lists expansion cyclic shifts used by variable nodes connected to check node c. Dim: nNeighbors x 1

%%
%START

for i = 1 : nNeighbors
    
    
    c2v_abs = cC2V(:,1); 
    idx = (cC2V(:,4) == i);
    c2v_abs(idx) = cC2V(idx,2);
    
    c2v_sgn = cC2V(:,3) .* V2C(:,i,2);
    
    C2V(:,c,i) = circshift(c2v_abs .* c2v_sgn,NeighborShift(i));
    
end


    
    
