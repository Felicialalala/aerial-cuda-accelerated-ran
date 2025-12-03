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

function cC2V = compute_cC2V(c,V2C,Zc,TannerPar)

%compute cC2V for check node c

%inputs:
%c    --> index of check node to be updated
%V2C  --> V2C message for check node c. Dim: Zc x nNeighors(c) x 2 (abs,sgn)
%Zc   --> lifting size

%ouputs:
%cC2V --> compressed C2V message for check node c. Dim: Zc x 4 (min1,min2,sgnPrd,min1_idx)


%%
%PARAMATERS

nNeighbors = TannerPar.numNeighbors(c);         % number of variable nodes connected to check node c

%%
%START

alg = 1;
if alg == 1
    cC2V = zeros(4, Zc); 
    V2C1 = V2C(:,1:nNeighbors,1).';
    V2C2 = V2C(:,1:nNeighbors,2).';
    [min1, min1_idx] = min(V2C1);
    for i = 1:Zc
        V2C1(min1_idx(i), i) = Inf;
    end
    [min2] = min(V2C1);
    sgnPrb = ones(1, Zc);
    for i = 1 : nNeighbors
        sgnPrb = sgnPrb.*V2C2(i,:);
    end
    cC2V(1,:) = min1;
    cC2V(2,:) = min2;
    cC2V(3,:) = sgnPrb;
    cC2V(4,:) = min1_idx; 
    cC2V = cC2V.';  
else
    cC2V = zeros(Zc,4);    
    for z = 1 : Zc
        min1 = Inf;
        min2 = Inf;
        sgnPrb = 1;

        for i = 1 : nNeighbors

            if V2C(z,i,1) <= min2 
                if V2C(z,i,1) <= min1
                    min2 = min1;
                    min1 = V2C(z,i,1);
                    min1_idx = i;
                else
                    min2 = V2C(z,i,1);
                end
            end

            sgnPrb = sgnPrb * V2C(z,i,2);
        end

        cC2V(z,1) = min1;
        cC2V(z,2) = min2;
        cC2V(z,3) = sgnPrb;
        cC2V(z,4) = min1_idx;   
    end    
end

return


        
        
      
                
