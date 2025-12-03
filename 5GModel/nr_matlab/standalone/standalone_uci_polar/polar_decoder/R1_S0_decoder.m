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

function [c0, P_prime, pm, nPaths] = R1_S0_decoder(nPaths, cwTree, L, pm)

%Function decodes leaf node for each active path:

% -Each path has two potential child solutions: 0 or 1 leaf. 
% -Function computes path metric for each child
% -If nChildren > L, the the L best are kept

%inputs:
% nPaths    --> current number of active paths
% cwTree    --> codeword tree object
% L         --> max number of active paths
% pm        --> current path metric. Dim: L x 1
% s         --> current stage

%outputs:
% c0       --> children bit estiamtes of current bit. Dim: L x 1
% P_prime  --> pointer from child to parent. Dim: L x 1
% pm       --> path metric of children
% nPaths   --> updated number of paths


%%
%START

%first compute path metric of children:
pm_children = zeros(nPaths,2);

%compute child path metrics:
for p = 0 : (nPaths - 1)

    % extract LLR of the current bit:
    LLR = get_cwLLR(cwTree, p ,0); 

    % compute expected bit:
    if LLR >= 0 
        b_expected = 0;
    else
        b_expected = 1;
    end

    for j = 0 : 1    
        % penalize unexpected children:
        if (b_expected == j)
            pm_children(p+1, j+1) = pm(p+1);
        else
            pm_children(p+1, j+1) = pm(p+1) + abs(LLR);
        end 
    end
end
    

%if nPaths < L, keep both children, doubling number of paths:
if nPaths < L
    P_prime = zeros(L,1);
    P_prime(1:2*nPaths) = repmat((0:(nPaths-1))',2,1);

    c0 = [ zeros(1,nPaths) ones(1,nPaths)];
    nPaths = 2*nPaths;

    pm(1 : nPaths) = pm_children(:);

else
    % nPaths = L, keep the L best children
    pm_children = pm_children(:);
    [pm_children,idx] = sort(pm_children,'ascend');

    P_prime = mod(idx(1:L) - 1, L);
    c0      = floor((idx(1:L) - 1) / L)';
    pm      = pm_children(1 : L);

end

P_prime = P_prime(1 : nPaths);






            
        
        
        
    



