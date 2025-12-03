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

function [c0, P_prime, pm, nPaths] = R1_decoder(nPaths, cwTree, L, pm, s)

%Function decodes R1 codeword:

% 1.) Computes ML solution by slicing LLRs
% 2.) Compute four "near ML" children by flipping the two least reliable
% 3.) compute path metric for each child
% 4.) If nChildren > L, the the L best are kept

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


% use Chase II method to generate 4 "near ML" children for each path,
% computre their path-metrics:
pm_children = zeros(nPaths,4);
c0 = zeros(2^s,nPaths,4);

for p = 0 : (nPaths - 1)

    %first extract the stage LLRs
    LLR = get_cwLLR(cwTree,p,s);

    %Compute ML estimate by slicing LLRs:
    mlEst = sliceLLR(LLR);

    %identify the two least reliable bits:
    [lrb_LLR,lrb_idx] = sort(abs(LLR),'ascend');

    %compute the bit flips:
    f1 = mod(1 + mlEst(lrb_idx(1)),2);
    f2 = mod(1 + mlEst(lrb_idx(2)),2);

    %first candidite is ML solution:
    pm_children(p+1,1) = pm(p+1);
    c0(:,p+1,1) = mlEst;

    %second candidite flips 1st lrb bit of ML solution:
    pm_children(p+1,2) = pm(p+1) + lrb_LLR(1);
    c0(:,p+1,2) = mlEst;
    c0(lrb_idx(1),p+1,2) = f1;

    %third candidite flips 2nd lrb bit of ML solution:
    pm_children(p+1,3) = pm(p+1) + lrb_LLR(2);
    c0(:,p+1,3) = mlEst;
    c0(lrb_idx(2),p+1,3) = f2;    

    %fourth candidate flips 1st and 2nd lrb of ML solution:
    pm_children(p+1,4) = pm(p+1) + lrb_LLR(1) + lrb_LLR(2);
    c0(:,p+1,4) = mlEst;
    c0(lrb_idx(1),p+1,4) = f1;
    c0(lrb_idx(2),p+1,4) = f2; 

end

c0 = reshape(c0,2^s,4*nPaths);



% if 4*nPaths <= L, keep all four children
if (4*nPaths) <= L

    %pointer to parents:
    P_prime = zeros(L,1);
    P_prime(1 : 4*nPaths) = repmat((0:(nPaths-1))',4,1);


    nPaths = 4*nPaths;
    pm(1 : nPaths) = pm_children(:);
else

    % if 4*nPaths > L, keep the L best children
    pm_children = pm_children(:);
    [pm_children,idx] = sort(pm_children,'ascend');

    P_prime = mod(idx(1:L) - 1, L);
    pm      = pm_children(1:L);
    c0      = c0(:,idx(1:L));
    nPaths  = L;
end

P_prime = P_prime(1 : nPaths);

            
        
        
        
    



