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

function [c0, P_prime, pm] = R0_decoder(nPaths, cwTree, L, pm, s)

% For each path function decodes R0 codeword and updates path metric

%inputs:
% nPaths    --> current number of active paths
% cwTree    --> codeword tree object
% L         --> max number of active paths
% pm        --> current path metric. Dim: L x 1
% s         --> current stage

%outputs:
% c0       --> children bit estiamtes of current bit. Dim: L x 1
% P0       --> pointer from child to parent. Dim: L x 1
% pm       --> path metric of children
% nPaths   --> updated number of paths

%%
%START

% the R0 codeword is all 0's:
c0      = zeros(2^s,nPaths);

% trivial pointers:
P_prime             = zeros(L,1);
P_prime(1 : nPaths) = (0 : (nPaths-1))';

% update the path-metric:
for p = 0 : (nPaths - 1)
    LLR = get_cwLLR(cwTree,p,s);
    
    for i = 0 : (2^s - 1)
        % penalize path metric if 0 bit "unexpected"
        if LLR(i+1) < 0 
            pm(p+1) = pm(p+1) + abs(LLR(i+1));     
        end
    end    
end

P_prime = P_prime(1 : nPaths);




