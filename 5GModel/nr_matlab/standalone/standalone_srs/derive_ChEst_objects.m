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

function W = derive_ChEst_objects(srsChEst)

% function derive SRS channel estimation objects (filters and sequences)

%outputs:
% srsChEst.W_cell --> cell containing estimation filters. 
%                           Dim: 4 x 1 (combOffset)

%%
%START

nPrbPerThreadBlock = srsChEst.nPrbPerThreadBlock;  % number of prb per threadblock
W = zeros(nPrbPerThreadBlock/2,nPrbPerThreadBlock*3,4); % (freq out) x (freq in) x (comb offset)

for combOffset = 0 : 3
    W(:,:,combOffset + 1) = compute_srs_ChEst_filter(combOffset,srsChEst);
end


        
        
