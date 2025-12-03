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

function r = build_freq_scrambling_sequence(c,M)

%function builds the frequency dmrs scrambling sequence from a Gold
%sequence

%inputs:
%c --> Gold sequence. Dim: 2*M x 1
%M --> length of desired scrambling sequence

%outputs:
%r --> frequency dmrs scrambling sequence. Dim: M x 1

%%
%START

r = zeros(M,1);

for i = 1 : M
    
    index = (i-1)*2;
    
    r(i) = 1 / sqrt(2) * ((1 - 2*c(index + 1)) + ...
        1i*(1 - 2*c(index + 2)));
    
end
    
