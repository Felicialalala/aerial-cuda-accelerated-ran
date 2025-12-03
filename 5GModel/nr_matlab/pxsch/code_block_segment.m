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

function TbCbs = code_block_segment(TbBits, C, K, F, K_prime, crc_table)

%function segments CRC encoded transport block into code blocks, adds filler
%and additional CRC if necassary.

%Follows 38.212 section 5.2.2

%inputs:
%TbBits --> CRC encoded transport block. Dim: B x 1

%outputs:
%TbCbs --> codeblock segmented transport block, possibly with added CRC and filler bits. Dim: K x C

% C = coding.C;             %number of codeblocks
% K = coding.K;             %number of systematic bits per codeblock
% F = coding.F;             %number of filler bits per codeblock
% K_prime = coding.K_prime; %number of CRC encoded bits per codeblock (no filler)

TbCbs = zeros(K,C);

if C == 1    
    %If only one code block, no CRC attached. Just add filler bits
    TbCbs(1 : (K - F)) = TbBits;
    TbCbs(K_prime + 1 : end) = -1;    
else    
    %First, split b among codeblocks
    TbCbs(1 : (K_prime - 24),:) = reshape(TbBits, K_prime - 24,C);    
    %Next, add CRC bits to each codeblock
    for c = 1 : C
        TbCbs(1 : K_prime,c) = add_CRC_LUT(TbCbs(1:(K_prime - 24),c),'24B',crc_table);
%         TbCbs(1 : K_prime,c) = crc_encode_mex(TbCbs(1:(K_prime - 24),c),'24B'); 
    end    
    %finally, add filler bits
    TbCbs(K_prime + 1 : end, :) = -1;    
end

return
