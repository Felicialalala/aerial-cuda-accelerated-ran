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

function a = pbch_payload_coding(x, SFN, n_hf, block_idx, N_id, L_max, k_SSB)

x = dec2bin(x, 24) - '0';

msb_k_ssb = (int8 (k_SSB >= 16));
SFN_LSB = dec2bin(SFN, 4)-'0';
a0_3 = SFN_LSB(end-3:end);
a4 = n_hf;
if L_max == 64
    block_idx_LSB = dec2bin(block_idx, 6)-'0';
    a5_7 = block_idx_LSB(end-5:end-3);
else
    a5_7 = [msb_k_ssb, zeros(1,2)];
end
abar = [x(:)', a0_3, a4, a5_7];

j_SFN = 0;
j_HRF = 10;
j_SSB = 11;
j_other = 14;
G = [16 23 18 17 8 30 10 6 24 7 0 5 3 2 1 4 9 11 12 13 14 15 19 20 21 22 25 26 27 28 29 31];

v = a0_3(end-2)*2 + a0_3(end-1);
if L_max == 4 || L_max == 8
    M = 32-3;
elseif L_max == 64
    M = 32-6;
end
c = build_Gold_sequence(N_id,v*M+32);
c = c(v*M+1:end);
noScram = zeros(1, 32);
a = zeros(1, 32);

for i = 0:31    
    if i >= 24 && i < 28
        a(G(j_SFN+1)+1) = abar(i+1);
        if (i == 25 || i == 26)
            noScram(G(j_SFN+1)+1) = 1;
        end
        j_SFN = j_SFN + 1;        
    elseif i == 28
        a(G(j_HRF+1)+1) = abar(i+1);        
        noScram(G(j_HRF+1)+1) = 1;
    elseif i >= 29 && i <= 31
        a(G(j_SSB+1)+1) = abar(i+1);
        if (L_max == 64)
            noScram(G(j_SSB+1)+1) = 1;
        end
        j_SSB = j_SSB + 1;
    else
        if i == 0 || i >= 7
            a(G(j_other+1)+1) = abar(i+1);                                 
            j_other = j_other + 1;
        else
            a(G(j_SFN+1)+1) = abar(i+1);            
            j_SFN = j_SFN + 1;   
        end
    end
end

j = 0;
for i = 0:31    
    if ~noScram(i+1)
        a(i+1) = mod(a(i+1) + c(j+1), 2);
        j = j+1;
    end
end
