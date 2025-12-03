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

function out = scramble(in, E, nRNTI, nID, PuschMsgaFlag, nRAPID)
%
% This function scrambles an encoded (and rate-matched) bit sequence,
%
% Input:    in:             the bit sequence to be scrambled
%           E:              the rate-matched bit sequence length
%           nRNTI:          RNTI as described in Sec. 6.3.1.1, TS38.211
%           nID:            dataScramblingId or N_ID^cell as described in Sec. 6.3.1.1, TS38.211
%           PuschMsgaFlag:  flag for PUSCH msgA as described in Sec. 6.3.1.1, TS38.211
%           nRAPID:         the index of the random-access preamble transmitted for msgA
%
% Output:   out:            the scrambled bit sequence
%
%

%% validate inputs
if length(in) ~= E
    error('Invalid input: the length of input bit sequence does not match with the provided sequence length E');
end

%% scramble
if PuschMsgaFlag %% Sec. 6.3.1.1, TS 38.211
    c_init = nRNTI*2^16 + nRAPID*2^10 + nID;
else
    c_init = nRNTI*2^15 + nID;
end

if E < 4
    c = build_Gold_sequence(c_init, 4);
else
    c = build_Gold_sequence(c_init, E); 
end

out = zeros(E, 1);

for bitIdx = 1:E
    bitTemp = in(bitIdx);
    if bitTemp == -1 % 'x'
        out(bitIdx) = 1;
    elseif bitTemp == -2 % 'y'
        out(bitIdx) = out(bitIdx - 1);
    else
        out(bitIdx) = mod(bitTemp + c(bitIdx),2);
    end
end
end
