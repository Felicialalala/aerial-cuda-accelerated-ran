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

function out = SimplexDescramble(inLLRs, K, E, Qm, nRNTI, nID, PuschMsgaFlag, nRAPID)
%
% This function descrambles a receiver LLR array,
%
% Input:    inLLRs:         the array of LLRs to be descrambled
%           K:              the number of information bits
%           E:              the rate-matched LLR sequence length
%           Qm:             modulation order, should be in {1, 2, 4, 6, 8}
%           nRNTI:          RNTI as described in Sec. 6.3.1.1, TS38.211
%           nID:            dataScramblingId or N_ID^cell as described in Sec. 6.3.1.1, TS38.211
%           PuschMsgaFlag:  flag for PUSCH msgA as described in Sec. 6.3.1.1, TS38.211
%           nRAPID:         the index of the random-access preamble transmitted for msgA
%
% Output:   out:            the descrambled LLR array
%
%

%% validate inputs
if length(inLLRs) ~= E
    error('Invalid input: the length of input LLR sequence does not match with the provided sequence length E');
end

if Qm ~= 1 && Qm ~= 2 && Qm ~= 4 && Qm ~= 6 && Qm ~= 8
    error('Invalid input: Modulation order Qm must be in {1, 2, 4, 6, 8}');
end

if K ~= 1 && K ~= 2
    error('Invalid input: Simplex code only applies to 1-bit or 2-bit information');
end

if K == 1
    if E < Qm
        error('Invalid input: For 1-bit information, the length of rate-matched sequence E must be no less than Qm');
    end
end

%% descramble

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

if K == 1
    if Qm == 1
        for bitIdx = 1:E
            out(bitIdx) = (1 - 2*c(bitIdx)) * inLLRs(bitIdx); % flip LLR if c(bitIdx) == 1
        end
    else
        for bitIdx = 1:E
            temp = mod(bitIdx, Qm);
            if temp == 1
                out(bitIdx) = (1 - 2*c(bitIdx)) * inLLRs(bitIdx); % flip LLR if c(bitIdx) == 1
            elseif temp == 2
                out(bitIdx) = (1- 2*c(bitIdx - 1)) * inLLRs(bitIdx); % flip LLR if c(bitIdx - 1) == 1
            end
        end
    end
else % K == 2
    if Qm == 1 || Qm == 2
        for bitIdx = 1:E
            out(bitIdx) = (1 - 2*c(bitIdx)) * inLLRs(bitIdx); % flip LLR if c(bitIdx) == 1
        end
    else
        for bitIdx = 1:E
            temp = mod(bitIdx, Qm);
            if temp == 1 || temp == 2
                out(bitIdx) = (1 - 2*c(bitIdx)) * inLLRs(bitIdx); % flip LLR if c(bitIdx) == 1
            end 
        end
    end
end

end
