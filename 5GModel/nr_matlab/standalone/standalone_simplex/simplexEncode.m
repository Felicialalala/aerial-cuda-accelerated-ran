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

function rmSimplexCw = simplexEncode(payload, K, E, Qm)
%
% This function encodes 1-bit or 2-bit information through Simplex code with rate matching,
%
% Input:    payload: the vector of information bits to encode (should contain 1 bit or 2 bits)
%           K:       the number of information bits
%           E:       the length of bit sequence after rate matching
%           Qm:      modulation order, should be in {1, 2, 4, 6, 8}
%           
%
% Output:   out: the vector of encoded (and rate-matched) bits, 
%           ** In out, -1 indicates 'x', and -2 indicates 'y' (refer to Sec. 5.3.3, TS 38.212)
%
%

%% validate inputs
if K ~= length(payload)
    error('Invalid input: the length of input payload array does not match with the provided number of information bits K');
end

if K ~= 1 && K ~= 2
    error('Invalid input: Simplex code only applies to 1-bit or 2-bit information');
end

if Qm ~= 1 && Qm ~= 2 && Qm ~= 4 && Qm ~= 6 && Qm ~= 8
    error('Invalid input: Modulation order Qm must be in {1, 2, 4, 6, 8}');
end

if K == 1
    if E < Qm
        error('Invalid input: For 1-bit information, the length of rate-matched sequence E must be no less than Qm');
    end
end

%% encode


if K == 1
   simplexCw    = -1*ones(Qm, 1);
   simplexCw(1) = payload(1);
   if Qm > 1
       simplexCw(2) = -2;
   end
else % K == 2
   simplexCw    = -1*ones(3*Qm, 1);
   c2           = mod(payload(1) + payload(2), 2);
   simplexCw(1) = payload(1);
   simplexCw(2) = payload(2);
    if Qm == 1
        simplexCw(3) = c2;
    else
        simplexCw(Qm + 1) = c2;
        simplexCw(2*Qm + 2) = c2;
        simplexCw(Qm + 2) = payload(1);
        simplexCw(2*Qm + 1) = payload(2);
    end
end


%% rate matching

rmSimplexCw = zeros(E, 1);
if K == 1
    for bitIdx = 1 : E
        IdxTemp = mod(bitIdx-1, Qm)+1;
        rmSimplexCw(bitIdx) = simplexCw(IdxTemp);
    end
else % K == 2
    for bitIdx = 1 : E
        IdxTemp = mod(bitIdx-1, 3*Qm)+1;
        rmSimplexCw(bitIdx) = simplexCw(IdxTemp);
    end
end
end

