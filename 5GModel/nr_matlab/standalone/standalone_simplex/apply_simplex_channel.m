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

function rmBitLLRs = apply_simplex_channel(snr, nBitsPerQam, rmBits)

%inputs:
% snr          --> input SNR (dB)
% nBitsPerQam  --> number of bits per QAM
% rmBits       --> array of rate-matched bits. Dim: E x 1 

%outputs:
% rmBitLLRs --> array of LLRs of rate-matched bits. Dim: E x 1

%% validate inputs
if nBitsPerQam ~= 1 && nBitsPerQam ~= 2 && nBitsPerQam ~= 4 && nBitsPerQam ~= 6 && nBitsPerQam ~= 8
    error('Invalid input: nBitsPerQam must be in {1, 2, 4, 6, 8}');
end


%% start

nRmBits = length(rmBits);
nQams   = nRmBits / nBitsPerQam;
N0      = 10^(-snr / 10); 

x         = mod_bits_to_qams(rmBits, nBitsPerQam, nQams);         % mod
y         = x + sqrt(N0 / 2) * (randn(nQams,1) + randn(nQams,1)); % channel
rmBitLLRs = deMod_qamEsts_to_bitLLRs(y,N0,nQams,nBitsPerQam, 0);     % deMod


end

