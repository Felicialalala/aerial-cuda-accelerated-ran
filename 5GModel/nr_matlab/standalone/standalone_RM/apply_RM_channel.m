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

function rmBitLLRs = apply_RM_channel(snr, rmBits)

%only evaluate RM code for BPSK
%inputs:
% snr          --> input SNR (dB)
% rmBits       --> array of rate-matched bits. Dim: E x 1 

%outputs:
% rmBitLLRs --> array of LLRs of rate-matched bits. Dim: E x 1

%% start

nRmBits   = length(rmBits);

N0        = 10^(-snr / 10); 

x         = 1 - 2*rmBits; % mod

y         = x + sqrt(N0 / 2) * (randn(nRmBits,1) + randn(nRmBits,1)); % channel

rmBitLLRs = real(2 * y / N0); % demod

end

