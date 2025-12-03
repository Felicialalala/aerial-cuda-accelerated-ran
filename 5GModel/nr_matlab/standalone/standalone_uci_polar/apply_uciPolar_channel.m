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

function polUciSegLLRs_cell = apply_uciPolar_channel(snr, nPolUciSegs, polUciSegsEncoded_cell)

% function BPSK modulates bits, applies AWGN,then computes LLRs

%inputs:
% snr                    --> input SNR
% nPolUciSegs            --> number of polar uci segments
% polUciSegsEncoded_cell --> cell containing polar encoded segments

%%
%START

N0                 = 10^(-snr / 10);
polUciSegLLRs_cell = cell(nPolUciSegs, 1);

for segIdx = 0 : (nPolUciSegs - 1)
    bits     = polUciSegsEncoded_cell{segIdx + 1};
    
    txSignal = -2*(bits - 0.5);
    rxSignal = txSignal + sqrt(N0) * randn(size(txSignal));
    
    polUciSegLLRs_cell{segIdx + 1} = 2 * rxSignal / N0;
end

end

