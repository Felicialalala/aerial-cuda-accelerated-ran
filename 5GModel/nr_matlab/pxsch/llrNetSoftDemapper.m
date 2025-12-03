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

function out = llrNetSoftDemapper(in, T, QAM_bits, QAM_noise_var)
  
    PAM_noise_var = QAM_noise_var / 2;
    num_symbols = size(in, 1);
    assert(num_symbols == 1)
    out = zeros(num_symbols * QAM_bits, 1);
    
    input = [real(in)*3.0, imag(in)*3.0];
    n0 = 8;
    n1 = 4; 
    
    weight_0 = [[ 0.0058, -0.6743],
            [ 0.0021, -0.2509],
            [-0.5530, -0.0419],
            [ 0.5395,  0.0221],
            [ 0.3179, -0.0273],
            [ 0.0011,  0.5740],
            [ 0.0028, -0.1160],
            [-0.3987,  0.0146]];
            
    bias_0 = [-0.2127,  0.1184,  0.3271, -0.4823,  0.2116, -0.2491, -0.2803, -0.3550];
    
    weight_1 = [[ 0.0173,  0.0079, -0.3718,  0.5660,  0.1948, -0.0165, -0.1694, -0.4164],
            [-0.4814, -0.1283, -0.1623,  0.1515, -0.2611,  0.6248, -0.0513,  0.2159],
            [ 0.0093, -0.0241, -0.0974, -0.3561, -0.1616,  0.0101, -0.0692, -0.4729],
            [-0.3954,  0.0888,  0.1118, -0.1078,  0.1780, -0.4141, -0.1620, -0.1487]];
    
    bias_1 = [0.0972, 0.1598, 0.3587, 0.2990];
    
    
    result_0 = zeros(1, n0);
    
    for idx = 1: n0
       temp = input*weight_0(idx, :)' + bias_0(idx);
       result_0(idx) = max(0.0, temp);
    end  
    
    for idx = 1: n1
        out(idx) =  (result_0 * weight_1(idx, :)' + bias_1(idx)) * (1/real(PAM_noise_var)) * 0.8;
    end

return