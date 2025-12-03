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

function x = mod_bits_to_qams(inputBits, nBitsPerQam, nQams)

%inputs:
% inputBits   --> collection of input bits to be modulated. Dim: nBits x 1
% nBitsPerQam --> number of bits per Qam
% nQams       --> number of modulated qams.

% outputs:
% x --> modulated qams. Dim: nQams x 1

%%
%SETUP

load('qam_mapping.mat');

switch nBitsPerQam
    case 2 % QPSK     
        QAM_mapping = QPSK_mapping;
    case 4 % 16QAM
        QAM_mapping = QAM16_mapping;
    case 6 % 64QAM
        QAM_mapping = QAM64_mapping;
    case 8 % 256QAM
        QAM_mapping = QAM256_mapping;
end

%%
%START

if (nBitsPerQam == 1)
    x = 1 - 2*inputBits;
else
    inputBits = reshape(inputBits, nBitsPerQam, nQams);
    qamIdx    = zeros(1,nQams);
    
    for i = 1 : nBitsPerQam
        qamIdx = qamIdx + inputBits(i,:)*2^(i-1);
    end
    
    x = QAM_mapping(qamIdx + 1);
end
