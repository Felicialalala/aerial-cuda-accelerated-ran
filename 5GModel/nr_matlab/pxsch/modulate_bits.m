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

function out = modulate_bits(in,modulation,pdschTable)

%function modulates bits into QAM symbols

%QAM MAPPING

% load('qam_mapping.mat');

switch modulation

    case 'pi/2-BPSK'
        bits_per_QAM = 1;
        QAM_mapping = pdschTable.BPSK_mapping;
        
    case 'QPSK'
        bits_per_QAM = 2;
        QAM_mapping = pdschTable.QPSK_mapping;
        
    case '16QAM'
        bits_per_QAM = 4;
        QAM_mapping = pdschTable.QAM16_mapping;
        
    case '64QAM'
        bits_per_QAM = 6;
        QAM_mapping = pdschTable.QAM64_mapping;
        
    case '256QAM'
        bits_per_QAM = 8;
        QAM_mapping = pdschTable.QAM256_mapping;
end

%START

num_qams = length(in) / bits_per_QAM;
out = zeros(num_qams,1);

% for i = 1 : num_qams
%     index = (i-1)*bits_per_QAM + 1 : i*bits_per_QAM;
%     bits = (flip(in(index)))';
% %     bits = flip(bits');
%     qam_index = b2d(bits);
%     out(i) = QAM_mapping(qam_index + 1);
% end

in2 = reshape(in, bits_per_QAM, num_qams);
in3 = zeros(1, num_qams);
for k = 1:bits_per_QAM
    in3 = in2(k,:)*2^(k-1) + in3;
end
out = QAM_mapping(in3 + 1);

% apply pi/2 rotation
if strcmp(modulation, 'pi/2-BPSK')
    out = out .* exp(1j*pi/2*mod([0:length(out)-1]', 2));
end

return
