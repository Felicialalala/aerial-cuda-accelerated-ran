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

% Init Sphere decoder object for PUSCH MIMO equalizer
function sphere_decoder = init_sphere_decoder(SimCtrl,table,qamstr)
    if SimCtrl.alg.enable_sphere_decoder
        switch qamstr
            case 'QPSK'
                QAM_mapping = table.QPSK_mapping;
            case '16QAM'
                QAM_mapping = table.QAM16_mapping;
            case '64QAM'
                QAM_mapping = table.QAM64_mapping;
            case '256QAM'
                QAM_mapping = table.QAM256_mapping;
        end
        M = length(QAM_mapping);
        bps = log2(M);
        sym = qammod((0:M-1).',M,0:M-1,'UnitAveragePower',true);
        symMap = zeros(1,M);
        for ii = 1:M
            symMap(ii) = find(QAM_mapping==sym(ii))-1;
        end
%         bitTable = int2bit(symMap,bps, false)';
        bitTable = (flipud(dec2bin(symMap,bps)')-'0')';        
        sphere_decoder = comm.SphereDecoder('Constellation',sym, 'BitTable',bitTable,'InitialRadius', 'ZF solution','DecisionType','Soft');
    else
        sphere_decoder = [];
    end
end