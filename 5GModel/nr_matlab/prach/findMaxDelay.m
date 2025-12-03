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

function delay = findMaxDelay(SysPar, prachTable)
%
% find max delay for a given PRACH configuration
%

FR = SysPar.carrier.FR;
duplex = SysPar.carrier.duplex;
mu = SysPar.carrier.mu;
prachCfgIdx = SysPar.prach{1}.configurationIndex;
restrictedSet = SysPar.prach{1}.restrictedSet;
zeroCorrelationZone = SysPar.prach{1}.zeroCorrelationZone;

[preambleFormat, SFN_x, SFN_y, subframeNum, startingSym, N_slot_subframe, ...
    N_t_RA_slot, N_dur_RA] = readPrachCfgTable(prachCfgIdx, FR, duplex, prachTable);

%% find dela_f_RA and L_RA
switch preambleFormat
    case {'0', '1', '2'}
        delta_f_RA = 1250;
        L_RA = 839;
    case '3'
        delta_f_RA = 5000;
        L_RA = 839;
    otherwise
        delta_f_RA = 15000*2^mu;
        L_RA = 139;
end  

%% find N_CS from zeroCorrelationZone
switch delta_f_RA
    case 1250
        switch restrictedSet
            case 0
                typeIdx = 1;
            case 1
                typeIdx = 2;
            case 2
                typeIdx = 3;
            otherwise
                error('restrictedSet is not supported ...\n');
        end
        % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-5
        N_CS_table = load('table_NCS_1p25k.txt');
        N_CS = N_CS_table(zeroCorrelationZone + 1, typeIdx);        
    case 5000
        switch restrictedSet
            case 'unrestricted'
                typeIdx = 1;
            case 'typeA'
                typeIdx = 2;
            case 'typeB'
                typeIdx = 3;
            otherwise
                error('restrictedSet is not supported ...\n');                
        end
        % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-6
        N_CS_table = load('table_NCS_5k.txt');
        N_CS = N_CS_table(zeroCorrelationZone + 1, typeIdx); 
    otherwise
        % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-7
        N_CS_table = load('table_NCS_15kplus.txt');
        N_CS = N_CS_table(zeroCorrelationZone + 1);
end

delay =  N_CS/(delta_f_RA*L_RA);
        
return        

