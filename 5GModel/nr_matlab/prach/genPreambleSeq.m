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

function prach = genPreambleSeq(prach, carrier, nodeType)
% function prach = genPreambleSeq(prach, carrier, nodeType)
%
% This function generates PRACH paramters, ZC sequence and preamble in 
% freq domain.
%
% Input:    prach: prach related configuration 
%           carrier: carrier related configuration
%           nodeType: 'UE' or 'gNB'
%
% Output:   prach: add fields for PRACH paramters, ZC sequence and 
%           preamble in freq domain
%

isUE = strcmp(nodeType, 'UE'); 

% read input from carrier and prach
FR = carrier.FR;
duplex = carrier.duplex;
mu = carrier.mu;
prachCfgIdx = prach.configurationIndex;
restrictedSet = prach.restrictedSet;
rootSequenceIndex = prach.rootSequenceIndex;
zeroCorrelationZone = prach.zeroCorrelationZone;
if isUE
    prmbIdx = prach.prmbIdx;
end

% look at prachCfgTable
[preambleFormat, SFN_x, SFN_y, subframeNum, startingSym, N_slot_subframe, ...
    N_t_RA_slot, N_dur_RA] = readPrachCfgTable(prachCfgIdx, FR, duplex);

% find dela_f_RA and L_RA
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

% find N_CS from zeroCorrelationZone
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
            case 0
                typeIdx = 1;
            case 1
                typeIdx = 2;
            case 2
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

% load logIdx2u table for logical root mapping
if L_RA == 839
    % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-3
    logIdx2u_table = load('table_logIdx2u_839.txt');
elseif L_RA == 139
    % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-4
    logIdx2u_table = load('table_logIdx2u_139.txt');
else
    error('L_RA length is not supported ... \n');
end

% generate ZC sequence and preamble
if isUE   
    % calculate u and C_v
    [u, C_v] = findZcPar(prmbIdx, rootSequenceIndex, ...
        L_RA, restrictedSet, N_CS, logIdx2u_table);
        
    % generate ZC sequence and preamble
    [y_uv, x_uv, x_u] = genZcPreamble(L_RA, C_v, u);    
else % gNB
    % generate possible root preambles for all index (0-63)
    Nprmb = 64;
    for prmbIdx = 0:Nprmb-1
        [u, C_v] = findZcPar(prmbIdx, rootSequenceIndex, ...
            L_RA, restrictedSet, N_CS, logIdx2u_table);
        if prmbIdx == 0
            uList = u;
            uCount = 1;
        else
            if u ~= uList(uCount)
                uCount = uCount + 1;
                uList(uCount) = u;
            end
        end
        y_u_ref(uCount, :) = genZcPreamble(L_RA, 0, u);
        u_ref(prmbIdx+1) = u;
        C_v_ref(prmbIdx+1) = C_v;
    end
end
% save output into prach
if isUE
    prach.u = u;
    prach.C_v = C_v;
    prach.x_u = x_u;
    prach.x_uv = x_uv;
    prach.y_uv = y_uv;
    % y_uv_test is used to compare with matlab 5gtb, no need to implement
    prach.y_uv_test = y_uv*sqrt(L_RA); 
else
    prach.uCount = uCount;
    prach.y_u_ref = y_u_ref;
    prach.u_ref = u_ref;
    prach.C_v_ref = C_v_ref;
end
prach.L_RA = L_RA;
prach.N_CS = N_CS;

return
