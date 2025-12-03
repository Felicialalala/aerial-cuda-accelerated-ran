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

function prach = findPreambleTfLoc(prach, carrier, table)
% function prach = findPreambleTfLoc(prach, carrier, nodeType)
%
% This function find freq and time domain location to transmit preamble
%
% Input:    prach: prach related configuration 
%           carrier: carrier related configuration
%           nodeType: 'UE' or 'gNB'
%
% Output:   prach: add fields for PRACH paramters, ZC sequence and 
%           preamble in freq domain
%

prachTable = table;

% read input from carrier and prach
FR = carrier.FR;
duplex = carrier.duplex;
mu = carrier.mu;
mu0 = carrier.mu0;
N_sc_RB = carrier.N_sc_RB;
N_grid_start_mu = carrier.N_grid_start_mu;
N_grid_size_mu = carrier.N_grid_size_mu;

N_BWP_start = carrier.N_BWP_start;
delta_f = carrier.delta_f;
Nfft = carrier.Nfft;
k0_mu = carrier.k0_mu;

prachCfgIdx = prach.configurationIndex;
M_RA = prach.msg1_FDM;
n_RA_start = prach.n_RA_start;
ssbIdx = prach.ssbIdx; 
zeroCorrelationZone = prach.zeroCorrelationZone;
restrictedSet = prach.restrictedSet;

% look at prachCfgTable
[preambleFormat, SFN_x, SFN_y, subframeNum, startingSym, N_slot_subframe, ...
    N_t_RA_slot, N_dur_RA] = readPrachCfgTable(prachCfgIdx, FR, duplex, prachTable);

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

switch preambleFormat
    case {'0'}
        N_rep = 1;
    case 'B4'
        N_rep = 12;
    case {'1'}
        N_rep = 2;
    otherwise
        error('preambleFormat is not supported ...\n');
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
        N_CS_table = prachTable.table_NCS_1p25k;
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
        N_CS_table = prachTable.table_NCS_5k;
        N_CS = N_CS_table(zeroCorrelationZone + 1, typeIdx);
    otherwise
        % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-7
        N_CS_table = prachTable.table_NCS_15kplus;
        N_CS = N_CS_table(zeroCorrelationZone + 1);
end


% find allowable slot for RA occations
switch delta_f_RA
    case {1250, 5000, 15000, 60000}
        n_slot_RA = 0;
    case {30000, 120000}
        if N_slot_subframe == 1
            n_slot_RA = 1;
        else
            n_slot_RA = [0, 1];
        end
    otherwise 
        error('delta_f_RA is not supported ... \n');
end

% find RA occation based on ssbIdx, freq domain first, then time domain, 
% then next slot 

n_RA = mod(ssbIdx, M_RA);
n_t_RA = mod((ssbIdx-n_RA)/M_RA, N_t_RA_slot);
if length(n_slot_RA) == 1
    n_slot_RA_sel = n_slot_RA;
else
    n_slot_RA_sel = mod(((ssbIdx-n_RA)/M_RA-n_t_RA)/N_t_RA_slot, max(n_slot_RA));
end

% find the start symbol index within the subframe
if L_RA == 139
    startRaSym = startingSym + n_t_RA*N_dur_RA; 
else
    startRaSym = startingSym;
end
startRaSym = startRaSym + n_slot_RA_sel*14; % 14 is # of symbols in a slot

% find time domain preamble length from preambleFormat table
isLastRaSlot = (n_t_RA == N_t_RA_slot-1);  % last occation within a slot   
[L_RA, delta_f_RA, N_u, N_CP_RA] = readPreambleFormatTable ...
    (preambleFormat, mu, isLastRaSlot);

% find kBar for subcarrier offset in unit of delta_f_RA
% 3GPP 38.211 (V15.4) Table 6.3.3.2-1
% kBar_table = load('table_kBar.txt');
kBar_table = prachTable.kBar_table;
[M, ~] = size(kBar_table);
find_flag = 0;
for m = 1:M
    if (L_RA == kBar_table(m, 1)) && (delta_f_RA == kBar_table(m, 2)*1000) ...
            && (delta_f == kBar_table(m, 3)*1000)
        N_RB_RA = kBar_table(m, 4);
        kBar = kBar_table(m, 5); % preamble first subcarrier shift
        find_flag = 1;
        break;
    end
end
if find_flag == 0
    error('kBar table error ... \n');
end

% calculate k1 for freq domain starting PRB in unit of delta_f
k1 = k0_mu + (N_BWP_start - N_grid_start_mu)*N_sc_RB + ...
    n_RA_start*N_sc_RB + n_RA*N_RB_RA*N_sc_RB - N_grid_size_mu*N_sc_RB/2;
K = delta_f/delta_f_RA;

% save output into prach
prach.SFN_x = SFN_x;
prach.SFN_y = SFN_y; 
prach.subframeNum = subframeNum;
prach.startingSym = startingSym;
prach.N_slot_subframe = N_slot_subframe;
prach.N_t_RA_slot = N_t_RA_slot;
prach.N_dur_RA = N_dur_RA;

prach.preambleFormat = preambleFormat;
prach.delta_f_RA = delta_f_RA;
prach.n_RA = n_RA;
prach.startRaSym = startingSym;
prach.n_slot_RA_sel = n_slot_RA_sel;
prach.N_u = N_u;
prach.N_CP_RA = N_CP_RA;
prach.kBar = kBar;
prach.K = K;
prach.k1 = k1;
prach.L_RA = L_RA;
prach.N_rep = N_rep;
prach.N_CS = N_CS;

% loads or calculates the lowpass filters used for preamble 
% time domain samples generation or reception.
if L_RA == 839
    calculateFIR = 0;
    if calculateFIR    
        [prachFIRx2, prachFIRx3] = calcPrachFir(Nfft, L_RA, N_sc_RB, K);
    else
        load prachFIRx2;
        load prachFIRx3;
    end

    if delta_f_RA == 1250 && delta_f == 15e3
        prach.NfirStage = 3;    
        prach.fir{1}.coef = prachFIRx3;
        prach.sampRate(1) = 3;
        prach.fir{2}.coef = prachFIRx2;
        prach.sampRate(2) = 2;
        prach.fir{3}.coef = prachFIRx2;
        prach.sampRate(3) = 2;   
    elseif delta_f_RA == 1250 && delta_f == 30e3
        prach.NfirStage = 4;    
        prach.fir{1}.coef = prachFIRx3;
        prach.sampRate(1) = 3;
        prach.fir{2}.coef = prachFIRx2;
        prach.sampRate(2) = 2;
        prach.fir{3}.coef = prachFIRx2;
        prach.sampRate(3) = 2;   
        prach.fir{4}.coef = prachFIRx2;
        prach.sampRate(4) = 2;   
    else
        error('delta_f and delta_f_RA combination is not supported ...\n')
    end
end
return
