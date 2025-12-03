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

function [nComp, errCnt, nTV, detErr] = testCompGenTV_csirs(caseSet, compTvMode, subSetMod, relNum)

tic;
if nargin == 0
    caseSet = 'full';
    compTvMode = 'both';
    subSetMod = [0, 1];
    relNum = 10000;
elseif nargin == 1
    compTvMode = 'both';
    subSetMod = [0, 1];
    relNum = 10000;
elseif nargin == 2
    subSetMod = [0, 1];
    relNum = 10000;
elseif nargin == 3
    relNum = 10000;
end

switch compTvMode
    case 'both'
        genTV = 1;
        testCompliance = 1;
    case 'genTV'
        genTV = 1;
        testCompliance = 0;
    case 'testCompliance'
        genTV = 0;
        testCompliance = 1;
    otherwise
        error('compTvMode is not supported...\n');
end

selected_TC = [4001:4999];
disabled_TC = [];
[~,TcIdx] = ismember(disabled_TC, selected_TC);
selected_TC(TcIdx) = [];

compact_TC = [4001:4999];
full_TC = [4001:4999];

if isnumeric(caseSet)
    TcToTest = caseSet;
else
    switch caseSet
        case 'compact'
            TcToTest = compact_TC;
        case 'full'
            TcToTest = full_TC;
        case 'selected'
            TcToTest = selected_TC;
        otherwise
            error('caseSet is not supported...\n');
    end
end

CFG = {...
    % TC#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID TxPwr  slotIdx FreqDomain
    % sweep Row and Density
    4001,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4002,  1,  2,  0,   0,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4003,  1,  2,  0,   1,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4004,  1,  2,  0,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4005,  1,  3,  1,   0,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4006,  1,  3,  1,   1,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4007,  1,  3,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4008,  1,  4,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4009,  1,  5,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4010,  1,  6,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4011,  1,  7,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4012,  1,  8,  2,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4013,  1,  9,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4014,  1, 10,  2,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4015,  1, 11,  1,   0,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4016,  1, 11,  1,   1,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4017,  1, 11,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4018,  1, 12,  2,   0,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4019,  1, 12,  2,   1,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4020,  1, 12,  2,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4021,  1, 13,  1,   0,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4022,  1, 13,  1,   1,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4023,  1, 13,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4024,  1, 14,  2,   0,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4025,  1, 14,  2,   1,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4026,  1, 14,  2,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4027,  1, 15,  3,   0,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4028,  1, 15,  3,   1,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4029,  1, 15,  3,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4030,  1, 16,  1,   0,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4031,  1, 16,  1,   1,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4032,  1, 16,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4033,  1, 17,  2,   0,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4034,  1, 17,  2,   1,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4035,  1, 17,  2,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4036,  1, 18,  3,   0,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4037,  1, 18,  3,   1,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    4038,  1, 18,  3,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};
    % TC#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID TxPwr  slotIdx FreqDomain
    4039,  1,  4,  1,   2,   100,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % RB0
    4040,  1,  4,  1,   2,     0,  273,   0,    8,    0,   0,     0,    {ones(1,12)};  % nRB
    4041,  1,  4,  1,   2,     0,    4,  13,    8,    0,   0,     0,    {ones(1,12)};  % sym0
    4042,  1, 13,  1,   2,     0,    4,   0,   12,    0,   0,     0,    {ones(1,12)};  % sym1
    4043,  1,  4,  1,   2,     0,    4,   0,    8,   41,   0,     0,    {ones(1,12)};  % nID
    4044,  1,  4,  1,   2,     0,    4,   0,    8,    0,  -3,     0,    {ones(1,12)};  % PwCtrl
    % TC#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID TxPwr  slotIdx FreqDomain
    4045,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {[1 1 1 0]}; % FreqDomain
    4046,  1,  2,  0,   0,     0,    4,   0,    8,    0,   0,     0,    {[ones(1,11), 0]}; % FreqDomain
    4047,  1,  3,  1,   0,     0,    4,   0,    8,    0,   0,     0,    {[1 1 1 0 0 0]}; % FreqDomain
    4048,  1,  4,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {[1 0 0]}; % FreqDomain
    4049,  1,  6,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {[1 0 1 0 1 1]}; % FreqDomain
    4050,  1, 18,  3,   1,     0,    4,   0,    8,    0,   0,     0,    {[0 1 1 1 0 1]}; % FreqDomain    
    4051,  1,  4,  1,   2,     0,    4,   0,    8,    0,   0,     3,    {ones(1,12)};  % slotIdx    
    4052,  1,  4,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % batching, different location 
    4053,  1,  4,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % batching, different row number    
    4054,  1,  4,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % batching, different number of ports        
    4055,  1,  4,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % mu=0, gridSize = 106
    4056,  0,  1,  0,   3,     0,   52,   5,    8,   41,   0,     0,    {[0 1 0 0]};   % TRS
    4057,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % precoding
    4058,  1,  4,  1,   2,     0,    4,   0,    8,    0,  -3,     0,    {ones(1,12)};  % modComp
    4059,  1,  4,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % batching, 16 CSIRS 
    4060,  1,  4,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % batching, 32 CSIRS
    4061,  1,  4,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % 4-BWP
    4062,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {[0 0 0 1]};   % NZP + ZP
    4063,  1, 18,  3,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % 32 ports with CSIRS compression
    4064,  1, 18,  3,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % 4063 with modComp
    4065,  1, 16,  1,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % 32 ports with CSIRS compression to 2 ports
    4066,  1, 17,  2,   2,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)};  % 32 ports with CSIRS compression to 4 ports

    % TC#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID PwCtrl slotIdx FreqDomain
    4101,  1,  1,  0,   3,     0,  273,   0,    8,   41,  -3,     0,    {[1 0 0 0]};
    4102,  1,  4,  1,   2,     0,  273,  13,    8,   41,   3,     1,    {[0 1 0]};
    4103,  1,  8,  2,   2,     0,  273,  12,    8,    0,   6,    10,    {[0 0 1 0 1 0]};   
    
    % different BW
    % mu = 0
    % TC#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID PwCtrl slotIdx FreqDomain
    4201,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 5 MHz
    4202,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 10 MHz
    4203,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 15 MHz
    4204,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 20 MHz
    4205,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 25 MHz
    4206,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 30 MHz
    4207,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 40 MHz
    4208,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 50 MHz
    4209,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 60 MHz
    4210,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 70 MHz
    4211,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 80 MHz
    4212,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 90 MHz
    4213,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 100 MHz
    % mu = 0
    4214,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 5 MHz
    4215,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 10 MHz
    4216,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 15 MHz
    4217,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 20 MHz
    4218,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 25 MHz
    4219,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 30 MHz
    4220,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 40 MHz
    4221,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 50 MHz
    % additional BW test cases
    4222,  1,  1,  0,   3,     0,    4,   0,    8,    0,   0,     0,    {ones(1,12)}; % 40 MHz
    
    % perf test
    % TC#  CSI Row CDM Density RB0  nRB  sym0  sym1   nID PwCtrl slotIdx FreqDomain
    4801,  1,  4,  1,   2,     0,  273,   5,    8,    0,   0,     0,    {ones(1,12)}; % F08
    4802,  0,  1,  0,   3,     0,  273,   5,    8,    0,   0,     0,    {ones(1,12)}; % F09 TRS
    4803,  1,  16, 1,   2,     0,  273,   9,    12,   0,   0,     0,    {ones(1,12)}; % F14
    % 4T4R FDD
    4804,  1,  4,  1,   2,     0,  106,   5,    8,    0,   0,     0,    {ones(1,12)}; % F08
    % F14 20 MHz    
    4805,  1,  16, 1,   2,     0,   51,   9,    12,   0,   0,     0,    {ones(1,12)};
    % F08 20 MHz
    4806,  1,  4,  1,   2,     0,   51,   5,    8,    0,   0,     0,    {ones(1,12)};
    % F09, 20 MHz, 16 ports
    4807,  1,  11, 1,   2,     0,   51,   5,    6,    0,   0,     0,    {ones(1,12)}; % F14

    % request TCs
    4901,  0,  1,  0,   3,     0,   52,   6,    8,   41,   0,     0,    {[0 1 0 0]};  % 4056
    4902,  0,  1,  0,   3,     0,  273,   6,    8,    0,   0,     0,    {ones(1,12)}; % 4802
    4903,  1,  4,  1,   2,     0,  273,  13,    8,    0,   0,     0,    {ones(1,12)}; % 4040
    4904,  1,  4,  1,   2,     0,  273,  13,    8,    0,   0,     0,    {ones(1,12)}; % 4801
    4905,  1,  5,  1,   2,     0,  273,  10,    8,    0,   0,     0, {[1 0 1 0 1 1]}; % 3907 w/ NZP
    };

[NallTest, ~] = size(CFG);
errCnt = 0;
detErr = 0;
nTV = 0;
nComp = 0;

if (isnumeric(caseSet))
    caseSetStr = num2str(caseSet);
else
    caseSetStr = caseSet;
end
fprintf('CSIRS: genTV = %d, testCompliance = %d, caseSet = %s', genTV, testCompliance, caseSetStr);
fprintf('\nTC#   CSI Row CDM Density  RB0  nRB  sym0 sym1 nID  TxPwr slotIdx PASS Det\n');
fprintf('---------------------------------------------------------------------------\n');

parfor n = 1:NallTest
    testAlloc = [];
    testAlloc.dl = 1;
    testAlloc.ul = 0;
    testAlloc.csirs = 1;
    Ncsirs = 1;
    idxSet = n;
    caseNum = CFG{idxSet, 1};
    if ismember(caseNum, TcToTest) && (mod(caseNum, subSetMod(2)) == subSetMod(1))
        rng(caseNum);
        SysPar = initSysPar(testAlloc);
        SysPar.SimCtrl.relNum = relNum;
        SysPar.SimCtrl.N_frame = 1;
        if genTV
            nTV = nTV + 1;
            SysPar.SimCtrl.genTV.enable = 1;
            SysPar.SimCtrl.genTV.TVname = sprintf('TVnr_%04d', caseNum);
            if ismember(caseNum, disabled_TC)
                SysPar.SimCtrl.genTV.TVname = sprintf('disabled_TVnr_%04d', caseNum);
            end
            SysPar.SimCtrl.N_slot_run = CFG{idxSet, 12} + 1;
%             SysPar.SimCtrl.genTV.forceSlotIdxFlag = 1;            
            SysPar.SimCtrl.genTV.slotIdx = CFG{idxSet, 12};
        end
        SysPar.csirs{1}.CSIType = CFG{idxSet, 2}; % 0:TRS, 1:CSI-RS NZP, 2:CSI-RS ZP
        SysPar.csirs{1}.Row = CFG{idxSet, 3}; % 3;
        SysPar.csirs{1}.CDMType = CFG{idxSet, 4}; % 1; % 0: noCDM, 1: fd-CDM2, 2: cdm4-FD2-TD2, 3: cdm8-FD2-TD4
        SysPar.csirs{1}.FreqDensity = CFG{idxSet, 5}; % 2; % 0: dot5 (even RB), 1: dot5 (odd RB), 2: one, 3: three
        SysPar.csirs{1}.StartRB = CFG{idxSet, 6};
        SysPar.csirs{1}.NrOfRBs = CFG{idxSet, 7};
        SysPar.csirs{1}.SymbL0 = CFG{idxSet, 8};
        SysPar.csirs{1}.SymbL1 = CFG{idxSet, 9};
        SysPar.csirs{1}.ScrambId = CFG{idxSet, 10};
        SysPar.csirs{1}.powerControlOffsetSS = CFG{idxSet, 11}/3 + 1;                
        SysPar.csirs{1}.FreqDomain = cell2mat(CFG{idxSet, 13});
        [row2nPort, row2nCdm] = getCsirsConfig();
        nPort = row2nPort(SysPar.csirs{1}.Row);
        SysPar.carrier.Nant_gNB = nPort;
        nCdm = row2nCdm(SysPar.csirs{1}.Row);
        
        if caseNum == 4052 % different location
            SysPar.csirs{2} = SysPar.csirs{1};            
            SysPar.csirs{2}.StartRB = 100;
            SysPar.testAlloc.csirs = 2;
        elseif caseNum == 4053 % different Row number
            SysPar.csirs{2} = SysPar.csirs{1};
            SysPar.csirs{2}.Row = 5;
            SysPar.csirs{2}.CDMType = 1; % 0: noCDM, 1: fd-CDM2, 2: cdm4-FD2-TD2, 3: cdm8-FD2-TD4
            SysPar.csirs{2}.FreqDensity = 2; % 0: dot5 (even RB), 1: dot5 (odd RB), 2: one, 3: three
            SysPar.csirs{2}.StartRB = 40;
            SysPar.csirs{2}.NrOfRBs = 8;
            SysPar.csirs{2}.SymbL0 = 4;
            SysPar.csirs{2}.SymbL1 = 8;
            SysPar.csirs{2}.ScrambId = 41;
            SysPar.csirs{2}.powerControlOffset = 8;
            SysPar.csirs{2}.FreqDomain = ones(1,12);
            SysPar.testAlloc.csirs = 2;
        elseif caseNum == 4054 % different number of ports
            SysPar.csirs{2} = SysPar.csirs{1};
            SysPar.csirs{2}.Row = 1;
            SysPar.csirs{2}.CDMType = 0; % 0: noCDM, 1: fd-CDM2, 2: cdm4-FD2-TD2, 3: cdm8-FD2-TD4
            SysPar.csirs{2}.FreqDensity = 3; % 0: dot5 (even RB), 1: dot5 (odd RB), 2: one, 3: three
            SysPar.csirs{2}.StartRB = 0;
            SysPar.csirs{2}.NrOfRBs = 273;
            SysPar.csirs{2}.SymbL0 = 13;
            SysPar.csirs{2}.SymbL1 = 8;
            SysPar.csirs{2}.ScrambId = 41;
            SysPar.csirs{2}.powerControlOffset = 8;
            SysPar.csirs{2}.FreqDomain = [1 0 0 0];
            SysPar.testAlloc.csirs = 2;            
        elseif caseNum == 4055
            SysPar.carrier.mu = 0;
            SysPar.carrier.N_grid_size_mu = 106;
        elseif caseNum == 4056
            % apply precoding [1; 1] to mimic the walk-around (tx on 2 ports)
            SysPar.carrier.Nant_gNB = 2;
%             SysPar.csirs{1}.prcdBf = 13; 
            SysPar.csirs{2} = SysPar.csirs{1};
            SysPar.csirs{2}.SymbL0 = 9;
            SysPar.testAlloc.csirs = 2;
        elseif caseNum == 4057
            SysPar.csirs{1}.prcdBf = 4;
            SysPar.carrier.Nant_gNB = 4;
        elseif caseNum == 4058
            SysPar.SimCtrl.genTV.fhMsgMode = 2; 
        elseif ismember(caseNum, [4059, 4060]) 
            if caseNum == 4059
                Ncsirs = 16;
            elseif caseNum == 4060
                Ncsirs = 32;
            end
            for idxCsirs = 2:Ncsirs
                SysPar.csirs{idxCsirs} = SysPar.csirs{1};
                SysPar.csirs{idxCsirs}.StartRB = 8*(idxCsirs-1);
            end
            SysPar.testAlloc.csirs = Ncsirs;
        elseif ismember(caseNum, [4061])
            Ncsirs = 4;
            for idxCsirs = 1:Ncsirs
                SysPar.csirs{idxCsirs} = SysPar.csirs{1};
                SysPar.csirs{idxCsirs}.BWPSize = 68;
                SysPar.csirs{idxCsirs}.BWPStart = 68*(idxCsirs-1);
                % Per FAPI spec, StartRB is related to common resource block #0 (CRB#0), not BWP 
                SysPar.csirs{idxCsirs}.StartRB = 68*(idxCsirs-1) + idxCsirs*4;
            end
            SysPar.testAlloc.csirs = Ncsirs;
        elseif caseNum == 4062
            Ncsirs = 2;
            SysPar.csirs{2} = SysPar.csirs{1};
            SysPar.csirs{2}.CSIType = 2;
            SysPar.csirs{2}.FreqDomain = [0 0 1 0];
            SysPar.testAlloc.csirs = Ncsirs;
        elseif ismember(caseNum, [4021:4038, 4042, 4050, 4803, 4063:4066])
            SysPar.SimCtrl.nPort_enable_csirs_compression = nPort;
            SysPar.carrier.Nant_gNB = nCdm;
            if caseNum == 4064
                SysPar.SimCtrl.genTV.fhMsgMode = 2; 
            end
        elseif caseNum == 4201
            SysPar.carrier.N_grid_size_mu = 11; % 5 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4202
            SysPar.carrier.N_grid_size_mu = 24; % 10 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4203
            SysPar.carrier.N_grid_size_mu = 38; % 15 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4204
            SysPar.carrier.N_grid_size_mu = 51; % 20 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4205
            SysPar.carrier.N_grid_size_mu = 65; % 25 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4206
            SysPar.carrier.N_grid_size_mu = 78; % 30 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4207
            SysPar.carrier.N_grid_size_mu = 106; % 40 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4208
            SysPar.carrier.N_grid_size_mu = 133; % 50 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4209
            SysPar.carrier.N_grid_size_mu = 162; % 60 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4210
            SysPar.carrier.N_grid_size_mu = 189; % 70 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4211
            SysPar.carrier.N_grid_size_mu = 217; % 80 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4212
            SysPar.carrier.N_grid_size_mu = 245; % 90 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4213
            SysPar.carrier.N_grid_size_mu = 273; % 100 MHz 
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4214
            SysPar.carrier.mu = 0;
            SysPar.carrier.N_grid_size_mu = 25; % 5 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4215
            SysPar.carrier.mu = 0;
            SysPar.carrier.N_grid_size_mu = 52; % 10 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4216
            SysPar.carrier.mu = 0;
            SysPar.carrier.N_grid_size_mu = 79; % 15 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4217
            SysPar.carrier.mu = 0;
            SysPar.carrier.N_grid_size_mu = 106; % 20 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4218
            SysPar.carrier.mu = 0;
            SysPar.carrier.N_grid_size_mu = 133; % 25 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4219
            SysPar.carrier.mu = 0;
            SysPar.carrier.N_grid_size_mu = 160; % 30 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4220
            SysPar.carrier.mu = 0;
            SysPar.carrier.N_grid_size_mu = 216; % 40 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4221
            SysPar.carrier.mu = 0;
            SysPar.carrier.N_grid_size_mu = 270; % 50 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;
        elseif caseNum == 4222
            SysPar.carrier.N_grid_size_mu = 100; % 40 MHz
            SysPar.csirs{1}.BWPSize = SysPar.carrier.N_grid_size_mu;     
        elseif caseNum == 4804
            SysPar.carrier.N_grid_size_mu = 106; 
            SysPar.carrier.duplex = 0;
            SysPar.prach{1}.configurationIndex = 198;
            SysPar.testAlloc.prach = 0;
        elseif caseNum == 4901
            SysPar.carrier.Nant_gNB = 2;
            SysPar.csirs{2} = SysPar.csirs{1};
            SysPar.csirs{2}.SymbL0 = 10;
            SysPar.testAlloc.csirs = 2;
        elseif caseNum == 4902
            SysPar.csirs{2} = SysPar.csirs{1};
            SysPar.csirs{2}.SymbL0 = 10;
            SysPar.testAlloc.csirs = 2;
        end
        
        for idxUe = 1:length(SysPar.csirs)
            digBFInterfaces = 1;
            switch SysPar.csirs{idxUe}.prcdBf
                case 0
                    digBFInterfaces = SysPar.carrier.Nant_gNB;
                case {1, 2, 5, 6}
                    digBFInterfaces = 2;
                case {3, 4, 7, 8, 9, 10, 11, 12}
                    digBFInterfaces = 4;
                otherwise
                    error('prcdBf is not supported ... \n');
            end
            SysPar.csirs{idxUe}.digBFInterfaces = digBFInterfaces;
            SysPar.csirs{idxUe}.beamIdx = [1:digBFInterfaces];
        end        
        
        if strcmp(caseSet, 'full') || strcmp(caseSet, 'compact')
            SysPar.SimCtrl.enableUeRx = 1;
        end
        
        % to generate TVs for CSI-RS RX in UE
        if (~ismember(caseNum, [4056, 4057, 4062, 4802, 4901, 4902]))
            SysPar.SimCtrl.enableUeRx = 1;
            SysPar.SimCtrl.genTV.enableUE = 1;
        end
        
        % update Nre_max which is used to calculate beta value in BFP
        SysPar.SimCtrl.oranComp.Nre_max = SysPar.carrier.N_grid_size_mu*12;
        
        % save SysPar into Cfg_<TC#>.yaml config file
        if SysPar.SimCtrl.genTV.genYamlCfg
            fileName = sprintf('Cfg_%04d.yaml', caseNum);
            WriteYaml(fileName, SysPar);
        end
        
        if SysPar.SimCtrl.genTV.enable && SysPar.SimCtrl.genTV.launchPattern
            if ~ismember(caseNum, disabled_TC)
                LPFileName = 'launch_pattern_nrSim';
            else
                LPFileName = 'disabled_launch_pattern_nrSim';
            end
            slotIdx =  SysPar.SimCtrl.genTV.slotIdx;
            genSingleSlotLPFile(LPFileName, caseNum, slotIdx);
        end

        if ismember(caseNum, [4057]) % precoding TCs
            SysPar.SimCtrl.alg.enablePrcdBf = 1;
        end
        
        [SysPar, UE, gNB] = nrSimulator(SysPar);
        
        Detected = 1;
        if SysPar.SimCtrl.enableUeRx
            results = SysPar.SimCtrl.results.csirs;
            nCsirs = length(results);
            for idxCsirs = 1:nCsirs
                if (results{idxCsirs}.errCnt > 0)
                    Detected = 0;
                end
            end
            
            % bypass detection check for precoding cases
            if ismember(caseNum, [4057])
                Detected = 1;
            end
            
            if ~Detected
                detErr = detErr + 1;
            end
        end

        testPass = 1;        
        if ismember(caseNum, [4057]) || SysPar.SimCtrl.nPort_enable_csirs_compression <= 32 % precoding TCs, csirs compression
            bypassCompTest = 1;
        else
            bypassCompTest = 0;
        end        
        if testCompliance && ~bypassCompTest
            nComp = nComp + 1;
            
            csirs = gNB.Phy.Config.csirs;
            carrier = gNB.Phy.Config.carrier;
            table = gNB.Phy.Config.table;
            Xtf_5g = hCSIRSGen(csirs, carrier, table);
            
            % compare the last slot
            Xtf_nr = gNB.Phy.tx.Xtf;
            
            err_Xtf = sum(sum(sum(abs(Xtf_nr - Xtf_5g))));
            
            testPass = (err_Xtf < 1e-4);
            if ~testPass
                errCnt = errCnt + 1;
            end
        end
        fprintf('%4d  %2d  %2d  %2d  %4d    %3d   %3d   %2d   %2d   %2d   %2d      %2d   %2d   %2d\n',...
            CFG{idxSet, 1}, CFG{idxSet, 2}, CFG{idxSet, 3}, CFG{idxSet, 4}, ...
            CFG{idxSet, 5}, CFG{idxSet, 6}, CFG{idxSet, 7}, CFG{idxSet, 8}, ...
            CFG{idxSet, 9}, CFG{idxSet, 10}, CFG{idxSet, 11}, ...
            CFG{idxSet, 12}, testPass, Detected);
    end
end

fprintf('---------------------------------------------------------------------------\n');
fprintf('Total Compliance TC = %d, PASS = %d, FAIL = %d, Total TV generated = %d\n\n', nComp, nComp-errCnt, errCnt, nTV);
toc; 
fprintf('\n');