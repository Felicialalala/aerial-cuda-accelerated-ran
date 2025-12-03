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

function [nComp, errCnt, nTV, detErr] = testCompGenTV_prach(caseSet, compTvMode, subSetMod, relNum)

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

selected_TC = [5001:5999];
disabled_TC = [5023, 5109:5112]; % format 1
[~,TcIdx] = ismember(disabled_TC, selected_TC);
selected_TC(TcIdx) = [];

compact_TC = [5001:5999];
full_TC = [5001:5999];

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

% Based on TS 38.141 8.4 (V15.4)
CFG = {...
    % change only one parameter from the base case    
    % TC#   duplex  mu  cfg restrictSet root  zone  prmbIdx  Nant  N_nc delay   SNR    CFO 
    5001,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% format B4
    5002,       0,  0,  16,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% format 0
    5003,       1,  1, 155,     0,      137,    5,     2,      2,   1,   0e-6,  100,    0;% root
    5004,       1,  1, 155,     0,        0,   15,     2,      2,   1,   0e-6,  100,    0;% zone
    5005,       1,  1, 155,     0,        0,    5,    63,      2,   1,   0e-6,  100,    0;% prmbIdx
    5006,       1,  1, 155,     0,        0,    5,     2,     16,   1,   0e-6,  100,    0;% Nant
    5007,       1,  1, 155,     0,        0,    5,     2,      2,   4,   0e-6,  100,    0;% N_nc
    5008,       1,  1, 155,     0,        0,    5,     2,      2,   1,   1e-6,  100,    0;% delay
    5009,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,    0,    0;% SNR
    5010,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,  500;% CFO
    5011,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 2-UE
    5012,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 4-UE
    5013,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 16-UE
    5014,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% rx power +10dB
    5015,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% rx power -10dB
    5016,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% very small signal
    5017,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% very big signal
    5018,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% forceRxZero
    5019,       1,  1, 155,     0,        0,    5,     2,      2,   1,   1e-6,   0,   500;% BFP-16
    5020,       1,  1, 155,     0,        0,    5,     2,      2,   1,   1e-6,   0,   500;% BFP-14  
    5021,       1,  1, 155,     0,        0,    5,     2,      2,   1,   1e-6,   0,   500;% BFP-9  
    5022,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% BFP-9 16-UE
    5023,       1,  1,  33,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% format 1
    5024,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% n_RA_start    
    5025,       1,  1, 155,     0,        0,    5,     2,      2,   1,   1e-6,   0,   500;% FixedPoint-16  
    
    % change multiple parameters from the base case 
    % TC#   duplex  mu  cfg restrictSet root  zone  prmbIdx  Nant  N_nc delay   SNR   
    5101,       0,  0, 209,     0,       27,    6,    13,      2,   2,   2e-6,  -10,  100;% format B4
    5102,       0,  1, 217,     0,       15,    5,     9,      4,   4,   1e-6,    0, -200;% format B4
    5103,       1,  0, 155,     0,       39,   12,     0,      8,   1,   3e-6,   15,  300;% format B4
    5104,       1,  1, 168,     0,      101,    7,    32,      1,   2,   1e-6,   30, -400;% format B4
    5105,       0,  0,  16,     0,       27,   12,    13,      1,   1,   2e-6,    0, -200;% format 0
    5106,       0,  1,  27,     0,       15,    3,     9,      8,   1,   1e-6,   -5,  100;% format 0
    5107,       1,  0,   7,     0,       39,   12,     0,      4,   1,   2e-6,   15, -100;% format 0
    5108,       1,  1,  27,     0,      101,   12,    32,      2,   1,   1e-6,   30,  100;% format 0
    5109,       0,  0,  44,     0,       27,   12,    13,      1,   1,   2e-6,   10, -100;% format 1
    5110,       0,  1,  46,     0,       15,    3,     9,      8,   1,   1e-6,   -5,  100;% format 1
    5111,       1,  0,  33,     0,       39,   12,     0,      4,   1,   2e-6,   15, -100;% format 1
    5112,       1,  1,  33,     0,      101,    7,    32,      2,   1,   1e-6,   30,  400;% format 1
    
    
    % different BW
    % mu = 1
    5201,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 5 MHz
    5202,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 10 MHz
    5203,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 15 MHz
    5204,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 20 MHz
    5205,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 25 MHz
    5206,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 30 MHz
    5207,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 40 MHz
    5208,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 50 MHz
    5209,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 60 MHz
    5210,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 70 MHz
    5211,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 80 MHz
    5212,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 90 MHz
    5213,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 100 MHz
    % mu = 0
    5214,       1,  0, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 5 MHz
    5215,       1,  0, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 10 MHz
    5216,       1,  0, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 15 MHz
    5217,       1,  0, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 20 MHz
    5218,       1,  0, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 25 MHz
    5219,       1,  0, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 30 MHz
    5220,       1,  0, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 40 MHz
    5221,       1,  0, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 50 MHz
    % additional BW test cases
    5222,       1,  1, 155,     0,        0,    5,     2,      2,   1,   0e-6,  100,    0;% 40 MHz
    
    % Perf test vectors
    % TC#   duplex  mu  cfg restrictSet root  zone  prmbIdx  Nant  N_nc delay   SNR  CFO
    % F08 full cell
    5801,       1,  1, 158,     0,      0,    5,     2,      4,   1,   2e-6,  20,  0;
    % F08 ave cell
    5802,       1,  1, 158,     0,      0,    5,     2,      4,   1,   2e-6,  20,  0;
    % F14 full cell
    5803,       1,  1, 158,     0,      0,    5,     2,      16,   1,   2e-6,  20,  0;
    % F14 ave cell
    5804,       1,  1, 158,     0,      0,    5,     2,      16,   1,   2e-6,  20,  0;
    % F09
    5805,       1,  1, 158,     0,      0,    5,     2,      8,    1,   2e-6,  20,  0;
    % FDD 4T4R
    5806,       0,  1,  16,     0,      0,    5,     2,      4,    1,   0e-6,  100,    0;% format 0
    % F14 8 PRACH
    5807,       1,  1, 158,     0,      0,    5,     2,      16,   1,   2e-6,  20,  0;

    % specific configuration for TV generation
    % TC#   duplex  mu  cfg restrictSet root  zone  prmbIdx  Nant  N_nc delay   SNR
    5901,       1,  1, 158,     0,      137,    5,     3,      1,   1,   2e-6,  -10,  500;% demo_msg1
    % TVs for prach_test
    5911,       0,  0,  27,     0,       22,    1,    32,      4,   1,   0e-6,  100,    0;% F0, mu = 0
    5912,       0,  1,  27,     0,       22,    1,    32,      8,   1,   0e-6,  100,    0;% F0, mu = 1
    5913,       0,  0, 217,     0,        0,   11,     0,      4,   1,   0e-6,  100,    0;% B4, mu = 0
    5914,       0,  1, 217,     0,        0,   11,     0,      8,   1,   0e-6,  100,    0;% B4, mu = 1
          
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
fprintf('PRACH: genTV = %d, testCompliance = %d, caseSet = %s', genTV, testCompliance, caseSetStr);
fprintf('\nTC  duplex mu cfgIdx resSet rootIdx  corrZone prmbIdx   Nant  N_nc  delay  SNR   CFO  PASS  det\n');
fprintf('-----------------------------------------------------------------------------------------------\n');

parfor n = 1:NallTest
    testAlloc = [];
    testAlloc.dl = 0;
    testAlloc.ul = 1;
    testAlloc.prach = 1;
    idxSet = n;
    caseNum = CFG{idxSet, 1};
    prachFormat = 'B4';
    if ismember(caseNum, [5023, 5109:5112])
        prachFormat = '1';
    elseif ismember(caseNum, [5002, 5105:5108])
        prachFormat = '0';
    end
    if ismember(caseNum, TcToTest) && (mod(caseNum, subSetMod(2)) == subSetMod(1))
        rng(caseNum);
        SysPar = initSysPar(testAlloc);
        SysPar.SimCtrl.relNum = relNum;        
        SysPar.SimCtrl.N_frame = 1;
        SysPar.SimCtrl.N_slot_run = 0;
        if genTV
            nTV = nTV + 1;
            SysPar.SimCtrl.genTV.enable = 1;
            SysPar.SimCtrl.genTV.TVname = sprintf('TVnr_%04d', caseNum);
            if ismember(caseNum, disabled_TC)
                SysPar.SimCtrl.genTV.TVname = sprintf('disabled_TVnr_%04d', caseNum);
            end
            if strcmp(prachFormat, '1')
                SysPar.SimCtrl.genTV.slotIdx = [0:19];
            elseif CFG{idxSet, 3} == 1 % mu = 1
                SysPar.SimCtrl.genTV.slotIdx = [0:1];
                SysPar.prach{1}.allSubframes = 1;
                SysPar.SimCtrl.N_slot_run = 2;
            else
                SysPar.SimCtrl.genTV.slotIdx = [0];
                SysPar.prach{1}.allSubframes = 1;
                SysPar.SimCtrl.N_slot_run = 1;
            end
%             SysPar.SimCtrl.genTV.forceSlotIdxFlag = 1;
%             SysPar.SimCtrl.N_slot_run = 2;
%             SysPar.SimCtrl.genTV.slotIdx = 6;            
        end        
        SysPar.carrier.duplex =  CFG{idxSet, 2};
        SysPar.carrier.mu =  CFG{idxSet, 3};
        if SysPar.carrier.mu == 0
            SysPar.carrier.N_grid_size_mu = 106;
        end
        SysPar.prach{1}.configurationIndex = CFG{idxSet, 4};
        SysPar.prach{1}.restrictedSet = CFG{idxSet, 5};
        SysPar.prach{1}.rootSequenceIndex = CFG{idxSet, 6};
        SysPar.prach{1}.zeroCorrelationZone = CFG{idxSet, 7};
        SysPar.prach{1}.prmbIdx = CFG{idxSet, 8};
        SysPar.carrier.Nant_gNB = CFG{idxSet, 9};
        SysPar.carrier.Nant_UE = 1;
        SysPar.prach{1}.N_nc  = CFG{idxSet, 10};        
        SysPar.Chan{1}.delay = CFG{idxSet, 11};
        SysPar.Chan{1}.SNR = CFG{idxSet, 12};
        SysPar.Chan{1}.CFO = CFG{idxSet, 13};    
        if caseNum == 5011
            SysPar.SimCtrl.N_UE = 2;
            SysPar.Chan{2} = SysPar.Chan{1};
            SysPar.prach{1}.idxUE = [0, 1];
        elseif caseNum == 5012
            nUe = 4;
            SysPar.SimCtrl.N_UE = nUe;
            for idxChan = 1:nUe-1
                SysPar.Chan{idxChan+1} = SysPar.Chan{1};
            end
            SysPar.prach{1}.idxUE = [0:nUe-1];
        elseif ismember(caseNum, [5013, 5020])
            nUe = 16;
            nRo = 4;
            nUePerRo = nUe/nRo;
            SysPar.SimCtrl.N_UE = nUe;
            for idxUe = 1:nUe-1
                SysPar.Chan{idxUe+1} = SysPar.Chan{1};                
            end
            SysPar.prach{1}.msg1_FDM = nRo;
            SysPar.prach{1}.idxUE  = [0:nUePerRo-1];
            for idxRo = 1:nRo-1
                SysPar.prach{idxRo+1} = SysPar.prach{1};
                SysPar.prach{idxRo+1}.idxUE = idxRo * nUePerRo + [0:nUePerRo-1];
                SysPar.prach{idxRo+1}.n_RA_start = 12*idxRo;
            end
            SysPar.testAlloc.prach = nRo;
        elseif caseNum == 5014
            SysPar.Chan{1}.gain = sqrt(10);
        elseif caseNum == 5015
            SysPar.Chan{1}.gain = sqrt(0.1);
        elseif caseNum == 5016
            SysPar.Chan{1}.gain = 0.01;
        elseif caseNum == 5017
            SysPar.Chan{1}.gain = 100;
        elseif caseNum == 5018
            SysPar.SimCtrl.forceRxZero = 1;
        elseif caseNum == 5024
            nUe = 4;
            nRo = 4;
            nUePerRo = nUe/nRo;
            SysPar.SimCtrl.N_UE = nUe;
            for idxUe = 1:nUe-1
                SysPar.Chan{idxUe+1} = SysPar.Chan{1};
            end
            SysPar.prach{1}.msg1_FDM = nRo;
            SysPar.prach{1}.idxUE  = [0:nUePerRo-1];
            for idxRo = 1:nRo-1
                SysPar.prach{idxRo+1} = SysPar.prach{1};
                SysPar.prach{idxRo+1}.idxUE = idxRo * nUePerRo + [0:nUePerRo-1];
                SysPar.prach{idxRo+1}.n_RA_start = 2*12*idxRo + 30 + idxRo;
            end
            SysPar.testAlloc.prach = nRo;
        elseif caseNum == 5201
            SysPar.carrier.N_grid_size_mu = 11; % 5 MHz
        elseif caseNum == 5202
            SysPar.carrier.N_grid_size_mu = 24; % 10 MHz
        elseif caseNum == 5203
            SysPar.carrier.N_grid_size_mu = 38; % 15 MHz
        elseif caseNum == 5204
            SysPar.carrier.N_grid_size_mu = 51; % 20 MHz
        elseif caseNum == 5205
            SysPar.carrier.N_grid_size_mu = 65; % 25 MHz
        elseif caseNum == 5206
            SysPar.carrier.N_grid_size_mu = 78; % 30 MHz
        elseif caseNum == 5207
            SysPar.carrier.N_grid_size_mu = 106; % 40 MHz
        elseif caseNum == 5208
            SysPar.carrier.N_grid_size_mu = 133; % 50 MHz
        elseif caseNum == 5209
            SysPar.carrier.N_grid_size_mu = 162; % 60 MHz
        elseif caseNum == 5210
            SysPar.carrier.N_grid_size_mu = 189; % 70 MHz
        elseif caseNum == 5211
            SysPar.carrier.N_grid_size_mu = 217; % 80 MHz
        elseif caseNum == 5212
            SysPar.carrier.N_grid_size_mu = 245; % 90 MHz
        elseif caseNum == 5213
            SysPar.carrier.N_grid_size_mu = 273; % 100 MHz
        elseif caseNum == 5214
            SysPar.carrier.N_grid_size_mu = 25; % 5 MHz
        elseif caseNum == 5215
            SysPar.carrier.N_grid_size_mu = 52; % 10 MHz
        elseif caseNum == 5216
            SysPar.carrier.N_grid_size_mu = 79; % 15 MHz
        elseif caseNum == 5217
            SysPar.carrier.N_grid_size_mu = 106; % 20 MHz
        elseif caseNum == 5218
            SysPar.carrier.N_grid_size_mu = 133; % 25 MHz
        elseif caseNum == 5219
            SysPar.carrier.N_grid_size_mu = 160; % 30 MHz
        elseif caseNum == 5220
            SysPar.carrier.N_grid_size_mu = 216; % 40 MHz
        elseif caseNum == 5221
            SysPar.carrier.N_grid_size_mu = 270; % 50 MHz
        elseif caseNum == 5222
            SysPar.carrier.N_grid_size_mu = 100; % 40 MHz            
        elseif caseNum == 5803
            SysPar.SimCtrl.N_UE = 2;
            SysPar.Chan{2} = SysPar.Chan{1};
            SysPar.prach{1}.idxUE = [0, 1];
        elseif caseNum == 5806
            SysPar.carrier.N_grid_size_mu = 106;
        elseif caseNum == 5807
            SysPar.SimCtrl.N_UE = 8;
            for ue = 1 : 8
                SysPar.Chan{ue} = SysPar.Chan{1};
                SysPar.prach{ue} = SysPar.prach{1};
                SysPar.prach{ue}.idxUE = ue - 1;
                SysPar.prach{ue}.n_RA_start = 12*(ue - 1);
            end
        end
        
        if caseNum == 5019
            SysPar.SimCtrl.BFPforCuphy = 16;
        elseif caseNum == 5020
            SysPar.SimCtrl.BFPforCuphy = 14;
        elseif ismember(caseNum, [5021, 5022])
            SysPar.SimCtrl.BFPforCuphy = 9;
        elseif caseNum == 5025
            SysPar.SimCtrl.FixedPointforCuphy = 16;
        end

        SysPar.SimCtrl.timeDomainSim = 1;        

        for idxUe = 1:length(SysPar.prach)
            digBFInterfaces = SysPar.carrier.Nant_gNB;
            SysPar.prach{idxUe}.digBFInterfaces = digBFInterfaces;
            SysPar.prach{idxUe}.beamIdx = [1:digBFInterfaces];
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
        
        [SysPar, UE, gNB] = nrSimulator(SysPar);        
        SimCtrl = SysPar.SimCtrl; 
        
        results = SimCtrl.results.prach;
        nPrach = length(results);
        Detected = 1;
        for idxPrach = 1:nPrach
            if results{idxPrach}.falseCnt || results{idxPrach}.missCnt
                Detected = 0;
            end
        end
        
        % bypass detection check for forceRxZero case
        if caseNum == 5018
            Detected = 1;
        end
        
        if ~Detected
            detErr = detErr + 1;
        end
        
        testPass = 1;
        if testCompliance
            nComp = nComp + 1;
            
            prach = SysPar.prach{1};
            carrier = SysPar.carrier;
            prachTable = UE{1}.Mac.Config.table;
            [sym_5g, NSlot, LRA] = hPRACHGen(prach, carrier, prachTable);
            sym_5g = sym_5g(1:LRA);
            
            if SysPar.prach{1}.allSubframes
                NSlot = mod(NSlot, 2);
            end
            if strcmp(prachFormat, '1') || SysPar.SimCtrl.N_slot_run == 1 ...
                    || SysPar.SimCtrl.genTV.forceSlotIdxFlag
                NSlot = 0;
            end
            
            sym_nr = UE{1}.Phy.tx.Xtf_prach_frame(1:LRA, NSlot+1)*sqrt(LRA);
            
            err_sym = sum(abs(sym_nr-sym_5g));
            
            testPass = (err_sym < 1e-4);
            if ~testPass
                errCnt = errCnt + 1;
            end
        end
        fprintf('%4d %4d %3d %4d %5d %9d %8d %8d %6d  %4d  %6.1f %4d  %4d %4d %4d\n',...
            CFG{idxSet, 1}, CFG{idxSet, 2}, CFG{idxSet, 3}, CFG{idxSet, 4}, ...
            CFG{idxSet, 5}, CFG{idxSet, 6}, CFG{idxSet, 7}, CFG{idxSet, 8}, ...
            CFG{idxSet, 9}, CFG{idxSet, 10}, 1e6*CFG{idxSet, 11}, CFG{idxSet, 12},...
            CFG{idxSet, 13}, testPass, Detected);
    end
end
fprintf('-----------------------------------------------------------------------------------------------\n');
fprintf('Total Compliance TC = %d, PASS = %d, FAIL = %d, Total TV generated = %d\n\n', nComp, nComp-errCnt, errCnt, nTV);
toc; 
fprintf('\n');