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

function [nComp, errCnt, nTV, detErr] = testCompGenTV_ssb(caseSet, compTvMode, subSetMod, relNum)

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

selected_TC = [1001:1999];
disabled_TC = []; % modComp
[~,TcIdx] = ismember(disabled_TC, selected_TC);
selected_TC(TcIdx) = [];

compact_TC = [1001:1999];
full_TC = [1001:1999];

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
    % change only one parameter from the base case
  % TC#    mu   Nid  n_hf  L_max k_SSB offsetPA SFN blockIdx
    1001,   1,    0,    0,    8,    0,      0,    0,   0; % base case
    1002,   0,    0,    0,    8,    0,      0,    0,   0; % mu = 0
    1003,   1, 1007,    0,    8,    0,      0,    0,   0; % Nid > 0
    1004,   1,    0,    1,    8,    0,      0,    0,   0; % n_hf = 1
    1005,   1,    0,    0,    4,    0,      0,    0,   0; % L_max = 4
    1006,   1,    0,    0,    8,   22,      0,    0,   0; % k_SSB > 0
    1007,   1,    0,    0,    8,    0,    263,    0,   0; % offsetPA > 0
    1008,   1,    0,    0,    8,    0,      0, 1023,   0; % SFN > 0
    1009,   1,    0,    0,    8,    0,      0,    0,   7; % blockIdx > 0    
    1010,   1,    0,    0,    8,    0,      0,    0,   0; % precoding
    1011,   1,    0,    0,    8,    0,      0,    0,   0; % betaPss
    1012,   1,   41,    0,    8,   22,    248,    0,   0; % modComp  
    
    % change multiple parameters from the base case
  % TC#    mu   Nid  n_hf  L_max k_SSB offsetPA SFN blockIdx
    1101,   0,   40,    1,    4,    3,     15,   10,   2;
    1102,   1,  100,    0,    8,   18,     80,   20,   5;
    1103,   0,   40,    1,    4,    3,     15,   10,{0,1}; % 2 SSBs in the first slot
    1104,   1,  100,    0,    8,   18,     80,   20,{6,7}; % 2 SSBs in the last slot
    1105,   0,   40,    1,    4,    3,     15,   10,{0,1}; % 2 SSBs in the first slot, precoding
    1106,   1,  100,    0,    8,   18,     80,   20,{6,7}; % 2 SSBs in the last slot, precoding
    1107,   0,   40,    1,    4,    3,     15,   10,{0,1}; % 2 SSBs in the first slot, precoding
    1108,   1,  100,    0,    8,   18,     80,   20,{6,7}; % 2 SSBs in the last slot, precoding
    
    % different BW
  % TC#    mu   Nid  n_hf  L_max k_SSB offsetPA SFN blockIdx    
    % mu = 1
%     1201,   1,    0,    0,    8,    0,      0,    0,   0; % 5 MHz
    1202,   1,    0,    0,    8,    0,      0,    0,   0; % 10 MHz
    1203,   1,    0,    0,    8,    0,      0,    0,   0; % 15 MHz
    1204,   1,    0,    0,    8,    0,      0,    0,   0; % 20 MHz
    1205,   1,    0,    0,    8,    0,      0,    0,   0; % 25 MHz
    1206,   1,    0,    0,    8,    0,      0,    0,   0; % 30 MHz
    1207,   1,    0,    0,    8,    0,      0,    0,   0; % 40 MHz
    1208,   1,    0,    0,    8,    0,      0,    0,   0; % 50 MHz
    1209,   1,    0,    0,    8,    0,      0,    0,   0; % 60 MHz
    1210,   1,    0,    0,    8,    0,      0,    0,   0; % 70 MHz
    1211,   1,    0,    0,    8,    0,      0,    0,   0; % 80 MHz
    1212,   1,    0,    0,    8,    0,      0,    0,   0; % 90 MHz
    1213,   1,    0,    0,    8,    0,      0,    0,   0; % 100 MHz
    % mu = 0    
    1214,   0,    0,    0,    8,    0,      0,    0,   0; %  5 MHz
    1215,   0,    0,    0,    8,    0,      0,    0,   0; % 10 MHz
    1216,   0,    0,    0,    8,    0,      0,    0,   0; % 15 MHz
    1217,   0,    0,    0,    8,    0,      0,    0,   0; % 20 MHz
    1218,   0,    0,    0,    8,    0,      0,    0,   0; % 25 MHz
    1219,   0,    0,    0,    8,    0,      0,    0,   0; % 30 MHz
    1220,   0,    0,    0,    8,    0,      0,    0,   0; % 40 MHz
    1221,   0,    0,    0,    8,    0,      0,    0,   0; % 50 MHz
    % additional BW test cases
    1222,   1,    0,    0,    8,    0,      0,    0,   0; % 40 MHz
        
    % Perf test vectors
  % TC#    mu   Nid  n_hf  L_max k_SSB offsetPA SFN blockIdx
    1801,   1,    0,    0,    8,    0,      0,    0,   0; % base case   
    1802,   1,    0,    0,    8,    0,      0,    0,   0; % FDD 106 PRB 
    1803,   1,    0,    0,    8,    0,      0,    0,   {0,1}; 
    
    % specific configuration for TV generation
  % TC#    mu   Nid  n_hf  L_max k_SSB offsetPA SFN blockIdx
    1901,   1,   41,    0,    8,   22,    248,    0,   1;  % demo_ssb   
    1902,   1,   41,    0,    8,    0,    263,    0,   0;  % from 1007 for CP pipeline
    % requested TCs
    1903,   1,   41,    0,    8,   22,      0,    0,   2; % ssbIdx = 2
    1904,   1,   41,    0,    8,   22,      0,    0,   3; % ssbIdx = 3
    1905,   1,   41,    0,    8,   22,      0,    0,   4; % ssbIdx = 4
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
fprintf('SSB: genTV = %d, testCompliance = %d, caseSet = %s', genTV, testCompliance, caseSetStr);
fprintf('\nTC#    mu   Nid n_hf L_max k_SSB offsetPA SFN blockIdx Pass Det\n');
fprintf('---------------------------------------------------------------\n');

parfor n = 1:NallTest
    testAlloc = [];
    testAlloc.dl = 1;
    testAlloc.ul = 0;
    testAlloc.ssb = 1;
    idxSet = n;
    caseNum = CFG{idxSet, 1};
    if ismember(caseNum, TcToTest) && (mod(caseNum, subSetMod(2)) == subSetMod(1))
        rng(caseNum);
        SysPar = initSysPar(testAlloc);
        SysPar.SimCtrl.relNum = relNum;
        SysPar.SimCtrl.N_frame = 1;
        SysPar.SimCtrl.N_slot_run = 0;
%         SysPar.SimCtrl.genTV.forceSlotIdxFlag = 1;
        if genTV
            nTV = nTV + 1;
            SysPar.SimCtrl.genTV.enable = 1;
            SysPar.SimCtrl.genTV.TVname = sprintf('TVnr_%04d', caseNum);
            if ismember(caseNum, disabled_TC)
                SysPar.SimCtrl.genTV.TVname = sprintf('disabled_TVnr_%04d', caseNum);
            end
        end
        SysPar.carrier.mu = CFG{idxSet, 2};
        if SysPar.carrier.mu == 0
            SysPar.carrier.N_grid_size_mu = 106;
        end
        SysPar.carrier.N_ID_CELL =  CFG{idxSet, 3};
        SysPar.ssb.n_hf =  CFG{idxSet, 4};
        SysPar.ssb.L_max = CFG{idxSet, 5};
        SysPar.ssb.ssbSubcarrierOffset = CFG{idxSet, 6};
        SysPar.ssb.SsbOffsetPointA = CFG{idxSet, 7};
        SysPar.carrier.SFN_start = CFG{idxSet, 8};
        if SysPar.ssb.L_max == 8
            SysPar.ssb.symIdxInFrame = [2, 8, 16, 22, 30, 36, 44, 50];
        elseif SysPar.ssb.L_max == 4
            SysPar.ssb.symIdxInFrame = [2, 8, 16, 22];
        end
        if iscell(CFG{idxSet, 9})
            blockIdx =  cell2mat(CFG{idxSet, 9});
            blockIdx_display = 'M';
        else
            blockIdx =  CFG{idxSet, 9};      
            blockIdx_display = num2str(blockIdx);
        end
        SysPar.ssb.ssbBitMap = zeros(1, 8);
        SysPar.ssb.ssbBitMap(blockIdx+1) = 1;
        SysPar.SimCtrl.genTV.slotIdx = floor(SysPar.ssb.symIdxInFrame(blockIdx+1)/14);
        if SysPar.ssb.n_hf == 1
            nSlotsHalfFrame = 5*2^(SysPar.carrier.mu);
            SysPar.ssb.symIdxInFrame = SysPar.ssb.symIdxInFrame + 14*nSlotsHalfFrame;
            SysPar.SimCtrl.genTV.slotIdx = SysPar.SimCtrl.genTV.slotIdx + nSlotsHalfFrame;
        end
        
        if caseNum == 1010
            SysPar.ssb.prcdBf_vec(1) = 4;
        elseif ismember(caseNum, [1011, 1012, 1903:1905])
            SysPar.ssb.betaPss = 1;
            if caseNum == 1012
                SysPar.SimCtrl.genTV.fhMsgMode = 2;
            end
        elseif caseNum == 1102
            SysPar.ssb.beamIdx_0 = 2;
        elseif caseNum == 1103
            SysPar.ssb.beamIdx_0 = 1;
            SysPar.ssb.beamIdx_1 = 2;
        elseif caseNum == 1104
            SysPar.ssb.beamIdx_0 = 1;            
            SysPar.ssb.beamIdx_1 = 2;
        elseif caseNum == 1105
            SysPar.ssb.prcdBf_vec(1) = 3;
            SysPar.ssb.prcdBf_vec(2) = 4;
        elseif caseNum == 1106
            SysPar.ssb.prcdBf_vec(1) = 4;
            SysPar.ssb.prcdBf_vec(2) = 3;
        elseif caseNum == 1107
            SysPar.ssb.prcdBf_vec(1) = 2;
            SysPar.ssb.prcdBf_vec(2) = 4;
        elseif caseNum == 1108
            SysPar.ssb.prcdBf_vec(1) = 3;
            SysPar.ssb.prcdBf_vec(2) = 1;
%         elseif caseNum == 1201
%             SysPar.carrier.N_grid_size_mu = 11; % 5 MHz
        elseif caseNum == 1202
            SysPar.carrier.N_grid_size_mu = 24; % 10 MHz
        elseif caseNum == 1203
            SysPar.carrier.N_grid_size_mu = 38; % 15 MHz
        elseif caseNum == 1204
            SysPar.carrier.N_grid_size_mu = 51; % 20 MHz
        elseif caseNum == 1205
            SysPar.carrier.N_grid_size_mu = 65; % 25 MHz
        elseif caseNum == 1206
            SysPar.carrier.N_grid_size_mu = 78; % 30 MHz
        elseif caseNum == 1207
            SysPar.carrier.N_grid_size_mu = 106; % 40 MHz
        elseif caseNum == 1208
            SysPar.carrier.N_grid_size_mu = 133; % 50 MHz
        elseif caseNum == 1209
            SysPar.carrier.N_grid_size_mu = 162; % 60 MHz
        elseif caseNum == 1210
            SysPar.carrier.N_grid_size_mu = 189; % 70 MHz
        elseif caseNum == 1211
            SysPar.carrier.N_grid_size_mu = 217; % 80 MHz
        elseif caseNum == 1212
            SysPar.carrier.N_grid_size_mu = 245; % 90 MHz
        elseif caseNum == 1213
            SysPar.carrier.N_grid_size_mu = 273; % 100 MHz
        elseif caseNum == 1214
            SysPar.carrier.N_grid_size_mu = 25; % 5 MHz
        elseif caseNum == 1215
            SysPar.carrier.N_grid_size_mu = 52; % 10 MHz
        elseif caseNum == 1216
            SysPar.carrier.N_grid_size_mu = 79; % 15 MHz
        elseif caseNum == 1217
            SysPar.carrier.N_grid_size_mu = 106; % 20 MHz
        elseif caseNum == 1218
            SysPar.carrier.N_grid_size_mu = 133; % 25 MHz
        elseif caseNum == 1219
            SysPar.carrier.N_grid_size_mu = 160; % 30 MHz
        elseif caseNum == 1220
            SysPar.carrier.N_grid_size_mu = 216; % 40 MHz
        elseif caseNum == 1221
            SysPar.carrier.N_grid_size_mu = 270; % 50 MHz
        elseif caseNum == 1222
            SysPar.carrier.N_grid_size_mu = 100; % 40 MHz            
        elseif caseNum == 1901
            SysPar.ssb.mib = [
               0,...             % BCH_choice
               0,0,0,0,0,0,...   % 6 SFN MSBs, replaced at run-time
               1,...             % scsCommon, 1=scs30
               0,1,1,0,...       % ssbScOffset (4 LSBs of ssbSubcarrierOffset)
               0,...             % dmrsTypeAPosition, 0=pos2
               1,1,0,0,...       % pdcchConfigSIB1: 4 MSBs are per Table 13-4 38.213, set to 12d
               0,0,0,0,...       % pdcchConfigSIB1: 4 LSBs are per Table 13-11 38.213, set to 0d
               1,...             % cellBarred, 1=not barred
               1,...             % intraFreqReselection
               0];               % spare bit
        elseif caseNum == 1802
            SysPar.carrier.N_grid_size_mu = 106; 
            SysPar.carrier.duplex = 0;
            SysPar.prach{1}.configurationIndex = 198;
            SysPar.testAlloc.prach = 0;
        end
        
        for idxSsb = 1:length(SysPar.ssb.prcdBf_vec)
            digBFInterfaces = 1;
            switch SysPar.ssb.prcdBf_vec(idxSsb)
                case 0
                    digBFInterfaces = 1;
                case {1, 2, 5, 6}
                    digBFInterfaces = 2;
                case {3, 4, 7, 8, 9, 10, 11, 12}
                    digBFInterfaces = 4;
                otherwise
                    error('prcdBf is not supported ... \n');
            end
            switch idxSsb
                case 1
                    SysPar.ssb.digBFInterfaces_0 = digBFInterfaces;
                    SysPar.ssb.beamIdx_0 = [1:digBFInterfaces];
                case 2
                    SysPar.ssb.digBFInterfaces_1 = digBFInterfaces;
                    SysPar.ssb.beamIdx_1 = [1:digBFInterfaces];
                case 3
                    SysPar.ssb.digBFInterfaces_2 = digBFInterfaces;
                    SysPar.ssb.beamIdx_2 = [1:digBFInterfaces];
                case 4
                    SysPar.ssb.digBFInterfaces_3 = digBFInterfaces;
                    SysPar.ssb.beamIdx_3 = [1:digBFInterfaces];
                otherwise
                    error('idxSsb is not supported ... \n');
            end
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

        if ismember(caseNum, [1010, 1105, 1106, 1107, 1108]) % precoding TCs
            SysPar.SimCtrl.alg.enablePrcdBf = 1;
        end
        
        if strcmp(caseSet, 'full') || strcmp(caseSet, 'compact')
            SysPar.SimCtrl.enableUeRx = 1;
        end
        [SysPar, UE, gNB] = nrSimulator(SysPar);     
        
        Detected = 1;
        if SysPar.SimCtrl.enableUeRx
            results = SysPar.SimCtrl.results.ssb;
            nSsb = length(results);
            for idxSsb = 1:nSsb
                if (results{idxSsb}.errCnt > 0)
                    Detected = 0;
                end
            end
            
            if ~Detected
                detErr = detErr + 1;
            end
        end
                
        testPass = 1;        
        if ismember(caseNum, [1010, 1105, 1106, 1107, 1108]) % precoding TCs 
            bypassCompTest = 1;
        else
            bypassCompTest = 0;
        end        
        if testCompliance && ~bypassCompTest
            nComp = nComp + 1;

            f0 = (SysPar.ssb.ssbSubcarrierOffset + SysPar.ssb.SsbOffsetPointA*12)/2^SysPar.carrier.mu;
            Xtf_nr = gNB.Phy.tx.Xtf_frame(f0+1:f0+240,:, 1);
            ssb = gNB.Phy.Config.ssb;
            carrier = gNB.Phy.Config.carrier;
            ssb_type = SysPar.ssb.caseType;
            Xtf_5g = hSSBGen(ssb, carrier, ssb_type, SysPar);
            err = sum(sum(abs(Xtf_nr-Xtf_5g)));

            testPass = (err < 1e-4);
            if ~testPass
                errCnt = errCnt + 1;
            end
        end
        fprintf('%4d %4d %4d %3d %4d %6d %6d %6d %5s %6d %4d\n',...
            CFG{idxSet, 1}, CFG{idxSet, 2}, CFG{idxSet, 3}, CFG{idxSet, 4}, ...
            CFG{idxSet, 5}, CFG{idxSet, 6}, CFG{idxSet, 7}, CFG{idxSet, 8}, ...
            blockIdx_display, testPass, Detected);
    end
end
fprintf('---------------------------------------------------------------\n');
fprintf('Total Compliance TC = %d, PASS = %d, FAIL = %d, Total TV generated = %d\n\n', nComp, nComp-errCnt, errCnt, nTV);
toc; 
fprintf('\n');