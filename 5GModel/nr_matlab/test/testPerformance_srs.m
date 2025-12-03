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

function [nPerf, nFail] = testPerformance_srs(caseSet, chanType, N_frame, SNR_offset, batchsimCfg)

tic;
if nargin == 0
    caseSet = 'full';
    chanType = 'AWGN';
    N_frame = 1;
    SNR_offset = 0;
    batchsimCfg.batchsimMode = 0; % 0: disable batchsimMode
elseif nargin == 1
    chanType = 'AWGN';
    N_frame = 1;
    SNR_offset = 0;
    batchsimCfg.batchsimMode = 0;
elseif nargin == 2
    N_frame = 1;
    SNR_offset = 0;
    batchsimCfg.batchsimMode = 0;
elseif nargin == 3
    SNR_offset = 0;
    batchsimCfg.batchsimMode = 0;
elseif nargin == 4
    batchsimCfg.batchsimMode = 0;
end


selected_TC = [];
compact_TC = [];
full_TC = [8001:8999];

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
   % TC#  rnti  Nap nSym  Nrep sym0 cfgIdx seqId bwIdx cmbSz cmbOffset cs  fPos fShift frqH grpH resType Tsrs  Toffset idxSlot SNR
   % 272 PRBs
    8001  17     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0     0;
    8002  17     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0     6;
    8003  17     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    12;
    8004  17     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    18;
    8005  17     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    24;
    8006  17     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    30;
    8007  17     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0     0;
    8008  17     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0     6;
    8009  17     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    12;
    8010  17     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    18;
    8011  17     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    24;
    8012  17     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    30;
    8013  17     4    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0     0;
    8014  17     4    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0     6;
    8015  17     4    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    12;
    8016  17     4    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    18;
    8017  17     4    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    24;
    8018  17     4    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    30;
    % 16 PRBs
    8101  17     1    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0     0;
    8102  17     1    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0     6;
    8103  17     1    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0    12;
    8104  17     1    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0    18;
    8105  17     1    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0    24;
    8106  17     1    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0    30;
    8107  17     2    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0     0;
    8108  17     2    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0     6;
    8109  17     2    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0    12;
    8110  17     2    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0    18;
    8111  17     2    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0    24;
    8112  17     2    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0    30;
    8113  17     4    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0     0;
    8114  17     4    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0     6;
    8115  17     4    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0    12;
    8116  17     4    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0    18;
    8117  17     4    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0    24;
    8118  17     4    1     1   13    63     3    1     4        1      3    2    1      3    1     0      1      0       0    30;     
   % 4 PRBs
    8201  17     1    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0     0;
    8202  17     1    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0     6;
    8203  17     1    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0    12;
    8204  17     1    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0    18;
    8205  17     1    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0    24;
    8206  17     1    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0    30;
    8207  17     2    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0     0;
    8208  17     2    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0     6;
    8209  17     2    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0    12;
    8210  17     2    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0    18;
    8211  17     2    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0    24;
    8212  17     2    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0    30;
    8213  17     4    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0     0;
    8214  17     4    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0     6;
    8215  17     4    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0    12;
    8216  17     4    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0    18;
    8217  17     4    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0    24;
    8218  17     4    1     1   13    63     3    3     2        1      3    2    1      3    1     0      1      0       0    30;
    };

[NallTest, ~] = size(CFG);
nTV = 0;
srsDetected = [];

fprintf('Test SRS detection:\n');
fprintf('\nTC#  rnti  Nap nSym  Nrep sym0 cfgIdx seqId bwIdx cmbSz cmbOffset cs  SNR    det snrErrDb toErrUs hestErrDb\n');
fprintf('-----------------------------------------------------------------------------------------------------------\n');

num_generated_subscenarios = 0;
parfor n = 1:NallTest
    testAlloc = [];
    testAlloc.dl = 0;
    testAlloc.ul = 1;
    testAlloc.srs = 1;
    idxSet = n;
    caseNum = CFG{idxSet, 1};
    if ismember(caseNum, TcToTest)
        rng(caseNum);
        SysPar = initSysPar(testAlloc);
        SysPar.SimCtrl.N_frame = N_frame;
        SysPar.SimCtrl.N_slot_run = 0;
        
        SysPar.srs{1}.RNTI = CFG{idxSet, 2};
        SysPar.srs{1}.numAntPorts = CFG{idxSet, 3};
        SysPar.srs{1}.numSymbols = CFG{idxSet, 4};
        SysPar.srs{1}.numRepetitions = CFG{idxSet, 5};
        SysPar.srs{1}.timeStartPosition = CFG{idxSet, 6};
        SysPar.srs{1}.configIndex = CFG{idxSet, 7};
        SysPar.srs{1}.sequenceId = CFG{idxSet, 8};
        SysPar.srs{1}.bandwidthIndex = CFG{idxSet, 9};
        SysPar.srs{1}.combSize = CFG{idxSet, 10};
        SysPar.srs{1}.combOffset = CFG{idxSet, 11};
        SysPar.srs{1}.cyclicShift = CFG{idxSet, 12};
        SysPar.srs{1}.frequencyPosition = CFG{idxSet, 13};
        SysPar.srs{1}.frequencyShift = CFG{idxSet, 14};
        SysPar.srs{1}.frequencyHopping = CFG{idxSet, 15};
        SysPar.srs{1}.groupOrSequenceHopping = CFG{idxSet, 16};
        SysPar.srs{1}.resourceType = CFG{idxSet, 17};
        SysPar.srs{1}.Tsrs = CFG{idxSet, 18};
        SysPar.srs{1}.Toffset = CFG{idxSet, 19};
        
        SysPar.SimCtrl.timeDomainSim = 1;
        SysPar.Chan{1}.delay = 0.5e-6;
        SysPar.Chan{1}.SNR = CFG{idxSet, 21} + SNR_offset;
        SysPar.Chan{1}.type = chanType;
        
        SysPar.SimCtrl.checkSrsHestErr = 1;
        
        if batchsimCfg.batchsimMode==0
            [SysPar, UE, gNB] = nrSimulator(SysPar);
        else
            for idx_seed = 1:length(batchsimCfg.seed_list)
                my_seed = batchsimCfg.seed_list(idx_seed);
                SysPar.SimCtrl.seed = my_seed;
                SysPar.SimCtrl.batchsim.save_results = 1;
                subscenario_name = sprintf('scenario_TC%d___seed_%d___SNR_%2.2f',caseNum, my_seed,SysPar.Chan{1}.SNR);
                subscenario_folder_name = fullfile(batchsimCfg.ws_folder_name,subscenario_name);
                if ~exist(subscenario_folder_name, 'dir')
                   mkdir(subscenario_folder_name)
                end
                full_cfg_template_yaml_file_name = fullfile(subscenario_folder_name,'cfg_template.yaml');
                fprintf('Generating subscenario: %s\n', full_cfg_template_yaml_file_name)
                WriteYaml(full_cfg_template_yaml_file_name, SysPar);
                num_generated_subscenarios = num_generated_subscenarios + 1;
            end
            continue;
        end       
        SimCtrl = SysPar.SimCtrl;        
        
        totalCnt = SimCtrl.results.srs{1}.totalCnt;
        snrErr = sqrt(SimCtrl.results.srs{1}.snrErr/totalCnt);
        toErr = sqrt(SimCtrl.results.srs{1}.toErr/totalCnt);
        hestErr =  10*log10(SimCtrl.results.srs{1}.hestErr/totalCnt);
        
        % if snrErr > 3 || toErr > 0.3
        if toErr > 0.3
            srsDetected(n) = 0;
        else
            srsDetected(n) = 1;
        end
        
        fprintf('%4d %4d  %2d   %2d  %4d  %4d  %4d  %4d  %4d  %4d  %4d    %4d %4d  %4d   %4.1f    %5.2f     %5.1f\n',...
            CFG{idxSet, 1}, CFG{idxSet, 2}, CFG{idxSet, 3}, CFG{idxSet, 4}, ...
            CFG{idxSet, 5}, CFG{idxSet, 6}, CFG{idxSet, 7}, CFG{idxSet, 8}, ...
            CFG{idxSet, 9},  CFG{idxSet, 10},  CFG{idxSet, 11},  CFG{idxSet, 12}, ...
            CFG{idxSet, 21}, srsDetected(n), snrErr, toErr, hestErr);
    end
end

if batchsimCfg.batchsimMode==0
    fprintf('-----------------------------------------------------------------------------------------------------------\n');
    nPerf = 0;
    nFail = 0;
    for idxSet = 1:NallTest
        caseNum = CFG{idxSet, 1};
        if ismember(caseNum, TcToTest)
            nPerf = nPerf + 1;
            if srsDetected(idxSet) == 0
                nFail = nFail + 1;
            end      
        end
    end
    fprintf('Total TC = %d, PASS = %d, FAIL = %d\n\n', nPerf, nPerf-nFail, nFail);
else
    fprintf('Total number of subcenarios generated per SNR: %d\n',num_generated_subscenarios);
end        
toc; 
fprintf('\n');