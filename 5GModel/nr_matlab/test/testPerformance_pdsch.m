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

function [nPerf, nFail, CBER, TBER, CFOERR, TOERR, SNR] = testPerformance_pdsch(caseSet, SNRoffset, Nframe, pdschTestMode, batchsimCfg, relNum)

if nargin == 0
    caseSet = 'full';
    SNRoffset = 0;
    Nframe = 1;
    pdschTestMode = 0;
    batchsimCfg.batchsimMode = 0; % 0: disable batchsimMode, 1: phase2 perf study, 2: performance match test for 5GModel and cuPHY
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 1
    SNRoffset = 0;
    Nframe = 1;
    pdschTestMode = 0;
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 2
    Nframe = 1;
    pdschTestMode = 0;
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 3
    pdschTestMode = 0;
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 4
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 5
    relNum = 10000;
end

selected_TC = [3001:3800];
compact_TC = [3001:3800];
full_TC = [3001:3800];

if isnumeric(caseSet)
    TcToTest = caseSet;
else
    switch pdschTestMode
        case 0
            TcToTest = [3001]; % PDSCH data        
        case 7
            TcToTest = [3701:3764]; % mMIMO SRS/BF/PDSCH
        otherwise
            error('pdschTestMode is not supported...\n');
    end
end
if isfield(batchsimCfg, 'caseSet')
    if ~isempty(batchsimCfg.caseSet)
        TcToTest_superset_from_batchsimCfg = batchsimCfg.caseSet;
    else
        TcToTest_superset_from_batchsimCfg = 1:1e5;
    end
else
    TcToTest_superset_from_batchsimCfg = 1:1e5;
end

if isfield(batchsimCfg, 'test_version')
    test_version = batchsimCfg.test_version; % '38.141-1.v15.14'
else
    test_version = '38.104.v16.4';
end

if 1 % strcmp(test_version, '38.104.v16.4')

    CFG = {...
    % TS38-104 V16.4 Table 8.2.1.2-7 100MHz, 30kHz SCS       
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
     3001,  'G-FR1-A3-14', 'TDLB100-400-Low',   2,  -2.8, 200,   0e-6;

    % Self defined test cases
    % TC for mMIMO SRS/BFW/PDSCH 
    % SU-MIMO/nl = 1 x 2
    % nSlotSrsUpdate = 20
      3701,  'G-FR1-A4-28',            'AWGN',   4,  -7.2,   0,   0e-6; % known AWGN channel
      3702,  'G-FR1-A4-28',            'AWGN',   4,  -4.6,   0,   0e-6; % estimated AWGN channel
      3703,  'G-FR1-A4-28',       'CDLA30-10',   4,  -7.4,   0,   0e-6; % known CDLA channel
      3704,  'G-FR1-A4-28',       'CDLA30-10',   4,  -5.0,   0,   0e-6; % estimated CDLA channel
      3705,  'G-FR1-A4-28',     'CDLB100-400',   4,   8.2,   0,   0e-6; % known CDLB channel
      3706,  'G-FR1-A4-28',     'CDLB100-400',   4,   8.6,   0,   0e-6; % estimated CDLB channel
      3707,  'G-FR1-A4-28',     'CDLC300-100',   4,  -3.0,   0,   0e-6; % known CDLC channel
      3708,  'G-FR1-A4-28',     'CDLC300-100',   4,  -2.0,   0,   0e-6; % estimated CDLC channel

    % nSlotSrsUpdate = 1
      3709,  'G-FR1-A4-28',            'AWGN',   4,  -7.2,   0,   0e-6; % known AWGN channel
      3710,  'G-FR1-A4-28',            'AWGN',   4,  -4.4,   0,   0e-6; % estimated AWGN channel
      3711,  'G-FR1-A4-28',       'CDLA30-10',   4,  -7.4,   0,   0e-6; % known CDLA channel
      3712,  'G-FR1-A4-28',       'CDLA30-10',   4,  -5.0,   0,   0e-6; % estimated CDLA channel
      3713,  'G-FR1-A4-28',     'CDLB100-400',   4,  -7.0,   0,   0e-6; % known CDLB channel
      3714,  'G-FR1-A4-28',     'CDLB100-400',   4,  -5.0,   0,   0e-6; % estimated CDLB channel
      3715,  'G-FR1-A4-28',     'CDLC300-100',   4,  -7.2,   0,   0e-6; % known CDLC channel
      3716,  'G-FR1-A4-28',     'CDLC300-100',   4,  -5.2,   0,   0e-6; % estimated CDLC channel
    % MU-MIMO/nl = 2 x 1
    % nSlotSrsUpdate = 20
      3717,  'G-FR1-A4-14',            'AWGN',   4,  -6.6,   0,   0e-6; % known AWGN channel
      3718,  'G-FR1-A4-14',            'AWGN',   4,  -4.4,   0,   0e-6; % estimated AWGN channel
      3719,  'G-FR1-A4-14',       'CDLA30-10',   4,  -8.4,   0,   0e-6; % known CDLA channel
      3720,  'G-FR1-A4-14',       'CDLA30-10',   4,  -4.4,   0,   0e-6; % estimated CDLA channel
      3721,  'G-FR1-A4-14',     'CDLB100-400',   4,  40.0,   0,   0e-6; % known CDLB channel
      3722,  'G-FR1-A4-14',     'CDLB100-400',   4,  40.0,   0,   0e-6; % estimated CDLB channel
      3723,  'G-FR1-A4-14',     'CDLC300-100',   4,  -3.0,   0,   0e-6; % known CDLC channel
      3724,  'G-FR1-A4-14',     'CDLC300-100',   4,  -3.6,   0,   0e-6; % estimated CDLC channel
    % nSlotSrsUpdate = 1
      3725,  'G-FR1-A4-14',            'AWGN',   4,  -6.6,   0,   0e-6; % known AWGN channel
      3726,  'G-FR1-A4-14',            'AWGN',   4,  -4.4,   0,   0e-6; % estimated AWGN channel
      3727,  'G-FR1-A4-14',       'CDLA30-10',   4,  -8.4,   0,   0e-6; % known CDLA channel
      3728,  'G-FR1-A4-14',       'CDLA30-10',   4,  -4.4,   0,   0e-6; % estimated CDLA channel
      3729,  'G-FR1-A4-14',     'CDLB100-400',   4,  -7.6,   0,   0e-6; % known CDLB channel
      3730,  'G-FR1-A4-14',     'CDLB100-400',   4,  -4.8,   0,   0e-6; % estimated CDLB channel
      3731,  'G-FR1-A4-14',     'CDLC300-100',   4,  -7.8,   0,   0e-6; % known CDLC channel
      3732,  'G-FR1-A4-14',     'CDLC300-100',   4,  -5.6,   0,   0e-6; % estimated CDLC channel
    % MU-MIMO/nl = 8 x 1
    % nSlotSrsUpdate = 20
      3733,  'G-FR1-B4-14',            'AWGN',   4,  -8.6,   0,   0e-6; % known AWGN channel
      3734,  'G-FR1-B4-14',            'AWGN',   4,  -3.6,   0,   0e-6; % estimated AWGN channel
      3735,  'G-FR1-B4-14',       'CDLA30-10',   4,  -8.0,   0,   0e-6; % known CDLA channel
      3736,  'G-FR1-B4-14',       'CDLA30-10',   4,  -2.6,   0,   0e-6; % estimated CDLA channel
      3737,  'G-FR1-B4-14',     'CDLB100-400',   4,    40,   0,   0e-6; % known CDLB channel
      3738,  'G-FR1-B4-14',     'CDLB100-400',   4,    40,   0,   0e-6; % estimated CDLB channel
      3739,  'G-FR1-B4-14',     'CDLC300-100',   4,    40,   0,   0e-6; % known CDLC channel
      3740,  'G-FR1-B4-14',     'CDLC300-100',   4,    40,   0,   0e-6; % estimated CDLC channel
    % nSlotSrsUpdate = 1
      3741,  'G-FR1-B4-14',            'AWGN',   4,  -8.6,   0,   0e-6; % known AWGN channel
      3742,  'G-FR1-B4-14',            'AWGN',   4,  -3.6,   0,   0e-6; % estimated AWGN channel
      3743,  'G-FR1-B4-14',       'CDLA30-10',   4,  -8.6,   0,   0e-6; % known CDLA channel
      3744,  'G-FR1-B4-14',       'CDLA30-10',   4,  -2.8,   0,   0e-6; % estimated CDLA channel
      3745,  'G-FR1-B4-14',     'CDLB100-400',   4,    40,   0,   0e-6; % known CDLB channel
      3746,  'G-FR1-B4-14',     'CDLB100-400',   4,    40,   0,   0e-6; % estimated CDLB channel
      3747,  'G-FR1-B4-14',     'CDLC300-100',   4,  -6.8,   0,   0e-6; % known CDLC channel
      3748,  'G-FR1-B4-14',     'CDLC300-100',   4,  -2.6,   0,   0e-6; % estimated CDLC channel
    % MU-MIMO/nl = 16 x 1
    % nSlotSrsUpdate = 20
      3749,  'G-FR1-B4-14',            'AWGN',   4,  -8.6,   0,   0e-6; % known AWGN channel
      3750,  'G-FR1-B4-14',            'AWGN',   4,  -1.8,   0,   0e-6; % estimated AWGN channel
      3751,  'G-FR1-B4-14',       'CDLA30-10',   4,  -6.0,   0,   0e-6; % known CDLA channel
      3752,  'G-FR1-B4-14',       'CDLA30-10',   4,   1.8,   0,   0e-6; % estimated CDLA channel
      3753,  'G-FR1-B4-14',     'CDLB100-400',   4,    40,   0,   0e-6; % known CDLB channel
      3754,  'G-FR1-B4-14',     'CDLB100-400',   4,    40,   0,   0e-6; % estimated CDLB channel
      3755,  'G-FR1-B4-14',     'CDLC300-100',   4,    40,   0,   0e-6; % known CDLC channel
      3756,  'G-FR1-B4-14',     'CDLC300-100',   4,    40,   0,   0e-6; % estimated CDLC channel
    % nSlotSrsUpdate = 1
      3757,  'G-FR1-B4-14',            'AWGN',   4,  -8.6,   0,   0e-6; % known AWGN channel
      3758,  'G-FR1-B4-14',            'AWGN',   4,  -2.0,   0,   0e-6; % estimated AWGN channel
      3759,  'G-FR1-B4-14',       'CDLA30-10',   4,  -8.4,   0,   0e-6; % known CDLA channel
      3760,  'G-FR1-B4-14',       'CDLA30-10',   4,  -0.2,   0,   0e-6; % estimated CDLA channel
      3761,  'G-FR1-B4-14',     'CDLB100-400',   4,    40,   0,   0e-6; % known CDLB channel
      3762,  'G-FR1-B4-14',     'CDLB100-400',   4,    40,   0,   0e-6; % estimated CDLB channel
      3763,  'G-FR1-B4-14',     'CDLC300-100',   4,    40,   0,   0e-6; % known CDLC channel
      3764,  'G-FR1-B4-14',     'CDLC300-100',   4,    40,   0,   0e-6; % estimated CDLC channel      
    };

end

SRS_CFG = {
   % TC#  rnti  Nap nSym  Nrep sym0 cfgIdx seqId bwIdx cmbSz cmbOffset cs  fPos fShift frqH grpH resType Tsrs  Toffset idxSlot
    3701   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3702   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3703   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3704   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3705   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3706   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3707   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3708   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3709   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3710   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3711   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3712   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3713   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3714   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3715   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3716   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    
    3717   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3718   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3719   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3720   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3721   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3722   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3723   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3724   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3725   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3726   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3727   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3728   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3729   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3730   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3731   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3732   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3733   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3734   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3735   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3736   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3737   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3738   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3739   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3740   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3741   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3742   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3743   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3744   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3745   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3746   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3747   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3748   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3749   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3750   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3751   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3752   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3753   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3754   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3755   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3756   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3757   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3758   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3759   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3760   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3761   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3762   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3763   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    3764   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    };


% update CFG if necessary
enable_update_FRC_PRB_number = 0;
new_prb_number = 273;
if enable_update_FRC_PRB_number    
    for idx_case_number = 7950:7983  
        for idx_row = 1:size(CFG,1)
            if CFG{idx_row}==idx_case_number
                tmp = CFG{idx_row, 2};
                CFG{idx_row, 2} = strrep(tmp,'024',num2str(new_prb_number));
            end
        end
    end   
end

% export CFG into csv file
if isfield(batchsimCfg, 'export_cfg')
    if batchsimCfg.export_cfg
        if isfield(batchsimCfg,'export_fileName')
            cfg_table = cell2table(CFG);
            cfg_table.Properties.VariableNames = {'TC', 'FRC', 'Chan', 'rxAnt', 'SNR', 'CFO', 'delay'};
            %filter table and just keep valid TCs
            idx_row = ismember(cfg_table.TC,caseSet);
            writetable(cfg_table(idx_row,:),batchsimCfg.export_fileName);
        else
            error('Please specify the exporting path!')
        end
        nPerf = nan;
        nFail = nan;
        CBER  = nan; 
        TBER  = nan;
        CFOERR= nan; 
        TOERR = nan;
        SNR   = nan;
        return;
    end
end

[NallTest, ~] = size(CFG);
N_SNR = length(SNRoffset);
SNR = zeros(NallTest, N_SNR);
CBER = zeros(NallTest, N_SNR);
TBER = zeros(NallTest, N_SNR);
CFOERR = zeros(NallTest, N_SNR);
TOERR = zeros(NallTest, N_SNR);

switch pdschTestMode
    case 0
        fprintf('Test PDSCH data detection performance:\n');
    case 7
        fprintf('Test mMIMO SRS/BFW/PDSCH performance:\n');
    case 99
        fprintf('Save test configuration:\n');
    otherwise
        fprintf('Test PUSCH other performance:\n');
end

fprintf('\nTC#    Nframe      FRC            Chan       rxAnt  SNR  SNRoffset  CFO   delay   nCB  errCB   CBer   nTB  errTB   TBer\n');
fprintf('-----------------------------------------------------------------------------------------------------------------------\n');

% initialize a txt file to save info used for cuPHY test bench
output_infor_for_cuphy = cell(1,1); 

load('CDLparam.mat', 'CDLparam');

num_generated_subscenarios = 0;
parfor idxSet = 1:NallTest
    testAlloc = [];
    testAlloc.dl = 1;
    testAlloc.ul = 0;
    testAlloc.pdsch = 1;
    testAlloc_ssb = [];
    testAlloc_sib1 = [];
    testAlloc_pdcch = [];
    caseNum = CFG{idxSet, 1};
    if ismember(caseNum, TcToTest) && ismember(caseNum, TcToTest_superset_from_batchsimCfg)
        snr = [];
        cber = [];
        tber = [];

        for idxSNR = 1:N_SNR
            if ismember(caseNum, [3701:3799])
                rng(3701,'threefry');
                testAlloc.ul = 1;
            else
                rng(caseNum,'threefry'); % clearly indicate the rng generator type although it is 'threefry' by default in parallel mode. 
            end
            SysPar = initSysPar(testAlloc);
            SysPar.SimCtrl.relNum = relNum;
            SysPar.SimCtrl.N_frame = Nframe;
            SysPar.SimCtrl.N_slot_run = 0;
            SysPar.SimCtrl.puschHARQ.EnableAutoHARQ = 1; % automatically set newDataIndication and rvIndex
            SysPar.SimCtrl.puschHARQ.MaxTransmissions = 4; % PUSCH number of transmissions per TB for HARQ performance testing
            SysPar.SimCtrl.timeDomainSim = 1;
            SysPar.SimCtrl.normalize_pusch_tx_power_over_layers = 1;
            SysPar.SimCtrl.alg.enableCfoEstimation = 1;
            SysPar.SimCtrl.alg.enableCfoCorrection = 1;
            SysPar.SimCtrl.alg.enableToEstimation = 1;
            SysPar.SimCtrl.alg.enableToCorrection = 0;
            SysPar.SimCtrl.alg.TdiMode = 1;
            SysPar.SimCtrl.alg.enableIrc = 1;
            SysPar.SimCtrl.alg.enableNoiseEstForZf = 0;
            SysPar.SimCtrl.alg.listLength = 8;
            SysPar.pdsch{1} = loadFRC(SysPar.pdsch{1}, CFG{idxSet, 2});
            SysPar.Chan{1}.type =  CFG{idxSet, 3};
            % SysPar.carrier.Nant_gNB = CFG{idxSet, 4};
            % SysPar.carrier.Nant_UE = SysPar.pusch{1}.nrOfLayers;
            SysPar.carrier.Nant_gNB = SysPar.pdsch{1}.nrOfLayers; 
            SysPar.carrier.Nant_UE = CFG{idxSet, 4};            
            snr_offset = SNRoffset(idxSNR);
            if snr_offset == 1000 % SNR = 1000dB is used for generating a noiseless TV for cuPHY SNR sweeping usage
                SysPar.Chan{1}.SNR =  snr_offset;
            else
                SysPar.Chan{1}.SNR =  CFG{idxSet, 5} + snr_offset;
            end
            SysPar.Chan{1}.CFO = CFG{idxSet, 6};
            SysPar.Chan{1}.delay = CFG{idxSet, 7};
            SysPar.SimCtrl.enableUeRx = 1;
            SysPar.SimCtrl.usePuschRxForPdsch = 1;
            SysPar.SimCtrl.alg.enableWeightedAverageCfo = 0;

            SysPar.SimCtrl.fp_flag_pusch_equalizer = 0;

            if ismember(caseNum, [3701:3716])                
                SysPar.Chan{1}.model_source='custom';
                SysPar.SimCtrl.enableUlDlCoSim = 1;
                SysPar.SimCtrl.enableDlTxBf = 1;
                SysPar.SimCtrl.enableSrsState = 1;
                SysPar.SimCtrl.bfw.enable_prg_chest = 1;
                if ismember(caseNum, [3701:2:3716])
                    SysPar.SimCtrl.BfKnownChannel = 1;
                else
                    SysPar.SimCtrl.BfKnownChannel = 0;
                end
                run_CDL = 1;
                if run_CDL && ~strcmp(SysPar.Chan{1}.type, 'AWGN')
                    switch SysPar.Chan{1}.type
                        case 'CDLA30-10'
                            SysPar.Chan{1}.DelayProfile = 'CDL-A';
                            SysPar.Chan{1}.DelaySpread = 30e-9;
                            SysPar.Chan{1}.MaximumDopplerShift = 10;
                        case 'CDLB100-400'
                            SysPar.Chan{1}.DelayProfile = 'CDL-B';
                            SysPar.Chan{1}.DelaySpread = 100e-9;
                            SysPar.Chan{1}.MaximumDopplerShift = 400;
                        case 'CDLC300-100'
                            SysPar.Chan{1}.DelayProfile = 'CDL-C';
                            SysPar.Chan{1}.DelaySpread = 300e-9;
                            SysPar.Chan{1}.MaximumDopplerShift = 100;
                        otherwise
                            warning('Chan.type is not supported ...\n')
                    end
                    SysPar.Chan{1}.type = 'CDL';
                    SysPar.Chan{1}.gNB_AntArraySize = [4,8,2];
                    SysPar.Chan{1}.UE_AntArraySize = [1,1,2];
                end
                if ismember(caseNum, [3701:3708])
                    SysPar.SimCtrl.nSlotSrsUpdate = 20;
                else
                    SysPar.SimCtrl.nSlotSrsUpdate = 1;
                end
                % SysPar.SimCtrl.N_slot_run = 3;
                SysPar.carrier.Nant_gNB = 64;
                SysPar.carrier.Nant_UE = 2;
                SysPar.SimCtrl.CellConfigPorts = 2;
                % SysPar.Chan{1}.SNR = 20;
                SysPar.SimCtrl.checkSrsHestErr = 1;
                testAlloc.srs = 1;
                SysPar.testAlloc = testAlloc;
                SrsCfg = cell2mat(SRS_CFG);
                SrsCfgList = SrsCfg(:, 1);
                SrsCfgIdx = find(caseNum == SrsCfgList);
                SysPar.srs{1}.RNTI = SRS_CFG{SrsCfgIdx, 2};
                SysPar.srs{1}.numAntPorts = SRS_CFG{SrsCfgIdx, 3};
                SysPar.srs{1}.numSymbols = SRS_CFG{SrsCfgIdx, 4};
                SysPar.srs{1}.numRepetitions = SRS_CFG{SrsCfgIdx, 5};
                SysPar.srs{1}.timeStartPosition = SRS_CFG{SrsCfgIdx, 6};
                SysPar.srs{1}.configIndex = SRS_CFG{SrsCfgIdx, 7};
                SysPar.srs{1}.sequenceId = SRS_CFG{SrsCfgIdx, 8};
                SysPar.srs{1}.bandwidthIndex = SRS_CFG{SrsCfgIdx, 9};
                SysPar.srs{1}.combSize = SRS_CFG{SrsCfgIdx, 10};
                SysPar.srs{1}.combOffset = SRS_CFG{SrsCfgIdx, 11};
                SysPar.srs{1}.cyclicShift = SRS_CFG{SrsCfgIdx, 12};
                SysPar.srs{1}.frequencyPosition = SRS_CFG{SrsCfgIdx, 13};
                SysPar.srs{1}.frequencyShift = SRS_CFG{SrsCfgIdx, 14};
                SysPar.srs{1}.frequencyHopping = SRS_CFG{SrsCfgIdx, 15};
                SysPar.srs{1}.groupOrSequenceHopping = SRS_CFG{SrsCfgIdx, 16};
                SysPar.srs{1}.resourceType = SRS_CFG{SrsCfgIdx, 17};
                SysPar.srs{1}.Tsrs = SRS_CFG{SrsCfgIdx, 18};
                SysPar.srs{1}.Toffset = SRS_CFG{SrsCfgIdx, 19};
                
                SysPar.pdsch{1}.RNTI = SysPar.srs{1}.RNTI;
                SysPar.pdsch{1}.NrOfSymbols = 14;
                SysPar.pdsch{1}.rbSize = 272;
                SysPar.pdsch{1}.prgSize = 2;
            elseif ismember(caseNum, [3717:3764])
                if ismember(caseNum, [3717:3732])
                    nUe = 2;
                elseif ismember(caseNum, [3733:3748])
                    nUe = 8;
                else
                    nUe = 16;
                end
                SysPar.SimCtrl.N_UE = nUe;
                SysPar.Chan{1}.model_source='custom';
                SysPar.SimCtrl.enableUlDlCoSim = 1;
                SysPar.SimCtrl.enableDlTxBf = 1;
                SysPar.SimCtrl.enableSrsState = 1;
                SysPar.SimCtrl.bfw.enable_prg_chest = 1;
                if ismember(caseNum, [3717:2:3764])
                    SysPar.SimCtrl.BfKnownChannel = 1;
                else
                    SysPar.SimCtrl.BfKnownChannel = 0;
                end
                run_CDL = 1;
                if nUe == 2
                    if run_CDL && ~strcmp(SysPar.Chan{1}.type, 'AWGN')
                        switch SysPar.Chan{1}.type
                            case 'CDLA30-10'
                                SysPar.Chan{1}.DelayProfile = 'CDL-A';
                                SysPar.Chan{1}.DelaySpread = 30e-9;
                                SysPar.Chan{1}.MaximumDopplerShift = 10;
                            case 'CDLB100-400'
                                SysPar.Chan{1}.DelayProfile = 'CDL-B';
                                SysPar.Chan{1}.DelaySpread = 100e-9;
                                SysPar.Chan{1}.MaximumDopplerShift = 400;
                            case 'CDLC300-100'
                                SysPar.Chan{1}.DelayProfile = 'CDL-C';
                                SysPar.Chan{1}.DelaySpread = 300e-9;
                                SysPar.Chan{1}.MaximumDopplerShift = 100;
                            otherwise
                                warning('Chan.type is not supported ...\n')
                        end
                        SysPar.Chan{1}.type = 'CDL';
                        SysPar.Chan{1}.gNB_AntArraySize = [4,8,2];
                        SysPar.Chan{1}.UE_AntArraySize = [1,1,2];
                    end
                else
                    if run_CDL && ~strcmp(SysPar.Chan{1}.type, 'AWGN')
                        switch SysPar.Chan{1}.type
                            case 'CDLA30-10'
                                % load CDLparam;
                                SysPar.Chan{1}.DelayProfile = 'CDL_customized';
                                SysPar.Chan{1}.CDL_DPA = CDLparam.DPA_A;
                                SysPar.Chan{1}.CDL_PCP = CDLparam.PCP_A;
                                SysPar.Chan{1}.DelaySpread = 30e-9;
                                SysPar.Chan{1}.MaximumDopplerShift = 10;
                            case 'CDLB100-400'
                                % load CDLparam;
                                SysPar.Chan{1}.DelayProfile = 'CDL_customized';
                                SysPar.Chan{1}.CDL_DPA = CDLparam.DPA_B;
                                SysPar.Chan{1}.CDL_PCP = CDLparam.PCP_B;
                                SysPar.Chan{1}.DelaySpread = 100e-9;
                                SysPar.Chan{1}.MaximumDopplerShift = 400;
                            case 'CDLC300-100'
                                % load CDLparam;
                                SysPar.Chan{1}.DelayProfile = 'CDL_customized';
                                SysPar.Chan{1}.CDL_DPA = CDLparam.DPA_C;
                                SysPar.Chan{1}.CDL_PCP = CDLparam.PCP_C;
                                SysPar.Chan{1}.DelaySpread = 300e-9;
                                SysPar.Chan{1}.MaximumDopplerShift = 100;
                            otherwise
                                warning('Chan.type is not supported ...\n')
                        end
                        SysPar.Chan{1}.type = 'CDL';
                        SysPar.Chan{1}.gNB_AntArraySize = [4,8,2];
                        SysPar.Chan{1}.UE_AntArraySize = [1,1,2];
                    end
                end

                for idxUe = 1:nUe
                    SysPar.Chan{idxUe} =SysPar.Chan{1};
                    if strcmp(SysPar.Chan{1}.DelayProfile, 'CDL_customized')
                        angle_ue = (idxUe-0.5)*120/nUe - 60;
                        SysPar.Chan{idxUe}.CDL_DPA(:, 4) = angle_ue;
                    end
                end
                if ismember(caseNum, [3717:3724, 3733:3740, 3749:3756])
                    SysPar.SimCtrl.nSlotSrsUpdate = 20;
                else
                    SysPar.SimCtrl.nSlotSrsUpdate = 1;
                end
                % SysPar.SimCtrl.N_slot_run = 3;
                SysPar.carrier.Nant_gNB = 64;
                SysPar.carrier.Nant_UE = 1;
                SysPar.SimCtrl.CellConfigPorts = nUe;
                % SysPar.Chan{1}.SNR = 20;
                SysPar.SimCtrl.checkSrsHestErr = 1;
                testAlloc.srs = nUe;
                testAlloc.pdsch = nUe;
                SysPar.testAlloc = testAlloc;
                SrsCfg = cell2mat(SRS_CFG);
                SrsCfgList = SrsCfg(:, 1);
                SrsCfgIdx = find(caseNum == SrsCfgList);
                for idxUe = 1:nUe
                    SysPar.srs{idxUe} = SysPar.srs{1};
                    SysPar.srs{idxUe}.RNTI = SRS_CFG{SrsCfgIdx, 2} + idxUe-1;
                    SysPar.srs{idxUe}.numAntPorts = 1; % SRS_CFG{SrsCfgIdx, 3};
                    SysPar.srs{idxUe}.numSymbols = SRS_CFG{SrsCfgIdx, 4};
                    SysPar.srs{idxUe}.numRepetitions = SRS_CFG{SrsCfgIdx, 5};
                    if nUe > 8
                        if idxUe < 5
                            SysPar.srs{idxUe}.timeStartPosition = SRS_CFG{SrsCfgIdx, 6} - 3;
                        elseif idxUe < 9                        
                            SysPar.srs{idxUe}.timeStartPosition = SRS_CFG{SrsCfgIdx, 6} - 2;
                        elseif idxUe < 13
                            SysPar.srs{idxUe}.timeStartPosition = SRS_CFG{SrsCfgIdx, 6} - 1;
                        else
                            SysPar.srs{idxUe}.timeStartPosition = SRS_CFG{SrsCfgIdx, 6};
                        end
                    elseif nUe > 4
                        if idxUe < 5
                            SysPar.srs{idxUe}.timeStartPosition = SRS_CFG{SrsCfgIdx, 6} - 1;
                        else
                            SysPar.srs{idxUe}.timeStartPosition = SRS_CFG{SrsCfgIdx, 6};
                        end
                    else
                        SysPar.srs{idxUe}.timeStartPosition = SRS_CFG{SrsCfgIdx, 6};
                    end
                    SysPar.srs{idxUe}.configIndex = SRS_CFG{SrsCfgIdx, 7};
                    SysPar.srs{idxUe}.sequenceId = SRS_CFG{SrsCfgIdx, 8};
                    SysPar.srs{idxUe}.bandwidthIndex = SRS_CFG{SrsCfgIdx, 9};
                    SysPar.srs{idxUe}.combSize = SRS_CFG{SrsCfgIdx, 10};
                    SysPar.srs{idxUe}.combOffset = mod(SRS_CFG{SrsCfgIdx, 11} + idxUe-1, 4);
                    SysPar.srs{idxUe}.cyclicShift = SRS_CFG{SrsCfgIdx, 12};
                    SysPar.srs{idxUe}.frequencyPosition = SRS_CFG{SrsCfgIdx, 13};
                    SysPar.srs{idxUe}.frequencyShift = SRS_CFG{SrsCfgIdx, 14};
                    SysPar.srs{idxUe}.frequencyHopping = SRS_CFG{SrsCfgIdx, 15};
                    SysPar.srs{idxUe}.groupOrSequenceHopping = SRS_CFG{SrsCfgIdx, 16};
                    SysPar.srs{idxUe}.resourceType = SRS_CFG{SrsCfgIdx, 17};
                    SysPar.srs{idxUe}.Tsrs = SRS_CFG{SrsCfgIdx, 18};
                    SysPar.srs{idxUe}.Toffset = SRS_CFG{SrsCfgIdx, 19};
                    SysPar.srs{idxUe}.idxUE = idxUe-1;
                    SysPar.pdsch{idxUe} = SysPar.pdsch{1};
                    SysPar.pdsch{idxUe}.RNTI = SysPar.srs{idxUe}.RNTI;
                    SysPar.pdsch{idxUe}.portIdx = mod(idxUe - 1, 8);
                    SysPar.pdsch{idxUe}.SCID = floor((idxUe-1)/8);
                    % if nUe > 4
                    %     SysPar.pdsch{idxUe}.NrOfSymbols = 12;
                    % else
                    %     SysPar.pdsch{idxUe}.NrOfSymbols = 13;
                    % end
                    SysPar.pdsch{idxUe}.NrOfSymbols = 14;
                    SysPar.pdsch{idxUe}.rbSize = 272;
                    SysPar.pdsch{idxUe}.prgSize = 2;
                    SysPar.pdsch{idxUe}.idxUE = idxUe-1;
                end
            end

            if isfield(batchsimCfg, 'enable_UL_Rx_RF_impairments') 
                if batchsimCfg.enable_UL_Rx_RF_impairments
                    SysPar.SimCtrl.enable_UL_Rx_RF_impairments = 1;
                    SysPar.RF.UL_Rx_tot_NF_dB = 0;
                    SysPar.RF.UL_Rx_gain_dB = 33.5;
                    SysPar.RF.UL_Rx_IIP3_dBm = -3.5
                    SysPar.RF.UL_Rx_IQ_imblance_gain_dB = 0.005;
                    SysPar.RF.UL_Rx_IQ_imblance_phase_degree = 0.063;
                    SysPar.RF.UL_Rx_PN_level_offset_dB = 0;
                    SysPar.RF.UL_Rx_PN_spectral_mask_freqOffset_Hz = [1e5, 1e6, 1e7];
                    SysPar.RF.UL_Rx_PN_spectral_mask_power_dBcPerHz = [-104.0, -125.0, -149.0];
                    SysPar.RF.UL_Rx_DC_offset_real_volt = 2e-7;
                    SysPar.carrier.N_grid_size_mu = 273; % to make sure the sampling rate is twice larger than max freq. offset of phase noise
                end
            end

            if pdschTestMode == 99
                full_cfg_template_yaml_file_name = ['cfg-', num2str(caseNum), '.yaml'];
                WriteYaml(full_cfg_template_yaml_file_name, SysPar);
                continue;
            elseif batchsimCfg.batchsimMode==0
                [SysPar, UE, gNB] = nrSimulator(SysPar);
            else
                for idx_seed = 1:length(batchsimCfg.seed_list)
                    my_seed = batchsimCfg.seed_list(idx_seed);
                    SysPar.SimCtrl.seed = my_seed;
                    SysPar.SimCtrl.batchsim.save_results = 1;
                    SysPar.SimCtrl.batchsim.save_results_short = 1;
                    if ~ismember(caseNum, [7911:7918]) % for PUSCH measurement, we don't need to set MIMO equalizer related params. Just use the default config.
                        SysPar.SimCtrl.alg.enableIrc = batchsimCfg.SimCtrl.alg.enableIrc;
                        SysPar.SimCtrl.alg.enableNoiseEstForZf =  batchsimCfg.SimCtrl.alg.enableNoiseEstForZf;
                        SysPar.SimCtrl.alg.enable_use_genie_nCov =  batchsimCfg.SimCtrl.alg.enable_use_genie_nCov;
                        SysPar.SimCtrl.alg.genie_nCov_method =  batchsimCfg.SimCtrl.alg.genie_nCov_method;
                        SysPar.SimCtrl.alg.enable_nCov_shrinkage =  batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage;
                        SysPar.SimCtrl.alg.nCov_shrinkage_method =  batchsimCfg.SimCtrl.alg.nCov_shrinkage_method;
                        SysPar.SimCtrl.alg.enable_get_genie_meas =  batchsimCfg.SimCtrl.alg.enable_get_genie_meas;        
                        SysPar.SimCtrl.enable_get_genie_channel_matrix =  batchsimCfg.SimCtrl.enable_get_genie_channel_matrix;
                        if batchsimCfg.enable_pusch_use_perfect_channel_est_for_equalizer
                            SysPar.SimCtrl.enable_get_genie_channel_matrix = batchsimCfg.SimCtrl.enable_get_genie_channel_matrix;
                            SysPar.SimCtrl.alg.enable_use_genie_channel_for_equalizer = batchsimCfg.SimCtrl.alg.enable_use_genie_channel_for_equalizer;
                            SysPar.SimCtrl.alg.TdiMode = batchsimCfg.SimCtrl.alg.TdiMode;
                            SysPar.SimCtrl.alg.enableCfoCorrection = batchsimCfg.SimCtrl.alg.enableCfoCorrection;
                        end
                    end 
                    % tdi mode
%                     SysPar.SimCtrl.alg.TdiMode = 2;
                    % config sub-slot processing
                    enable_sub_slot_proc = 0;%1;
                    if enable_sub_slot_proc == 1
                        SysPar.SimCtrl.subslot_proc_option = 2;
                        SysPar.SimCtrl.alg.enable_avg_nCov_prbs_fd = 1;
                        SysPar.SimCtrl.alg.win_size_avg_nCov_prbs_fd = 3;
                        SysPar.SimCtrl.alg.threshold_avg_nCov_prbs_fd_dB = 3;
                        SysPar.SimCtrl.alg.enable_instant_equ_coef_cfo_corr = 1;
                    end
%                     SysPar.SimCtrl.useCuphySoftDemapper = 1;%2;
%                     SysPar.SimCtrl.BFPforCuphy = 9;
                    subscenario_name = sprintf('scenario_TC%d___seed_%d___SNR_%2.2f',caseNum,my_seed,SysPar.Chan{1}.SNR);
                    subscenario_folder_name = fullfile(batchsimCfg.ws_folder_name,subscenario_name);
                    if ~exist(subscenario_folder_name, 'dir')
                       mkdir(subscenario_folder_name)
                    end
                    if batchsimCfg.batchsimMode == 2
                        % freeze Tx and Chan
                        SysPar.SimCtrl.enable_freeze_tx_and_channel = 1;
                    elseif batchsimCfg.batchsimMode == 3
                        % freeze Tx signal (except DMRS symbols which depends on slot idx)
                        SysPar.SimCtrl.enable_freeze_tx = 1;
                    end
                    if batchsimCfg.batchsimMode == 2 || (batchsimCfg.batchsimMode == 3)
                        % enable logging tx Xtf into TV
                        SysPar.SimCtrl.enable_snapshot_gNB_UE_into_SimCtrl = 1;
                        SysPar.SimCtrl.genTV.enable_logging_tx_Xtf = 1;
                        SysPar.SimCtrl.genTV.enable_logging_carrier_and_channel_info = 1;
                        % disable HARQ
                        if batchsimCfg.cfgMode ~= 2 % in conformance TV generation, we need harq for PUSCH data
                            SysPar.SimCtrl.puschHARQ.EnableAutoHARQ = 0;
                            SysPar.SimCtrl.puschHARQ.MaxTransmissions = 1;
                        end
                    end
                    % set the fake SNR for ZF to be a very low value as cuPHY has perf issue when it is high.
%                     SysPar.SimCtrl.alg.fakeSNRdBForZf = -36;
                    %
                    if (SysPar.Chan{1}.SNR == 1000) || (batchsimCfg.cfgMode==2) % cfgMode 2 is used for conformance TV generation
                        SysPar.SimCtrl.N_frame = 1;
                        SysPar.SimCtrl.N_slot_run = 0;
                        SysPar.SimCtrl.genTV.enable = 1;
                        SysPar.SimCtrl.genTV.tvDirName = subscenario_folder_name;
                        SysPar.SimCtrl.genTV.FAPI = 1;
                        if batchsimCfg.cfgMode==2
                            SysPar.SimCtrl.genTV.TVname = 'e2e_cfm_pusch';
                            SysPar.SimCtrl.genTV.slotIdx = [4, 5, 14, 15];
                            SysPar.SimCtrl.genTV.launchPattern = 1;
                            SysPar.SimCtrl.genTV.FAPIyaml = 1;
                            num_pusch_pdu = length(SysPar.pusch);
                            for idx_pdu = 1:num_pusch_pdu
                                SysPar.pusch{idx_pdu}.DmrsScramblingId = 0; % required by conformance tests.
                            end
                        else
                            SysPar.SimCtrl.genTV.TVname = 'TV';
                            SysPar.SimCtrl.genTV.slotIdx = 0;
                        end
                    end  
                    if snr_offset == 1000
                        SNR_range_for_cuPHY = CFG{idxSet, 5}+SNRoffset(SNRoffset<1000);
                        str_SNR_range_for_cuPHY = sprintf('%2.2f,', SNR_range_for_cuPHY);
                        str_SNR_range_for_cuPHY = str_SNR_range_for_cuPHY(1:end-1); % remove the last comma
                        output_infor_for_cuphy = [output_infor_for_cuphy;{sprintf('TV_subfolder_name=%s, SNR_range=[%s], N_slots=%d;\n',subscenario_name, str_SNR_range_for_cuPHY, Nframe*10*2^(SysPar.carrier.mu))}];
                    end
                    full_cfg_template_yaml_file_name = fullfile(subscenario_folder_name,'cfg_template.yaml');
                    fprintf('Generating subscenario: %s\n', full_cfg_template_yaml_file_name)
                    WriteYaml(full_cfg_template_yaml_file_name, SysPar);
                    num_generated_subscenarios = num_generated_subscenarios + 1;

                    % for E2E conformance tests TVs, we also need to add SSB and SIB1 slots so that gNB and the Keysight devices can sycn up together
                    if batchsimCfg.cfgMode==2
                        % SSB
                        testAlloc_ssb.dl = 1;
                        testAlloc_ssb.ul = 0;
                        testAlloc_ssb.ssb = 1;
                        testAlloc_ssb.pdcch = 0;
                        testAlloc_ssb.pdsch = 0;
                        testAlloc_ssb.csirs = 0;
                        testAlloc_ssb.prach = 0;
                        testAlloc_ssb.pucch = 0;
                        testAlloc_ssb.pusch = 0;
                        testAlloc_ssb.srs = 0;
                        SysPar_ssb = initSysPar(testAlloc_ssb); 
                        SysPar_ssb_yaml = ReadYaml('./test/e2e_cfm_ssb.yaml', 0, 0);
                        SysPar_ssb = updateStruct(SysPar_ssb, SysPar_ssb_yaml);
                        SysPar_ssb.SimCtrl.genTV.tvDirName = subscenario_folder_name;
                        SysPar_ssb.SimCtrl.genTV.TVname = 'e2e_cfm_ssb';
                        SysPar_ssb.SimCtrl.genTV.launchPattern = 1;
                        SysPar_ssb.SimCtrl.genTV.FAPIyaml = 1;
                        full_cfg_template_yaml_file_name_ssb = fullfile(subscenario_folder_name,'cfg_template_ssb.yaml');
                        fprintf('Generating subscenario: %s\n', full_cfg_template_yaml_file_name_ssb)
                        WriteYaml(full_cfg_template_yaml_file_name_ssb, SysPar_ssb);
                        % SIB1
                        testAlloc_sib1.dl = 1;
                        testAlloc_sib1.ul = 0;
                        testAlloc_sib1.ssb = 0;
                        testAlloc_sib1.pdcch = 1;
                        testAlloc_sib1.pdsch = 1;
                        testAlloc_sib1.csirs = 0;
                        testAlloc_sib1.prach = 0;
                        testAlloc_sib1.pucch = 0;
                        testAlloc_sib1.pusch = 0;
                        testAlloc_sib1.srs = 0;
                        SysPar_sib1 = initSysPar(testAlloc_sib1); 
                        SysPar_sib1_yaml = ReadYaml('./test/e2e_cfm_sib1.yaml', 0, 0);
                        SysPar_sib1 = updateStruct(SysPar_sib1, SysPar_sib1_yaml);
                        SysPar_sib1.SimCtrl.genTV.tvDirName = subscenario_folder_name;
                        SysPar_sib1.SimCtrl.genTV.TVname = 'e2e_cfm_sib1';
                        SysPar_sib1.SimCtrl.genTV.launchPattern = 1;
                        SysPar_sib1.SimCtrl.genTV.FAPIyaml = 1;
                        full_cfg_template_yaml_file_name_sib1 = fullfile(subscenario_folder_name,'cfg_template_sib1.yaml');
                        fprintf('Generating subscenario: %s\n', full_cfg_template_yaml_file_name_sib1)
                        WriteYaml(full_cfg_template_yaml_file_name_sib1, SysPar_sib1);
                        % PDCCH for PUSCH
                        testAlloc_pdcch.dl = 1;
                        testAlloc_pdcch.ul = 0;
                        testAlloc_pdcch.ssb = 0;
                        testAlloc_pdcch.pdcch = 1;
                        testAlloc_pdcch.pdsch = 0;
                        testAlloc_pdcch.csirs = 0;
                        testAlloc_pdcch.prach = 0;
                        testAlloc_pdcch.pucch = 0;
                        testAlloc_pdcch.pusch = 0;
                        testAlloc_pdcch.srs = 0;
                        SysPar_pdcch = initSysPar(testAlloc_pdcch); 
                        SysPar_pdcch_yaml = ReadYaml('./test/e2e_cfm_ul_pdcch.yaml', 0, 0);
                        SysPar_pdcch = updateStruct(SysPar_pdcch, SysPar_pdcch_yaml);
                        SysPar_pdcch.SimCtrl.genTV.tvDirName = subscenario_folder_name;
                        SysPar_pdcch.SimCtrl.genTV.TVname = 'e2e_cfm_pdcch';
                        SysPar_pdcch.SimCtrl.genTV.launchPattern = 1;
                        SysPar_pdcch.SimCtrl.genTV.FAPIyaml = 1;
                        %update MCS
                        MCS_this_test = SysPar.pusch{1}.mcsIndex;
                        MCS_bin = de2bi(MCS_this_test,5,'left-msb');
                        DCI_payload_bits = SysPar_pdcch.pdcch{1}.DCI{1}.Payload;
                        DCI_payload_bits(23:27) = MCS_bin;
                        SysPar_pdcch.pdcch{1}.DCI{1}.Payload = DCI_payload_bits;
                        full_cfg_template_yaml_file_name_pdcch = fullfile(subscenario_folder_name,'cfg_template_pdcch.yaml');
                        fprintf('Generating subscenario: %s\n', full_cfg_template_yaml_file_name_pdcch)
                        WriteYaml(full_cfg_template_yaml_file_name_pdcch, SysPar_pdcch);
                      
                    end
                end
                continue;
            end
            SimCtrl = SysPar.SimCtrl;
        
            snr(idxSNR) = SysPar.Chan{1}.SNR; 
            results = SimCtrl.results;
            pdsch = results.pdsch;
            nPdsch = length(pdsch);
            cbCnt = 0;
            cbErrorCnt = 0;
            tbCnt = 0;
            tbErrorCnt = 0;
            for idxpdsch = 1:nPdsch
                cbCnt = cbCnt + pdsch{idxpdsch}.cbCnt;
                cbErrorCnt = cbErrorCnt + pdsch{idxpdsch}.cbErrorCnt;
                tbCnt = tbCnt + pdsch{idxpdsch}.tbCnt;
                tbErrorCnt = tbErrorCnt + pdsch{idxpdsch}.tbErrorCnt;
            end
            cber(idxSNR) = cbErrorCnt/cbCnt;
            tber(idxSNR) = tbErrorCnt/tbCnt;

            fprintf('%4d  % 4d   %12s  %15s  %4d   %4.1f     %4.1f   %4d   %4.1f  %5d  %5d  %4.3f  %4d  %4d   %4.3f\n',...
                CFG{idxSet, 1}, Nframe, CFG{idxSet, 2}, CFG{idxSet, 3}, ...
                CFG{idxSet, 4}, CFG{idxSet, 5}, snr_offset, CFG{idxSet, 6}, ...
                1e6*CFG{idxSet, 7}, cbCnt, cbErrorCnt, cber(idxSNR), ...
                tbCnt, tbErrorCnt, tber(idxSNR));
        end
        if batchsimCfg.batchsimMode>0 || pdschTestMode == 99
            continue;
        end
        SNR(idxSet,:) = snr;
        CBER(idxSet,:) = cber;
        TBER(idxSet,:) = tber;
        % CSI1BER(idxSet,:) = csi1ber;
        % CSI2BER(idxSet,:) = csi2ber;
        % CFOERR(idxSet,:) = cfoErrHz;
        % TOERR(idxSet, :) = toErrMicroSec;
    end
end

if batchsimCfg.batchsimMode==0 && pdschTestMode < 99
    fprintf('\n-----------------------------------------------------------------------------------------------------------------------\n');
    nPerf = 0;
    nFail = 0;
    for idxSet = 1:NallTest
        caseNum = CFG{idxSet, 1};
        if ismember(caseNum, TcToTest)
            nPerf = nPerf + 1;
            if ismember(caseNum, [3701:3764]) % mMIMO PDSCH test, using SRS-based beamforming
                if find(TBER(idxSet,:) > 0.4)
                    nFail = nFail + 1;
                end
            else
                if find(TBER(idxSet,:) > 0.5) % should be 0.3 per 3GPP test requirement
                    nFail = nFail + 1;
                end                
            end
        end
    end
    fprintf('Total TC = %d, PASS = %d, FAIL = %d\n\n', nPerf, nPerf-nFail, nFail);
else
    if (batchsimCfg.batchsimMode == 2) || (batchsimCfg.batchsimMode == 3)
        txt_file_name = fullfile(batchsimCfg.ws_folder_name,'info_for_cuPHY_unit_test.txt');
        fid = fopen(txt_file_name,'w+'); % create file and write to it 
        output_infor_for_cuphy = output_infor_for_cuphy(~cellfun(@isempty,output_infor_for_cuphy));
        for idx = 1:length(output_infor_for_cuphy)
            fprintf(fid, '%s', output_infor_for_cuphy{idx});
        end
        fclose(fid);
    end
    fprintf('Total number of subcenarios generated: %d\n',num_generated_subscenarios)
end
 
fprintf('\n');
return

% function plotMcsBler(SNR, TBER)
% 
% SNR1 = SNR(end-27:end,:);
% TBER1 = TBER(end-27:end, :);
% 
% [nTC, nSNR] = size(SNR1);
% 
% figure; 
% for idxTC = 1:nTC
%     semilogy(SNR1(idxTC,:), TBER1(idxTC, :), 'LineWidth',2); hold on; grid on;
% end
% 
% title('TS 38.214 Table 5.1.3.1-2, 1T1R, MCS 0-27, 24 PRBs, 2000 slots'); xlabel('SNR (dB)'); ylabel('TBER');
% 
% return

