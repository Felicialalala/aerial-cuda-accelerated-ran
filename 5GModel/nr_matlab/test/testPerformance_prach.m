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

function [nPerf, nFail, Pfd, Pmd] = testPerformance_prach(caseSet, SNRoffset, Nframe, falseDetectionTest, batchsimCfg, relNum)

tic;
if nargin == 0
    caseSet = 'full';
    SNRoffset = 0;
    Nframe = 1;
    falseDetectionTest = 0;
    batchsimCfg.batchsimMode = 0; % 0: disable batchsimMode, 1: phase2 perf study, 2: performance match test for 5GModel and cuPHY
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 1
    SNRoffset = 0;
    Nframe = 1;
    falseDetectionTest = 0;
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 2
    Nframe = 1;    
    falseDetectionTest = 0;
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 3
    falseDetectionTest = 0;
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

compact_TC = [5001:5999];
full_TC = [5001:5999];
selected_TC = [5001:5999];

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

if strcmp(test_version, '38.104.v16.4')    
    CFG = {...
    % Test cases based on 38.104 V16.4 Table 8.4.2.2-1/2/3
    % TC#   mu  cfg    root zone  prmb  Nant N_PRB     channelType    CFO  N_nc   SNR
     5001,  1,  217,    0,   14,    0,   2,   273,            'AWGN',   0,   1,  -16.5; % B4, mu = 1
     5002,  1,  217,    0,   14,    0,   2,   273, 'TDLC300-100-Low', 400,   1,   -9.9; % B4, mu = 1
     5003,  1,  217,    0,   14,    0,   4,   273,            'AWGN',   0,   1,  -19.0; % B4, mu = 1
     5004,  1,  217,    0,   14,    0,   4,   273, 'TDLC300-100-Low', 400,   1,  -14.5; % B4, mu = 1
     5005,  1,  217,    0,   14,    0,   8,   273,            'AWGN',   0,   1,  -21.1; % B4, mu = 1
     5006,  1,  217,    0,   14,    0,   8,   273, 'TDLC300-100-Low', 400,   1,  -17.6; % B4, mu = 1
     5007,  1,   27,   22,    1,   32,   2,   273,            'AWGN',   0,   1,  -14.5; % F0, mu = 1
     5008,  1,   27,   22,    1,   32,   2,   273, 'TDLC300-100-Low', 400,   1,   -6.6; % F0, mu = 1
     5009,  1,   27,   22,    1,   32,   4,   273,            'AWGN',   0,   1,  -16.7; % F0, mu = 1
     5010,  1,   27,   22,    1,   32,   4,   273, 'TDLC300-100-Low', 400,   1,  -11.9; % F0, mu = 1
     5011,  1,   27,   22,    1,   32,   8,   273,            'AWGN',   0,   1,  -18.9; % F0, mu = 1
     5012,  1,   27,   22,    1,   32,   8,   273, 'TDLC300-100-Low', 400,   1,  -15.8; % F0, mu = 1
     5013,  0,  217,    0,   11,    0,   2,   106,            'AWGN',   0,   1,  -16.8; % B4, mu = 0
     5014,  0,  217,    0,   11,    0,   2,   106, 'TDLC300-100-Low', 400,   1,   -8.8; % B4, mu = 0
     5015,  0,  217,    0,   11,    0,   4,   106,            'AWGN',   0,   1,  -19.0; % B4, mu = 0
     5016,  0,  217,    0,   11,    0,   4,   106, 'TDLC300-100-Low', 400,   1,  -13.8; % B4, mu = 0
     5017,  0,  217,    0,   11,    0,   8,   106,            'AWGN',   0,   1,  -21.2; % B4, mu = 0
     5018,  0,  217,    0,   11,    0,   8,   106, 'TDLC300-100-Low', 400,   1,  -17.3; % B4, mu = 0
     5019,  0,   27,   22,    1,   32,   2,   106,            'AWGN',   0,   1,  -14.5; % F0, mu = 0
     5020,  0,   27,   22,    1,   32,   2,   106, 'TDLC300-100-Low', 400,   1,   -6.6; % F0, mu = 0
     5021,  0,   27,   22,    1,   32,   4,   106,            'AWGN',   0,   1,  -16.7; % F0, mu = 0
     5022,  0,   27,   22,    1,   32,   4,   106, 'TDLC300-100-Low', 400,   1,  -11.9; % F0, mu = 0
     5023,  0,   27,   22,    1,   32,   8,   106,            'AWGN',   0,   1,  -18.9; % F0, mu = 0
     5024,  0,   27,   22,    1,   32,   8,   106, 'TDLC300-100-Low', 400,   1,  -15.8; % F0, mu = 0
    };    
elseif strcmp(test_version, '38.141-1') % SNR requirement based on 38.141-1    
    CFG = {...
    % Test cases based on TS 38.141-1 Table 8.4.1.5-1/2/3
    % TC#   mu  cfg    root zone  prmb  Nant N_PRB     channelType    CFO  N_nc   SNR
     5001,  1,  217,    0,   14,    0,   2,   273,            'AWGN',   0,   1,  -16.2; % B4, mu = 1
     5002,  1,  217,    0,   14,    0,   2,   273, 'TDLC300-100-Low', 400,   1,   -9.3; % B4, mu = 1
     5003,  1,  217,    0,   14,    0,   4,   273,            'AWGN',   0,   1,  -18.7; % B4, mu = 1
     5004,  1,  217,    0,   14,    0,   4,   273, 'TDLC300-100-Low', 400,   1,  -13.9; % B4, mu = 1
     5005,  1,  217,    0,   14,    0,   8,   273,            'AWGN',   0,   1,  -20.8; % B4, mu = 1
     5006,  1,  217,    0,   14,    0,   8,   273, 'TDLC300-100-Low', 400,   1,  -17.0; % B4, mu = 1
     5007,  1,   27,   22,    1,   32,   2,   273,            'AWGN',   0,   1,  -14.2; % F0, mu = 1
     5008,  1,   27,   22,    1,   32,   2,   273, 'TDLC300-100-Low', 400,   1,   -6.0; % F0, mu = 1
     5009,  1,   27,   22,    1,   32,   4,   273,            'AWGN',   0,   1,  -16.4; % F0, mu = 1
     5010,  1,   27,   22,    1,   32,   4,   273, 'TDLC300-100-Low', 400,   1,  -11.3; % F0, mu = 1
     5011,  1,   27,   22,    1,   32,   8,   273,            'AWGN',   0,   1,  -18.6; % F0, mu = 1
     5012,  1,   27,   22,    1,   32,   8,   273, 'TDLC300-100-Low', 400,   1,  -15.2; % F0, mu = 1
     5013,  0,  217,    0,   11,    0,   2,   106,            'AWGN',   0,   1,  -16.5; % B4, mu = 0
     5014,  0,  217,    0,   11,    0,   2,   106, 'TDLC300-100-Low', 400,   1,   -8.2; % B4, mu = 0
     5015,  0,  217,    0,   11,    0,   4,   106,            'AWGN',   0,   1,  -18.7; % B4, mu = 0
     5016,  0,  217,    0,   11,    0,   4,   106, 'TDLC300-100-Low', 400,   1,  -13.2; % B4, mu = 0
     5017,  0,  217,    0,   11,    0,   8,   106,            'AWGN',   0,   1,  -20.9; % B4, mu = 0
     5018,  0,  217,    0,   11,    0,   8,   106, 'TDLC300-100-Low', 400,   1,  -16.7; % B4, mu = 0
     5019,  0,   27,   22,    1,   32,   2,   106,            'AWGN',   0,   1,  -14.2; % F0, mu = 0
     5020,  0,   27,   22,    1,   32,   2,   106, 'TDLC300-100-Low', 400,   1,   -6.0; % F0, mu = 0
     5021,  0,   27,   22,    1,   32,   4,   106,            'AWGN',   0,   1,  -16.4; % F0, mu = 0
     5022,  0,   27,   22,    1,   32,   4,   106, 'TDLC300-100-Low', 400,   1,  -11.3; % F0, mu = 0
     5023,  0,   27,   22,    1,   32,   8,   106,            'AWGN',   0,   1,  -18.6; % F0, mu = 0
     5024,  0,   27,   22,    1,   32,   8,   106, 'TDLC300-100-Low', 400,   1,  -15.2; % F0, mu = 0
    };
elseif strcmp(test_version, '38.141-1.v15.14') % SNR requirement based on 38.141-1 for conformance tests
    CFG = {...
    % Test cases based on TS 38.141-1 V15.14 Table 8.4.1.5-1/2/3
    % TC#   mu  cfg    root zone  prmb  Nant N_PRB     channelType    CFO  N_nc   SNR
     5001,  1,  217,    0,   14,    0,   2,   273,            'AWGN',   0,   1,  -16.2; % B4, mu = 1
     5002,  1,  217,    0,   14,    0,   2,   273, 'TDLC300-100-Low', 400,   1,   -9.3; % B4, mu = 1
     5003,  1,  217,    0,   14,    0,   4,   273,            'AWGN',   0,   1,  -18.7; % B4, mu = 1
     5004,  1,  217,    0,   14,    0,   4,   273, 'TDLC300-100-Low', 400,   1,  -13.9; % B4, mu = 1
     5005,  1,  217,    0,   14,    0,   8,   273,            'AWGN',   0,   1,  -20.8; % B4, mu = 1
     5006,  1,  217,    0,   14,    0,   8,   273, 'TDLC300-100-Low', 400,   1,  -17.0; % B4, mu = 1

     5013,  0,  217,    0,   11,    0,   2,   106,            'AWGN',   0,   1,  -16.5; % B4, mu = 0
     5014,  0,  217,    0,   11,    0,   2,   106, 'TDLC300-100-Low', 400,   1,   -8.2; % B4, mu = 0
     5015,  0,  217,    0,   11,    0,   4,   106,            'AWGN',   0,   1,  -18.7; % B4, mu = 0
     5016,  0,  217,    0,   11,    0,   4,   106, 'TDLC300-100-Low', 400,   1,  -13.2; % B4, mu = 0
     5017,  0,  217,    0,   11,    0,   8,   106,            'AWGN',   0,   1,  -20.9; % B4, mu = 0
     5018,  0,  217,    0,   11,    0,   8,   106, 'TDLC300-100-Low', 400,   1,  -16.7; % B4, mu = 0

    };

end

% export CFG into csv file
if isfield(batchsimCfg, 'export_cfg')
    if batchsimCfg.export_cfg
        if isfield(batchsimCfg,'export_fileName')
            cfg_table = cell2table(CFG);
            cfg_table.Properties.VariableNames = {'TC', 'mu', 'cfg', 'root', 'zone', 'prmb', 'Nant','N_PRB','channelType','CFO','N_nc','SNR'};
            %filter table and just keep valid TCs
            idx_row = ismember(cfg_table.TC,caseSet);
            writetable(cfg_table(idx_row,:),batchsimCfg.export_fileName);
        else
            error('Please specify the exporting path!')
        end
        nPerf = nan;
        nFail = nan;
        Pfd  = nan; 
        Pmd  = nan;        
        return;
    end
end

[NallTest, ~] = size(CFG);
N_SNR = length(SNRoffset);

Pfd = zeros(NallTest, N_SNR);
Pmd = zeros(NallTest, N_SNR);

prachTable = loadTable;

if falseDetectionTest
    fprintf('Test PRACH false detection rate:\n');
else
    fprintf('Test PRACH miss detection rate:\n');
end

fprintf('\nTC#  Nframe mu  cfg root zone  prmb Nant N_PRB     channelType     CFO N_nc   SNR  SNRoffset  total  false  miss\n');
fprintf('----------------------------------------------------------------------------------------------------------------\n');

% initialize a txt file to save info used for cuPHY test bench
output_infor_for_cuphy = cell(1,1);

num_generated_subscenarios = 0;
parfor idxSet = 1:NallTest
    testAlloc = [];
    testAlloc.dl = 0;
    testAlloc.ul = 1;
    testAlloc.prach = 1;
    testAlloc_ssb = [];
    testAlloc_sib1 = [];
    caseNum = CFG{idxSet, 1};
    if ismember(caseNum, TcToTest)  && ismember(caseNum, TcToTest_superset_from_batchsimCfg)
        falseRate = [];
        missRate = [];
        for idxSNR = 1:N_SNR
            rng(caseNum,'threefry'); % clearly indicate the rng generator type although it is 'threefry' by default in parallel mode.  
            SysPar = initSysPar(testAlloc);
            SysPar.SimCtrl.relNum = relNum;
            SysPar.SimCtrl.N_frame = Nframe;
            SysPar.SimCtrl.N_slot_run = 0;
            SysPar.SimCtrl.timeDomainSim = 1;
            
            SysPar.SimCtrl.prachFalseAlarmTest = falseDetectionTest;
            SysPar.carrier.duplex = 0;
            SysPar.carrier.mu =  CFG{idxSet, 2};
            SysPar.carrier.FR = 1;
            SysPar.prach{1}.configurationIndex = CFG{idxSet, 3};
            SysPar.prach{1}.rootSequenceIndex = CFG{idxSet, 4};
            SysPar.prach{1}.zeroCorrelationZone = CFG{idxSet, 5};
            SysPar.prach{1}.prmbIdx = CFG{idxSet, 6};
            SysPar.carrier.Nant_gNB = CFG{idxSet, 7};
            SysPar.carrier.Nant_UE = 1;
            SysPar.carrier.N_grid_size_mu = CFG{idxSet, 8};
            SysPar.Chan{1}.type = CFG{idxSet, 9};
            SysPar.Chan{1}.CFO = CFG{idxSet, 10};
            SysPar.prach{1}.N_nc = CFG{idxSet, 11};
            maxDelay =  findMaxDelay(SysPar, prachTable);
            SysPar.Chan{1}.delay = 0.5*maxDelay;
            snr_offset = SNRoffset(idxSNR);
            if snr_offset == 1000 % SNR = 1000dB is used for generating a noiseless TV for cuPHY SNR sweeping usage
                SysPar.Chan{1}.SNR =  snr_offset;
            else
                SysPar.Chan{1}.SNR =  CFG{idxSet, 12} + snr_offset;
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
            
            if batchsimCfg.batchsimMode==0
                [SysPar, UE, gNB] = nrSimulator(SysPar);
            else
                for idx_seed = 1:length(batchsimCfg.seed_list)
                    my_seed = batchsimCfg.seed_list(idx_seed);
                    SysPar.SimCtrl.seed = my_seed;
                    SysPar.SimCtrl.batchsim.save_results = 1;
                    SysPar.SimCtrl.batchsim.save_results_short = 1;
                    subscenario_name = sprintf('scenario_TC%d___seed_%d___SNR_%2.2f',caseNum, my_seed,SysPar.Chan{1}.SNR);
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
                    % enable logging tx Xtf into TV
                    SysPar.SimCtrl.enable_snapshot_gNB_UE_into_SimCtrl = 1;
                    SysPar.SimCtrl.genTV.enable_logging_tx_Xtf = 1;
                    SysPar.SimCtrl.genTV.enable_logging_carrier_and_channel_info = 1;
                    % disable HARQ
                    SysPar.SimCtrl.puschHARQ.EnableAutoHARQ = 0;
                    SysPar.SimCtrl.puschHARQ.MaxTransmissions = 1;
                    % set the fake SNR for ZF to be a very low value as cuPHY has perf issue when it is high.
%                     SysPar.SimCtrl.alg.fakeSNRdBForZf = -36;
                    if (SysPar.Chan{1}.SNR == 1000) || (batchsimCfg.cfgMode==2)
                        SysPar.SimCtrl.N_frame = 1;
                        SysPar.SimCtrl.N_slot_run = 0;
                        SysPar.SimCtrl.genTV.enable = 1;
                        SysPar.SimCtrl.genTV.tvDirName = subscenario_folder_name;
                        SysPar.SimCtrl.genTV.FAPI = 1;
                        if batchsimCfg.cfgMode==2
                            SysPar.SimCtrl.genTV.TVname = 'e2e_cfm_prach';
                            SysPar.SimCtrl.genTV.slotIdx = 4;
                            SysPar.SimCtrl.genTV.launchPattern = 1;
                            SysPar.SimCtrl.genTV.FAPIyaml = 1;
                        else
                            SysPar.SimCtrl.genTV.TVname = 'TV';
                            SysPar.SimCtrl.genTV.slotIdx = 0;
                        end
                    end  
                    if snr_offset == 1000
                        SNR_range_for_cuPHY = CFG{idxSet, 12}+SNRoffset(SNRoffset<1000);
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
                    end

                end
                continue;
            end
            SimCtrl = SysPar.SimCtrl;
        
            totalCnt = SimCtrl.results.prach{1}.totalCnt;
            falseCnt = SimCtrl.results.prach{1}.falseCnt;
            missCnt = SimCtrl.results.prach{1}.missCnt;
            
            falseRate(idxSNR) = falseCnt/totalCnt;  
            missRate (idxSNR) = missCnt/totalCnt;
            
            fprintf('%4d %4d %4d %4d %4d %4d %4d %4d  %4d %18s  %4d %3d  %5.1f   %4.1f      %4d  %4d  %4d\n',...
                CFG{idxSet, 1}, Nframe, CFG{idxSet, 2}, CFG{idxSet, 3}, ...
                CFG{idxSet, 4}, CFG{idxSet, 5}, CFG{idxSet, 6}, CFG{idxSet, 7},...
                CFG{idxSet, 8}, CFG{idxSet, 9}, CFG{idxSet, 10}, CFG{idxSet, 11},...
                CFG{idxSet, 12}, snr_offset, totalCnt,  falseCnt, missCnt);            
        end
        if batchsimCfg.batchsimMode>0
            continue;
        end
        Pfd(idxSet,:) = falseRate;
        Pmd(idxSet,:) = missRate;
    end
end

if batchsimCfg.batchsimMode==0
    fprintf('----------------------------------------------------------------------------------------------------------------\n');
    nPerf = 0;
    nFail = 0;
    for idxSet = 1:NallTest
        caseNum = CFG{idxSet, 1};
        if ismember(caseNum, TcToTest)
            nPerf = nPerf + 1;
            if find(Pfd(idxSet,:) > 0.01)
                nFail = nFail + 1;
            end
            if find(Pmd(idxSet,:) > 0.001)
                nFail = nFail + 1;
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
    fprintf('Total number of subcenarios generated: %d\n',num_generated_subscenarios);
end    
toc; 
fprintf('\n');
