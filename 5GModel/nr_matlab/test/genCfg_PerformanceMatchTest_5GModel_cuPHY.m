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

% This script is used to call testPerformance_xxx.m to generate subscenarios files that can be launched to LSF cluster with batchsim framework

function genCfg_PerformanceMatchTest_5GModel_cuPHY(cfgMode, batchsimMode)

    if nargin == 0
        cfgMode = 0;
    elseif nargin == 1
        batchsimMode = 2;
    end

    if cfgMode == 0        % simple mode for fast simulation
        global_nFrames_per_seed = 2;
        global_seed_list        = 1;
        global_SNR_offset       = -1:1;
    elseif cfgMode == 1    % full mode for long time simulations, e.g., Perf Match Tests
        global_nFrames_per_seed = 50;
        global_seed_list        = 1:20;
        global_SNR_offset       = -7:3;
    elseif cfgMode == 2    % simple ode for conformance tests TV generation
        global_nFrames_per_seed = 1;
        global_seed_list        = 1;
        global_SNR_offset       = [-6, 6];
    end

    timestamp = (datetime('now','TimeZone','America/Los_Angeles','format','yyyy-MM-dd-HH-mm-ss'));
    top_folder_name = sprintf('/home/Aerial-simulations/PerfMatchTest_5GModel_cuPHY/%s_mode_%d/', timestamp, cfgMode); % append timestamp to avoid data overriding by mistake
    batchsimCfg.cfgMode = cfgMode; 
    % batchsimMode = 0: disable batchsimMode, 1: phase2 perf study, 2: perf match test
    % for 5GModel and cuPHY with channel realization freezing 3: perf match test mode with GPU TDL channel
    batchsimCfg.batchsimMode = batchsimMode;

    enable_gen_cfg_pusch   = 1;
    enable_gen_cfg_pucch   = 1;
    enable_gen_cfg_prach   = 1;
    enable_gen_cfg_srs     = 0;
    
    
    %/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ PUSCH /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    if enable_gen_cfg_pusch
        enable_pusch_data                   = 1;
        enable_pusch_uci                    = 1;
        enable_pusch_tf_precoding           = 1;
        enable_pusch_0p00001bler            = 0;
        enable_pusch_ul_meas                = 0;
        enable_pusch_awgn_bler_all_mcs      = 0;
        enable_pusch_fading_bler_all_mcs    = 0;

        %  MIMO equalizer related config
        mimo_equalizer = 'MMSE';
        if strcmp(mimo_equalizer,'ZF')
            batchsimCfg.SimCtrl.alg.enableIrc = 0.0;
            batchsimCfg.SimCtrl.alg.enableNoiseEstForZf = 0.0;
            batchsimCfg.SimCtrl.alg.enable_use_genie_nCov = 0.0;
            batchsimCfg.SimCtrl.alg.genie_nCov_method = ' ';
            batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage = 0.0;
            batchsimCfg.SimCtrl.alg.nCov_shrinkage_method = 0;
            batchsimCfg.SimCtrl.alg.enable_get_genie_meas = 0.0;        
            batchsimCfg.SimCtrl.enable_get_genie_channel_matrix = 0.0;
        elseif strcmp(mimo_equalizer,'MMSE')
            batchsimCfg.SimCtrl.alg.enableIrc = 0.0;
            batchsimCfg.SimCtrl.alg.enableNoiseEstForZf = 1.0;
            batchsimCfg.SimCtrl.alg.enable_use_genie_nCov = 0.0;
            batchsimCfg.SimCtrl.alg.genie_nCov_method = ' ';
            batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage = 0.0;
            batchsimCfg.SimCtrl.alg.nCov_shrinkage_method = 0;
            batchsimCfg.SimCtrl.alg.enable_get_genie_meas = 0.0;        
            batchsimCfg.SimCtrl.enable_get_genie_channel_matrix = 0.0;
        elseif strcmp(mimo_equalizer,'MMSE-IRC')
            batchsimCfg.SimCtrl.alg.enableIrc = 1.0;
            batchsimCfg.SimCtrl.alg.enableNoiseEstForZf = 0.0;
            batchsimCfg.SimCtrl.alg.enable_use_genie_nCov = 0.0;
            batchsimCfg.SimCtrl.alg.genie_nCov_method = ' ';
            batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage = 0.0;
            batchsimCfg.SimCtrl.alg.nCov_shrinkage_method = 0;
            batchsimCfg.SimCtrl.alg.enable_get_genie_meas = 0.0;        
            batchsimCfg.SimCtrl.enable_get_genie_channel_matrix = 0.0;     
        elseif strcmp(mimo_equalizer,'MMSE-IRC-genie_nCov_opt2')
            batchsimCfg.SimCtrl.alg.enableIrc = 1.0;
            batchsimCfg.SimCtrl.alg.enableNoiseEstForZf = 0.0;
            batchsimCfg.SimCtrl.alg.enable_use_genie_nCov = 1.0; 
            batchsimCfg.SimCtrl.alg.genie_nCov_method = 'genie_interfChanResponse_based';        
            batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage = 0.0;
            batchsimCfg.SimCtrl.alg.nCov_shrinkage_method = 0;
            batchsimCfg.SimCtrl.alg.enable_get_genie_meas = 1.0;        
            batchsimCfg.SimCtrl.enable_get_genie_channel_matrix = 1.0;
        elseif strcmp(mimo_equalizer,'MMSE-IRC-w-nCov_shrinkage_RBLW')    
            batchsimCfg.SimCtrl.alg.enableIrc = 1.0;
            batchsimCfg.SimCtrl.alg.enableNoiseEstForZf = 0.0;
            batchsimCfg.SimCtrl.alg.enable_use_genie_nCov = 0.0;
            batchsimCfg.SimCtrl.alg.genie_nCov_method = ' ';
            batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage = 1.0;
            batchsimCfg.SimCtrl.alg.nCov_shrinkage_method = 0; % 'RBLW';
            batchsimCfg.SimCtrl.alg.enable_get_genie_meas = 0.0;        
            batchsimCfg.SimCtrl.enable_get_genie_channel_matrix = 0.0;    
        elseif strcmp(mimo_equalizer,'MMSE-IRC-w-nCov_shrinkage_OAS')   
            batchsimCfg.SimCtrl.alg.enableIrc = 1.0;
            batchsimCfg.SimCtrl.alg.enableNoiseEstForZf = 0.0;
            batchsimCfg.SimCtrl.alg.enable_use_genie_nCov = 0.0;
            batchsimCfg.SimCtrl.alg.genie_nCov_method = ' ';
            batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage = 1.0;
            batchsimCfg.SimCtrl.alg.nCov_shrinkage_method = 1; % 'OAS';
            batchsimCfg.SimCtrl.alg.enable_get_genie_meas = 0.0;        
            batchsimCfg.SimCtrl.enable_get_genie_channel_matrix = 0.0;    
        end
        batchsimCfg.enable_pusch_use_perfect_channel_est_for_equalizer = 0;
        if batchsimCfg.enable_pusch_use_perfect_channel_est_for_equalizer
            batchsimCfg.SimCtrl.enable_get_genie_channel_matrix = 1;
            batchsimCfg.SimCtrl.alg.enable_use_genie_channel_for_equalizer = 1;
            batchsimCfg.SimCtrl.alg.TdiMode = 2;
            batchsimCfg.SimCtrl.alg.enableCfoCorrection = 0;
        end
        
        % PUSCH data
        if enable_pusch_data
            if cfgMode == 1
                nFrame_pusch_data_per_seed = 2*global_nFrames_per_seed;    
            else
                nFrame_pusch_data_per_seed = global_nFrames_per_seed; 
            end
            SNR_offset = global_SNR_offset; %-5:1:5;
            batchsimCfg.seed_list = global_seed_list; %1:5;            
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_data/');
            selected_TCs = [7004, 7005, 7006, 7012, 7013]; 
            testPerformance_pusch(selected_TCs, [SNR_offset, 1000], nFrame_pusch_data_per_seed, 0, batchsimCfg); % SNR = 1000dB is used for generating a noiseless TV for cuPHY SNR sweeping usage. In testPerformance_xxx, it will force to absolute SNR to 1000dB instead of offset
        end
        % UCI
        if enable_pusch_uci
            if cfgMode == 1
                nFrame_pusch_uci_per_seed = 10*global_nFrames_per_seed;
            else
                nFrame_pusch_uci_per_seed = global_nFrames_per_seed;  
            end  
            SNR_offset = global_SNR_offset; %-5:1:5;
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_uci/');
            selected_TCs = [7051:7054];
            testPerformance_pusch(selected_TCs, [SNR_offset, 1000], nFrame_pusch_uci_per_seed, 1, batchsimCfg);
        end
        % transform precoding
        if enable_pusch_tf_precoding
            nFrame_pusch_tf_precoding_per_seed = global_nFrames_per_seed; 
            SNR_offset = -6:1.5:10;
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_tf_precoding/');
            testPerformance_pusch('selected', [SNR_offset, 1000], nFrame_pusch_tf_precoding_per_seed, 2, batchsimCfg);
        end
        % PUSCH data 0.001% BLER
        if enable_pusch_0p00001bler
            if cfgMode == 1
                nFrame_pusch_data_0p00005bler_per_seed = 10*global_nFrames_per_seed;
                batchsimCfg.seed_list = 1:25;
            else
                nFrame_pusch_data_0p00005bler_per_seed = global_nFrames_per_seed; 
                batchsimCfg.seed_list = global_seed_list;
            end
            SNR_offset = -2:0.25:2;           
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_0p00001bler/');
            testPerformance_pusch('selected', [SNR_offset, 1000], nFrame_pusch_data_0p00005bler_per_seed, 3, batchsimCfg);
        end
        % PUSCH UL measurement
        if enable_pusch_ul_meas
            nFrame_pusch_ul_meas_per_seed = global_nFrames_per_seed;     
            SNR_offset = -10:2:30;
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_ul_measurement/');
            testPerformance_pusch('selected', [SNR_offset, 1000], nFrame_pusch_ul_meas_per_seed, 10, batchsimCfg);
        end
        % AWGN, BLER for all MCS
        if enable_pusch_awgn_bler_all_mcs
            if cfgMode == 1
                nFrame_pusch_data_bler_awgn_all_mcs_per_seed = 5*global_nFrames_per_seed;  
            else
                nFrame_pusch_data_bler_awgn_all_mcs_per_seed = global_nFrames_per_seed;
            end
            SNR_offset = -1:0.25:1.75;
            batchsimCfg.seed_list = 1; % just one seed should be fine for PerfMatchTest
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_bler_awgn_all_mcs/');
            testPerformance_pusch('selected', [SNR_offset, 1000], nFrame_pusch_data_bler_awgn_all_mcs_per_seed, 11, batchsimCfg);
        end
        % fading, BLER for all MCS
        if enable_pusch_fading_bler_all_mcs
            if cfgMode == 1
                nFrame_pusch_data_bler_fading_all_mcs_per_seed = 5*global_nFrames_per_seed;
            else
                nFrame_pusch_data_bler_fading_all_mcs_per_seed = global_nFrames_per_seed; 
            end
            SNR_offset = -6:2:10; % -1:0.25:1.75;%
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_bler_fading_all_mcs/');
            testPerformance_pusch('selected', [SNR_offset, 1000], nFrame_pusch_data_bler_fading_all_mcs_per_seed, 12, batchsimCfg);
        end
    
    end
    
    %/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ PUCCH /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    if enable_gen_cfg_pucch
        list_pucchFormat = [0, 1, 2, 3];
        list_pucchTestMode = [1, 2, 3, 4, 9];
    
        % Format 0 
        pucchFormat = 0; 
        if ismember(pucchFormat,list_pucchFormat) 
            nFrame_pucch_0_per_seed = global_nFrames_per_seed; %50; 
            batchsimCfg.seed_list = global_seed_list; %1:5;    
            for pucchTestMode = [1, 3] % 1: DTX to ACK, 2: NACK to ACK detection, 3: ACK missed detection, 9: measurement
                if ismember(pucchTestMode,list_pucchTestMode)
                    if cfgMode == 1
                        if ismember(pucchTestMode, [1])
                            SNR_offset = [-10,0,10];
                        elseif ismember(pucchTestMode, [2,3])
                            SNR_offset = global_SNR_offset; %-5:1:5; 
                        elseif pucchTestMode==9
                            SNR_offset = -10:2:30;
                        end
                    else
                        SNR_offset = global_SNR_offset;
                    end
                    batchsimCfg.ws_folder_name = fullfile(top_folder_name, sprintf('/ws_scenarios_pucchFormat%d_pucchTestMode%d/', pucchFormat, pucchTestMode)); 
                    selected_TCs = [6003, 6004];
                    testPerformance_pucch(selected_TCs, [SNR_offset, 1000], nFrame_pucch_0_per_seed, pucchFormat, pucchTestMode, batchsimCfg);
                end
            end          
        end
    
        % Format 1 
        pucchFormat = 1; 
        if ismember(pucchFormat,list_pucchFormat)
            nFrame_pucch_1_per_seed = global_nFrames_per_seed; %50; 
            batchsimCfg.seed_list = global_seed_list; %1:5;    
            for pucchTestMode = [1, 2, 3] % 1: false detection, 2: NACK to ACK detection, 3: ACK missed detection, 9: measurement
                if ismember(pucchTestMode,list_pucchTestMode)
                    if ismember(pucchTestMode, [1])
                        if cfgMode == 1
                            SNR_offset = [-10,0,10, 20, 30, 40];
                        else
                            SNR_offset = global_SNR_offset;
                        end
                        selected_TCs = [6108];
                    elseif ismember(pucchTestMode, [2,3])
                        SNR_offset = global_SNR_offset; %-5:1:5; 
                        if pucchTestMode==2
                            selected_TCs = [6102];
                        else
                            selected_TCs = [6108];
                        end
                    elseif pucchTestMode==9
                        if cfgMode == 1
                            SNR_offset = -10:2:30;
                        else
                            SNR_offset = global_SNR_offset;
                        end
                    end
                    batchsimCfg.ws_folder_name = fullfile(top_folder_name, sprintf('/ws_scenarios_pucchFormat%d_pucchTestMode%d/', pucchFormat, pucchTestMode)); 
                    testPerformance_pucch(selected_TCs, [SNR_offset, 1000], nFrame_pucch_1_per_seed, pucchFormat, pucchTestMode, batchsimCfg);
                end
            end          
        end 
    
        % Format 2 
        pucchFormat = 2; 
        if ismember(pucchFormat,list_pucchFormat)
            nFrame_pucch_2_per_seed = global_nFrames_per_seed; %50; 
            batchsimCfg.seed_list = global_seed_list; %1:5;    
            for pucchTestMode = [1, 3, 4] % 1: false detection, 3: ACK missed detection, 4: UCI BLER, 9: measurement
                if ismember(pucchTestMode,list_pucchTestMode)
                    if pucchTestMode==1
                        if cfgMode == 1
                            SNR_offset = [-10,0,10];
                        else
                            SNR_offset = global_SNR_offset;
                        end
                        selected_TCs = [6202];
                    elseif ismember(pucchTestMode,[3,4])
                        SNR_offset = global_SNR_offset; %-5:1:5; 
                        if pucchTestMode == 3
                            selected_TCs = [6202];
                        else
                            selected_TCs = [6212];
                        end
                    elseif pucchTestMode==9
                        if cfgMode == 1
                            SNR_offset = -10:2:30;
                        else
                            SNR_offset = global_SNR_offset;
                        end                        
                    end
                    batchsimCfg.ws_folder_name = fullfile(top_folder_name, sprintf('/ws_scenarios_pucchFormat%d_pucchTestMode%d/', pucchFormat, pucchTestMode)); 
                    testPerformance_pucch(selected_TCs, [SNR_offset, 1000], nFrame_pucch_2_per_seed, pucchFormat, pucchTestMode, batchsimCfg);
                end
            end          
        end    
    
        % Format 3 
        pucchFormat = 3; 
        if ismember(pucchFormat,list_pucchFormat)
            nFrame_pucch_3_per_seed = global_nFrames_per_seed; %50; 
            batchsimCfg.seed_list = global_seed_list; %1:5;    
            for pucchTestMode = [4] % 4: UCI BLER, 9: measurement
                if ismember(pucchTestMode,list_pucchTestMode)
                    if pucchTestMode==4
                        SNR_offset = global_SNR_offset; %-5:1:5; 
                        selected_TCs = [6303, 6304, 6308];
                    elseif pucchTestMode==9
                        if cfgMode == 1
                            SNR_offset = -10:2:30;
                        else
                            SNR_offset = global_SNR_offset;
                        end 
                    end
                    batchsimCfg.ws_folder_name = fullfile(top_folder_name, sprintf('/ws_scenarios_pucchFormat%d_pucchTestMode%d/', pucchFormat, pucchTestMode)); 
                    testPerformance_pucch(selected_TCs, [SNR_offset, 1000], nFrame_pucch_3_per_seed, pucchFormat, pucchTestMode, batchsimCfg);
                end
            end          
        end 
    end
    
    %/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ PRACH /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    if enable_gen_cfg_prach
        list_falseDetectionTest = [0, 1]; % 1: false detection rate, 0: miss detection rate
        for falseDetectionTest = list_falseDetectionTest
            if cfgMode == 1
                nFrame_prach_per_seed = 2*global_nFrames_per_seed;
            else
                nFrame_prach_per_seed = global_nFrames_per_seed;  
            end  
            if falseDetectionTest==0 % miss detection
                SNR_offset = global_SNR_offset; %-5:1:5;
            else % false alarm
                if cfgMode == 1
                    SNR_offset = [-10, 0, 10];
                else
                    SNR_offset = global_SNR_offset;
                end                 
            end
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, sprintf('/ws_scenarios_prach_falseDetectionTest%d/', falseDetectionTest));
            selected_TCs = [5003, 5004];
            testPerformance_prach(selected_TCs, [SNR_offset, 1000], nFrame_prach_per_seed, falseDetectionTest, batchsimCfg);
        end
    end
    
    %/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ SRS /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    if enable_gen_cfg_srs
        list_chanType = {'AWGN', 'TDLA30-10-Low', 'TDLB100-400-Low', 'TDLC300-100-Low'};
        for idx_chanType = 1:length(list_chanType)
            chanType = list_chanType{idx_chanType};
            if cfgMode == 1
                N_frame_per_seed = 10*global_nFrames_per_seed;
            else
                N_frame_per_seed = global_nFrames_per_seed;  
            end
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, sprintf('/ws_scenarios_srs_chanType_%s/', chanType));
            list_SNR_offset = global_SNR_offset; %-5:1:5;
            for SNR_offset = list_SNR_offset
                testPerformance_srs('full', chanType, N_frame_per_seed, SNR_offset, batchsimCfg)
            end
        end
    end
