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
% cfgM = 0: simple mode for fast simulation, cfgMode = 1: full mode for long simulations, cfgMode = 2: gen TVs for conformance tests
% batchsimMode = 0: disable batchsimMode, 1: phase2 perf study, 2: perf match test for 5GModel and cuPHY with channel realization freezing 3: perf match test mode with GPU TDL channel
function genCfg_testPerformance_batchsimLaunch(cfgMode, caseSet, batchsimMode, enable_UL_Rx_impairments)

    if nargin == 0
        cfgMode = 0;
        caseSet = 'full';
		batchsimMode = 1;
        enable_UL_Rx_impairments = 0;
    elseif nargin == 1
        caseSet = 'full';
		batchsimMode = 1;
        enable_UL_Rx_impairments = 0;
	elseif nargin == 2
		batchsimMode = 1;
        enable_UL_Rx_impairments = 0;
    elseif nargin == 3
        enable_UL_Rx_impairments = 0;
    end
    if cfgMode == 0     % for quick tests
        global_nFrames_per_seed = 10;
        global_seed_list        = 1;
        global_SNR_offset       = -1:1;
    elseif cfgMode == 1 % for long time simulations
        global_nFrames_per_seed = 50;
		global_SNR_offset       = -8:3;
%         global_SNR_offset       = -5:0.5:0;
		if (batchsimMode == 2) || (batchsimMode == 3) % for perf match tests with freezing channel, we need many seeds		
        	global_seed_list    = 1:20;
		else
            global_seed_list    = 1:5;
        end
	elseif cfgMode == 2    % for conformance tests TV generation, one frame and one seed should be good
        global_nFrames_per_seed = 1;
        global_seed_list        = 1;
        global_SNR_offset       = [-6, 0, 6];	
    end

    timestamp = (datetime('now','TimeZone','America/Los_Angeles','format','yyyy-MM-dd-HH-mm-ss'));
    top_folder_name = sprintf('/home/Aerial-simulations/sim_results/%s_mode_%d/', timestamp, cfgMode); % append timestamp to avoid data overriding by mistake
	batchsimCfg.cfgMode = cfgMode;         
    batchsimCfg.batchsimMode = batchsimMode;
    
    enable_gen_cfg_pusch   = 1;
    enable_gen_cfg_pucch   = 1;
    enable_gen_cfg_prach   = 1;
    enable_gen_cfg_srs     = 0;
    
    if strcmp(caseSet, 'full')
        batchsimCfg.caseSet = 0:1e5;
    elseif strcmp(caseSet, 'TR4')
        batchsimCfg.caseSet = [7004:7006, 7012:7013, 7051:7054, 7056, 6003:6004, 6102, 6108, 6202, 6212, 6303:6304, 6308, 5003:5004];
    elseif strcmp(caseSet, 'test')
        batchsimCfg.caseSet = [7052];
    elseif strcmp(caseSet, 'lowSNRmarginTCs')
        batchsimCfg.caseSet = [7001, 7004, 7007, 7051:7053, 7055:7060];
    elseif strcmp(caseSet, 'twoLayersTCs')
        batchsimCfg.caseSet = [7012, 7013];
    end
    
    if cfgMode == 2    % for conformance tests TV generation
        batchsimCfg.test_version = '38.141-1.v15.14';
    end

    batchsimCfg.enable_UL_Rx_RF_impairments = enable_UL_Rx_impairments;

    %/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ PUSCH /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    if enable_gen_cfg_pusch
        enable_pusch_data                   = 1;
        enable_pusch_uci                    = 1;
        enable_pusch_tf_precoding           = 1;
        enable_pusch_0p00001bler            = 0;
        enable_bler_tdl_cdl                 = 0;
        enable_pusch_ul_meas                = 0;
        enable_pusch_awgn_bler_all_mcs      = 0;
        enable_pusch_fading_bler_all_mcs    = 0;

        % PUSCH MIMO equalizer related config
        pusch_mimo_equalizer = 'MMSE-IRC-w-nCov_shrinkage_RBLW';
        if strcmp(pusch_mimo_equalizer,'ZF')
            batchsimCfg.SimCtrl.alg.enableIrc = 0.0;
            batchsimCfg.SimCtrl.alg.enableNoiseEstForZf = 0.0;
            batchsimCfg.SimCtrl.alg.enable_use_genie_nCov = 0.0;
            batchsimCfg.SimCtrl.alg.genie_nCov_method = ' ';
            batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage = 0.0;
            batchsimCfg.SimCtrl.alg.nCov_shrinkage_method = 0;
            batchsimCfg.SimCtrl.alg.enable_get_genie_meas = 0.0;        
            batchsimCfg.SimCtrl.enable_get_genie_channel_matrix = 0.0;
        elseif strcmp(pusch_mimo_equalizer,'MMSE')
            batchsimCfg.SimCtrl.alg.enableIrc = 0.0;
            batchsimCfg.SimCtrl.alg.enableNoiseEstForZf = 1.0;
            batchsimCfg.SimCtrl.alg.enable_use_genie_nCov = 0.0;
            batchsimCfg.SimCtrl.alg.genie_nCov_method = ' ';
            batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage = 0.0;
            batchsimCfg.SimCtrl.alg.nCov_shrinkage_method = 0;
            batchsimCfg.SimCtrl.alg.enable_get_genie_meas = 0.0;        
            batchsimCfg.SimCtrl.enable_get_genie_channel_matrix = 0.0;
        elseif strcmp(pusch_mimo_equalizer,'MMSE-IRC')
            batchsimCfg.SimCtrl.alg.enableIrc = 1.0;
            batchsimCfg.SimCtrl.alg.enableNoiseEstForZf = 0.0;
            batchsimCfg.SimCtrl.alg.enable_use_genie_nCov = 0.0;
            batchsimCfg.SimCtrl.alg.genie_nCov_method = ' ';
            batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage = 0.0;
            batchsimCfg.SimCtrl.alg.nCov_shrinkage_method = 0;
            batchsimCfg.SimCtrl.alg.enable_get_genie_meas = 0.0;        
            batchsimCfg.SimCtrl.enable_get_genie_channel_matrix = 0.0;     
        elseif strcmp(pusch_mimo_equalizer,'MMSE-IRC-genie_nCov_opt2')
            batchsimCfg.SimCtrl.alg.enableIrc = 1.0;
            batchsimCfg.SimCtrl.alg.enableNoiseEstForZf = 0.0;
            batchsimCfg.SimCtrl.alg.enable_use_genie_nCov = 1.0; 
            batchsimCfg.SimCtrl.alg.genie_nCov_method = 'genie_interfChanResponse_based';        
            batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage = 0.0;
            batchsimCfg.SimCtrl.alg.nCov_shrinkage_method = 0;
            batchsimCfg.SimCtrl.alg.enable_get_genie_meas = 1.0;        
            batchsimCfg.SimCtrl.enable_get_genie_channel_matrix = 1.0;
        elseif strcmp(pusch_mimo_equalizer,'MMSE-IRC-w-nCov_shrinkage_RBLW')    
            batchsimCfg.SimCtrl.alg.enableIrc = 1.0;
            batchsimCfg.SimCtrl.alg.enableNoiseEstForZf = 0.0;
            batchsimCfg.SimCtrl.alg.enable_use_genie_nCov = 0.0;
            batchsimCfg.SimCtrl.alg.genie_nCov_method = ' ';
            batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage = 1.0;
            batchsimCfg.SimCtrl.alg.nCov_shrinkage_method = 0; % 'RBLW';
            batchsimCfg.SimCtrl.alg.enable_get_genie_meas = 0.0;        
            batchsimCfg.SimCtrl.enable_get_genie_channel_matrix = 0.0;    
        elseif strcmp(pusch_mimo_equalizer,'MMSE-IRC-w-nCov_shrinkage_OAS')   
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
            batchsimCfg.SimCtrl.alg.TdiMode = 1;
            batchsimCfg.SimCtrl.alg.enableCfoCorrection = 0;
        end

        % PUSCH data
        if enable_pusch_data
            if cfgMode == 1
                nFrame_pusch_data_per_seed = 2*global_nFrames_per_seed; % run more frames for long time simulation on PUSCH data   
            else
                nFrame_pusch_data_per_seed = global_nFrames_per_seed; 
            end
            SNR_offset = global_SNR_offset; %-5:1:5;
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_data/');
			if (batchsimMode == 2) || (batchsimMode == 3)
				SNR_offset_list = [SNR_offset, 1000]; % in channel freezing test mode, cuPHY needs a noise-free TV
			else
				SNR_offset_list = SNR_offset;
			end
            testPerformance_pusch('selected', SNR_offset_list, nFrame_pusch_data_per_seed, 0, batchsimCfg);
        end
        % UCI on PUSCH
        if enable_pusch_uci
            if cfgMode == 1
                nFrame_pusch_uci_per_seed = 10*global_nFrames_per_seed;
            else
                nFrame_pusch_uci_per_seed = global_nFrames_per_seed;  
            end  
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_uci/');
			if (batchsimMode == 2) || (batchsimMode == 3)
				SNR_offset_list = [global_SNR_offset, 1000]; % in channel freezing test mode, cuPHY needs a noise-free TV
			else
				SNR_offset_list = global_SNR_offset;
			end
            testPerformance_pusch('selected', SNR_offset_list, nFrame_pusch_uci_per_seed, 1, batchsimCfg);
        end
        % transform precoding
        if enable_pusch_tf_precoding
            nFrame_pusch_tf_precoding_per_seed = global_nFrames_per_seed; 
			if (batchsimMode == 2) || (batchsimMode == 3)
				SNR_offset_list = [-6:1.5:10, 1000];
			else
				SNR_offset_list = global_SNR_offset;
			end
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_tf_precoding/');
            testPerformance_pusch('selected', SNR_offset_list, nFrame_pusch_tf_precoding_per_seed, 2, batchsimCfg);
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
			if (batchsimMode == 2) || (batchsimMode == 3)
				SNR_offset_list = [SNR_offset, 1000]; % in channel freezing test mode, cuPHY needs a noise-free TV
			else
				SNR_offset_list = SNR_offset;
			end          
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_0p00001bler/');
            testPerformance_pusch('selected', SNR_offset_list, nFrame_pusch_data_0p00005bler_per_seed, 3, batchsimCfg);
        end
        % PUSCH BLER with TDL/CDL channel model
        if enable_bler_tdl_cdl
            if cfgMode == 1
                nFrame_pusch_data_per_seed = 2*global_nFrames_per_seed; % run more frames for long time simulation on PUSCH data   
            else
                nFrame_pusch_data_per_seed = global_nFrames_per_seed; 
            end
            SNR_offset = global_SNR_offset; %-5:1:5;
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_bler_tdl_cdl/');
			if (batchsimMode == 2) || (batchsimMode == 3)
				SNR_offset_list = [SNR_offset, 1000]; % in channel freezing test mode, cuPHY needs a noise-free TV
			else
				SNR_offset_list = SNR_offset;
			end
            testPerformance_pusch('selected', SNR_offset_list, nFrame_pusch_data_per_seed, 8, batchsimCfg);
        end
        % PUSCH UL measurement
        if enable_pusch_ul_meas
            nFrame_pusch_ul_meas_per_seed = global_nFrames_per_seed;     
            SNR_offset = -10:2:30;
			if (batchsimMode == 2) || (batchsimMode == 3)
				SNR_offset_list = [SNR_offset, 1000]; % in channel freezing test mode, cuPHY needs a noise-free TV
			else
				SNR_offset_list = SNR_offset;
			end 
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_ul_measurement/');
            testPerformance_pusch('selected', SNR_offset_list, nFrame_pusch_ul_meas_per_seed, 10, batchsimCfg);
        end
        % AWGN, BLER for all MCS
        if enable_pusch_awgn_bler_all_mcs
            if cfgMode == 1
                nFrame_pusch_data_bler_awgn_all_mcs_per_seed = 5*global_nFrames_per_seed;  
            else
                nFrame_pusch_data_bler_awgn_all_mcs_per_seed = global_nFrames_per_seed;
            end
			if (batchsimMode == 2) || (batchsimMode == 3)
				SNR_offset_list = [-1:0.25:1.75, 1000];
            	batchsimCfg.seed_list = 1; % just one seed should be fine for PerfMatchTest when channel is AWGN
			else
            	SNR_offset_list = -1:0.25:1.75;
            	batchsimCfg.seed_list = global_seed_list; %1:5;
			end
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_bler_awgn_all_mcs/');
            testPerformance_pusch('selected', SNR_offset_list, nFrame_pusch_data_bler_awgn_all_mcs_per_seed, 11, batchsimCfg);
        end
        % fading, BLER for all MCS
        if enable_pusch_fading_bler_all_mcs
            if cfgMode == 1
                nFrame_pusch_data_bler_fading_all_mcs_per_seed = 5*global_nFrames_per_seed;
            else
                nFrame_pusch_data_bler_fading_all_mcs_per_seed = global_nFrames_per_seed; 
            end
            SNR_offset = -6:2:10;
			if (batchsimMode == 2) || (batchsimMode == 3)
				SNR_offset_list = [SNR_offset, 1000]; % in channel freezing test mode, cuPHY needs a noise-free TV
			else
				SNR_offset_list = SNR_offset;
			end 
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, '/ws_scenarios_pusch_bler_fading_all_mcs/');
            testPerformance_pusch('selected', SNR_offset_list, nFrame_pusch_data_bler_fading_all_mcs_per_seed, 12, batchsimCfg);
        end
    
    end
    
    %/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ PUCCH /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    if enable_gen_cfg_pucch
        list_pucchFormat = [0, 1, 2, 3];
        list_pucchTestMode = [1, 2, 3, 4, 9]; %1: false detection, 2: NACK to ACK detection, 3: ACK missed detection, 9: measurement
    
        % Format 0 
        pucchFormat = 0; 
        if ismember(pucchFormat,list_pucchFormat) 
            nFrame_pucch_0_per_seed = global_nFrames_per_seed; %50; 
            batchsimCfg.seed_list = global_seed_list; %1:5; 
			if strcmp(caseSet, 'full')
				pucchTestMode_PF0 = [1, 2, 3, 9];
			else
				pucchTestMode_PF0 = [1, 3];
			end   
            for pucchTestMode = pucchTestMode_PF0
                if ismember(pucchTestMode,list_pucchTestMode)
                    if ismember(pucchTestMode, [1])
                        SNR_offset = [-10,0,10];
                    elseif ismember(pucchTestMode, [2,3])
                        SNR_offset = global_SNR_offset; %-5:1:5; 
                    elseif pucchTestMode==9
                        SNR_offset = -10:2:30;
                    end
					if (batchsimMode == 2) || (batchsimMode == 3)
						SNR_offset_list = [SNR_offset, 1000]; % in channel freezing test mode, cuPHY needs a noise-free TV
					else
						SNR_offset_list = SNR_offset;
					end 
                    batchsimCfg.ws_folder_name = fullfile(top_folder_name, sprintf('/ws_scenarios_pucchFormat%d_pucchTestMode%d/', pucchFormat, pucchTestMode)); 
                    testPerformance_pucch('selected', SNR_offset_list, nFrame_pucch_0_per_seed, pucchFormat, pucchTestMode, batchsimCfg);
                end
            end          
        end
    
        % Format 1 
        pucchFormat = 1; 
        if ismember(pucchFormat,list_pucchFormat)
            nFrame_pucch_1_per_seed = global_nFrames_per_seed; %50; 
            batchsimCfg.seed_list = global_seed_list; %1:5;
			if strcmp(caseSet, 'full')
				pucchTestMode_PF1 = [1, 2, 3, 9];
			else
				pucchTestMode_PF1 = [1, 2, 3];
			end      
            for pucchTestMode = pucchTestMode_PF1 % 1: false detection, 2: NACK to ACK detection, 3: ACK missed detection, 9: measurement
                if ismember(pucchTestMode,list_pucchTestMode)
                    if ismember(pucchTestMode, [1])
                        SNR_offset = [-10, 0, 10, 20, 30, 40];
                    elseif ismember(pucchTestMode, [2,3])
                        SNR_offset = global_SNR_offset; %-5:1:5; 
                    elseif pucchTestMode==9
                        SNR_offset = -10:2:30;
                    end
					if (batchsimMode == 2) || (batchsimMode == 3)
						SNR_offset_list = [SNR_offset, 1000]; % in channel freezing test mode, cuPHY needs a noise-free TV
					else
						SNR_offset_list = SNR_offset;
					end 
                    batchsimCfg.ws_folder_name = fullfile(top_folder_name, sprintf('/ws_scenarios_pucchFormat%d_pucchTestMode%d/', pucchFormat, pucchTestMode)); 
                    testPerformance_pucch('selected', SNR_offset_list, nFrame_pucch_1_per_seed, pucchFormat, pucchTestMode, batchsimCfg);
                end
            end          
        end 
    
        % Format 2 
        pucchFormat = 2; 
        if ismember(pucchFormat,list_pucchFormat)
            nFrame_pucch_2_per_seed = global_nFrames_per_seed; %50; 
            batchsimCfg.seed_list = global_seed_list; %1:5;   
			if strcmp(caseSet, 'full')
				pucchTestMode_PF2 = [1, 3, 4, 9];
			else
				pucchTestMode_PF2 = [1, 3, 4];
			end       
            for pucchTestMode = pucchTestMode_PF2 % 1: false detection, 3: ACK missed detection, 4: UCI BLER, 9: measurement
                if ismember(pucchTestMode,list_pucchTestMode)
                    if pucchTestMode==1
                        SNR_offset = [-10,0,10];
                    elseif ismember(pucchTestMode,[3,4])
                        SNR_offset = global_SNR_offset; %-5:1:5; 
                    elseif pucchTestMode==9
                        SNR_offset = -10:2:30;
                    end
					if (batchsimMode == 2) || (batchsimMode == 3)
						SNR_offset_list = [SNR_offset, 1000]; % in channel freezing test mode, cuPHY needs a noise-free TV
					else
						SNR_offset_list = SNR_offset;
					end 
                    batchsimCfg.ws_folder_name = fullfile(top_folder_name, sprintf('/ws_scenarios_pucchFormat%d_pucchTestMode%d/', pucchFormat, pucchTestMode)); 
                    testPerformance_pucch('selected', SNR_offset_list, nFrame_pucch_2_per_seed, pucchFormat, pucchTestMode, batchsimCfg);
                end
            end          
        end    
    
        % Format 3 
        pucchFormat = 3; 
        if ismember(pucchFormat,list_pucchFormat)
            nFrame_pucch_3_per_seed = global_nFrames_per_seed; %50; 
            batchsimCfg.seed_list = global_seed_list; %1:5; 
			if strcmp(caseSet, 'full')
				pucchTestMode_PF3 = [4, 9];
			else
				pucchTestMode_PF3 = [4];
			end     
            for pucchTestMode = pucchTestMode_PF3 % 4: UCI BLER, 9: measurement
                if ismember(pucchTestMode,list_pucchTestMode)
                    if pucchTestMode==4
                        SNR_offset = global_SNR_offset; %-5:1:5; 
                    elseif pucchTestMode==9
                        SNR_offset = -10:2:30;
                    end
					if (batchsimMode == 2) || (batchsimMode == 3)
						SNR_offset_list = [SNR_offset, 1000]; % in channel freezing test mode, cuPHY needs a noise-free TV
					else
						SNR_offset_list = SNR_offset;
					end 
                    batchsimCfg.ws_folder_name = fullfile(top_folder_name, sprintf('/ws_scenarios_pucchFormat%d_pucchTestMode%d/', pucchFormat, pucchTestMode)); 
                    testPerformance_pucch('selected', SNR_offset_list, nFrame_pucch_3_per_seed, pucchFormat, pucchTestMode, batchsimCfg);
                end
            end          
        end 
    end
    
    %/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ PRACH /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    if enable_gen_cfg_prach
        list_falseDetectionTest = [0, 1]; % 1: false detection rate, 0: miss detection rate
        for falseDetectionTest = list_falseDetectionTest
            if cfgMode == 1
                nFrame_prach_per_seed = 10*global_nFrames_per_seed;
            else
                nFrame_prach_per_seed = global_nFrames_per_seed;  
            end  
            if falseDetectionTest==0 % miss detection
                SNR_offset = global_SNR_offset; %-5:1:5;
            else % false alarm
                SNR_offset = [-10, 0, 10];
            end
			
            batchsimCfg.seed_list = global_seed_list; %1:5;
            batchsimCfg.ws_folder_name = fullfile(top_folder_name, sprintf('/ws_scenarios_prach_falseDetectionTest%d/', falseDetectionTest));
			if (batchsimMode == 2) || (batchsimMode == 3)
				SNR_offset_list = [SNR_offset, 1000]; % in channel freezing test mode, cuPHY needs a noise-free TV
			else
				SNR_offset_list = SNR_offset;
			end 
            testPerformance_prach('selected', SNR_offset_list, nFrame_prach_per_seed, falseDetectionTest, batchsimCfg);
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
