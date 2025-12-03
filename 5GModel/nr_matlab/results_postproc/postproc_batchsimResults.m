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

% Post-process perfMatchTest simulation results
function postproc_batchsimResults(top_folder, wsname_pattern, enable_proc_cuPHY_testBench_results, enable_proc_5gmodel_testBench_results)
    warning('off','all')
    if nargin == 0
        top_folder = '/home/Aerial-simulations/Phase2_perf/2022-11-08-20-43-57_mode_1';
        wsname_pattern = 'ws';
        enable_proc_cuPHY_testBench_results = 0;
        enable_proc_5gmodel_testBench_results = 1;
    elseif nargin == 1
        wsname_pattern = 'ws';
        enable_proc_cuPHY_testBench_results = 0;
        enable_proc_5gmodel_testBench_results = 1;
    elseif nargin == 2
        enable_proc_cuPHY_testBench_results = 0;
        enable_proc_5gmodel_testBench_results = 1;
    elseif nargin == 3
        enable_proc_5gmodel_testBench_results = 1;
    end    
    
    tmpSubFolders   = dir(top_folder);
    dirFlags        = [tmpSubFolders.isdir];
    subFolders      = tmpSubFolders(dirFlags); 
    subFolderNames  = {subFolders(3:end).name}; % Start at 3 to skip . and ..
    num_subFolders  = length(subFolderNames);
    
    % load CFG from testPerformance_xxx.m
    pusch_sim_cfg_fileName = fullfile(top_folder,'cfg_pusch_sim.csv');
    pucch_sim_cfg_fileName = fullfile(top_folder,'cfg_pucch_sim.csv');
    prach_sim_cfg_fileName = fullfile(top_folder,'cfg_prach_sim.csv');
    caseSetName = 'selected';
    if strcmp(caseSetName,'selected')
        TC_set_for_cfg_gen_pusch = 7001:7999;
        TC_set_for_cfg_gen_pucch = 6001:6999;
        TC_set_for_cfg_gen_prach = 5001:5999;
    elseif strcmp(caseSetName,'TR4')
        TC_set_for_cfg_gen_pusch = [7004:7006, 7012:7013, 7051:7054, 7059, 6003:6004, 6102, 6108, 6202, 6212, 6303:6304, 6308, 5003:5004];
        TC_set_for_cfg_gen_pucch = TC_set_for_cfg_gen_pusch;
        TC_set_for_cfg_gen_prach = TC_set_for_cfg_gen_pusch;
    end
    if ~isfile(pusch_sim_cfg_fileName)
        batchsimCfg.export_cfg = 1;
        batchsimCfg.export_fileName = pusch_sim_cfg_fileName;
        testPerformance_pusch(TC_set_for_cfg_gen_pusch, 0, 1, 0, batchsimCfg);
        cfg_table_pusch = readtable(pusch_sim_cfg_fileName);
    else
        cfg_table_pusch = readtable(pusch_sim_cfg_fileName);
    end
    if ~isfile(pucch_sim_cfg_fileName)
        batchsimCfg.export_cfg = 1;
        batchsimCfg.export_fileName = pucch_sim_cfg_fileName;
        testPerformance_pucch(TC_set_for_cfg_gen_pucch, 0, 1, 0, 0, batchsimCfg);
        cfg_table_pucch = readtable(pucch_sim_cfg_fileName);
    else
        cfg_table_pucch = readtable(pucch_sim_cfg_fileName);
    end
    if ~isfile(prach_sim_cfg_fileName)
        batchsimCfg.export_cfg = 1;
        batchsimCfg.export_fileName = prach_sim_cfg_fileName;
        testPerformance_prach(TC_set_for_cfg_gen_prach, 0, 1, 0, batchsimCfg);
        cfg_table_prach = readtable(prach_sim_cfg_fileName);
    else
        cfg_table_prach = readtable(prach_sim_cfg_fileName);
    end
    
    
    for idx_subFolder = 1:num_subFolders
        this_subFolderName = subFolderNames{idx_subFolder};    
        if contains(this_subFolderName, wsname_pattern) 
            fprintf('Processing %s...\n', this_subFolderName)
            this_subFolderFullPath = fullfile(top_folder,this_subFolderName);
            tmpSubscenariosFolders = dir(this_subFolderFullPath);
            dirFlags    = [tmpSubscenariosFolders.isdir];
            subscenariosFolders = tmpSubscenariosFolders(dirFlags);
            subscenariosFolderNames = {subscenariosFolders(3:end).name};
            cfg_all = cell2mat(cellfun(@(x) sscanf(x, 'scenario_TC%d___seed_%d___SNR_%f'),subscenariosFolderNames,'UniformOutput',false));
            unique_TCs = unique(cfg_all(1,:));
%             unique_TCs = [7004, 7005, 7006, 7012, 7013];%[7004, 7005, 7006, 7012, 7013]; %[7004, 7005, 7006, 7012]; %
            num_unique_TCs = length(unique_TCs);
            num_missing_subscenarios = 0;
            if contains(this_subFolderName, 'pusch')
                if contains(this_subFolderName, 'pusch_ul_measurement')
                    VariableTypes = {'double','string','string','double', 'double',    'double','double','cell',   'cell',             'cell',             'cell',      'cell'};
                    VariableNames = {'TC',    'FRC',   'Chan',  'rxAnt',  'targetSNR', 'CFO',   'delay', 'simSNR', 'error_preSNR_dB', 'error_postSNR_dB', 'cfoErrHz',  'toErrMicroSec'};                
                elseif contains(this_subFolderName, 'all_mcs')
                    VariableTypes = {'double','string','string','double', 'double',     'double','double','double',        'cell',   'cell'};
                    VariableNames = {'TC',    'FRC',   'Chan',  'rxAnt',  'exampleSNR', 'CFO',   'delay', 'min_SNR_in_LA', 'simSNR', 'BLER'};
                else
                    VariableTypes = {'double','string','string','double', 'double',    'double','double','double',                'double',                  'double',     'logical', 'cell',   'cell'};
                    VariableNames = {'TC',    'FRC',   'Chan',  'rxAnt',  'targetSNR', 'CFO',   'delay', 'ErrorRate_atTargetSNR', 'achieved_SNR_atTargetER', 'SNR_margin', 'Pass',    'simSNR', 'BLER'}; 
                end
            elseif contains(this_subFolderName, 'pucch') 
                tmp_extr = sscanf(this_subFolderName, 'ws_scenarios_pucchFormat%d_pucchTestMode%d');
                pucchFormat = tmp_extr(1);
                pucchTestMode = tmp_extr(2);
                if pucchTestMode==1
                    VariableTypes = {'double','double','double','string', 'double','double','double',     'double','double', 'string',                 'logical', 'cell',   'cell'};
                    VariableNames = {'TC',    'mu',    'PRB',   'Chan',   'rxAnt', 'nSym',  'targetSNR',  'CFO',   'delay',  'achieved_ACK_FalseAlarm', 'Pass',    'simSNR', 'BLER'}; 
                elseif pucchTestMode==2
                    VariableTypes = {'double','double','double','string', 'double','double','double',     'double','double', 'double',                           'double',                  'double',    'logical', 'cell',   'cell'};
                    VariableNames = {'TC',    'mu',    'PRB',   'Chan',   'rxAnt', 'nSym',  'targetSNR',  'CFO',   'delay',  'achieved_NACK2ACK_ER_atTargetSNR', 'achieved_SNR_atTargetER', 'SNR_margin','Pass',    'simSNR', 'BLER'};      
                elseif pucchTestMode==3
                    VariableTypes = {'double','double','double','string', 'double','double','double',     'double','double', 'double',                      'double',                      'double',     'logical', 'cell',   'cell'};
                    VariableNames = {'TC',    'mu',    'PRB',   'Chan',   'rxAnt', 'nSym',  'targetSNR',  'CFO',   'delay',  'achieved_ACK_MD_atTargetSNR', 'achieved_SNR_atTargetACK_MD', 'SNR_margin', 'Pass',    'simSNR', 'BLER'}; 
                elseif pucchTestMode==4
                    VariableTypes = {'double','double','double','string', 'double','double','double',     'double','double', 'double',                      'double',              'double',     'logical', 'cell',   'cell'};
                    VariableNames = {'TC',    'mu',    'PRB',   'Chan',   'rxAnt', 'nSym',  'targetSNR',  'CFO',   'delay',  'achieved_UCI_ER_atTargetSNR', 'achieved_SNR_atTargetUCI_ER', 'SNR_margin', 'Pass',    'simSNR', 'BLER'}; 
                elseif pucchTestMode==9
                    VariableTypes = {'double','double','double','string', 'double','double','double',     'double','double', 'string',   'string',          'string'         };
                    VariableNames = {'TC',    'mu',    'PRB',   'Chan',   'rxAnt', 'nSym',  'targetSNR',  'CFO',   'delay',  'simSNR',   'toErrMicroSec',   'error_preSNR_dB'};     
                end
            elseif contains(this_subFolderName, 'prach')
                tmp_extr = sscanf(this_subFolderName, 'ws_scenarios_prach_falseDetectionTest%d');
                falseDetectionTest = tmp_extr(1);
                if falseDetectionTest==1 % FA test
                    VariableTypes = {'double','double','double','double', 'double','double','double','double','string',     'double', 'double', 'double',     'string',                           'logical', 'cell',   'cell'};
                    VariableNames = {'TC',    'mu',    'cfg',   'root',   'zone',  'prmb',  'Nant',  'N_PRB', 'channelType','CFO',    'N_nc',   'targetSNR',  'achieved_FalseAlarm_atTargetSNR',  'Pass',    'simSNR', 'BLER'}; 
                else % MD test
                    VariableTypes = {'double','double','double','double', 'double','double','double','double','string',     'double', 'double', 'double',     'double',                  'double',                   'double',        'logical', 'cell',   'cell'};
                    VariableNames = {'TC',    'mu',    'cfg',   'root',   'zone',  'prmb',  'Nant',  'N_PRB', 'channelType','CFO',    'N_nc',   'targetSNR',  'achieved_MD_atTargetSNR', 'achieved_SNR_atTargetMD',  'SNR_margin_MD', 'Pass',    'simSNR', 'BLER'}; 
                end
            end
            num_columns = length(VariableNames);
            analysis_results_table = table('Size',[num_unique_TCs, num_columns],'VariableTypes', VariableTypes, 'VariableNames', VariableNames);
            analysis_results_table_cuphy = table('Size',[num_unique_TCs, num_columns],'VariableTypes', VariableTypes, 'VariableNames', VariableNames);
            for idx_TC = 1:num_unique_TCs
                this_TC = unique_TCs(idx_TC);
                unique_SNRs = unique(cfg_all(3,cfg_all(1,:)==this_TC));                 
                num_unique_SNRs = length(unique_SNRs);
                unique_SNRs_exclude1000 = unique_SNRs;
                unique_SNRs_exclude1000(unique_SNRs_exclude1000==1000) = []; 
                num_unique_SNRs_exclude1000 = length(unique_SNRs_exclude1000);
                % initialize variables
                list_SNRs       = nan*ones(100,1); % assuming maximum 100 SNRs per seed
                Cnt             = zeros(100,1);
                ErrorCnt        = zeros(100,1);
                sinrdB          = zeros(100,1);
                postEqSinrdB    = zeros(100,1);
                cfoErrHz        = zeros(100,1); 
                toErrMicroSec   = zeros(100,1); 
                pucch_error_timeOffset_usec  = nan*ones(100,1);
                pucch_error_preSNR_dB       = nan*ones(100,1);
                prachFalseCnt   = zeros(100,1);
                prachMissCnt    = zeros(100,1);
                % for cuPHY results
                Cnt_cuphy             = zeros(100,1);
                ErrorCnt_cuphy        = zeros(100,1);
                pucch_ErrorRate_cuphy = zeros(100,1);
                sinrdB_cuphy          = zeros(100,1);
                postEqSinrdB_cuphy    = zeros(100,1);
                cfoErrHz_cuphy        = zeros(100,1); 
                toErrMicroSec_cuphy   = zeros(100,1); 
                pucch_error_timeOffset_usec_cuphy  = nan*ones(100,1);
                pucch_error_preSNR_dB_cuphy        = nan*ones(100,1);
                prach_totCnt_cuphy      = zeros(100,1);
                prach_falseCnt_cuphy    = zeros(100,1);
                prach_missedCnt_cuphy   = zeros(100,1);
                for idx_SNR = 1:num_unique_SNRs
                    this_SNR = unique_SNRs(idx_SNR);
                    list_SNRs(idx_SNR) = this_SNR;
                    unique_seeds = unique(cfg_all(2,(cfg_all(1,:)==this_TC)&(cfg_all(3,:)==this_SNR)));
%                     unique_seeds = unique_seeds(1:1);
                    unique_seeds_save{idx_TC} = unique_seeds;
                    num_unique_seeds = length(unique_seeds);
                    tmp_pucch_error_timeOffset_usec = [];
                    tmp_pucch_error_preSNR_dB = [];
                    tmp_pucch_error_timeOffset_usec_cuphy = [];
                    tmp_pucch_error_preSNR_dB_cuphy = [];
                    for idx_seed = 1:num_unique_seeds
                        this_seed = unique_seeds(idx_seed);
                        this_subscenarioFolderName = subscenariosFolderNames{find((cfg_all(1,:)==this_TC)&(cfg_all(2,:)==this_seed)&(cfg_all(3,:)==this_SNR))};                        

                        if (this_SNR ~= 1000) && enable_proc_5gmodel_testBench_results% postproc sim results from 5GModel
                            % load and process results.mat
                            results_file_name = fullfile(this_subFolderFullPath,fullfile(this_subscenarioFolderName,'/results/results.mat'));
                            try
                                tmp_results = load(results_file_name);
                            catch
                                fprintf('Missing %s\n',results_file_name)
                                num_missing_subscenarios = num_missing_subscenarios+1;
                                continue;
                            end
                            if isfield(tmp_results, 'SysPar')
                                SysPar = tmp_results.SysPar;
                            elseif isfield(tmp_results, 'SysParShort')
                                SysPar = tmp_results.SysParShort;
                            else
                                error('Neither SysPar nor SysParShort found in %s\n', results_file_name);
                            end
                            results = SysPar.SimCtrl.results;
                            if contains(this_subFolderName, 'pusch') % post-proc PUSCH results
                                pusch = results.pusch;
                                Cnt(idx_SNR) = Cnt(idx_SNR) + pusch{1}.tbCnt;
                                if ismember(this_TC,[7051:7052]) % CSI part1
                                    ErrorCnt(idx_SNR) = ErrorCnt(idx_SNR) + pusch{1}.csi1ErrCnt;
                                elseif ismember(this_TC,[7053:7054]) % CSI part2
                                    ErrorCnt(idx_SNR) = ErrorCnt(idx_SNR) + pusch{1}.csi2ErrCnt;
                                elseif ismember(this_TC,[7911:7918]) % UL measurement
                                    sinrdB(idx_SNR) = sinrdB(idx_SNR) + sqrt(mean(abs(pusch{1}.sinrdB - pusch{1}.genie_sinrdB ).^2));
                                    postEqSinrdB(idx_SNR) = postEqSinrdB(idx_SNR) + sqrt(mean(abs(pusch{1}.postEqSinrdB - pusch{1}.genie_postEqSinrdB).^2));
                                    cfoErrHz(idx_SNR) = cfoErrHz(idx_SNR) + sqrt(mean(abs(pusch{1}.cfoEstHz - SysPar.Chan{1}.CFO).^2));
                                    toErrMicroSec(idx_SNR) = toErrMicroSec(idx_SNR) + sqrt(mean(abs(pusch{1}.toEstMicroSec - SysPar.Chan{1}.delay*1e6).^2));
                                else                                
                                    ErrorCnt(idx_SNR) = ErrorCnt(idx_SNR) + pusch{1}.tbErrorCnt;
                                end
                            elseif contains(this_subFolderName, 'pucch')
                                pucch = results.pucch;
                                Npucch = length(pucch);                            
                                for idxPucch = 1:Npucch
                                    if pucchTestMode == 4
                                        Cnt(idx_SNR) = pucch{idxPucch}.totalUciCnt + Cnt(idx_SNR);
                                    else
                                        Cnt(idx_SNR) = pucch{idxPucch}.totalCnt + Cnt(idx_SNR);
                                    end
%                                     switch pucchFormat
%                                         case {0, 1}
%                                             if pucchTestMode == 1
%                                                 ErrorCnt(idx_SNR) = ErrorCnt(idx_SNR) + pucch{idxPucch}.falseCnt;
%                                             elseif pucchTestMode == 2
%                                                 ErrorCnt(idx_SNR) = ErrorCnt(idx_SNR) + pucch{idxPucch}.nack2ack;
%                                             elseif pucchTestMode == 3
%                                                 ErrorCnt(idx_SNR) = ErrorCnt(idx_SNR) + pucch{idxPucch}.missack;
%                                             end
%                                         case{2, 3}
%                                             ErrorCnt(idx_SNR) = ErrorCnt(idx_SNR) + pucch{idxPucch}.errorCnt;
%                                     end
                                    if pucchTestMode == 1
                                        ErrorCnt(idx_SNR) = ErrorCnt(idx_SNR) + pucch{idxPucch}.falseCnt;
                                    elseif pucchTestMode == 2
                                        ErrorCnt(idx_SNR) = ErrorCnt(idx_SNR) + pucch{idxPucch}.nack2ack;
                                    elseif pucchTestMode == 3
                                        ErrorCnt(idx_SNR) = ErrorCnt(idx_SNR) + pucch{idxPucch}.missack;
                                    elseif pucchTestMode == 4
                                        ErrorCnt(idx_SNR) = ErrorCnt(idx_SNR) + pucch{idxPucch}.errorCnt;
                                    end

                                    if pucchTestMode == 9 % measurement
                                        tmp_pucch_error_timeOffset_usec = [tmp_pucch_error_timeOffset_usec, abs(pucch{idxPucch}.taEstMicroSec-SysPar.Chan{1}.delay*1e6).^2];
                                        tmp_pucch_error_preSNR_dB = [tmp_pucch_error_preSNR_dB, abs(pucch{idxPucch}.snrdB - SysPar.Chan{1}.SNR).^2]; 
                                    end
        
                                end
                            elseif contains(this_subFolderName, 'prach')
                                prach = results.prach{1};
                                Cnt(idx_SNR) = Cnt(idx_SNR) + prach.totalCnt;
                                prachFalseCnt(idx_SNR) = prachFalseCnt(idx_SNR) + prach.falseCnt;
                                prachMissCnt(idx_SNR) = prachMissCnt(idx_SNR) + prach.missCnt;
                            end

                        elseif this_SNR == 1000 % load sim results from cuPHY
                            
                            if contains(this_subFolderName, 'pusch') % load and post-proc PUSCH results
                                results_file_name = fullfile(this_subFolderFullPath,fullfile(this_subscenarioFolderName,'/cuphyBlerCurve.h5'));
                                try
                                    snr_cuphy = round(double(h5read(results_file_name, '/snr'))*1000)/1000;
                                    if ismember(this_TC,[7051:7052]) % CSI part1
                                        tbCnt_cuphy = double(h5read(results_file_name, '/numCsi1Ucis'));
                                        tbErrorCnt_cuphy = double(h5read(results_file_name, '/numErrCsi1Ucis'));
                                    elseif ismember(this_TC,[7053:7054]) % CSI part2
                                        tbCnt_cuphy = double(h5read(results_file_name, '/numCsi2Ucis'));
                                        tbErrorCnt_cuphy = double(h5read(results_file_name, '/numErrCsi2Ucis'));
                                    else
                                        tbCnt_cuphy = double(h5read(results_file_name, '/numTBs'));
                                        tbErrorCnt_cuphy = double(h5read(results_file_name, '/numErrTBs'));
                                    end
                                catch
                                    fprintf('Missing %s\n',results_file_name)
                                    num_missing_subscenarios = num_missing_subscenarios+1;
                                    continue;
                                end

                                for idx_SNR_cuphy = 1:num_unique_SNRs_exclude1000
                                    this_snr_cuphy = unique_SNRs_exclude1000(idx_SNR_cuphy);
                                    idx_catch = find(snr_cuphy==this_snr_cuphy);
                                    Cnt_cuphy(idx_SNR_cuphy) = Cnt_cuphy(idx_SNR_cuphy) + tbCnt_cuphy(idx_catch);
                                    if ismember(this_TC,[7051:7052]) % CSI part1
                                        ErrorCnt_cuphy(idx_SNR_cuphy) = ErrorCnt_cuphy(idx_SNR_cuphy) + tbErrorCnt_cuphy(idx_catch);
                                    elseif ismember(this_TC,[7053:7054]) % CSI part2
                                        ErrorCnt_cuphy(idx_SNR_cuphy) = ErrorCnt_cuphy(idx_SNR_cuphy) + tbErrorCnt_cuphy(idx_catch);
                                    elseif ismember(this_TC,[7911:7918]) % UL measurement
                                        error('undefined yet!')
                                        sinrdB_cuphy(idx_SNR_cuphy) = sinrdB_cuphy(idx_SNR_cuphy) + sqrt(mean(abs(pusch{1}.sinrdB - pusch{1}.genie_sinrdB ).^2));
                                        postEqSinrdB_cuphy(idx_SNR_cuphy) = postEqSinrdB_cuphy(idx_SNR_cuphy) + sqrt(mean(abs(pusch{1}.postEqSinrdB - pusch{1}.genie_postEqSinrdB).^2));
                                        cfoErrHz_cuphy(idx_SNR_cuphy) = cfoErrHz_cuphy(idx_SNR_cuphy) + sqrt(mean(abs(pusch{1}.cfoEstHz - SysPar.Chan{1}.CFO).^2));
                                        toErrMicroSec_cuphy(idx_SNR_cuphy) = toErrMicroSec_cuphy(idx_SNR_cuphy) + sqrt(mean(abs(pusch{1}.toEstMicroSec - SysPar.Chan{1}.delay*1e6).^2));
                                    else                                
                                        ErrorCnt_cuphy(idx_SNR_cuphy) = ErrorCnt_cuphy(idx_SNR_cuphy) + tbErrorCnt_cuphy(idx_catch);
                                    end
                                end
                                
                            elseif contains(this_subFolderName, 'pucch')
                                results_file_name = fullfile(this_subFolderFullPath,fullfile(this_subscenarioFolderName,'/cuphyPerfCurve.h5'));
                                try
                                    snr_cuphy = round(double(h5read(results_file_name, '/snr'))*1000)/1000;
                                    perf_cuphy = double(h5read(results_file_name, '/perf'));
                                catch
                                    fprintf('Missing %s\n',results_file_name)
                                    num_missing_subscenarios = num_missing_subscenarios+1;
                                    continue;
                                end

                                for idx_SNR_cuphy = 1:num_unique_SNRs_exclude1000
                                    this_snr_cuphy = unique_SNRs_exclude1000(idx_SNR_cuphy);
                                    idx_catch = find(snr_cuphy==this_snr_cuphy);
                                    pucch_ErrorRate_cuphy(idx_SNR_cuphy) = pucch_ErrorRate_cuphy(idx_SNR_cuphy) + perf_cuphy(idx_catch);
                                end

                            elseif contains(this_subFolderName, 'prach')
                                results_file_name = fullfile(this_subFolderFullPath,fullfile(this_subscenarioFolderName,'/cuphyPerfCurve.h5'));
                                try
                                    snr_cuphy = round(double(h5read(results_file_name, '/snr'))*1000)/1000;
                                    tmp_prach_totCnt_cuphy = double(h5read(results_file_name, '/numTotCnt'));
                                    tmp_prach_falseCnt_cuphy = double(h5read(results_file_name, '/numFalseDet'));
                                    tmp_prach_missedCnt_cuphy = double(h5read(results_file_name, '/numMiss'));
                                catch
                                    fprintf('Missing %s\n',results_file_name)
                                    num_missing_subscenarios = num_missing_subscenarios+1;
                                    continue;
                                end

                                for idx_SNR_cuphy = 1:num_unique_SNRs_exclude1000
                                    this_snr_cuphy = unique_SNRs_exclude1000(idx_SNR_cuphy);
                                    idx_catch = find(snr_cuphy==this_snr_cuphy);
                                    prach_totCnt_cuphy(idx_SNR_cuphy) = prach_totCnt_cuphy(idx_SNR_cuphy) + tmp_prach_totCnt_cuphy(idx_catch);
                                    prach_falseCnt_cuphy(idx_SNR_cuphy) = prach_falseCnt_cuphy(idx_SNR_cuphy) + tmp_prach_falseCnt_cuphy(idx_catch);
                                    prach_missedCnt_cuphy(idx_SNR_cuphy) = prach_missedCnt_cuphy(idx_SNR_cuphy) + tmp_prach_missedCnt_cuphy(idx_catch);
                                end
                            end
                           

                        else
%                             error('Unexpected SNR values!')
                        end
                        
                    end % end of idx_seed
                    pucch_error_timeOffset_usec(idx_SNR) = sqrt(mean(tmp_pucch_error_timeOffset_usec));
                    pucch_error_preSNR_dB(idx_SNR) = sqrt(mean(tmp_pucch_error_preSNR_dB));
                end % end of idx_SNR
                if enable_proc_5gmodel_testBench_results
                    sinrdB = sinrdB/num_unique_seeds;
                    postEqSinrdB = postEqSinrdB/num_unique_seeds;
                    cfoErrHz = cfoErrHz/num_unique_seeds;
                    toErrMicroSec = toErrMicroSec/num_unique_seeds;
                    ErrorRate = ErrorCnt(1:num_unique_SNRs_exclude1000)./Cnt(1:num_unique_SNRs_exclude1000);
                    prachFalseRate = prachFalseCnt(1:num_unique_SNRs_exclude1000)./Cnt(1:num_unique_SNRs_exclude1000);
                    prachMissRate = prachMissCnt(1:num_unique_SNRs_exclude1000)./Cnt(1:num_unique_SNRs_exclude1000);
                end

                % for cuphy sim results
                if enable_proc_cuPHY_testBench_results
                    sinrdB_cuphy = sinrdB_cuphy/num_unique_seeds;
                    postEqSinrdB_cuphy = postEqSinrdB_cuphy/num_unique_seeds;
                    cfoErrHz_cuphy = cfoErrHz_cuphy/num_unique_seeds;
                    toErrMicroSec_cuphy = toErrMicroSec_cuphy/num_unique_seeds;
                    ErrorRate_cuphy = ErrorCnt_cuphy(1:num_unique_SNRs_exclude1000)./Cnt_cuphy(1:num_unique_SNRs_exclude1000);                
                    pucch_ErrorRate_cuphy = pucch_ErrorRate_cuphy(1:num_unique_SNRs_exclude1000)./num_unique_seeds;
                    prach_falseRate_cuphy = prach_falseCnt_cuphy(1:num_unique_SNRs_exclude1000)./prach_totCnt_cuphy(1:num_unique_SNRs_exclude1000);
                    prach_missedRate_cuphy = prach_missedCnt_cuphy(1:num_unique_SNRs_exclude1000)./prach_totCnt_cuphy(1:num_unique_SNRs_exclude1000);
                end
                list_SNRs = list_SNRs(1:num_unique_SNRs_exclude1000); 
                SNRs = list_SNRs(1:num_unique_SNRs_exclude1000);     
                SNRs_cuphy = list_SNRs(1:num_unique_SNRs_exclude1000);  
                if contains(this_subFolderName, 'pusch')
                    num_cols_cfg = size(cfg_table_pusch,2);
                    analysis_results_table(idx_TC,1:num_cols_cfg) = cfg_table_pusch(cfg_table_pusch.TC==this_TC,:);
                    analysis_results_table_cuphy(idx_TC,1:num_cols_cfg) = cfg_table_pusch(cfg_table_pusch.TC==this_TC,:);
                    if contains(this_subFolderName, 'pusch_ul_measurement')
                        SNRs = list_SNRs(~isnan(list_SNRs)).';
                        sinrdB = round(sinrdB(~isnan(list_SNRs)).'*100)/100;
                        postEqSinrdB = round(postEqSinrdB(~isnan(list_SNRs)).'*100)/100;
                        cfoErrHz = round(cfoErrHz(~isnan(list_SNRs)).'*100)/100;
                        toErrMicroSec = round(toErrMicroSec(~isnan(list_SNRs)).'*100)/100;
                        analysis_results_table(idx_TC,num_cols_cfg+1) = {regexprep(num2str(SNRs),'\s+',', ')};
                        analysis_results_table(idx_TC,num_cols_cfg+2) = {regexprep(num2str(sinrdB),'\s+',', ')};
                        analysis_results_table(idx_TC,num_cols_cfg+3) = {regexprep(num2str(postEqSinrdB),'\s+',', ')};
                        analysis_results_table(idx_TC,num_cols_cfg+4) = {regexprep(num2str(cfoErrHz),'\s+',', ')};
                        analysis_results_table(idx_TC,num_cols_cfg+5) = {regexprep(num2str(toErrMicroSec),'\s+',', ')};
                    elseif contains(this_subFolderName, 'all_mcs')
                        if enable_proc_5gmodel_testBench_results
                            ErrorRate_raw = ErrorRate;
                            SNRs_raw = SNRs;
                            [ErrorRate, idx_unique] = unique(ErrorRate);
                            SNRs = SNRs(idx_unique);                        
                            target_ErrorRate_for_LA = 0.1;
                            if length(ErrorRate)==1
                                achieved_SNR = -inf;
                                warning(sprintf('All ErrorRate has the same value, thus impossible for interpolation. Will set achieved_SNR to -INF. %2.2f in %s, TC%d\n', ErrorRate, this_subFolderName, this_TC))
                            else
                                achieved_SNR = interp1(ErrorRate, SNRs,target_ErrorRate_for_LA,'linear');
                            end
                            if isnan(achieved_SNR)
                                achieved_SNR = interp1(ErrorRate, SNRs,target_ErrorRate_for_LA,'linear','extrap');
                                warning(sprintf('Extrapolation in %s, TC%d\n', this_subFolderName, this_TC))
                            end   
                            achieved_SNR = round(achieved_SNR*100)/100; % keep just two fractional digits
                            analysis_results_table(idx_TC,num_cols_cfg+1) = {achieved_SNR};
                            analysis_results_table(idx_TC,num_cols_cfg+2) = {regexprep(num2str(SNRs.'),'\s+',', ')};
                            analysis_results_table(idx_TC,num_cols_cfg+3) = {regexprep(num2str(ErrorRate.'),'\s+',', ')};
                        end
                        % cuphy
                        if enable_proc_cuPHY_testBench_results
                            ErrorRate_cuphy = ErrorRate_cuphy(~isnan(ErrorRate_cuphy)).';
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+1) = {regexprep(num2str(SNRs),'\s+',', ')};
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+2) = {regexprep(num2str(ErrorRate_cuphy),'\s+',', ')};
                        end
                    else
                        if contains(this_subFolderName, 'pusch_0p00001bler')
                            target_ErrorRate = 1e-5;
                        elseif contains(this_subFolderName, 'pusch_uci')
                            if ismember(this_TC,[7051:7052]) % CSI part1
                                target_ErrorRate = 0.001;
                            elseif ismember(this_TC,[7053:7054]) % CSI part2
                                target_ErrorRate = 0.01;
                            else
                                error('Undefined!')
                            end
                        else
                            target_ErrorRate = 0.3;
                        end
                        if enable_proc_5gmodel_testBench_results
                            ErrorRate_raw = ErrorRate;
                            SNRs_raw = SNRs;
                            [ErrorRate, idx_unique] = unique(ErrorRate);
                            SNRs = SNRs(idx_unique);
                            if length(ErrorRate)==1
                                achieved_SNR = -inf;
                                warning(sprintf('All ErrorRate is has the same value, thus impossible for interpolation. Will set achieved_SNR to -INF. %2.2f in %s, TC%d\n', ErrorRate, this_subFolderName, this_TC))
                            else
                                achieved_SNR = interp1(ErrorRate, SNRs,target_ErrorRate,'linear');
                            end
                            if isnan(achieved_SNR)
                                achieved_SNR = interp1(ErrorRate, SNRs,target_ErrorRate,'linear','extrap');
                                warning(sprintf('Extrapolation in %s, TC%d\n', this_subFolderName, this_TC))
                            end                    
                            achieved_SNR = round(achieved_SNR*1000)/1000; % keep just three fractional digits
                            analysis_results_table(idx_TC,1:7) = cfg_table_pusch(cfg_table_pusch.TC==this_TC,:);
                            SNR_3GPP_target = cfg_table_pusch.SNR(cfg_table_pusch.TC==this_TC);
                            achieved_ER_atTargetSNR = ErrorRate_raw(SNRs_raw==SNR_3GPP_target);
                            analysis_results_table(idx_TC,8) = {achieved_ER_atTargetSNR};
                            analysis_results_table(idx_TC,9) = {achieved_SNR};                    
                            analysis_results_table(idx_TC,10)  = {SNR_3GPP_target - achieved_SNR};
                            analysis_results_table(idx_TC,11) = {achieved_SNR<=SNR_3GPP_target};
                            analysis_results_table(idx_TC,12) = {regexprep(num2str(SNRs_raw.'),'\s+',', ')}; % simSNR
                            analysis_results_table(idx_TC,13) = {regexprep(num2str(ErrorRate_raw.'),'\s+',', ')}; % BLER
                        end

                        % cuphy
                        if enable_proc_cuPHY_testBench_results
                            ErrorRate_raw_cuphy = ErrorRate_cuphy;
                            SNRs_raw_cuphy = SNRs_cuphy;
                            [ErrorRate_cuphy, idx_unique] = unique(ErrorRate_cuphy);
                            SNRs_cuphy = SNRs_cuphy(idx_unique);                        
                            if length(ErrorRate_cuphy)==1
                                achieved_SNR_cuphy = -inf;
                                warning(sprintf('All ErrorRate is has the same value, thus impossible for interpolation. Will set achieved_SNR to -INF. %2.2f in %s, TC%d\n', ErrorRate_cuphy, this_subFolderName, this_TC))
                            else
                                achieved_SNR_cuphy = interp1(ErrorRate_cuphy, SNRs_cuphy,target_ErrorRate,'linear');
                            end
                            if isnan(achieved_SNR_cuphy)
                                achieved_SNR_cuphy = interp1(ErrorRate_cuphy, SNRs_cuphy,target_ErrorRate,'linear','extrap');
                                warning(sprintf('Extrapolation in %s, TC%d\n', this_subFolderName, this_TC))
                            end                    
                            achieved_SNR_cuphy = round(achieved_SNR_cuphy*1000)/1000; % keep just three fractional digits
                            analysis_results_table_cuphy(idx_TC,1:7) = cfg_table_pusch(cfg_table_pusch.TC==this_TC,:);
                            SNR_3GPP_target = cfg_table_pusch.SNR(cfg_table_pusch.TC==this_TC);
                            achieved_ER_atTargetSNR_cuphy = ErrorRate_raw_cuphy(SNRs_raw_cuphy==SNR_3GPP_target);
                            analysis_results_table_cuphy(idx_TC,8) = {achieved_ER_atTargetSNR_cuphy};
                            analysis_results_table_cuphy(idx_TC,9) = {achieved_SNR_cuphy};                    
                            analysis_results_table_cuphy(idx_TC,10)  = {SNR_3GPP_target - achieved_SNR_cuphy};
                            analysis_results_table_cuphy(idx_TC,11) = {achieved_SNR_cuphy<=SNR_3GPP_target};
                            analysis_results_table_cuphy(idx_TC,12) = {regexprep(num2str(SNRs_raw_cuphy.'),'\s+',', ')}; % simSNR
                            analysis_results_table_cuphy(idx_TC,13) = {regexprep(num2str(ErrorRate_raw_cuphy.'),'\s+',', ')}; % BLER
                        end
                    end
                elseif contains(this_subFolderName, 'pucch') 
                    if pucchTestMode == 1
                        target_ErrorRate = 0.01;
                    elseif pucchTestMode == 2
                        target_ErrorRate = 0.001;
                    elseif pucchTestMode == 3
                        target_ErrorRate = 0.01;
                    elseif pucchTestMode == 4 % UCI Bler
                        target_ErrorRate = 0.01;
                    end                
                    % writing results to table
                    num_cols_cfg = size(cfg_table_pucch,2);
                    analysis_results_table(idx_TC,1:num_cols_cfg) = cfg_table_pucch(cfg_table_pucch.TC==this_TC,:);
                    analysis_results_table_cuphy(idx_TC,1:num_cols_cfg) = cfg_table_pucch(cfg_table_pucch.TC==this_TC,:);
                    if ismember(pucchTestMode, [1]) % 1: DTX2ACK 
                        SNRs_raw = SNRs;
                        if enable_proc_5gmodel_testBench_results
                            ErrorRate_raw = ErrorRate;
                            [ErrorRate, idx_unique] = unique(ErrorRate);
                            SNRs = SNRs(idx_unique);
                            analysis_results_table(idx_TC,num_cols_cfg+1) = {regexprep(num2str(ErrorRate.'),'\s+',', ')};  
                            analysis_results_table(idx_TC,num_cols_cfg+2) = {all(ErrorRate<=target_ErrorRate)};
                            analysis_results_table(idx_TC,num_cols_cfg+3) = {regexprep(num2str(SNRs_raw.'),'\s+',', ')}; % simSNR
                            analysis_results_table(idx_TC,num_cols_cfg+4) = {regexprep(num2str(ErrorRate_raw.'),'\s+',', ')}; % BLER
                        end
                        % cuPHY
                        if enable_proc_cuPHY_testBench_results
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+1) = {regexprep(num2str(pucch_ErrorRate_cuphy.'),'\s+',', ')};  
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+2) = {all(pucch_ErrorRate_cuphy<=target_ErrorRate)};
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+3) = {regexprep(num2str(SNRs_raw.'),'\s+',', ')}; % simSNR
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+4) = {regexprep(num2str(pucch_ErrorRate_cuphy.'),'\s+',', ')}; % BLER
                        end
                    elseif ismember(pucchTestMode, [2, 3, 4]) % 2: NACK2ACK, 3: ACK miss detection rate, 4: UCI Bler
                        SNRs_raw = SNRs;
                        if enable_proc_5gmodel_testBench_results
                            ErrorRate_raw = ErrorRate;
                            [ErrorRate, idx_unique] = unique(ErrorRate);
                            SNRs = SNRs(idx_unique);
                            if length(ErrorRate)==1
                                achieved_SNR = -inf;
                                warning(sprintf('All ErrorRate is has the same value, thus impossible for interpolation. Will set achieved_SNR to -INF. %2.2f in %s, TC%d\n', ErrorRate, this_subFolderName, this_TC))
                            else
                                achieved_SNR = interp1(ErrorRate, SNRs,target_ErrorRate,'linear');
                            end
                            if isnan(achieved_SNR)
                                achieved_SNR = interp1(ErrorRate, SNRs,target_ErrorRate,'linear','extrap');
                                warning(sprintf('Extrapolation in %s, TC%d\n', this_subFolderName, this_TC))
                            end
                            achieved_SNR = round(achieved_SNR*100)/100; % keep just two fractional digits
                            SNR_3GPP_target = cfg_table_pucch.SNR(cfg_table_pucch.TC==this_TC);
                            achieved_ER_atTargetSNR = ErrorRate_raw(SNRs_raw==SNR_3GPP_target);
                            analysis_results_table(idx_TC,num_cols_cfg+1) = {achieved_ER_atTargetSNR};
                            analysis_results_table(idx_TC,num_cols_cfg+2) = {achieved_SNR};
                            analysis_results_table(idx_TC,num_cols_cfg+3)  = {SNR_3GPP_target - achieved_SNR};
                            analysis_results_table(idx_TC,num_cols_cfg+4) = {achieved_SNR<=SNR_3GPP_target};
                            analysis_results_table(idx_TC,num_cols_cfg+5) = {regexprep(num2str(SNRs_raw.'),'\s+',', ')}; % simSNR
                            analysis_results_table(idx_TC,num_cols_cfg+6) = {regexprep(num2str(ErrorRate_raw.'),'\s+',', ')}; % BLER
                        end

                        % cuPHY
                        if enable_proc_cuPHY_testBench_results
                            pucch_ErrorRate_raw_cuphy = pucch_ErrorRate_cuphy;
                            [ErrorRate, idx_unique] = unique(pucch_ErrorRate_raw_cuphy);
                            SNRs = SNRs_raw(idx_unique);
                            if length(ErrorRate)==1
                                achieved_SNR = -inf;
                                warning(sprintf('All ErrorRate is has the same value, thus impossible for interpolation. Will set achieved_SNR to -INF. %2.2f in %s, TC%d\n', ErrorRate, this_subFolderName, this_TC))
                            else
                                achieved_SNR = interp1(ErrorRate, SNRs,target_ErrorRate,'linear');
                            end
                            if isnan(achieved_SNR)
                                achieved_SNR = interp1(ErrorRate, SNRs,target_ErrorRate,'linear','extrap');
                                warning(sprintf('Extrapolation in %s, TC%d\n', this_subFolderName, this_TC))
                            end
                            achieved_SNR = round(achieved_SNR*100)/100; % keep just two fractional digits
                            SNR_3GPP_target = cfg_table_pucch.SNR(cfg_table_pucch.TC==this_TC);
                            achieved_ER_atTargetSNR = pucch_ErrorRate_raw_cuphy(SNRs_raw==SNR_3GPP_target);
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+1) = {achieved_ER_atTargetSNR};
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+2) = {achieved_SNR};
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+3)  = {SNR_3GPP_target - achieved_SNR};
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+4) = {achieved_SNR<=SNR_3GPP_target};
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+5) = {regexprep(num2str(SNRs_raw.'),'\s+',', ')}; % simSNR
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+6) = {regexprep(num2str(pucch_ErrorRate_raw_cuphy.'),'\s+',', ')}; % BLER
                        end
                    elseif pucchTestMode==9 % measurement
                        pucch_error_timeOffset_usec = round(pucch_error_timeOffset_usec(~isnan(SNRs))*100)/100;
                        pucch_error_preSNR_dB = round(pucch_error_preSNR_dB(~isnan(pucch_error_preSNR_dB))*100)/100;
                        analysis_results_table(idx_TC,num_cols_cfg+1) = {regexprep(num2str(SNRs.'),'\s+',', ')};
                        analysis_results_table(idx_TC,num_cols_cfg+2) = {regexprep(num2str(pucch_error_timeOffset_usec.'),'\s+',', ')}; 
                        analysis_results_table(idx_TC,num_cols_cfg+3) = {regexprep(num2str(pucch_error_preSNR_dB.'),'\s+',', ')}; 
                    end
                                                            
                elseif contains(this_subFolderName, 'prach')
                    SNRs_raw = SNRs;
                    if enable_proc_5gmodel_testBench_results
                        prachMissRate_raw = prachMissRate;
                        [prachMissRate, idx_unique] = unique(prachMissRate);
                        SNRs = SNRs(idx_unique);
                    end                    
                    target_falseErrorRate = 0.001;
                    target_missErrorRate = 0.01;
    
                    SNR_3GPP_target = cfg_table_prach.SNR(cfg_table_prach.TC==this_TC);
                    idx_with_target_SNR = find(SNRs_raw==SNR_3GPP_target);
                    num_cols_cfg = size(cfg_table_prach,2);
                    analysis_results_table(idx_TC,1:num_cols_cfg) = cfg_table_prach(cfg_table_prach.TC==this_TC,:);
                    analysis_results_table_cuphy(idx_TC,1:num_cols_cfg) = cfg_table_prach(cfg_table_prach.TC==this_TC,:);
                    if falseDetectionTest==1 % FA test
                        if enable_proc_5gmodel_testBench_results
                            analysis_results_table(idx_TC,num_cols_cfg+1) = {regexprep(num2str(prachFalseRate(idx_with_target_SNR).'),'\s+',', ')};
                            analysis_results_table(idx_TC,num_cols_cfg+2) = {all(prachFalseRate(idx_with_target_SNR)<=target_falseErrorRate)};
                            analysis_results_table(idx_TC,num_cols_cfg+3) = {regexprep(num2str(SNRs_raw.'),'\s+',', ')}; % simSNR
                            analysis_results_table(idx_TC,num_cols_cfg+4) = {regexprep(num2str(prachFalseRate.'),'\s+',', ')}; % BLER
                        end

                        % cuPHY
                        if enable_proc_cuPHY_testBench_results
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+1) = {regexprep(num2str(prach_falseRate_cuphy(idx_with_target_SNR).'),'\s+',', ')};
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+2) = {all(prach_falseRate_cuphy(idx_with_target_SNR)<=target_falseErrorRate)};
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+3) = {regexprep(num2str(SNRs_raw.'),'\s+',', ')}; % simSNR
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+4) = {regexprep(num2str(prach_falseRate_cuphy.'),'\s+',', ')}; % BLER
                        end
                    elseif falseDetectionTest==0 % MD test
                        if enable_proc_5gmodel_testBench_results
                            if length(prachMissRate)==1
                                achieved_SNR_missError = -inf;
                                warning(sprintf('All ErrorRate is has the same value, thus impossible for interpolation. Will set achieved_SNR to -INF. %2.2f in %s, TC%d\n', prachMissRate, this_subFolderName, this_TC))
                            else
                                achieved_SNR_missError = interp1(prachMissRate, SNRs,target_missErrorRate,'linear');
                            end
                            if isnan(achieved_SNR_missError)
                                achieved_SNR_missError = interp1(prachMissRate, SNRs,target_missErrorRate,'linear','extrap');
                                warning(sprintf('Extrapolation in %s, TC%d\n', this_subFolderName, this_TC))
                            end                            
                            achieved_ER_atTargetSNR = prachMissRate_raw(SNRs_raw==SNR_3GPP_target);
                            achieved_SNR_missError = round(achieved_SNR_missError*100)/100; % keep just two fractional digits
                            analysis_results_table(idx_TC,num_cols_cfg+1) = {achieved_ER_atTargetSNR};
                            analysis_results_table(idx_TC,num_cols_cfg+2) = {achieved_SNR_missError};
                            analysis_results_table(idx_TC,num_cols_cfg+3) = {SNR_3GPP_target - achieved_SNR_missError};
                            analysis_results_table(idx_TC,num_cols_cfg+4) = {achieved_SNR_missError<=SNR_3GPP_target};
                            analysis_results_table(idx_TC,num_cols_cfg+5) = {regexprep(num2str(SNRs_raw.'),'\s+',', ')}; % simSNR
                            analysis_results_table(idx_TC,num_cols_cfg+6) = {regexprep(num2str(prachMissRate_raw.'),'\s+',', ')}; % BLER
                        end
                        %cuPHY
                        if enable_proc_cuPHY_testBench_results
                            prach_missedRate_cuphy_raw = prach_missedRate_cuphy;
                            [prach_missedRate_cuphy_unique,idx] = unique(prach_missedRate_cuphy);
                            SNRs_raw = SNRs;
                            SNRs = SNRs(idx);
                            if length(prach_missedRate_cuphy)==1
                                achieved_SNR_missError_cuphy = -inf;
                                warning(sprintf('All ErrorRate is has the same value, thus impossible for interpolation. Will set achieved_SNR to -INF. %2.2f in %s, TC%d\n', prachMissRate, this_subFolderName, this_TC))
                            else
                                achieved_SNR_missError_cuphy = interp1(prach_missedRate_cuphy_unique, SNRs,target_missErrorRate,'linear');
                            end
                            if isnan(achieved_SNR_missError_cuphy)
                                achieved_SNR_missError_cuphy = interp1(prach_missedRate_cuphy_unique, SNRs,target_missErrorRate,'linear','extrap');
                                warning(sprintf('Extrapolation in %s, TC%d\n', this_subFolderName, this_TC))
                            end                            
                            achieved_ER_atTargetSNR_cuphy = prach_missedRate_cuphy(SNRs_raw==SNR_3GPP_target);
                            achieved_SNR_missError_cuphy = round(achieved_SNR_missError_cuphy*100)/100; % keep just two fractional digits
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+1) = {achieved_ER_atTargetSNR_cuphy};
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+2) = {achieved_SNR_missError_cuphy};
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+3) = {SNR_3GPP_target - achieved_SNR_missError_cuphy};
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+4) = {achieved_SNR_missError_cuphy<=SNR_3GPP_target};
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+5) = {regexprep(num2str(SNRs_raw.'),'\s+',', ')}; % simSNR
                            analysis_results_table_cuphy(idx_TC,num_cols_cfg+6) = {regexprep(num2str(prach_missedRate_cuphy.'),'\s+',', ')}; % BLER
                        end
                    end
                    
                else
                    error('Undefined!')
                end
                    
                            
            end
            if contains(this_subFolderName,'pusch')
                writetable(analysis_results_table,fullfile(top_folder,'analysis_results_pusch_5gmodel.xlsx'),'Sheet',this_subFolderName(14:end));
                if enable_proc_cuPHY_testBench_results
                    writetable(analysis_results_table_cuphy,fullfile(top_folder,'analysis_results_pusch_cuphy.xlsx'),'Sheet',this_subFolderName(14:end));
                end
            elseif contains(this_subFolderName,'pucch')
                writetable(analysis_results_table,fullfile(top_folder,'analysis_results_pucch_5gmodel.xlsx'),'Sheet',this_subFolderName(14:end));
                if enable_proc_cuPHY_testBench_results
                    writetable(analysis_results_table_cuphy,fullfile(top_folder,'analysis_results_pucch_cuphy.xlsx'),'Sheet',this_subFolderName(14:end));
                end
            elseif contains(this_subFolderName,'prach') 
                writetable(analysis_results_table,fullfile(top_folder,'analysis_results_prach_5gmodel.xlsx'),'Sheet',this_subFolderName(14:end));
                if enable_proc_cuPHY_testBench_results
                    writetable(analysis_results_table_cuphy,fullfile(top_folder,'analysis_results_prach_cuphy.xlsx'),'Sheet',this_subFolderName(14:end));
                end
            else
                writetable(analysis_results_table,fullfile(top_folder,'analysis_results_unknown.xlsx'),'Sheet',this_subFolderName(14:end));
            end
    
            % make some plots
            enable_plot = 1;
            if enable_plot
                font_size = 24;
                num_TCs = size(analysis_results_table,1);
%                 list_color = distinguishable_colors(num_TCs);
                list_color = linspecer(num_TCs);
                fig_handle = figure;
                if contains(this_subFolderName, 'pusch_data') || contains(this_subFolderName, 'pusch_tf') || contains(this_subFolderName, 'pusch_uci') || contains(this_subFolderName, 'pusch_bler_tdl_cdl')
                    for idx_TC = 1:num_TCs
                        if ~isempty(analysis_results_table.BLER{idx_TC})
                            semilogy(str2num(analysis_results_table.simSNR{idx_TC}), str2num(analysis_results_table.BLER{idx_TC}),'DisplayName',sprintf('TC %d, 5GModel',analysis_results_table.TC(idx_TC)),'LineWidth',3, 'Color',list_color(idx_TC,:))
                            hold on;
                        end
                    end
                    % cuphy
                    if enable_proc_cuPHY_testBench_results
                        for idx_TC = 1:num_TCs
                            if ~isempty(analysis_results_table.BLER{idx_TC})
                                semilogy(str2num(analysis_results_table_cuphy.simSNR{idx_TC}), str2num(analysis_results_table_cuphy.BLER{idx_TC}),'DisplayName',sprintf('TC %d, cuPHY',analysis_results_table_cuphy.TC(idx_TC)),'LineWidth',3, 'Color',list_color(idx_TC,:),'LineStyle','--')
                                hold on;
                            end
                        end
                    end
%                     ylim([1e-3,1])
%                     xlim([-10, 35])
                    xlabel('SNR [dB]')
                    ylabel('TBer')
                    grid minor;
%                     title(this_subFolderName, 'Interpreter', 'none')
%                     title(sprintf('%s, seed: [%s]', this_subFolderName, regexprep(num2str(unique_seeds_save{1} ),'\s+',', ')), 'Interpreter', 'none')
                    title(sprintf('%s, num of seeds: %d', this_subFolderName, length(unique_seeds_save{1}) ), 'Interpreter', 'none')
                    legend show;
                    my_legend = findobj(gcf,'Tag','legend');                    
                    set(findall(gcf,'-property','FontSize'),'FontSize',font_size) 
                    set(my_legend, 'FontSize', 18, 'Interpreter', 'none', 'Location', 'southeast')

                elseif contains(this_subFolderName, 'all_mcs') 
                    % plot BLER curves
                    for idx_TC = 1:num_TCs
                        if ~isempty(analysis_results_table.BLER{idx_TC})
                            semilogy(str2num(analysis_results_table.simSNR{idx_TC}), str2num(analysis_results_table.BLER{idx_TC}),'DisplayName',analysis_results_table.FRC{idx_TC},'LineWidth',3, 'Color',list_color(idx_TC,:))
                            hold on;
                        end
                    end
                    ylim([1e-3,1])
                    xlim([-10, 35])
                    xlabel('SNR [dB]')
                    ylabel('TBer')
                    grid minor;
                    title(this_subFolderName, 'Interpreter', 'none')
                    legend show;
                    my_legend = findobj(gcf,'Tag','legend');                    
                    set(findall(gcf,'-property','FontSize'),'FontSize',font_size) 
                    set(my_legend, 'FontSize', 18, 'Interpreter', 'none', 'Location', 'southeast')     

                    % plot throughput curves
                    fig_handle(2) = figure;                    
                    for idx_TC = 1:num_TCs
                        FRC = analysis_results_table.FRC{idx_TC};
                        tmp_input = sscanf(FRC,'L%dT%dM%dB%d');
                        numLayers = tmp_input(1);
                        mcsTable_idx  = tmp_input(2);
                        mcsValue = tmp_input(3);
                        numPRBs = tmp_input(4);
                        mcs_tables = load('McsTable.mat');
                        mcs_table = eval(sprintf('mcs_tables.McsTable%d',mcsTable_idx+1));
                        spef_this_TC = mcs_table(mcsValue+1,4);
                        if ~isempty(analysis_results_table.BLER{idx_TC}) && (numLayers==1) && (mcsTable_idx==1)
                            plot(str2num(analysis_results_table.simSNR{idx_TC}), spef_this_TC*(1-str2num(analysis_results_table.BLER{idx_TC})),'DisplayName',analysis_results_table.FRC{idx_TC},'LineWidth',3, 'Color',list_color(idx_TC,:))
                            hold on;
                        end
                    end
                    SNR_capacity = -10:0.5:27; 
                    plot(SNR_capacity, log2(1+db2pow(SNR_capacity)),'DisplayName','Shannon capacity', 'LineWidth',3, 'Color','k')
                    ylim([0,8])
                    xlim([-10, 35])
                    xlabel('SNR [dB]')
                    ylabel('Data rate [bps/Hz]')
                    grid minor;
                    title(this_subFolderName, 'Interpreter', 'none')
                    legend show;
                    my_legend = findobj(gcf,'Tag','legend');                    
                    set(findall(gcf,'-property','FontSize'),'FontSize',font_size) 
                    set(my_legend, 'FontSize', 18, 'Interpreter', 'none', 'Location', 'southeast')   
                    
                elseif contains(this_subFolderName, 'ul_measurement') 
                    tlo = tiledlayout(2,2,'TileSpacing','compact');
                    nexttile(tlo);
                    for idx_TC = 1:num_TCs
                        plot(str2num(analysis_results_table.simSNR{idx_TC}), str2num(analysis_results_table.error_preSNR_dB{idx_TC}),'DisplayName',[analysis_results_table.FRC{idx_TC},', ',analysis_results_table.Chan{idx_TC}],'LineWidth',3, 'Color',list_color(idx_TC,:))
                        hold on;
                    end
                    ylabel('Error of preSINR [dB]')
                    grid minor;
                    title('Error of preSNR')
                    legend show;
                    my_legend = findobj(gcf,'Tag','legend');
                    set(my_legend, 'Interpreter', 'none', 'Location', 'northwest')     
                    
    
                    nexttile(tlo);
                    for idx_TC = 1:num_TCs
                        plot(str2num(analysis_results_table.simSNR{idx_TC}), str2num(analysis_results_table.error_postSNR_dB{idx_TC}),'DisplayName',[analysis_results_table.FRC{idx_TC},', ',analysis_results_table.Chan{idx_TC}],'LineWidth',3, 'Color',list_color(idx_TC,:))
                        hold on;
                    end
                    ylabel('Error of postSINR [dB]')
                    grid minor;
                    title('Error of postSNR')
    
                    nexttile(tlo);
                    for idx_TC = 1:num_TCs
                        plot(str2num(analysis_results_table.simSNR{idx_TC}), str2num(analysis_results_table.cfoErrHz{idx_TC}),'DisplayName',[analysis_results_table.FRC{idx_TC},', ',analysis_results_table.Chan{idx_TC}],'LineWidth',3, 'Color',list_color(idx_TC,:))
                        hold on;
                    end
                    ylabel('Error of CFO [Hz]')
                    grid minor;
                    title('Error of CFO')
    
                    nexttile(tlo);
                    for idx_TC = 1:num_TCs
                        plot(str2num(analysis_results_table.simSNR{idx_TC}), str2num(analysis_results_table.toErrMicroSec{idx_TC}),'DisplayName',[analysis_results_table.FRC{idx_TC},', ',analysis_results_table.Chan{idx_TC}],'LineWidth',3, 'Color',list_color(idx_TC,:))
                        hold on;
                    end
                    ylabel('Error of TO [micro sec]')
                    grid minor;
                    title('Error of time offset')
    
                    title(tlo, this_subFolderName, 'Interpreter', 'none','FontSize',font_size)
                    xlabel(tlo,'sim SNR [dB]','FontSize',font_size)
                    set(findall(tlo,'-property','FontSize'),'FontSize',font_size) 
    
                elseif contains(this_subFolderName, 'pucchTestMode9') 
                    tlo = tiledlayout(1,2,'TileSpacing','compact');
                    nexttile(tlo);
                    for idx_TC = 1:num_TCs
                        plot(str2num(analysis_results_table.simSNR{idx_TC}), str2num(analysis_results_table.error_preSNR_dB{idx_TC}),'DisplayName',analysis_results_table.Chan{idx_TC},'LineWidth',3, 'Color',list_color(idx_TC,:))
                        hold on;
                    end
                    ylabel('Error of preSINR [dB]')
                    grid minor;
                    title('Error of preSNR')
                    legend show;
                    my_legend = findobj(gcf,'Tag','legend');
                    set(my_legend, 'Interpreter', 'none', 'Location', 'northwest')    
    
                    nexttile(tlo);
                    for idx_TC = 1:num_TCs
                        plot(str2num(analysis_results_table.simSNR{idx_TC}), str2num(analysis_results_table.toErrMicroSec{idx_TC}),'DisplayName',analysis_results_table.Chan{idx_TC},'LineWidth',3, 'Color',list_color(idx_TC,:))
                        hold on;
                    end
                    ylabel('Error of TO [micro sec]')
                    grid minor;
                    title('Error of timing offset')
    
                    title(tlo, this_subFolderName, 'Interpreter', 'none','FontSize',font_size)
                    xlabel(tlo,'sim SNR [dB]','FontSize',font_size)
                    set(findall(tlo,'-property','FontSize'),'FontSize',font_size) 


                elseif contains(this_subFolderName, 'pucchTestMode1') || contains(this_subFolderName, 'pucchTestMode2') || contains(this_subFolderName, 'pucchTestMode3') || contains(this_subFolderName, 'pucchTestMode4')
                    for idx_TC = 1:num_TCs
                        if ~isempty(analysis_results_table.BLER{idx_TC})
                            semilogy(str2num(analysis_results_table.simSNR{idx_TC}), str2num(analysis_results_table.BLER{idx_TC}),'DisplayName',sprintf('TC %d, 5GModel',analysis_results_table.TC(idx_TC)),'LineWidth',3, 'Color',list_color(idx_TC,:))
                            hold on;
                        end
                    end
                    % cuphy
                    if enable_proc_cuPHY_testBench_results
                        for idx_TC = 1:num_TCs
                            if ~isempty(analysis_results_table.BLER{idx_TC})
                                semilogy(str2num(analysis_results_table_cuphy.simSNR{idx_TC}), str2num(analysis_results_table_cuphy.BLER{idx_TC}),'DisplayName',sprintf('TC %d, cuPHY',analysis_results_table_cuphy.TC(idx_TC)),'LineWidth',3, 'Color',list_color(idx_TC,:),'LineStyle','--')
                                hold on;
                            end
                        end
                    end
%                     ylim([1e-3,1])
%                     xlim([-10, 35])
                    xlabel('SNR [dB]')
                    ylabel('ErrorRate')
                    grid minor;
%                     title(this_subFolderName, 'Interpreter', 'none')
%                     title(sprintf('%s, seed: [%s]', this_subFolderName, regexprep(num2str(unique_seeds_save{1} ),'\s+',', ')), 'Interpreter', 'none')
                    title(sprintf('%s, num of seeds: %d', this_subFolderName, length(unique_seeds_save{1}) ), 'Interpreter', 'none')
                    legend show;
                    my_legend = findobj(gcf,'Tag','legend');                    
                    set(findall(gcf,'-property','FontSize'),'FontSize',font_size) 
                    set(my_legend, 'FontSize', 18, 'Interpreter', 'none', 'Location', 'southeast')

                elseif contains(this_subFolderName, 'prach')
                    for idx_TC = 1:num_TCs
                        if ~isempty(analysis_results_table.BLER{idx_TC})
                            semilogy(str2num(analysis_results_table.simSNR{idx_TC}), str2num(analysis_results_table.BLER{idx_TC}),'DisplayName',sprintf('TC %d, 5GModel',analysis_results_table.TC(idx_TC)),'LineWidth',3, 'Color',list_color(idx_TC,:))
                            hold on;
                        end
                    end
                    % cuphy
                    if enable_proc_cuPHY_testBench_results
                        for idx_TC = 1:num_TCs
                            if ~isempty(analysis_results_table.BLER{idx_TC})
                                semilogy(str2num(analysis_results_table_cuphy.simSNR{idx_TC}), str2num(analysis_results_table_cuphy.BLER{idx_TC}),'DisplayName',sprintf('TC %d, cuPHY',analysis_results_table_cuphy.TC(idx_TC)),'LineWidth',3, 'Color',list_color(idx_TC,:),'LineStyle','--')
                                hold on;
                            end
                        end
                    end
%                     ylim([1e-3,1])
%                     xlim([-10, 35])
                    xlabel('SNR [dB]')
                    ylabel('Error rate')
                    grid minor;
%                     title(this_subFolderName, 'Interpreter', 'none')
%                     title(sprintf('%s, seed: [%s]', this_subFolderName, regexprep(num2str(unique_seeds_save{1} ),'\s+',', ')), 'Interpreter', 'none')
                    title(sprintf('%s, num of seeds: %d', this_subFolderName, length(unique_seeds_save{1}) ), 'Interpreter', 'none')
                    legend show;
                    my_legend = findobj(gcf,'Tag','legend');                    
                    set(findall(gcf,'-property','FontSize'),'FontSize',font_size) 
                    set(my_legend, 'FontSize', 18, 'Interpreter', 'none', 'Location', 'southeast')

                    end
                 
                plot_results_folder = top_folder;
                num_figures_this_subFolder = length(fig_handle);
                for idx_fig = 1:num_figures_this_subFolder
                    full_fig_name = sprintf('%s/%s_%d', plot_results_folder, this_subFolderName, idx_fig);             
                    set(fig_handle, 'units','normalized','outerposition',[0 0 1 1]); 
                    savefig(fig_handle(idx_fig),[full_fig_name,'.fig'])
                    exportgraphics(fig_handle(idx_fig),[full_fig_name,'.png'],'Resolution',300); % Saving Plots with Minimal White Space  
                end
                
                
            end
        end
    end

end % of of function

function colors = distinguishable_colors(n_colors,bg,func)
  % Parse the inputs
  if (nargin < 2)
    bg = [1 1 1];  % default white background
  else
    if iscell(bg)
      % User specified a list of colors as a cell aray
      bgc = bg;
      for i = 1:length(bgc)
	bgc{i} = parsecolor(bgc{i});
      end
      bg = cat(1,bgc{:});
    else
      % User specified a numeric array of colors (n-by-3)
      bg = parsecolor(bg);
    end
  end
  
  n_grid = 30;  % number of grid divisions along each axis in RGB space
  x = linspace(0,1,n_grid);
  [R,G,B] = ndgrid(x,x,x);
  rgb = [R(:) G(:) B(:)];
  if (n_colors > size(rgb,1)/3)
    error('You can''t readily distinguish that many colors');
  end
  
  % Convert to Lab color space, which more closely represents human perception
  if (nargin > 2)
    lab = func(rgb);
    bglab = func(bg);
  else
    C = makecform('srgb2lab');
    lab = applycform(rgb,C);
    bglab = applycform(bg,C);
  end

  mindist2 = inf(size(rgb,1),1);
  for i = 1:size(bglab,1)-1
    dX = bsxfun(@minus,lab,bglab(i,:)); % displacement all colors from bg
    dist2 = sum(dX.^2,2);  % square distance
    mindist2 = min(dist2,mindist2);  % dist2 to closest previously-chosen color
  end
  
  colors = zeros(n_colors,3);
  lastlab = bglab(end,:);   % initialize by making the "previous" color equal to background
  for i = 1:n_colors
    dX = bsxfun(@minus,lab,lastlab); % displacement of last from all colors on list
    dist2 = sum(dX.^2,2);  % square distance
    mindist2 = min(dist2,mindist2);  % dist2 to closest previously-chosen color
    [~,index] = max(mindist2);  % find the entry farthest from all previously-chosen colors
    colors(i,:) = rgb(index,:);  % save for output
    lastlab = lab(index,:);  % prepare for next iteration
  end
end

function c = parsecolor(s)
  if ischar(s)
    c = colorstr2rgb(s);
  elseif isnumeric(s) && size(s,2) == 3
    c = s;
  else
    error('MATLAB:InvalidColorSpec','Color specification cannot be parsed.');
  end
end

function c = colorstr2rgb(c)
  rgbspec = [1 0 0;0 1 0;0 0 1;1 1 1;0 1 1;1 0 1;1 1 0;0 0 0];
  cspec = 'rgbwcmyk';
  k = find(cspec==c(1));
  if isempty(k)
    error('MATLAB:InvalidColorString','Unknown color string.');
  end
  if k~=3 || length(c)==1,
    c = rgbspec(k,:);
  elseif length(c)>2,
    if strcmpi(c(1:3),'bla')
      c = [0 0 0];
    elseif strcmpi(c(1:3),'blu')
      c = [0 0 1];
    else
      error('MATLAB:UnknownColorString', 'Unknown color string.');
    end
  end
end

%----------------------------------------------------------------------------------------
function lineStyles=linspecer(N,varargin)
if nargin==0 % return a colormap
    lineStyles = linspecer(128);
    return;
end
if ischar(N)
    lineStyles = linspecer(128,N);
    return;
end
if N<=0 % its empty, nothing else to do here
    lineStyles=[];
    return;
end
% interperet varagin
qualFlag = 0;
colorblindFlag = 0;
if ~isempty(varargin)>0 % you set a parameter?
    switch lower(varargin{1})
        case {'qualitative','qua'}
            if N>12 % go home, you just can't get this.
                warning('qualitiative is not possible for greater than 12 items, please reconsider');
            else
                if N>9
                    warning(['Default may be nicer for ' num2str(N) ' for clearer colors use: whitebg(''black''); ']);
                end
            end
            qualFlag = 1;
        case {'sequential','seq'}
            lineStyles = colorm(N);
            return;
        case {'white','whitefade'}
            lineStyles = whiteFade(N);return;
        case 'red'
            lineStyles = whiteFade(N,'red');return;
        case 'blue'
            lineStyles = whiteFade(N,'blue');return;
        case 'green'
            lineStyles = whiteFade(N,'green');return;
        case {'gray','grey'}
            lineStyles = whiteFade(N,'gray');return;
        case {'colorblind'}
            colorblindFlag = 1;
        otherwise
            warning(['parameter ''' varargin{1} ''' not recognized']);
    end
end      
% *.95
% predefine some colormaps
  set3 = colorBrew2mat({[141, 211, 199];[ 255, 237, 111];[ 190, 186, 218];[ 251, 128, 114];[ 128, 177, 211];[ 253, 180, 98];[ 179, 222, 105];[ 188, 128, 189];[ 217, 217, 217];[ 204, 235, 197];[ 252, 205, 229];[ 255, 255, 179]}');
set1JL = brighten(colorBrew2mat({[228, 26, 28];[ 55, 126, 184]; [ 77, 175, 74];[ 255, 127, 0];[ 255, 237, 111]*.85;[ 166, 86, 40];[ 247, 129, 191];[ 153, 153, 153];[ 152, 78, 163]}'));
set1 = brighten(colorBrew2mat({[ 55, 126, 184]*.85;[228, 26, 28];[ 77, 175, 74];[ 255, 127, 0];[ 152, 78, 163]}),.8);
% colorblindSet = {[215,25,28];[253,174,97];[171,217,233];[44,123,182]};
colorblindSet = {[215,25,28];[253,174,97];[171,217,233]*.8;[44,123,182]*.8};
set3 = dim(set3,.93);
if colorblindFlag
    switch N
        %     sorry about this line folks. kind of legacy here because I used to
        %     use individual 1x3 cells instead of nx3 arrays
        case 4
            lineStyles = colorBrew2mat(colorblindSet);
        otherwise
            colorblindFlag = false;
            warning('sorry unsupported colorblind set for this number, using regular types');
    end
end
if ~colorblindFlag
    switch N
        case 1
            lineStyles = { [  55, 126, 184]/255};
        case {2, 3, 4, 5 }
            lineStyles = set1(1:N);
        case {6 , 7, 8, 9}
            lineStyles = set1JL(1:N)';
        case {10, 11, 12}
            if qualFlag % force qualitative graphs
                lineStyles = set3(1:N)';
            else % 10 is a good number to start with the sequential ones.
                lineStyles = cmap2linspecer(colorm(N));
            end
        otherwise % any old case where I need a quick job done.
            lineStyles = cmap2linspecer(colorm(N));
    end
end
lineStyles = cell2mat(lineStyles);
end
% extra functions
function varIn = colorBrew2mat(varIn)
for ii=1:length(varIn) % just divide by 255
    varIn{ii}=varIn{ii}/255;
end        
end
function varIn = brighten(varIn,varargin) % increase the brightness
if isempty(varargin)
    frac = .9; 
else
    frac = varargin{1}; 
end
for ii=1:length(varIn)
    varIn{ii}=varIn{ii}*frac+(1-frac);
end        
end
function varIn = dim(varIn,f)
    for ii=1:length(varIn)
        varIn{ii} = f*varIn{ii};
    end
end
function vOut = cmap2linspecer(vIn) % changes the format from a double array to a cell array with the right format
vOut = cell(size(vIn,1),1);
for ii=1:size(vIn,1)
    vOut{ii} = vIn(ii,:);
end
end

function cmap = colorm(varargin)
n = 100;
if ~isempty(varargin)
    n = varargin{1};
end
if n==1
    cmap =  [0.2005    0.5593    0.7380];
    return;
end
if n==2
     cmap =  [0.2005    0.5593    0.7380;
              0.9684    0.4799    0.2723];
          return;
end
frac=.95; % Slight modification from colorbrewer here to make the yellows in the center just a bit darker
cmapp = [158, 1, 66; 213, 62, 79; 244, 109, 67; 253, 174, 97; 254, 224, 139; 255*frac, 255*frac, 191*frac; 230, 245, 152; 171, 221, 164; 102, 194, 165; 50, 136, 189; 94, 79, 162];
x = linspace(1,n,size(cmapp,1));
xi = 1:n;
cmap = zeros(n,3);
for ii=1:3
    cmap(:,ii) = pchip(x,cmapp(:,ii),xi);
end
cmap = flipud(cmap/255);
end
function cmap = whiteFade(varargin)
n = 100;
if nargin>0
    n = varargin{1};
end
thisColor = 'blue';
if nargin>1
    thisColor = varargin{2};
end
switch thisColor
    case {'gray','grey'}
        cmapp = [255,255,255;240,240,240;217,217,217;189,189,189;150,150,150;115,115,115;82,82,82;37,37,37;0,0,0];
    case 'green'
        cmapp = [247,252,245;229,245,224;199,233,192;161,217,155;116,196,118;65,171,93;35,139,69;0,109,44;0,68,27];
    case 'blue'
        cmapp = [247,251,255;222,235,247;198,219,239;158,202,225;107,174,214;66,146,198;33,113,181;8,81,156;8,48,107];
    case 'red'
        cmapp = [255,245,240;254,224,210;252,187,161;252,146,114;251,106,74;239,59,44;203,24,29;165,15,21;103,0,13];
    otherwise
        warning(['sorry your color argument ' thisColor ' was not recognized']);
end
cmap = interpomap(n,cmapp);
end
% Eat a approximate colormap, then interpolate the rest of it up.
function cmap = interpomap(n,cmapp)
    x = linspace(1,n,size(cmapp,1));
    xi = 1:n;
    cmap = zeros(n,3);
    for ii=1:3
        cmap(:,ii) = pchip(x,cmapp(:,ii),xi);
    end
    cmap = (cmap/255); % flipud??
end