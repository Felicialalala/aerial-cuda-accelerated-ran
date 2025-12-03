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

function [errCnt, nTV] = genCfgTV_perf_ss(spreadsheet, tc_list, is_fapi)

    if nargin == 0
        error('Please specify the spreadsheet to use for the generation');
    else
        if nargin == 1
            tc_list = [];
            is_fapi = 0;
        else
            if nargin == 2
                is_fapi = 0;
            else
                if nargin > 3
                    error('Only 3 arguments can be accepted');
                end
            end
        end
    end

    % Check for the spreadsheet in some specific paths
    if (exist(spreadsheet,'file'))
        % Use the filename exactly as specified
    else
        fprintf(['Unable to find spreadsheet ',spreadsheet,'\n']);
        error('Exiting');
    end
    
    % Hashmap for modulation order given MCS table and index
    
    k_set_mcs_1 = 0:28;
    k_set_mcs_2 = 0:27;
    k_set_mcs_3 = 0:28;
    
    v_set_mcs_1 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6];
    v_set_mcs_2 = [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8];
    v_set_mcs_3 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6];
    
    mod_ord_1 = containers.Map(k_set_mcs_1, v_set_mcs_1);
    mod_ord_2 = containers.Map(k_set_mcs_2, v_set_mcs_2);
    mod_ord_3 = containers.Map(k_set_mcs_3, v_set_mcs_3);
    
    % Hashmap for max number of PRB in FT grid for given numerology index

    k_set_mu = [0, 1];
    v_set_mu = [106, 273];
    grid_mu_table = containers.Map(k_set_mu, v_set_mu);
    
    sheets = sheetnames(spreadsheet);
    
    if isempty(tc_list)
        tc_list = sheets;
    end

    errCnt = 0;
    nTV = 0;
    
    for i= 1:length(sheets)
    % check last 5 characters match PUSCH or PDSCH

        if sum(contains(tc_list, sheets{i})) == 0
            continue
        end
        
        if (strlength(sheets{i}) < 5)
            continue
        end
        
        buffer = split(sheets{i},'-');
        usecase_name = strip(buffer{1});
        pipeline_name = strip(buffer{2});
        
        if (pipeline_name == "PUSCH" || pipeline_name == "PDSCH")
            fprintf("\n**** Start Processing Sheet: %s ****\n", sheets{i});
            data = readtable(spreadsheet,'Sheet',sheets{i}, 'PreserveVariableNames', true);
            data = data(1:14,3:end);

            [~,nCases] = size(data);
            
            parfor tc = 1 : nCases
                nTV = nTV + 1;
                % extract test case paramaters:

                % column name
                tc_str = data.Properties.VariableNames{tc};
                
                % extracting parameters from the spreadsheet
                tc_par_mu = data{2, tc};
                tc_par_gNB_rf_chains = data{3, tc};
                tc_par_PRB = data{5, tc};
                tc_par_symbols = data{6, tc};
                tc_par_dmrs_symbols = data{7, tc};
                tc_par_dmrs_max_len = data{8, tc};
                tc_par_mcs_table = data{10, tc};
                tc_par_mcs_index = data{11, tc};
                tc_par_users = data{13, tc};
                tc_par_layers_per_user = data{14, tc};
                
                tc_par_mod_order = nan;
                
                if tc_par_mcs_table == 1
                    tc_par_mod_order = mod_ord_1(tc_par_mcs_index);
                else
                    if tc_par_mcs_table == 2
                        tc_par_mod_order = mod_ord_2(tc_par_mcs_index);
                    else
                        if tc_par_mcs_table == 3
                            tc_par_mod_order = mod_ord_3(tc_par_mcs_index);
                        end
                    end
                end         
                
                SysPar = initSysPar;
                
                if (pipeline_name == "PDSCH")
                    buffer = SysPar.pdsch{1};
                else
                    buffer = SysPar.pusch{1};
                end                

                SysPar = [];
                
                if (pipeline_name == "PDSCH")
                    SysPar.testAlloc.dl = 1;
                    SysPar.testAlloc.ul = 0;
                    SysPar.testAlloc.pdsch = tc_par_users;
                    SysPar.testAlloc.pusch = 0;
                    for idxUE = 1:tc_par_users
                        SysPar.pdsch{idxUE} = buffer;
                        SysPar.pdsch{idxUE}.rbStart = 0;
                        SysPar.pdsch{idxUE}.rbSize = tc_par_PRB;
                        SysPar.pdsch{idxUE}.StartSymbolIndex = 14 - tc_par_symbols;
                        SysPar.pdsch{idxUE}.NrOfSymbols = tc_par_symbols;
                        second_dmrs_symbol = tc_par_dmrs_max_len - 1;
                        SysPar.pdsch{idxUE}.DmrsSymbPos = [0 0 1 second_dmrs_symbol 0 0 0 0 0 0 0 0 0 0];
                        SysPar.pdsch{idxUE}.nrOfLayers = tc_par_layers_per_user;
                        SysPar.pdsch{idxUE}.mcsTable = tc_par_mcs_table - 1;
                        SysPar.pdsch{idxUE}.mcsIndex = tc_par_mcs_index;
                        SysPar.pdsch{idxUE}.portIdx = (idxUE-1)* tc_par_layers_per_user + [0:tc_par_layers_per_user-1];
                        SysPar.pdsch{idxUE}.portIdx = mod((idxUE-1)* tc_par_layers_per_user + [0:tc_par_layers_per_user-1], 8);    
                        SysPar.pdsch{idxUE}.SCID = floor((idxUE-1)*tc_par_layers_per_user/8);
                        SysPar.pdsch{idxUE}.idxUE = idxUE-1;
                        SysPar.pdsch{idxUE}.seed = sum(int32(tc_str)) + idxUE-1;
                    end
                    cfgFile = ['GPU_test_input/TV_cuphy_', tc_str, '_slot', num2str(0), '_MIMO', num2str(tc_par_users * tc_par_layers_per_user), 'x',num2str(tc_par_gNB_rf_chains), '_PRB', num2str(tc_par_PRB), '_DataSyms', num2str(tc_par_symbols - tc_par_dmrs_symbols),  '_qam' , num2str(2^tc_par_mod_order), '.yaml'];
                    tvFile =  ['TV_cuphy_', tc_str, '_slot', num2str(0), '_MIMO', num2str(tc_par_users * tc_par_layers_per_user), 'x',num2str(tc_par_gNB_rf_chains), '_PRB', num2str(tc_par_PRB), '_DataSyms', num2str(tc_par_symbols - tc_par_dmrs_symbols),  '_qam' , num2str(2^tc_par_mod_order)];
                else
                    SysPar.testAlloc.dl = 0;
                    SysPar.testAlloc.ul = 1;
                    SysPar.testAlloc.pdsch = 0;
                    SysPar.testAlloc.pusch = tc_par_users;
                    for idxUE = 1:tc_par_users
                        SysPar.pusch{idxUE} = buffer;
                        SysPar.pusch{idxUE}.rbStart = 0;
                        SysPar.pusch{idxUE}.rbSize = tc_par_PRB;
                        SysPar.pusch{idxUE}.StartSymbolIndex = 14 - tc_par_symbols;
                        SysPar.pusch{idxUE}.NrOfSymbols = tc_par_symbols;
                        second_dmrs_symbol = tc_par_dmrs_max_len - 1;
                        SysPar.pusch{idxUE}.DmrsSymbPos = [0 0 1 second_dmrs_symbol 0 0 0 0 0 0 0 0 0 0];
                        SysPar.pusch{idxUE}.nrOfLayers = tc_par_layers_per_user;
                        SysPar.pusch{idxUE}.mcsTable = tc_par_mcs_table - 1;
                        SysPar.pusch{idxUE}.mcsIndex = tc_par_mcs_index;
                        SysPar.pusch{idxUE}.portIdx = (idxUE-1)* tc_par_layers_per_user + [0:tc_par_layers_per_user-1];
                        SysPar.pusch{idxUE}.idxUE = idxUE-1;
                        SysPar.pusch{idxUE}.seed = sum(int32(tc_str)) + idxUE-1;
                    end
                    cfgFile = ['GPU_test_input/TV_cuphy_', tc_str, '_snrdb40.00_MIMO', num2str(tc_par_users * tc_par_layers_per_user), 'x',num2str(tc_par_gNB_rf_chains), '_PRB', num2str(tc_par_PRB), '_DataSyms', num2str(tc_par_symbols - tc_par_dmrs_symbols),  '_qam' , num2str(2^tc_par_mod_order), '.yaml'];
                    tvFile =  ['TV_cuphy_', tc_str, '_snrdb40.00_MIMO', num2str(tc_par_users * tc_par_layers_per_user), 'x',num2str(tc_par_gNB_rf_chains), '_PRB', num2str(tc_par_PRB), '_DataSyms', num2str(tc_par_symbols - tc_par_dmrs_symbols),  '_qam' , num2str(2^tc_par_mod_order)];
                end

                SysPar.testAlloc.ssb = 0;
                SysPar.testAlloc.pdcch = 0;
                SysPar.testAlloc.csirs = 0;
                SysPar.testAlloc.prach = 0;
                SysPar.testAlloc.pucch = 0;
                SysPar.testAlloc.srs = 0;  

                SysPar.carrier.N_grid_size_mu = grid_mu_table(tc_par_mu);
                SysPar.carrier.mu = tc_par_mu;
                SysPar.carrier.Nant_gNB = tc_par_gNB_rf_chains;
                SysPar.carrier.Nant_UE = tc_par_layers_per_user; 
                SysPar.SimCtrl.N_UE = tc_par_users;
                SysPar.SimCtrl.N_slot_run = 1;
                SysPar.SimCtrl.genTV.slotIdx = 0;
                SysPar.SimCtrl.genTV.FAPI = is_fapi;

                SysPar = updateAlgFlag(SysPar);
                
                [status,msg] = mkdir('GPU_test_input');
                WriteYaml(cfgFile, SysPar);
                errFlag = runSim(cfgFile, tvFile);
                errCnt = errCnt + errFlag;
            end
        end
    end
return
end
