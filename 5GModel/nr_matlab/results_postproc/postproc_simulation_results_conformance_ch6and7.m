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

% This script is used to post-process 5GModel simulation results for
% Chapter 6 and 7 conformance tests
% Author: Yuan Gao
% 2023/11
function postproc_simulation_results_conformance_ch6and7(ws_folder)
%     work_root_folder = '/home/Aerial/simulations/my_studies/study_RF/';
%     ws_folder = fullfile(work_root_folder,'/ws_scenarios_conformance_sim_ch6and7');
    if strcmp(ws_folder(end),'/')
        ws_folder(end) = [];
    end
    work_root_folder = fileparts(ws_folder);
    CFG = {...
    % TC#           FRC     Chan    rxAnt  SNR   CFO   delay     SNR range       nPRBs     description                           MCS
      601,   'G-FR1-A3-14', 'AWGN',   1,  -2.2,   0,   0e-6,     {60},           273,      'Ch 6 Tx EMV test',                   1;
      602,   'G-FR1-A3-14', 'AWGN',   1,  -2.2,   0,   0e-6,     {60},           273,      'Ch 6 Tx EMV test',                   10;
      603,   'G-FR1-A3-14', 'AWGN',   1,  -2.2,   0,   0e-6,     {60},           273,      'Ch 6 Tx EMV test',                   19;
      604,   'G-FR1-A3-14', 'AWGN',   1,  -2.2,   0,   0e-6,     {60},           273,      'Ch 6 Tx EMV test',                   27;
      701,   'G-FR1-A4-14', 'AWGN',   1,  4.0,    0,   0e-6,     {4:0.5:8},      51,       'Ch 7 Rx sensitivity test',           4;
      702,   'G-FR1-A1-5' , 'AWGN',   1,  4.0,  200,   0e-6,     {4:0.5:8},      51,       'Ch 7 Rx in-channel selectivity test',4;};
    
    list_TC     = [601:604, 701:702];
    list_seed   = [1:2];
    num_TCs     = length(list_TC);
    
    tmp_fig_name = [];
    num_missing_subscenarios = 0;
    for ii = 1:num_TCs
        TC          = list_TC(ii);
        idx_row     = find(cell2mat(CFG(:,1))==TC);
        list_SNR    = cell2mat(CFG{idx_row, 8});
        numPRBs     = CFG{idx_row,9};
        Chan_type   = CFG{idx_row,3};
        Description = CFG{idx_row,10};
        mcs         = CFG{idx_row,11};
        num_SNRs    = length(list_SNR);
        num_seeds   = length(list_seed);
        
        tbCnt = zeros(1, 1,num_SNRs);
        tbErrorCnt = zeros(1, 1,num_SNRs);
        EVM = zeros(1, 1,num_SNRs);
        % extract metrics    
        for idx1=1:1
        for idx2=1:1   
            for idx_SNR = 1:num_SNRs
                idx_start = 1;
                numSlots = 0;
                for idx_seed = 1:num_seeds
                    SNR = list_SNR(idx_SNR);                
                    seed = list_seed(idx_seed); 
                    
                    try                    
                        if ismember(TC, [601:604]) % Ch 6
                            results_file_name = sprintf('%s/scenario_TC_%d___SimCtrl_seed_%d___Chan_0_type_%s___Chan_0_SNR_%2.1f/results/results.mat', ...
                            ws_folder, TC, seed, Chan_type, SNR); 
                            tmp_results = load(results_file_name);
                            results = tmp_results.SysPar.SimCtrl.results;
                            pdsch = results.pdsch;
                            num_pdsch_PDUs = length(pdsch);
                            for idx_PDU = 1:num_pdsch_PDUs
                                EVM(idx1, idx2, idx_SNR) = EVM(idx1, idx2, idx_SNR) + sum(pdsch{idx_PDU}.evm.^2);
                                numSlots = numSlots + length(pdsch{idx_PDU}.evm);
                            end
                        elseif ismember(TC, [701, 702]) % Ch 7
                            results_file_name = sprintf('%s/scenario_TC_%d___SimCtrl_seed_%d___Chan_0_type_%s___Chan_0_SNR_%2.1f/results/results.mat', ...
                            ws_folder, TC, seed, Chan_type, SNR); 
                            tmp_results = load(results_file_name);
                            results = tmp_results.SysPar.SimCtrl.results;
                            pusch = results.pusch;
                            num_pusch_PDUs = 1; % In Rx sensitivity and in-channel selectivity test we just care about the first PDU.
                            for idx_PDU = 1:num_pusch_PDUs
                                tbCnt(idx1, idx2, idx_SNR) = tbCnt(idx1, idx2, idx_SNR) + pusch{idx_PDU}.tbCnt;
                                tbErrorCnt(idx1, idx2, idx_SNR) = tbErrorCnt(idx1, idx2, idx_SNR) + pusch{idx_PDU}.tbErrorCnt;
                            end
                        end
                    catch
    %                     fprintf('Missing %s\n',results_file_name)
                        num_missing_subscenarios = num_missing_subscenarios+1;
                    end
                end
            end
        end
        end
    
        EVM = sqrt(EVM./numSlots);
        TBER = tbErrorCnt./tbCnt;
        fprintf('Total missing subscenarios: %d\n', num_missing_subscenarios);
        
        % plot results
        if ismember(TC, [601:604])
            fprintf('TC %d Ch 6 Tx EMV test, EVM: %2.2f%%\n', TC, EVM*100);
        elseif ismember(TC, [701, 702]) % Ch 7
            fig_handle = figure('units','normalized','outerposition',[0 0 1 1]);
            list_color = {'r', 'g', 'k','m','c','y'};
            list_markers = {'o','d','x','s','+','h','<','>','v'};
            font_size = 24;
            for idx1 = 1:1
            for idx2 = 1:1
                idx_color = 1;
                curve_name = sprintf('5GModel');
                P_rx = -174 + pow2db(numPRBs*12*30e3) + list_SNR;
                semilogy(P_rx, squeeze(TBER(idx1, idx2,:)),'DisplayName',curve_name,'LineWidth',3, 'Color',list_color{idx_color},'Marker',list_markers{idx_color},'MarkerSize',18);
                hold on;  
            end
            end  
            if TC==701% plot 3GPP requirement on Rx sensitivity
                x = -94.6;
            elseif TC==702
                x = -90.8;
            end
            y = 0.05;
            plot(x,y,'DisplayName','3GPP requirement','LineStyle','none','LineWidth',3, 'Color','r','Marker','hexagram','MarkerSize',24); % ,'MarkerFaceColor','r'
            title(sprintf('%s', Description), 'Interpreter', 'none')
            ylim([0.001, 1])
            grid minor
            legend('Interpreter','none','Location','southwest')
            legend show;    
            xlabel('Rx power [dBm]')
            ylabel('BLER')
            set(findall(fig_handle,'-property','FontSize'),'FontSize',font_size)  
            plot_results_folder = fullfile(work_root_folder,'results/');
            if ~exist(plot_results_folder, 'dir')
                mkdir(plot_results_folder)
            end
            tmp_fig_name = sprintf('TC_%d', TC);                    
            full_fig_name = sprintf('%s/%s',plot_results_folder,tmp_fig_name) ; 
            set(fig_handle, 'units','normalized','outerposition',[0 0 1 1]); 
            savefig(fig_handle,[full_fig_name,'.fig'])
            exportgraphics(fig_handle,[full_fig_name,'.png'],'Resolution',300); % Saving Plots with Minimal White Space
        end
        
    
    end

end % end of funciton