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

function genCfgTV_perf_ss_bwc(spreadsheet)

sheets = sheetnames(spreadsheet);

for i= 1:length(sheets)
    pipeline_name = sheets{i};
    
    if ~ismember(pipeline_name, ["F14 - PDSCH", "F09 - PDSCH"])
        continue
    end
    
    
    data = readtable(spreadsheet,'Sheet',sheets{i}, 'PreserveVariableNames', true);
    data = data(1:14,3:end);
    
    
    [~,nCases] = size(data);
    
    nCompErrs     = 0;
    nCompChecks   = 0;
    nTvGen        = 0;
    detErr        = 0;
    
    for tc = 1 : nCases
        % extract test case paramaters:
        
        % column name
        tc_str = data.Properties.VariableNames{tc};
        
        % extracting parameters from the spreadsheet
        tc_par_gNB_rf_chains = data{3, tc};
        tc_par_layers = data{4, tc};
        tc_par_PRB = data{5, tc};
        tc_par_users = data{13, tc};
        tc_par_layers_per_user = data{14, tc};
        
        if ismember(pipeline_name(1:3),["F14", "F09"])
            nUeGrps = 1; % for F14
        else
            error("Error: only F09 and F14 are configured. \n")
        end
        
        tvFile =  ['TV_cuphy_', replace(tc_str, '-DS-','-BW-'), '_slot', num2str(0), '_MIMO', num2str(tc_par_users * tc_par_layers_per_user), 'x',num2str(tc_par_gNB_rf_chains), '_PRB', num2str(tc_par_PRB), '.h5'];
        
        % generate TVs based on standalone version
        % TODO: remove when cuphy phase 3 updated 
        if pipeline_name(1:3) == "F14"
            get_BFC_TV(tc_par_layers, 64, round(tc_par_PRB / 2), tvFile);
            % delete('./GPU_test_input/*.mat');
        end
        
        % generate TVs based on complete version
        tvFile =  ['bfw2_TV_cuphy_', replace(tc_str, '-DS-','-BW-'), '_slot', num2str(0), '_MIMO', num2str(tc_par_users * tc_par_layers_per_user), 'x',num2str(tc_par_gNB_rf_chains), '_PRB', num2str(tc_par_PRB), '.h5'];

        testAlloc     = [];
        testAlloc.dl  = 1;
        testAlloc.ul  = 0;
        testAlloc.bfw = nUeGrps;
        
        SysPar = initSysPar(testAlloc);
        
        SysPar.SimCtrl.genTV.slotIdx = 0;
        SysPar.SimCtrl.N_slot_run    = 1;
        SysPar.SimCtrl.genTV.enable  = 1;
        SysPar.SimCtrl.genTV.TVname  = sprintf(tvFile);
        nTvGen = nTvGen + 1;
        
        % configure channel:
        noiseEnergy_dB = -30;
        SysPar.Chan{1}.SNR = -noiseEnergy_dB;
        
        SysPar.SimCtrl.bf.N_SRS_CHEST_BUFF = tc_par_users;
        SysPar.SimCtrl.N_UE                = tc_par_users;
        for ueIdx = 0 : (SysPar.SimCtrl.bf.N_SRS_CHEST_BUFF - 1)
            SysPar.srsChEstBuff{ueIdx + 1}.ueIdx       = ueIdx;
            SysPar.srsChEstBuff{ueIdx + 1}.startPrbGrp = 0;
            SysPar.srsChEstBuff{ueIdx + 1}.nPrbGrp     = round(tc_par_PRB / 2);
            SysPar.srsChEstBuff{ueIdx + 1}.nUeAnt      = 4;
            SysPar.srsChEstBuff{ueIdx + 1}.prbGrpSize  = 2;
            SysPar.srsChEstBuff{ueIdx + 1}.gainDB      = 0;
            SysPar.srsChEstBuff{ueIdx + 1}.startValidPrg  = 0;
            SysPar.srsChEstBuff{ueIdx + 1}.nValidPrg      = 273;
            if pipeline_name(1:3) == "F14"
                SysPar.srsChEstBuff{ueIdx + 1}.nGnbAnt     = 64; % for F14
            elseif pipeline_name(1:3) == "F09"
                SysPar.srsChEstBuff{ueIdx + 1}.nGnbAnt     = 32; 
            else
                error("Error: only F09 and F14 are configured. \n")
            end
        end
        
        % Configure ue grp paramaters:        
        SysPar.SimCtrl.bf.N_UE_GRP = nUeGrps;
        for i = 0 : (nUeGrps - 1)
            SysPar.bfw{i + 1}.startPrbGrp  = 0;
            SysPar.bfw{i + 1}.startPrb     = 0;
            SysPar.bfw{i + 1}.nPrbGrp      = round(tc_par_PRB / 2);
            SysPar.bfw{i + 1}.nRxAnt       = SysPar.srsChEstBuff{1}.nGnbAnt;
            SysPar.bfw{i + 1}.nBfLayers    = tc_par_layers;
            SysPar.bfw{i + 1}.ueIdxs       = repelem([0:tc_par_users-1], tc_par_layers_per_user);
            SysPar.bfw{i + 1}.ueLayersIdxs = repmat([0:tc_par_layers_per_user-1],1, tc_par_users);
            SysPar.bfw{i + 1}.lambda       = 0;
            SysPar.bfw{i + 1}.coefBufIdx   = i;
            SysPar.bfw{i + 1}.prbGrpSize   = 2;
            SysPar.bfw{i + 1}.nPrb         = SysPar.bfw{i + 1}.nPrbGrp * SysPar.bfw{i + 1}.prbGrpSize;
        end
        
        [SysPar, UE, gNB] = nrSimulator(SysPar);
        
        % complience check:
        testPass = 1;
        testCompliance = 1;
        if testCompliance
            nCompChecks = nCompChecks + 1;
            bfw_results = SysPar.SimCtrl.results.bfw;
            for ueGrpIdx = 0 : (nUeGrps - 1)
                minBfSinr = bfw_results{ueGrpIdx + 1}.minBfSinr;
                if(minBfSinr < 26)
                    nCompErrs = nCompErrs + 1;
                    testPass = 0;
                end
            end
        end
        if testPass == 1            
            % fprintf('%s  PASS\n', tvFile);
        else
            fprintf('%s  NOT PASS Compliance check!\n', tvFile);
        end
        
        movefile(['./GPU_test_input/', tvFile, '_BFW_gNB_CUPHY_s0.h5'], ['./GPU_test_input/', tvFile]);
        
        fapi_name = ['./GPU_test_input/', strrep(tvFile,'TV_cuphy_','TV_fapi_'),'.h5'];
        movefile(fapi_name, strrep(fapi_name, '.h5.h5','.h5'));

        
   
    end
    
    
    
    fprintf('-----------\n');
    fprintf('Total Compliance TC = %d, PASS = %d, FAIL = %d, Total TV generated = %d\n\n', nCompChecks, nCompChecks-nCompErrs, nCompErrs, nTvGen);

end


