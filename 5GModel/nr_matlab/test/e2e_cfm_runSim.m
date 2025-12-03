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

function e2e_cfm_runSim(top_folder, wsname_pattern)

    tmpSubFolders   = dir(top_folder);
    dirFlags        = [tmpSubFolders.isdir];
    subFolders      = tmpSubFolders(dirFlags); 
    subFolderNames  = {subFolders(3:end).name};
    num_subFolders  = length(subFolderNames);
    parfor idx_subFolder = 1:num_subFolders
        this_subFolderName = subFolderNames{idx_subFolder};    
        if contains(this_subFolderName, wsname_pattern) 
            fprintf('Running %s...\n', this_subFolderName)
            this_subFolderFullPath = fullfile(top_folder,this_subFolderName);
            tmpSubscenariosFolders = dir(this_subFolderFullPath);
            dirFlags    = [tmpSubscenariosFolders.isdir];
            subscenariosFolders = tmpSubscenariosFolders(dirFlags);
            subscenariosFolderNames = {subscenariosFolders(3:end).name};
            cfg_all = cell2mat(cellfun(@(x) sscanf(x, 'scenario_TC%d___seed_%d___SNR_%f'),subscenariosFolderNames,'UniformOutput',false));
            unique_TCs = unique(cfg_all(1,:));
            num_unique_TCs = length(unique_TCs);
            num_subscenarios = length(subscenariosFolderNames);
            for idx_subscenarios = 1:num_subscenarios
                dir_name_this_subscn = fullfile(top_folder, subFolderNames{idx_subFolder}, subscenariosFolderNames{idx_subscenarios});
                file_list_this_subscn = dir(dir_name_this_subscn);
                dirFlags    = [file_list_this_subscn.isdir];
                fileNamesList_in_this_scn = file_list_this_subscn(~dirFlags);
                fileNames_in_this_scn = {fileNamesList_in_this_scn.name};
                num_files = length(fileNames_in_this_scn);
                for idx_file = 1:num_files
                    this_file_name = fileNames_in_this_scn{idx_file};
                    if contains(this_file_name,'.yaml')
                        full_name = fullfile(dir_name_this_subscn, this_file_name);
                        fprintf('Running %s...\n', full_name)
                        runSim(full_name);
                    end
                end
            end
        end
    end