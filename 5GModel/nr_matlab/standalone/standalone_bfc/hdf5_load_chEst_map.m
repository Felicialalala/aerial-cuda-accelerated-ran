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

function mapping = hdf5_load_chEst_map(filename)
    % HDF5_LOAD_CHEST_MAP Get channel estimate mapping from HDF5 file
    % 
    % Usage:
    %   mapping = hdf5_load_chEst_map('filename.h5')
    %
    % Inputs:
    %   filename  - Path to HDF5 file
    %
    % Returns:
    %   mapping   - Structure with fields:
    %               .pdu_indices - Array of PDU indices that have channel estimates
    %               .rntis       - Array of corresponding RNTI values
    %               .hest_fields - Cell array of IND*_Hest field names
    %
    % This function:
    % 1. Lists all field names in the HDF5 file
    % 2. Filters to find IND*_Hest fields (exact match, not HestNorm or HestToL2)
    % 3. For each IND*_Hest field, loads the corresponding PDU*.RNTI
    % 4. Returns mapping information for nrSimulator to use
    
    % Get file info to list all datasets without loading data
    file_info = h5info(filename);
    dataset_names = {file_info.Datasets.Name};
    
    % Filter to find IND*_Hest fields (exact match, exclude HestNorm, HestToL2, etc.)
    % Pattern: starts with 'IND', followed by digits, ends with '_Hest'
    hest_pattern = '^IND(\d+)_Hest$';
    hest_fields = {};
    hest_numbers = [];
    
    for i = 1:length(dataset_names)
        dataset_name = dataset_names{i};
        tokens = regexp(dataset_name, hest_pattern, 'tokens');
        if ~isempty(tokens)
            hest_fields{end+1} = dataset_name;
            hest_numbers(end+1) = str2double(tokens{1}{1});
        end
    end
    
    % Initialize output arrays
    pdu_indices = [];
    rntis = [];
    valid_hest_fields = {};
    
    % Open HDF5 file once for all operations
    fileID = H5F.open(filename, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
    
    try
        % Process each IND*_Hest field to get RNTI mapping
        for i = 1:length(hest_fields)
            hest_field = hest_fields{i};
            ind_number = hest_numbers(i);
            
            % Construct corresponding PDU field name
            pdu_field = sprintf('PDU%d', ind_number);
            
            % Check if corresponding PDU dataset exists
            if ismember(pdu_field, dataset_names)
                try
                    % Open PDU dataset and read RNTI field only
                    pduID = H5D.open(fileID, ['/' pdu_field]);
                    
                    % Read the entire PDU structure (it's small)
                    pdu_data = H5D.read(pduID);
                    H5D.close(pduID);
                    
                    % Extract RNTI if it exists
                    if isfield(pdu_data, 'RNTI')
                        rnti = pdu_data.RNTI;
                        
                        % Store mapping information
                        pdu_indices(end+1) = ind_number;
                        rntis(end+1) = rnti;
                        valid_hest_fields{end+1} = hest_field;
                    end
                catch ME
                    fprintf('Error processing %s: %s\n', hest_field, ME.message);
                end
            end
        end
    catch ME
        % Ensure file is closed even if there's an error
        H5F.close(fileID);
        rethrow(ME);
    end
    
    % Close the HDF5 file
    H5F.close(fileID);
    
    % Create output structure
    mapping = struct();
    mapping.pdu_indices = pdu_indices;
    mapping.rntis = rntis;
    mapping.hest_fields = valid_hest_fields;
end
