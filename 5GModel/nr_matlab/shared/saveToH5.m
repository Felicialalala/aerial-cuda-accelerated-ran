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

function saveToH5(folder, filename, varargin)
    % Create output directory
    if ~exist(folder, 'dir')
        mkdir(folder);
    end

    % Open HDF5 file
    filePath = fullfile(folder, [filename '.h5']);
    fileId = H5F.create(filePath, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

    % Save each variable
    if mod(length(varargin), 2) ~= 0
        error('varargin must contain an even number of elements (name-value pairs)');
    end
    
    for i = 1:2:length(varargin)
        varName = varargin{i};
        varData = varargin{i+1};
        
        if ~ischar(varName)
            warning('Variable name at position %d is not a character array, skipping', i);
            continue;
        end

        try
            if isnumeric(varData) || islogical(varData) || ischar(varData) || isstring(varData)
                hdf5_write_nv(fileId, varName, varData);
            else
                hdf5_write_nv_exp(fileId, varName, varData);
            end
        catch e
            warning('Failed to save variable "%s": %s', varName, e.message);
        end
    end

    % Close the file
    H5F.close(fileId);