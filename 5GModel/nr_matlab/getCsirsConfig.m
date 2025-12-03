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

function [row2nPort, row2nCdm] = getCsirsConfig()
% GETCSIRSCONFIG Get centralized CSI-RS configuration arrays
%
% This function provides access to the centralized CSI-RS configuration arrays
% that were previously hard-coded in multiple files throughout the codebase.
%
% Outputs:
%   row2nPort - Array mapping Row index to number of ports [1 1 2 4 4 8 8 8 12 12 16 16 24 24 24 32 32 32]
%   row2nCdm  - Array mapping Row index to number of CDM groups [1 1 2 2 2 2 2 4 2 4 2 4 2 4 8 2 4 8]
%
% Usage:
%   [row2nPort, row2nCdm] = getCsirsConfig();
%   nPort_csirs = row2nPort(Row);
%   nCdm_csirs = row2nCdm(Row);


% Load the table once and cache it (persistent variable for efficiency)
persistent cached_row2nPort cached_row2nCdm

if isempty(cached_row2nPort) || isempty(cached_row2nCdm)
    % Load the centralized table
    load('csirs/csirsLocTable.mat', 'csirsLocTable');
    cached_row2nPort = [csirsLocTable.Ports{:}];
    cached_row2nCdm = [csirsLocTable.CDMGroups{:}];
end

row2nPort = cached_row2nPort;
row2nCdm = cached_row2nCdm;

end 