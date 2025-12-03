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

function [N_symb_slot, N_slot_frame_mu, N_slot_subframe_mu] ...
    = readFrameStructureTable(mu, CpType)
%
% derive frame structure from mu value and cyclicPrefix type
%
switch CpType
    case 0
        if mu < 0 || mu > 4
            error('mu is not supported ...\n');
        end
        % 3GPP 38.211 V15.4 Table 4.3.2-1
        table = load('table_frameStruct_normalCP.txt');
        N_symb_slot = table(mu + 1, 1);
        N_slot_frame_mu = table(mu + 1, 2);
        N_slot_subframe_mu = table(mu + 1, 3); 
    case 1
        if mu ~= 2
            error('mu is not supported ...\n');
        end
        % 3GPP 38.211 V15.4 Table 4.3.2-2
        table = load('table_frameStruct_extendedCP.txt');
        N_symb_slot = table(1);
        N_slot_frame_mu = table(2);
        N_slot_subframe_mu = table(3);
    otherwise
        error('CpType is not supported ... \n');
end

    
