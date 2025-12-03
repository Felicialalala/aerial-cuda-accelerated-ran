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

function [preambleFormat, x, y, subframeNum, startingSym, N_slot_subframe, ...
    N_t_RA_slot, N_dur_RA] = readPrachCfgTable(prachCfgIdx, FR, duplex, prachTable)
% function [preambleFormat, x, y, subframeNum, startingSym, N_slot_subframe, ...
%   N_t_RA_slot, N_dur_RA] = readPrachCfgTable(prachCfgIdx, FR, duplex)
%
% This function reads TS 38.111 Table 6.3.3.2 2-4 to find preamble format 
% and RA occations in timing domain
%

if (FR == 1) && (duplex == 1) 
    % 3GPP 38.211 (V15.4) Table 6.3.3.2-3
%     load table_prachCfg_FR1TDD;
    tablePrachCfg = prachTable.prachCfg_FR1TDD;
elseif (FR == 1) && (duplex == 0)
    % 3GPP 38.211 (V15.4) Table 6.3.3.2-2
%     load table_prachCfg_FR1FDD;
    tablePrachCfg = prachTable.prachCfg_FR1FDD;
else
    % 3GPP 38.211 (V15.4) Table 6.3.3.2-4
%     load table_prachCfg_FR2;
    tablePrachCfg = prachTable.prachCfg_FR2;
end

preambleFormat = tablePrachCfg{prachCfgIdx+1, 2}{1};
x = tablePrachCfg{prachCfgIdx+1, 3};
y = tablePrachCfg{prachCfgIdx+1, 4}{1};
subframeNum = tablePrachCfg{prachCfgIdx+1, 5}{1};
startingSym = tablePrachCfg{prachCfgIdx+1, 6};
N_slot_subframe = tablePrachCfg{prachCfgIdx+1, 7};
N_t_RA_slot = tablePrachCfg{prachCfgIdx+1, 8};
N_dur_RA = tablePrachCfg{prachCfgIdx+1, 9};

return


