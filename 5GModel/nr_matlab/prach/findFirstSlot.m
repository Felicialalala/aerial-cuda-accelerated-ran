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

function [NFrame, NPRACHSlot, delta_f_RA, ActivePRACHSlot, subframeNum, N_slot_subframe]...
    = findFirstSlot(prach, carrier, prachTable)

FR = carrier.FR;
duplex = carrier.duplex;
mu = carrier.mu;
prachCfgIdx = prach.configurationIndex;
duplex = carrier.duplex;

ActivePRACHSlot = 0;

if mu == 0
    NslotPerSubframe = 1;
elseif mu == 1
    NslotPerSubframe = 2;
elseif mu == 2
    NslotPerSubframe = 4;
elseif mu == 3
    NslotPerSubframe = 8;
elseif mu == 4
    NslotPerSubframe = 16;
end

[preambleFormat, SFN_x, SFN_y, subframeNum, startingSym, N_slot_subframe, ...
    N_t_RA_slot, N_dur_RA] = readPrachCfgTable(prachCfgIdx, FR, duplex, prachTable);

NFrame = SFN_y;
NPRACHSlot = subframeNum*NslotPerSubframe;

switch preambleFormat
    case {'0', '1', '2'}
        delta_f_RA = 1250;
        L_RA = 839;
    case '3'
        delta_f_RA = 5000;
        L_RA = 839;
    otherwise
        delta_f_RA = 15000*2^mu;
        L_RA = 139;
end 

if mu == 1 && N_slot_subframe == 1
    ActivePRACHSlot = 1;
    NPRACHSlot = NPRACHSlot + 1;
end

if strcmp(preambleFormat, '1')
    if mu == 1
        NPRACHSlot = NPRACHSlot + 1;
    else
        NPRACHSlot = NPRACHSlot + 2;
    end
end
