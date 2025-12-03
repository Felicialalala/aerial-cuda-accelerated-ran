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

function [prachSym5g, NSlot, LRA] = hPRACHGen(prach, carrier, prachTable)                

prach = findPreambleTfLoc(prach, carrier, prachTable);
preambleFormat = prach.preambleFormat;

% generate preamble from Matlab 5G Toolbox
carrier5g = nrCarrierConfig;
%                 carrier5g.NSizeGrid = N_PRB;
prach5g = nrPRACHConfig;
[NFrame, NPRACHSlot, delta_f_RA, ActivePRACHSlot, ...
    subframeNum, N_slot_subframe] = findFirstSlot(prach, carrier, prachTable);
carrier5g.NFrame = NFrame;
carrier5g.NSlot = NPRACHSlot(1);
if strcmp(preambleFormat,'0')
    prach5g.NPRACHSlot =  subframeNum(1);
else
    prach5g.NPRACHSlot = NPRACHSlot(1);
end
prach5g.SubcarrierSpacing = delta_f_RA*1e-3;
prach5g.FrequencyRange = ['FR', num2str(carrier.FR)];
if carrier.duplex == 0
    prach5g.DuplexMode = 'FDD';
else
    prach5g.DuplexMode = 'TDD';    
end
carrier5g.SubcarrierSpacing = 15*2^carrier.mu;
prach5g.ConfigurationIndex = prach.configurationIndex;
prach5g.ActivePRACHSlot = ActivePRACHSlot;
switch prach.restrictedSet
    case 0
        prach5g.RestrictedSet = 'UnrestrictedSet';
    case 1
        prach5g.RestrictedSet = 'RestrictedSetTypeA';
    case 2
        prach5g.RestrictedSet = 'RestrictedSetTypeB';
    otherwise
        error('restrictedSet is not supported ... \n');
end
prach5g.SequenceIndex = prach.rootSequenceIndex;
prach5g.ZeroCorrelationZone = prach.zeroCorrelationZone;
prach5g.PreambleIndex = prach.prmbIdx;
prachSym5g = nrPRACH(carrier5g,prach5g);

NSlot = carrier5g.NSlot;
LRA = prach5g.LRA;

% if carrier.mu == 1 && N_slot_subframe == 1
%     NSlot = NSlot - 1;
% end

return                
