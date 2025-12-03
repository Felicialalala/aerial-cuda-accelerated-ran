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

function Xtf = hSRSGen(srs, carrier)

srs5g = nrSRSConfig;

srs5g.NumSRSPorts = srs.numAntPorts;
srs5g.NumSRSSymbols = srs.numSymbols;
srs5g.SymbolStart = srs.timeStartPosition;
srs5g.KTC = srs.combSize;
srs5g.KBarTC = srs.combOffset;
srs5g.CyclicShift = srs.cyclicShift;
srs5g.FrequencyStart = srs.frequencyShift;
srs5g.NRRC = srs.frequencyPosition;
srs5g.CSRS = srs.configIndex;
srs5g.BSRS = srs.bandwidthIndex;
srs5g.BHop = srs.frequencyHopping;
srs5g.Repetition = srs.numRepetitions;
if srs.resourceType == 0
    srs5g.SRSPeriod = 'on';
else
    srs5g.SRSPeriod = [srs.Tsrs, srs.Toffset];
end

switch srs.resourceType
    case 0       
        srs5g.ResourceType = 'aperiodic';
    case 1
        srs5g.ResourceType = 'semi-persistent';        
    case 2
        srs5g.ResourceType = 'periodic';  
end
switch srs.groupOrSequenceHopping
    case 0       
        srs5g.GroupSeqHopping = 'neither';
    case 1
        srs5g.GroupSeqHopping = 'groupHopping';        
    case 2
        srs5g.GroupSeqHopping = 'sequenceHopping';  
end
srs5g.NSRSID = srs.sequenceId;

carrier5g = nrCarrierConfig;
carrier5g.NCellID = carrier.N_ID_CELL;
carrier5g.SubcarrierSpacing = 15 * 2^(carrier.mu);
carrier5g.NSizeGrid = carrier.N_grid_size_mu;
carrier5g.NStartGrid = carrier.N_grid_start_mu;
carrier5g.NFrame = carrier.SFN_start; % Frame idx
global SimCtrl
if SimCtrl.genTV.forceSlotIdxFlag
    carrier5g.NSlot = SimCtrl.genTV.slotIdx(1);
else
    carrier5g.NSlot = mod((carrier.idxSlot + carrier.idxSubframe * ...
        carrier.N_slot_subframe_mu -1), carrier.N_slot_frame_mu);
end
    
[srsSym, srsInfo] = nrSRS(carrier5g,srs5g);
srsInd = nrSRSIndices(carrier5g,srs5g);
K = carrier5g.NSizeGrid*12;       % Number of subcarriers
L = carrier5g.SymbolsPerSlot;     % Number of OFDM symbols per slot
P = srs5g.NumSRSPorts;  % Number of antenna ports
gridSize = [K L P];
Xtf = complex(zeros(gridSize));
Xtf(srsInd) = srsSym;

return






