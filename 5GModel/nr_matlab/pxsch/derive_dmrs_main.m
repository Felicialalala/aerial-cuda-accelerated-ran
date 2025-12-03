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

function dmrs = derive_dmrs_main(pdu, pxschTable)

dmrs.n_scid = pdu.SCID;
idx = find(pdu.DmrsSymbPos);
if (length(idx) > 1) && (idx(2)-idx(1) == 1)
    maxLength = 2;
    dmrs.AdditionalPosition = sum(pdu.DmrsSymbPos)/2-1;     
else
    maxLength = 1;
    dmrs.AdditionalPosition = sum(pdu.DmrsSymbPos)-1;     
end
dmrs.maxLength = maxLength; 
dmrs.type = pdu.dmrsConfigType;

if pdu.numDmrsCdmGrpsNoData == 2
    dmrs.energy = 2;
elseif pdu.numDmrsCdmGrpsNoData == 1
    dmrs.energy = 1;
end

% derive_layer_dmrs_cfg
nl = pdu.nrOfLayers;            % number of layers transmited by user
portIdx = find(flip(pdu.dmrsPorts)); % dmrs antenna port assigned to each layer. Dim: nl x 1

% load('type1_dmrs_table.mat');

fOCC_table = pxschTable.fOCC_table;
grid_table = pxschTable.grid_table;
tOCC_table = pxschTable.tOCC_table;

fOCC_cfg = zeros(nl,1);
tOCC_cfg = zeros(nl,1);
grid_cfg = zeros(nl,1);

for layer = 1 : nl      
	fOCC_cfg(layer) = fOCC_table(portIdx(layer));
	tOCC_cfg(layer) = tOCC_table(portIdx(layer));
	grid_cfg(layer) = grid_table(portIdx(layer));    
end

dmrs.fOCC_cfg = fOCC_cfg;
dmrs.tOCC_cfg = tOCC_cfg;
dmrs.grid_cfg = grid_cfg;

% derive_grids
grid_cfg = [0 0 1 1]';

gridIdx = unique(grid_cfg);
nGrids = length(gridIdx);

dmrs.nGrids = nGrids;
dmrs.gridIdx = gridIdx;

% derive_dmrs_sizes
type = pdu.dmrsConfigType+1;
if type == 1 
    Nf_dmrs = 4; 
elseif type == 2
    Nf_dmrs = 6;
else
    error('dmrs type is not supported\n');
end
Nt_dmrs = sum(pdu.DmrsSymbPos);
N_dmrs = Nf_dmrs * Nt_dmrs;
dmrs.Nf_dmrs =  Nf_dmrs;
dmrs.Nt_dmrs = Nt_dmrs;
dmrs.N_dmrs = N_dmrs;

% derive_time_dmrs
dmrs.symIdx_dmrs = find(pdu.DmrsSymbPos);
dmrs.DmrsScramblingId = pdu.DmrsScramblingId;

return
