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

function H_est = pusch_rkhs_chEst(Y,table,slotNumber,Nf,Nt,N_dmrs_id,...
    nl, portIdx, n_scid, symIdx_dmrs, Nt_dmrs, nPrb, startPrb, df, ...
    maxLength, numDmrsCdmGrpsNoData, enableTfPrcd, ...
    N_slot_frame, N_symb_slot, puschIdentity,...
    groupOrSequenceHopping, PuschParamsUeg,nUe)

% perform channel estimation for all users.

% convert to C zero indexing:
startPrb    = startPrb - 1;
symIdx_dmrs = symIdx_dmrs - 1;
portIdx     = portIdx - 1;


%%
% UE GROUP PARAMATERS

ueGrpPrms = [];

% user paramaters:
ueGrpPrms.nUe                  = nUe;
ueGrpPrms.numDmrsCdmGrpsNoData = numDmrsCdmGrpsNoData;

% time/frequency allocation:
ueGrpPrms.startSc     = startPrb * 12;
ueGrpPrms.dmrsSymIdxs = symIdx_dmrs;
ueGrpPrms.nDmrsSym    = maxLength;
ueGrpPrms.startSc     = 12 * startPrb;

% grid paramaters:
ueGrpPrms.gridBitmask = uint8(0);
ueGrpPrms.gridIdxs    = zeros(nl,1);

%  MIMO dimensions:
ueGrpPrms.nGnbAnt = size(Y,3);
ueGrpPrms.nLayers = nl;



%%
%SEQUENCES

% scrambling sequence:

if enableTfPrcd == 0
    r_dmrs   = gen_dmrs_sym(slotNumber, Nf, Nt, N_dmrs_id);
    freqIdxs = startPrb*6 : (6*(startPrb + nPrb) - 1);
    r_dmrs   = r_dmrs(freqIdxs + 1,symIdx_dmrs + 1,n_scid + 1);
else
    [r_dmrs, ~, ~] = gen_dmrs_dftsofdm(Nf, startPrb, nPrb, N_slot_frame, N_symb_slot, symIdx_dmrs, ...
        slotNumber, puschIdentity, groupOrSequenceHopping);
end


%%
% PORT PARAMATERS


% initialize port paramaters:
gridPrms    = cell(2,1);

for i = 1 : 2
    gridPrms{i}.toccBitmask = uint8(0);
    gridPrms{i}.toccPrms    = cell(2,1);

    for j = 1 : 2
        gridPrms{i}.toccPrms{j}.foccBitmask = uint8(0);
        gridPrms{i}.toccPrms{j}.foccPrms    = cell(2,1);
    
        for k = 1 : 2
            gridPrms{i}.toccPrms{j}.foccPrms{k}.layerIdx = 0;
        end
    end
end


portIdx = uint8(portIdx);
for layerIdx = 0 : (nl - 1)

    % code paramaters
    foccIdx = bitand(portIdx(layerIdx + 1), 1); 
    gridIdx = bitshift(bitand(portIdx(layerIdx + 1),2), -1);
    toccIdx = bitshift(bitand(portIdx(layerIdx + 1),4), -2);

    % update active gird bitmask
    ueGrpPrms.gridBitmask            = bitor(ueGrpPrms.gridBitmask, gridIdx + 1);
    ueGrpPrms.gridIdxs(layerIdx + 1) = gridIdx;

    % update active tocc bitmap
    gridPrms{gridIdx + 1}.toccBitmask = bitor(gridPrms{gridIdx + 1}.toccBitmask, toccIdx + 1);

    % update active focc bitmap
    gridPrms{gridIdx + 1}.toccPrms{toccIdx + 1}.foccBitmask = bitor(gridPrms{gridIdx + 1}.toccPrms{toccIdx + 1}.foccBitmask, foccIdx + 1); 

    % layerIdx
    gridPrms{gridIdx + 1}.toccPrms{toccIdx + 1}.foccPrms{foccIdx + 1}.layerIdx = layerIdx;
end

%%
% COMPUTE BLOCKS

if(nPrb <= 64)
    nPrbPerComputeBlock = nPrb;
    nComputeBlocks      = 1;
else
    nPrbPerComputeBlock = 64;
    nComputeBlocks      = ceil(nPrb / 64);
end

computeBlocksCommonPrm           = [];
computeBlocksCommonPrm.nSc       = nPrbPerComputeBlock * 12;
computeBlocksCommonPrm.nDmrsSc   = nPrbPerComputeBlock * 6;
computeBlocksCommonPrm.nPrb      = nPrbPerComputeBlock;
computeBlocksCommonPrm.nZpDmrsSc = table.push_rkhs_tables.prbPrms{nPrbPerComputeBlock}.nZpDmrsSc;
computeBlocksCommonPrm.zpIdx     = table.push_rkhs_tables.prbPrms{nPrbPerComputeBlock}.zpIdx;
computeBlocksCommonPrm.nCpInt    = table.push_rkhs_tables.prbPrms{nPrbPerComputeBlock}.nCpInt;


compBlockPrms = cell(nComputeBlocks,1);

for computeBlockIdx = 0 : (nComputeBlocks - 1)
    compBlockPrm = [];

    compBlockPrm.startInputPrb         = startPrb + computeBlockIdx * nPrbPerComputeBlock;
    compBlockPrm.startInputSc          = compBlockPrm.startInputPrb * 12;
    compBlockPrm.nOutputSc             = nPrbPerComputeBlock * 12;
    compBlockPrm.startOutputScInBlock  = 0;
    compBlockPrm.scOffsetIntoChEstBuff = 12 * computeBlockIdx * nPrbPerComputeBlock;

    compBlockPrms{computeBlockIdx + 1} = compBlockPrm;
end

if(nComputeBlocks > 1)
    nEdgePrbs = (nPrb + startPrb) - (compBlockPrms{nComputeBlocks - 1}.startInputPrb + nPrbPerComputeBlock);

    compBlockPrms{nComputeBlocks}.startInputPrb        = startPrb + nPrb - nPrbPerComputeBlock;
    compBlockPrms{nComputeBlocks}.startInputSc         = 12 * compBlockPrms{nComputeBlocks}.startInputPrb;
    compBlockPrms{nComputeBlocks}.nOutputSc            = 12 * nEdgePrbs;
    compBlockPrms{nComputeBlocks}.startOutputScInBlock = 12 * (nPrbPerComputeBlock - nEdgePrbs);
    compBlockPrms{nComputeBlocks}.scOffsetIntoChEstBuff = 12 * (nPrb - nEdgePrbs);
end


%%
% NOISE MEASURMENT PARAMATERS

nNoiseMeasurments     = 0;
noiseMeasurmentMethod = 0;
rolloff               = 0;

if((ueGrpPrms.gridBitmask == 1) && (ueGrpPrms.numDmrsCdmGrpsNoData == 2)) %if empty grid avaliable, use it to measure noise
    nNoiseMeasurments     = computeBlocksCommonPrm.nDmrsSc * ueGrpPrms.nGnbAnt;
    noiseMeasurmentMethod = 1;

elseif(nl == 1) % else, if empty fOCC present, use it to measure noise
    rolloff               = max(computeBlocksCommonPrm.nZpDmrsSc / 4, 10);
    nNoiseMeasurments     = (computeBlocksCommonPrm.nZpDmrsSc / 2 - rolloff) * ueGrpPrms.nGnbAnt;
    noiseMeasurmentMethod = 2;

else % if no empty fOCC or grid, use quite regions of fOCC
    rolloff               = max(floor((computeBlocksCommonPrm.nZpDmrsSc / 2 - computeBlocksCommonPrm.nCpInt) / 4), 3);
    nNoiseMeasurments     = nl * (computeBlocksCommonPrm.nZpDmrsSc / 2 - computeBlocksCommonPrm.nCpInt - 2*rolloff) * ueGrpPrms.nGnbAnt;
    noiseMeasurmentMethod = 3;
end

computeBlocksCommonPrm.rolloff               = rolloff;
computeBlocksCommonPrm.nNoiseMeasurments     = nNoiseMeasurments;
computeBlocksCommonPrm.noiseMeasurmentMethod = noiseMeasurmentMethod;


%%
% USER PARAMATERS

% initialize user paramaters:
uePrms = cell(nUe,1);

% populate user paramaters:
nLayersTot = 0;
for ueIdx = 0 : (nUe - 1)
    nUeLayers = PuschParamsUeg.PuschParams(ueIdx + 1).nl;

    uePrms{ueIdx + 1}.layerStartIdx      = nLayersTot;
    uePrms{ueIdx + 1}.nLayers            = nUeLayers;
    uePrms{ueIdx + 1}.nUeLayersTimesNant = nUeLayers * ueGrpPrms.nGnbAnt;
end

%%
%WRAP

ueGrpPrms.computeBlocksCommonPrm = computeBlocksCommonPrm;


%%
% START

H_est = zeros(ueGrpPrms.nLayers, ueGrpPrms.nGnbAnt, 12*nPrb);


for compBlockIdx = 0 : (nComputeBlocks - 1)
    H_est = pusch_rkhs_chEst_kernel(Y, H_est, r_dmrs, compBlockPrms{compBlockIdx + 1}, ueGrpPrms, uePrms, gridPrms, table.push_rkhs_tables);
end


H_est = permute(H_est, [2 1 3]);

H_est2 = zeros(ueGrpPrms.nGnbAnt, ueGrpPrms.nLayers, 12*273);
freq_idxs = startPrb*12 : ((startPrb + nPrb)*12 - 1);
H_est2(:,:,freq_idxs + 1) = H_est;
H_est = H_est2;

      

    





