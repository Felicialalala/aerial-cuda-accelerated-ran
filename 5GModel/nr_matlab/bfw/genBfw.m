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

function [bfwBuf, bfwCompBuf] = genBfw(bfwPduList, carrier, srsChEstDatabase, bfwPowerNormAlg_selector)
global SimCtrl;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
srsChEstBuff          = srsChEstDatabase.srsChEstBuff;
nSrsUes               = length(srsChEstBuff);
for i = 0 : (nSrsUes - 1)
    temp = srsChEstDatabase.srsChEstBuff{i+1};
    srsChEstDatabase.srsChEstBuff{i+1}  = reshape(fp16nv(real(temp), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(temp), SimCtrl.fp16AlgoSel), [size(temp)]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[bfwBuf,bfwCompBuf, ueGrpsBfwPrms_cell] = genBfw_cuphy(bfwPduList, srsChEstDatabase, bfwPowerNormAlg_selector);

global SimCtrl;
idxSlot = carrier.idxSlotInFrame;

if SimCtrl.genTV.enable && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)
    TVname = [SimCtrl.genTV.TVname, '_BFW_gNB_CUPHY_s', num2str(idxSlot)];
    saveTV_bfw(SimCtrl.genTV.tvDirName, TVname, ueGrpsBfwPrms_cell, srsChEstDatabase, bfwBuf, bfwCompBuf);
    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

end


function [bfwBuf,bfwCompBuf, ueGrpsBfwPrms_cell] = genBfw_cuphy(bfwPduList, srsChEstDatabase, bfwPowerNormAlg_selector)
global SimCtrl;
%%
% PARAMATERS

srsChEstBuff          = srsChEstDatabase.srsChEstBuff;
chEstBuf_startPrbGrps = srsChEstDatabase.startPrbGrps;
srsPrbGrpSize         = srsChEstDatabase.prbGrpSize;
startValidPrgArr      = srsChEstDatabase.startValidPrg;
nValidPrgArr          = srsChEstDatabase.nValidPrg;

nUeGrps = length(bfwPduList);

% Convert FAPI PDU to cuPHY API:
ueGrpsBfwPrms_cell = cell(nUeGrps,1);

for ueGrpIdx = 0 : (nUeGrps - 1)
    bfwPdu        = bfwPduList{ueGrpIdx + 1};
    ueGrpsBfwPrms = [];
    
    % extract ueGrp common paramaters:
    ueGrpsBfwPrms.startPrb         = bfwPdu.rbStart;
    ueGrpsBfwPrms.nPrb             = bfwPdu.rbSize;
    ueGrpsBfwPrms.nRxAnt           = bfwPdu.gnbAntIdxEnd(1) - bfwPdu.gnbAntIdxStart(1) + 1;
    ueGrpsBfwPrms.nBfLayers        = sum(bfwPdu.numOfUeAnt);
    ueGrpsBfwPrms.lambda           = 0;
    ueGrpsBfwPrms.coefBufIdx       = ueGrpIdx;
    ueGrpsBfwPrms.compressBitWidth = bfwPdu.compressBitWidth;
    ueGrpsBfwPrms.beta             = bfwPdu.beta;
    ueGrpsBfwPrms.bfwPrbGrpSize    = bfwPdu.bfwPrbGrpSize;
    ueGrpsBfwPrms.nPrbGrp          = ceil(ueGrpsBfwPrms.nPrb / ueGrpsBfwPrms.bfwPrbGrpSize);
    
    % mapping from ueGrpLayerIdx to ueIdx:
    ueGrpsBfwPrms.ueIdxs = zeros(ueGrpsBfwPrms.nBfLayers, 1);
    
    ueGrpLayerIdx = 0;
    for ueIdxInUeGrp = 0 : (bfwPdu.nUes - 1)
        for ueAntIdx = 0 : (bfwPdu.numOfUeAnt(ueIdxInUeGrp + 1) - 1)
            ueGrpsBfwPrms.ueIdxs(ueGrpLayerIdx + 1) = bfwPdu.RNTI(ueIdxInUeGrp + 1)-1;
            ueGrpLayerIdx = ueGrpLayerIdx + 1;
        end
    end
    
    % mapping from ueGrpLayerIdx to ueLayerIdx:
    ueGrpsBfwPrms.ueLayersIdxs = zeros(ueGrpsBfwPrms.nBfLayers, 1);
    
    maxNumAntPerUe = 4;
    ueAntIdxs = zeros(maxNumAntPerUe, bfwPdu.nUes);
    ueAntIdxs(1,:) = bfwPdu.ueAntIdx0;
    ueAntIdxs(2,:) = bfwPdu.ueAntIdx1;
    ueAntIdxs(3,:) = bfwPdu.ueAntIdx2;
    ueAntIdxs(4,:) = bfwPdu.ueAntIdx3;
    
    ueGrpLayerIdx = 0;
    for ueIdxInUeGrp = 0 : (bfwPdu.nUes - 1)
        for ueAntIdx = 0 : (bfwPdu.numOfUeAnt(ueIdxInUeGrp + 1) - 1)
            ueGrpsBfwPrms.ueLayersIdxs(ueGrpLayerIdx + 1) = ueAntIdxs(ueAntIdx + 1, ueIdxInUeGrp + 1);
            ueGrpLayerIdx = ueGrpLayerIdx + 1;
        end
    end   
    
    %store cuphy paramaters into cell:
    ueGrpsBfwPrms_cell{ueGrpIdx + 1} = ueGrpsBfwPrms;
end

%%
% ALLOCATE BFW MEMORY
bfwBuf  = cell(nUeGrps , 1);

% Allocate Memory For BFW:
for ueGrpIdx = 0 : (nUeGrps - 1)
    nPrbGrp   = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nPrbGrp;
    nRxAnt    = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nRxAnt;
    nBfLayers = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nBfLayers;
    
    bfwBuf{ueGrpIdx + 1} = zeros(nRxAnt, nBfLayers, nPrbGrp);
    compressBitWidth = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.compressBitWidth;
    if compressBitWidth > 0
        bfwCompBuf{ueGrpIdx + 1} = zeros(1, nRxAnt*compressBitWidth*2/8+1, nPrbGrp, nBfLayers);
    else
        bfwBuf{ueGrpIdx + 1} = [];
    end
end


%%
% COMPUTE BFW

for ueGrpIdx = 0 : (nUeGrps - 1)
    startPrb          = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.startPrb;
    nPrb              = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nPrb;
    nRxAnt            = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nRxAnt;
    nBfLayers         = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nBfLayers;
    ueIdxs            = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.ueIdxs;
    ueLayersIdxs      = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.ueLayersIdxs;
    coefBufIdx        = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.coefBufIdx;
    lambda            = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.lambda;
    bfwPrbGrpSize     = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.bfwPrbGrpSize;
    compressBitWidth  = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.compressBitWidth;
    beta              = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.beta;
    nBfwPrbGrp        = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nPrbGrp;

    bfwToSrs_ueGrpSizeRatio = bfwPrbGrpSize / srsPrbGrpSize;
    srsUeGrpOffset          = floor(bfwToSrs_ueGrpSizeRatio / 2);
    startSrsPrbGrp          = floor(startPrb / srsPrbGrpSize);
    
    for bfwPrbGrpIdx = 0 : (nBfwPrbGrp - 1)
        
        % assemble MU-MIMO channel for this prbGrp
        H = zeros(nBfLayers, nRxAnt);
        for i = 0 : (nBfLayers - 1)
            chEstBufIdx  =  ueIdxs(i + 1);
            ueLayerIdx   =  ueLayersIdxs(i + 1);
            
            srsPrgIdx  = startSrsPrbGrp + bfwToSrs_ueGrpSizeRatio * bfwPrbGrpIdx + srsUeGrpOffset;
            startValidPrg = startValidPrgArr(chEstBufIdx + 1);
            nValidPrg = nValidPrgArr(chEstBufIdx + 1);

            if (srsPrgIdx >= startValidPrg) && (srsPrgIdx < (startValidPrg + nValidPrg))
                srs_prbGrpIdx = srsPrgIdx - chEstBuf_startPrbGrps(chEstBufIdx + 1);
            elseif srsPrgIdx < startValidPrg
                srs_prbGrpIdx = startValidPrg - chEstBuf_startPrbGrps(chEstBufIdx + 1);
            elseif srsPrgIdx >= (startValidPrg + nValidPrg)
                srs_prbGrpIdx = startValidPrg + nValidPrg - 1 - chEstBuf_startPrbGrps(chEstBufIdx + 1);
            end

            for gnbAntIdx = 0 : (nRxAnt - 1)
                H(i + 1, gnbAntIdx + 1) = srsChEstBuff{chEstBufIdx + 1}(srs_prbGrpIdx + 1, gnbAntIdx + 1, ueLayerIdx + 1);
            end

        end
        % quantize H to fp16
        H = reshape(fp16nv(real(H), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(H), SimCtrl.fp16AlgoSel), [size(H)]);
        % compute regularized Gram matrix:
        Gr = H * H' + lambda * eye(nBfLayers);
        
        % compute beamforming weights:
        bfw = H' * Gr^(-1);
        
        % normalize:
        switch bfwPowerNormAlg_selector
            case 0
                bfw = bfw / norm(bfw,'fro');

            case 1
                % Compute layer scaling factors:
                layerScalingFactors = zeros(nBfLayers, 1);
                for layerIdx = 0 : (nBfLayers - 1)
                    layerEnergy                       = sum(abs(bfw(:, layerIdx + 1)).^2);
                    layerScalingFactors(layerIdx + 1) = 1 / sqrt(layerEnergy);
                end

                % Normalizer layer beam patterns:
                for layerIdx = 0 : (nBfLayers - 1)
                    bfw(:, layerIdx + 1) = bfw(:, layerIdx + 1) * layerScalingFactors(layerIdx + 1);
                end

                % Compute antenna energys:
                antEnergies = zeros(nRxAnt, 1);
                for antIdx = 0 : (nRxAnt - 1)
                    antEnergies(antIdx + 1) = sum(abs(bfw(antIdx + 1, :)).^2);
                end

                % Compute antenna scaling factor:
                maxAntEnergy     = max(antEnergies);
                antScalingFactor = 1 / sqrt(maxAntEnergy);

                % Normalize:
                bfw = bfw * antScalingFactor;
        end
        % quantize bfw to fp16
        bfw = reshape(fp16nv(real(bfw), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(bfw), SimCtrl.fp16AlgoSel), [size(bfw)]);
      
        % store:
        bfwBuf{coefBufIdx + 1}(:, :, bfwPrbGrpIdx + 1) = bfw;
    end
    
    % BFP compression
    if compressBitWidth > 0
        bfwCompBuf{coefBufIdx + 1} = bfwBfpCompress(bfwBuf{coefBufIdx + 1},...
            compressBitWidth, beta, nRxAnt, nBfLayers, nBfwPrbGrp);
    else
        bfwCompBuf{coefBufIdx + 1}  =[];
    end
end

end


function saveTV_bfw(tvDirName, TVname, ueGrpsBfwPrms_cell, srsChEstDatabase, bfwBuf, bfwCompBuf)

global SimCtrl
[status,msg] = mkdir(tvDirName);
h5File       = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

% paramaters:
srsPrbGrpSize         = srsChEstDatabase.prbGrpSize;
srsChEstBuff          = srsChEstDatabase.srsChEstBuff;
chEstBuf_startPrbGrps = srsChEstDatabase.startPrbGrps;
startValidPrg         = srsChEstDatabase.startValidPrg;
nValidPrg             = srsChEstDatabase.nValidPrg;
nUeGrps               = length(ueGrpsBfwPrms_cell);
nSrsUes               = length(srsChEstBuff);

% layer paramaters:
cuphyBfwLayerPrm = [];
nLayersTot       = 0;
for ueGrpIdx = 0 : (nUeGrps - 1)
    nBfLayers    = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nBfLayers;
    ueIdxs       = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.ueIdxs;
    ueLayersIdxs = ueGrpsBfwPrms_cell{ueGrpIdx + 1}.ueLayersIdxs;
    
    for i = 0 : (nBfLayers - 1)
        cuphyBfwLayerPrm(nLayersTot + 1).chEstInfoBufIdx = uint16(ueIdxs(i + 1));
        cuphyBfwLayerPrm(nLayersTot + 1).ueLayerIndex    = uint8(ueLayersIdxs(i + 1));
        nLayersTot = nLayersTot + 1;
    end    
end
hdf5_write_nv_exp(h5File, 'cuphyBfwLayerPrm', cuphyBfwLayerPrm);

% sizes:
sizes = [];
sizes.nLayersTot      = uint16(nLayersTot);
sizes.nUeGrps         = uint16(nUeGrps);
sizes.nSrsUes         = uint16(nSrsUes);
sizes.srsPrbGrpSize   = uint16(srsPrbGrpSize);
hdf5_write_nv_exp(h5File, 'sizes', sizes);

% static paramaters:
bfwStatParms = [];
bfwStatParms.lambda                   = single(ueGrpsBfwPrms_cell{1}.lambda);
bfwStatParms.bfwPowerNormAlg_selector = uint8(SimCtrl.alg.bfwPowerNormAlg_selector);
bfwStatParms.beta                     = single(ueGrpsBfwPrms_cell{1}.beta);
hdf5_write_nv_exp(h5File, 'bfwStatParms', bfwStatParms);

% bfw paramaters:
cuphyBfwUeGrpPrm   = [];
bfLayerPrmStartIdx = 0;
for ueGrpIdx = 0 : (sizes.nUeGrps - 1)
    cuphyBfwUeGrpPrm(ueGrpIdx + 1).startPrbGrp        = uint16(ueGrpsBfwPrms_cell{ueGrpIdx + 1}.startPrb / ueGrpsBfwPrms_cell{ueGrpIdx + 1}.bfwPrbGrpSize);
    cuphyBfwUeGrpPrm(ueGrpIdx + 1).nPrbGrp            = uint16(ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nPrbGrp);
    cuphyBfwUeGrpPrm(ueGrpIdx + 1).startPrb           = uint16(ueGrpsBfwPrms_cell{ueGrpIdx + 1}.startPrb);
    cuphyBfwUeGrpPrm(ueGrpIdx + 1).nPrb               = uint16(ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nPrb);
    cuphyBfwUeGrpPrm(ueGrpIdx + 1).nRxAnt             = uint16(ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nRxAnt);
    cuphyBfwUeGrpPrm(ueGrpIdx + 1).nBfLayers          = uint8(ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nBfLayers);
    cuphyBfwUeGrpPrm(ueGrpIdx + 1).bfLayerPrmStartIdx = uint16(bfLayerPrmStartIdx);
    cuphyBfwUeGrpPrm(ueGrpIdx + 1).coefBufIdx         = uint16(ueGrpsBfwPrms_cell{ueGrpIdx + 1}.coefBufIdx);
    cuphyBfwUeGrpPrm(ueGrpIdx + 1).bfwPrbGrpSize      = uint16(ueGrpsBfwPrms_cell{ueGrpIdx + 1}.bfwPrbGrpSize);
    bfLayerPrmStartIdx = bfLayerPrmStartIdx + ueGrpsBfwPrms_cell{ueGrpIdx + 1}.nBfLayers;
end
hdf5_write_nv_exp(h5File, 'cuphyBfwUeGrpPrm', cuphyBfwUeGrpPrm);
   
% chEst buff:
for i = 0 : (nSrsUes - 1)
    nameStr = strcat('chEstBuf',num2str(i));
    hdf5_write_nv(h5File, nameStr, complex(single(srsChEstBuff{i + 1})));

    nameStr = strcat('chEstBufHalf',num2str(i));
    hdf5_write_nv(h5File, nameStr, complex(single(srsChEstBuff{i + 1})), 'fp16');
end
hdf5_write_nv(h5File, 'chEstBuf_startPrbGrps', uint16(chEstBuf_startPrbGrps));
hdf5_write_nv(h5File, 'startValidPrg', uint16(startValidPrg));
hdf5_write_nv(h5File, 'nValidPrg', uint16(nValidPrg));

% bfw buff:
for i = 0 : (nUeGrps - 1)
    nameStr = strcat('bfwBuf',num2str(i));
    hdf5_write_nv(h5File, nameStr, complex(single(bfwBuf{i + 1})));
end

% bfw compression buff:
for i = 0 : (nUeGrps - 1)
    nameStr = strcat('bfwCompBuf',num2str(i));
    hdf5_write_nv(h5File, nameStr, uint8(bfwCompBuf{i + 1}));
end

% save SRS TV number if exists
if isfield(SimCtrl.genTV, 'srsTv')
    hdf5_write_nv(h5File, 'srsTv', uint16(SimCtrl.genTV.srsTv));
end

end



