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

function tbErrList = detPusch_mat5g(pduList, carrier, Xtf, table)

nPdu = length(pduList);
tbErrList = [];

for idxPdu = 1:nPdu
    pdu = pduList{idxPdu};
    puschTable = table;
    % derive configuration
    dmrs    = derive_dmrs_main(pdu, puschTable);
    alloc   = derive_alloc_main(pdu, dmrs);
    coding  = derive_coding_main(pdu, alloc, puschTable);
    
    % generate QAM for data
    % Tb              = pdu.payload';
    % TbCrc = nrCRCEncode(Tb, coding.CRC);
    % TbCbs = nrCodeBlockSegmentLDPC(TbCrc, coding.BGN);
    % TbCodedCbs = nrLDPCEncode(TbCbs, coding.BGN);
    % G = coding.qam * alloc.N_data * alloc.nl; %total number of bits to be transmitted
    % rvIdx = coding.rvIdx;
    % if pdu.I_LBRM
    %     TbRateMatCbs = nrRateMatchLDPC(TbCodedCbs, G, rvIdx, coding.qamstr, alloc.nl, coding.Nref);
    % else
    %     TbRateMatCbs = nrRateMatchLDPC(TbCodedCbs, G, rvIdx, coding.qamstr, alloc.nl);
    % end
    % Qams = nrPUSCH(TbRateMatCbs,coding.qamstr, alloc.nl, alloc.dataScramblingId, alloc.RNTI);
        
    % configure PUSCH 
    
    carrier5g = nrCarrierConfig;
    carrier5g.NCellID = carrier.N_ID_CELL;
    carrier5g.NSizeGrid = carrier.N_grid_size_mu;
    carrier5g.NStartGrid = carrier.N_grid_start_mu;
    global SimCtrl
    if SimCtrl.genTV.forceSlotIdxFlag
        carrier5g.NSlot = SimCtrl.genTV.slotIdx(1);
    else
        carrier5g.NSlot = mod(carrier.idxSlot + carrier.idxSubframe * ...
            carrier.N_slot_subframe_mu, carrier.N_slot_frame_mu);
    end
    carrier5g.SubcarrierSpacing = 15*2^carrier.mu;
    
    pusch5g = nrPUSCHConfig;
    pusch5g.NSizeBWP = pdu.BWPSize;
    pusch5g.NStartBWP = pdu.BWPStart;
    pusch5g.Modulation = coding.qamstr;
    pusch5g.NumLayers = alloc.nl;
    pusch5g.SymbolAllocation = [pdu.StartSymbolIndex, pdu.NrOfSymbols];
    pusch5g.PRBSet = [pdu.rbStart:pdu.rbStart+pdu.rbSize-1];
%     pusch5g.NID = pdu.DmrsScramblingId;
    pusch5g.NID = pdu.dataScramblingId;
    pusch5g.RNTI = pdu.RNTI;
    if pdu.TransformPrecoding == 0
        pusch5g.TransformPrecoding = 1; % enable
    else
        pusch5g.TransformPrecoding = 0; % disable
    end
    pusch5g.DMRS.DMRSConfigurationType = pdu.dmrsConfigType+1;
    pusch5g.NumAntennaPorts = 1;
    pusch5g.TPMI = 0;
    if pdu.FrequencyHopping
        pusch5g.FrequencyHopping = 'enabled';
    else
        pusch5g.FrequencyHopping = 'neither';
    end
    idx = find(pdu.DmrsSymbPos);
    if pdu.DmrsMappingType == 0 % not defined in FAPI PDU, for compliance test only
        pusch5g.MappingType = 'A';
        pusch5g.DMRS.DMRSTypeAPosition = idx(1)-1-pdu.StartSymbolIndex;
    else
        pusch5g.MappingType = 'B';
        pusch5g.DMRS.DMRSTypeAPosition = 2;
    end
    pusch5g.DMRS.DMRSAdditionalPosition = (length(idx)-dmrs.maxLength)/dmrs.maxLength;
    pusch5g.DMRS.DMRSLength = dmrs.maxLength;
    pusch5g.DMRS.DMRSPortSet = alloc.portIdx-1;
    pusch5g.DMRS.NIDNSCID = pdu.DmrsScramblingId;
    pusch5g.DMRS.NRSID = pdu.puschIdentity;
    pusch5g.DMRS.NSCID = pdu.SCID;
    pusch5g.DMRS.NumCDMGroupsWithoutData = pdu.numDmrsCdmGrpsNoData;
    if pdu.groupOrSequenceHopping == 1
        pusch5g.DMRS.GroupHopping = 1;
    elseif pdu.groupOrSequenceHopping == 2
        pusch5g.DMRS.SequenceHopping = 1;
    end
% 
%     % UCI on PUSCH
%     isUciPresent = bitand(uint16(pdu.pduBitmap),uint16(2^1));
%     isDataPresent = bitand(uint16(pdu.pduBitmap),uint16(2^0));
%     if pdu.I_LBRM == 0
%         betaOffsetHarqAck_mapping = [1, 2, 2.5, 3.125, 4, 5, 6.26, 8, 10, ...
%             12.625, 15.875, 20, 31, 50, 80, 126];
%         pusch5g.BetaOffsetACK = betaOffsetHarqAck_mapping(pdu.betaOffsetHarqAck+1);
% 
%         betaOffsetCsi_mapping = [1.125, 1.250, 1.375, 1.625, 1.750, 2.000, ...
%             2.250, 2.500, 2.875, 3.125, 3.500, 4.000, 5.000, 6.250, 8.000, ...
%             10.000, 12.625, 15.875, 20.000];
%         pusch5g.BetaOffsetCSI1 = betaOffsetCsi_mapping(pdu.betaOffsetCsi1+1);
%         pusch5g.BetaOffsetCSI2 = betaOffsetCsi_mapping(pdu.betaOffsetCsi2+1);
% 
%         alphaScaling_mapping = [0.5, 0.65, 0.8, 1];
%         pusch5g.UCIScaling = alphaScaling_mapping(pdu.alphaScaling+1);
% 
%         tcr = pdu.targetCodeRate/10/1024;
%         tbs = pdu.TBSize*8;
%         oack = pdu.harqAckBitLength;
%         if (~isDataPresent) && (pdu.csiPart1BitLength > 0 && pdu.csiPart2BitLength ==0)%Ref. Sec. 6.3.2.1.1 38.212
%             if(pdu.harqAckBitLength <= 1)
%                 oack = length(pdu.harqPayload);
%             end
%         end
%         ocsi1 = pdu.csiPart1BitLength;
%         ocsi2 = sum(pdu.csiPart2BitLength);
%         rmInfo = nrULSCHInfo(pusch5g,tcr,tbs,oack,ocsi1,ocsi2);
% 
%         data = pdu.payload(:);
%         ack = pdu.harqPayload(:);
%         csi1 = pdu.csiPart1Payload(:);
%         csi2 = pdu.csiPart2Payload(:);
% 
%         encUL = nrULSCH;
%         encUL.TargetCodeRate = tcr;
%         setTransportBlock(encUL,data);
% 
%         rv = pdu.rvIndex;
%         culsch = encUL(pusch5g.Modulation,pusch5g.NumLayers,rmInfo.GULSCH,rv);
% 
%         cack  = nrUCIEncode(ack,rmInfo.GACK,pusch5g.Modulation);
%         ccsi1 = nrUCIEncode(csi1,rmInfo.GCSI1,pusch5g.Modulation);
%         ccsi2 = nrUCIEncode(csi2,rmInfo.GCSI2,pusch5g.Modulation);
% 
%         cw = nrULSCHMultiplex(pusch5g,tcr,tbs,culsch,cack,ccsi1,ccsi2);
%         Qams = nrPUSCH(carrier5g,pusch5g,cw);
%     end    
% 
%     Xtf = zeros(carrier.N_sc, carrier.N_symb_slot, alloc.nl);
%     ind = nrPUSCHIndices(carrier5g,pusch5g);
%     Xtf(ind) = Qams;
% 
%     % pipeline for DMRS
% 
%     r_dmrs = sqrt(dmrs.energy)*nrPUSCHDMRS(carrier5g, pusch5g);
%     ind_dmrs = nrPUSCHDMRSIndices(carrier5g,pusch5g);
%     Xtf(ind_dmrs) = r_dmrs;
%     Xtf1 = zeros(carrier.N_sc, carrier.N_symb_slot, carrier.numTxAnt);
% %     idxAnt = mod(alloc.portIdx-1, carrier.numTxAnt) + 1;
%     idxAnt = 1:alloc.nl; 
%     Xtf1(:,:,idxAnt) = Xtf;
%     if idxPdu == 1
%         Xtf_sum = Xtf1;
%     else
%         Xtf_sum = Xtf_sum + Xtf1;
%     end

    % PUSCH detection using 5GToolbox 

    harqEntity.RedundancyVersion = pdu.rvIndex;
    harqEntity.HARQProcessID = pdu.harqProcessID;

    puschextra.TargetCodeRate = pdu.targetCodeRate/10/1024;
    puschextra.XOverhead = 0;      

    [decbits, tbErrList{idxPdu}] = hPuschRx(carrier5g, pusch5g, Xtf, harqEntity, puschextra);

end

% Xtf = Xtf_sum;

end

function [decbits,blkerr] = hPuschRx(carrier, pusch, rxGrid, harqEntity, puschextra)


[puschIndices,puschIndicesInfo] = nrPUSCHIndices(carrier,pusch);
MRB = numel(puschIndicesInfo.PRBSet);
trBlkSize = nrTBS(pusch.Modulation,pusch.NumLayers,MRB,puschIndicesInfo.NREPerPRB,puschextra.TargetCodeRate,puschextra.XOverhead);

dmrsLayerSymbols = nrPUSCHDMRS(carrier,pusch);
dmrsLayerIndices = nrPUSCHDMRSIndices(carrier,pusch);
[estChannelGrid,noiseEst] = nrChannelEstimate_1(carrier,rxGrid,dmrsLayerIndices,dmrsLayerSymbols,'CDMLengths',pusch.DMRS.CDMLengths);


% Get PUSCH resource elements from the received grid
[puschRx,puschHest] = nrExtractResources(puschIndices,rxGrid,estChannelGrid);

% Equalization
[puschEq,csi] = nrEqualizeMMSE(puschRx,puschHest,noiseEst);

% Decode PUSCH physical channel
[ulschLLRs,rxSymbols] = nrPUSCHDecode(carrier,pusch,puschEq,noiseEst);

% Apply channel state information (CSI) produced by the equalizer
csi = nrLayerDemap(csi);
Qm = length(ulschLLRs) / length(rxSymbols);
csi = reshape(repmat(csi{1}.',Qm,1),[],1);
ulschLLRs = ulschLLRs .* csi;

% Decode the UL-SCH transport channel
decodeULSCH = nrULSCHDecoder;
decodeULSCH.MultipleHARQProcesses = true;
decodeULSCH.TargetCodeRate = puschextra.TargetCodeRate;
decodeULSCH.MaximumLDPCIterationCount = 10;
decodeULSCH.TransportBlockLength = trBlkSize;
[decbits,blkerr] = decodeULSCH(ulschLLRs,pusch.Modulation,pusch.NumLayers,harqEntity.RedundancyVersion,harqEntity.HARQProcessID);

end
