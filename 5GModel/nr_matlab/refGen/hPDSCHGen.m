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

function Xtf = hPDSCHGen(pdsch, carrier, table)


pduList = pdsch.pduList;
Xtf_remap = pdsch.Xtf_remap;
nPdu = length(pduList);

for idxPdu = 1:nPdu
    pdu = pduList{idxPdu};
    pdschTable = table;
%     Xtf_remap = pdu.Xtf_remap;
    % derive configuration
    dmrs    = derive_dmrs_main(pdu, pdschTable);
    alloc   = derive_alloc_main(pdu, dmrs, Xtf_remap);
    coding  = derive_coding_main(pdu, alloc, pdschTable);
    
    % pipeline for payload
    Tb              = pdu.payload';
    TbCrc = nrCRCEncode(Tb, coding.CRC);
    TbCbs = nrCodeBlockSegmentLDPC(TbCrc, coding.BGN);
    TbCodedCbs = nrLDPCEncode(TbCbs, coding.BGN);
%     G = coding.qam * alloc.N_data * alloc.nl; %total number of bits to be transmitted   
    G = coding.qam*alloc.N_data_used*alloc.nl;
    rvIdx = coding.rvIdx;    
    if pdu.I_LBRM
        TbRateMatCbs = nrRateMatchLDPC(TbCodedCbs, G, rvIdx, coding.qamstr, alloc.nl, coding.Nref);
    else
        TbRateMatCbs = nrRateMatchLDPC(TbCodedCbs, G, rvIdx, coding.qamstr, alloc.nl);
    end        
    Qams = nrPDSCH(TbRateMatCbs,coding.qamstr, alloc.nl, alloc.dataScramblingId, alloc.RNTI);
    
    carrier5g = nrCarrierConfig;
    carrier5g.NCellID = carrier.N_ID_CELL;
    carrier5g.NSizeGrid = carrier.N_grid_size_mu;
    carrier5g.NStartGrid = carrier.N_grid_start_mu;
    global SimCtrl
    if SimCtrl.genTV.forceSlotIdxFlag
        carrier5g.NSlot = SimCtrl.genTV.slotIdx(1);
    else
        carrier5g.NSlot = mod((carrier.idxSlot + carrier.idxSubframe * ...
            carrier.N_slot_subframe_mu -1), carrier.N_slot_frame_mu);
    end
    carrier5g.SubcarrierSpacing = 15*2^carrier.mu;
    
    pdsch5g = nrPDSCHConfig;
    pdsch5g.NSizeBWP = pdu.BWPSize;
    pdsch5g.NStartBWP = pdu.BWPStart;
    pdsch5g.Modulation = coding.qamstr;
    pdsch5g.NumLayers = alloc.nl;
    nlAbove16 = pdu.nlAbove16;
    if pdu.DmrsMappingType == 0 % not defined in FAPI PDU, for compliance test only
        pdsch5g.MappingType = 'A';
    else
        pdsch5g.MappingType = 'B';
    end
    pdsch5g.SymbolAllocation = [pdu.StartSymbolIndex, pdu.NrOfSymbols];
    if pdu.resourceAlloc == 0
%         pdsch5g.PRBSet = find(reshape(int2bit(pdu.rbBitmap,8, false),1,[])==1)-1;
        pdsch5g.PRBSet = find(reshape(flipud(dec2bin(pdu.rbBitmap,8)')-'0',1,[])==1)-1;
    else
        pdsch5g.PRBSet = [pdu.rbStart:pdu.rbStart+pdu.rbSize-1];
    end
    pdsch5g.VRBToPRBInterleaving = pdu.VRBtoPRBMapping;
    pdsch5g.NID = pdu.DmrsScramblingId;
    pdsch5g.RNTI = pdu.RNTI;
    pdsch5g.DMRS.DMRSConfigurationType = pdu.dmrsConfigType+1;
    if pdu.refPoint == 0
%     if pdu.RNTI < 65535
        pdsch5g.DMRS.DMRSReferencePoint = 'CRB0';
    else
        pdsch5g.DMRS.DMRSReferencePoint = 'PRB0';
    end
    idx = find(pdu.DmrsSymbPos);
    pdsch5g.DMRS.DMRSTypeAPosition = carrier.dmrsTypeAPos;
    pdsch5g.DMRS.DMRSAdditionalPosition = (length(idx)-dmrs.maxLength)/dmrs.maxLength;
    pdsch5g.DMRS.DMRSLength = dmrs.maxLength;
    pdsch5g.DMRS.DMRSPortSet = alloc.portIdx-1;
    pdsch5g.DMRS.NIDNSCID = pdu.DmrsScramblingId;
    pdsch5g.DMRS.NSCID = pdu.SCID;
    pdsch5g.DMRS.NumCDMGroupsWithoutData = pdu.numDmrsCdmGrpsNoData;    
    pdsch5g.ReservedRE = find_reservedRE(Xtf_remap, carrier.numTxPort);
    
    % compute Tx power level for dmrs and qam
    pdsch2csirs = pdu.powerControlOffset - 8;
    csirs2ssb = (pdu.powerControlOffsetSS-1) * 3;
    pdsch2ssb = pdsch2csirs + csirs2ssb;
    beta_qam = 10^(pdsch2ssb/20);
    beta_dmrs = 10^(pdsch2ssb/20);
    
    Xtf = zeros(carrier.N_sc, carrier.N_symb_slot, alloc.nl);
    ind = nrPDSCHIndices(carrier5g,pdsch5g);
    Xtf(ind) = beta_qam * Qams;
    
    % pipeline for DMRS
    
    r_dmrs = sqrt(dmrs.energy)*nrPDSCHDMRS(carrier5g, pdsch5g);
    ind_dmrs = nrPDSCHDMRSIndices(carrier5g,pdsch5g);
    Xtf(ind_dmrs) = beta_dmrs * r_dmrs;
    Xtf1 = zeros(carrier.N_sc, carrier.N_symb_slot, carrier.numTxPort);
    Xtf1(:,:,alloc.portIdx + 8*pdu.SCID + nlAbove16*16) = Xtf;
    if idxPdu == 1
        Xtf_sum = Xtf1;
    else
        Xtf_sum = Xtf_sum + Xtf1;
    end
end

Xtf = Xtf_sum;

return

function ReservedRE = find_reservedRE(Xtf_remap, numTxAnt)

Xtf_remap = Xtf_remap(:);
Xtf_remap = repmat(Xtf_remap, numTxAnt);
Xtf_remap = Xtf_remap(:);
ReservedRE = find(Xtf_remap == 1)-1;

return
