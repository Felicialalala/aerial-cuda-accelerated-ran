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

function gNB = gNBtransmitter(gNB)
% function gNB = gNBtransmitter(gNB)
%
% This function simulates gNB functionality
%
% Input:    gNB: structure for a single gNB
%
% Output:   gNB: structure for a single gNB
%

% Now we generate time domain samples slot by slot, which significantly
% increases the process latency and memeoy size.  We will need to generate
% time domain samples symbol by symbol in later version.
%

Mac = gNB.Mac;
Phy = gNB.Phy;

[FAPIpdu, Mac] = gNbDlMacSendPduToPhy(Mac);
Phy = gNbDlPhyGenSig(Phy, FAPIpdu);

global SimCtrl
if SimCtrl.timeDomainSim
    Phy = gNbDlGenTimeDomainSig(Phy);
end

gNB.Mac = Mac;
gNB.FAPIpdu = FAPIpdu;
gNB.Phy = Phy;

return;


function [FAPIpdu, Mac] = gNbDlMacSendPduToPhy(Mac)

carrier = Mac.Config.carrier;
global SimCtrl
if SimCtrl.genTV.forceSlotIdxFlag
    carrier.idxSlotInFrame = SimCtrl.genTV.slotIdx(1);
end
NsymPerSlot = carrier.N_symb_slot;
idxSlot = carrier.idxSlot;
idxSlotInFrame = carrier.idxSlotInFrame;
idxFrame = carrier.idxFrame;
idxPdu = 1;
ssbIdx = 1;
pdcchIdx = 1;
pdschIdx = 1;
csirsIdx = 1;
bfwIdx = 1;

FAPIpdu = [];

alloc = Mac.tx.alloc;
N_alloc = length(alloc);

lastPdcchIdx = 0;
for idxAlloc = 1:N_alloc
    thisAlloc = alloc{idxAlloc};
    allocType = thisAlloc.type;
    if strcmp(allocType, 'pdcch')
        lastPdcchIdx = idxAlloc;
    end
end

lastPdschIdx = 0;
for idxAlloc = 1:N_alloc
    thisAlloc = alloc{idxAlloc};
    allocType = thisAlloc.type;
    if strcmp(allocType, 'pdsch')
        lastPdschIdx = idxAlloc;
    end
end

lastCsirsIdx = 0;
for idxAlloc = 1:N_alloc
    thisAlloc = alloc{idxAlloc};
    allocType = thisAlloc.type;
    if strcmp(allocType, 'csirs')
        lastCsirsIdx = idxAlloc;
    end
end

lastBfwIdx = 0;
for idxAlloc = 1:N_alloc
    thisAlloc = alloc{idxAlloc};
    allocType = thisAlloc.type;
    if strcmp(allocType, 'bfw')
        lastBfwIdx = idxAlloc;
    end
end

for idxAlloc = 1:N_alloc
    thisAlloc = alloc{idxAlloc};
    allocType = thisAlloc.type;
    switch allocType
        case 'ssb'
            ssb = Mac.Config.ssb;
            % find if it is right frame for SSB
            periodFrame = ssb.periodFrame;
            ssb_exist = 0;
            if mod(idxFrame, periodFrame) == 0
                for symIdxInFrame = (0:NsymPerSlot-1) + idxSlotInFrame*NsymPerSlot
                    if ismember(symIdxInFrame, ssb.symIdxInFrame)
                        block_idx = Mac.Config.ssb.block_idx;
                        ssbBitMap = Mac.Config.ssb.ssbBitMap;
                        if ssbBitMap(block_idx+1)
                            Mac.Config.ssb.firstSymIdx = symIdxInFrame-idxSlotInFrame*NsymPerSlot;
                            [pdu, Mac] = gNbDlGenMac2PhyPdu(Mac, 'ssb', ssbIdx);
                            pdu.ssbPduIdx = ssbIdx;
                            pdu.lastSsbPdu = 0;
                            ssbIdx = ssbIdx + 1;
                            FAPIpdu{idxPdu} = pdu;
                            idxPdu = idxPdu + 1;
                            ssb_exist = 1;
                        else
                            if block_idx + 1 == ssb.L_max
                                block_idx = 0;
                            else
                                block_idx = block_idx + 1;
                            end
                            Mac.Config.ssb.block_idx = block_idx;
                        end
                    end
                end
            end
            if ssb_exist
                FAPIpdu{idxPdu-1}.lastSsbPdu = 1;
            end
        case 'pdcch'
            pdcch = Mac.Config.pdcch;
            [pdu, Mac] = gNbDlGenMac2PhyPdu(Mac, 'pdcch', pdcchIdx);
            pdu.pdcchPduIdx = pdcchIdx;
            pdcchIdx = pdcchIdx + 1;
            if idxAlloc == lastPdcchIdx
                pdu.lastPdcchPdu = 1;
            else
                pdu.lastPdcchPdu = 0;
            end
            FAPIpdu{idxPdu} = pdu;
            idxPdu = idxPdu + 1;
        case 'pdsch'
            pdsch = Mac.Config.pdsch;
            [pdu, Mac] = gNbDlGenMac2PhyPdu(Mac, 'pdsch', pdschIdx);
            pdu.pdschPduIdx = pdschIdx;
            pdschIdx = pdschIdx + 1;
            if idxAlloc == lastPdschIdx
                pdu.lastPdschPdu = 1;
            else
                pdu.lastPdschPdu = 0;
            end
            FAPIpdu{idxPdu} = pdu;
            idxPdu = idxPdu + 1;
        case 'csirs'
            csirs = Mac.Config.csirs;
            [pdu, Mac] = gNbDlGenMac2PhyPdu(Mac, 'csirs', csirsIdx);
            pdu.csirsPduIdx = csirsIdx;
            csirsIdx = csirsIdx + 1;
            if idxAlloc == lastCsirsIdx
                pdu.lastCsirsPdu = 1;
            else
                pdu.lastCsirsPdu = 0;
            end
            FAPIpdu{idxPdu} = pdu;
            idxPdu = idxPdu + 1;
        case 'bfw'
            bfw = Mac.Config.bfw;
            [pdu, Mac] = gNbDlGenMac2PhyPdu(Mac, 'bfw', bfwIdx);
            pdu.bfwPduIdx = bfwIdx;
            bfwIdx = bfwIdx + 1;
            if idxAlloc == lastBfwIdx
                pdu.lastBfwPdu = 1;
            else
                pdu.lastBfwPdu = 0;
            end
            FAPIpdu{idxPdu} = pdu;
            idxPdu          = idxPdu + 1;
        otherwise
            error('allocType is not supported ...\n');
    end
end

idxSlot = idxSlot + 1;
carrier = updateTimeIndex(carrier, idxSlot, SimCtrl.genTV.forceSlotIdxFlag);

Mac.Config.carrier = carrier;

return


function [pdu, Mac] = gNbDlGenMac2PhyPdu(Mac, pduType, allocIdx)
carrier = Mac.Config.carrier;
global SimCtrl
if SimCtrl.genTV.forceSlotIdxFlag
    carrier.idxSlotInFrame = SimCtrl.genTV.slotIdx(1);
end
table = Mac.Config.table;

switch pduType
    case 'ssb'
        ssb = Mac.Config.ssb;
        pdu.type = 'ssb';
        pdu.physCellId = carrier.N_ID_CELL;
        pdu.betaPss = ssb.betaPss;
        pdu.ssbBlockIndex = ssb.block_idx;
        pdu.ssbSubcarrierOffset = ssb.ssbSubcarrierOffset;
        pdu.SsbOffsetPointA = ssb.SsbOffsetPointA;
        pdu.bchPayloadFlag = 1;
        if (ssb.seed > 0) || isempty(ssb.mib)
            rng(ssb.seed);
            ssb.mib = num2str(round(rand(1, 24)));
        end
        payload =  bin2dec(num2str(ssb.mib));
        pdu.bchPayload = payload;
        [enablePrcdBf, PM_W] = loadPrecodingMatrix(ssb.prcdBf_vec(allocIdx), table);
        pdu.enablePrcdBf = enablePrcdBf;
        pdu.PM_W = PM_W;

        % TX precoding and beaforming PDU SCF-FAPI table 3-43
        pdu.numPRGs = 1; % only support 1
        pdu.prgSize = 20; % SSB nPRB = 20        
        pdu.PMidx = ssb.prcdBf_vec(allocIdx);
        switch allocIdx
            case 1
                pdu.digBFInterfaces = ssb.digBFInterfaces_0;
                pdu.beamIdx = ssb.beamIdx_0;
            case 2
                pdu.digBFInterfaces = ssb.digBFInterfaces_1;
                pdu.beamIdx = ssb.beamIdx_1;
            case 3
                pdu.digBFInterfaces = ssb.digBFInterfaces_2;
                pdu.beamIdx = ssb.beamIdx_2;
            case 4
                pdu.digBFInterfaces = ssb.digBFInterfaces_3;
                pdu.beamIdx = ssb.beamIdx_3;
            otherwise
                error('allocIdx is not supported ...\n');
        end                
        pdu.nSSBStartSymbol = ssb.firstSymIdx;
        ssb.block_idx = ssb.block_idx + 1;
        if ssb.block_idx == ssb.L_max
            ssb.block_idx = 0;
        end
        Mac.Config.ssb = ssb;
    case 'pdcch'
        pdcch = Mac.Config.pdcch{allocIdx};
        pdu.type = 'pdcch';
        pdu.dciUL = pdcch.dciUL;
        pdu.BWPSize = pdcch.BWPSize;
        pdu.BWPStart = pdcch.BWPStart;
        pdu.SubcarrierSpacing = carrier.mu;
        pdu.CyclicPrefix = carrier.CpType;
        pdu.StartSymbolIndex = pdcch.StartSymbolIndex;
        pdu.DurationSymbols = pdcch.DurationSymbols;
        % Use 64 bits for FreqDomainResource.
        % MSB corresponds to the lowest 6RB.
        coresetMap = pdcch.coresetMap;
        FreqDomainResource = zeros(1, 64);
        len = length(coresetMap);
        FreqDomainResource(1:len) = coresetMap;
        pdu.FreqDomainResource0 = bin2dec(num2str(FreqDomainResource(1:32)));
        pdu.FreqDomainResource1 = bin2dec(num2str(FreqDomainResource(33:64)));
        coresetIdx = pdcch.coresetIdx;
        if coresetIdx > 0
            pdu.CoreSetType = 1;
            pdu.CceRegMappingType = pdcch.CceRegMappingType;
            pdu.RegBundleSize = pdcch.RegBundleSize;
            pdu.InterleaverSize = pdcch.InterleaverSize;
            pdu.ShiftIndex = pdcch.ShiftIndex;
        else
            pdu.CoreSetType= 0;
            pdu.CceRegMappingType = 1;
            pdu.RegBundleSize = 6;
            pdu.InterleaverSize = 2;
            pdu.ShiftIndex = carrier.N_ID_CELL;
        end
        pdu.precoderGranularity = pdcch.precoderGranularity; % TBD
        pdu.numDlDci = pdcch.numDlDci;
        pdu.testModel = pdcch.testModel;
        
        pdu.idxUE = pdcch.idxUE;
        N_CCE = sum(pdcch.coresetMap)*pdcch.DurationSymbols;
        OccupiedCCE = zeros(1, N_CCE);
        for idxDCI = 1:pdu.numDlDci
            DCI = pdcch.DCI{idxDCI};
            pduDci.RNTI = DCI.RNTI;
            if coresetIdx > 0
                pduDci.ScramblingId = DCI.ScramblingId; % pdcch.dmrsId; % DCI.ScramblingId;
                pduDci.ScramblingRNTI = DCI.ScramblingRNTI;
            else
                pduDci.ScramblingId = carrier.N_ID_CELL;
                pduDci.ScramblingRNTI = 0;
            end
            rnti = pduDci.RNTI;
            aggrL =  DCI.AggregationLevel;
            if pdcch.forceCceIndex
                pduDci.CceIndex = DCI.cceIndex;
            else
                M_candidate = floor(N_CCE/aggrL);
                if M_candidate > 8
                    M_candidate = 8;
                elseif M_candidate == 7
                    M_candidate = 6;
                end
                isCSS = pdcch.isCSS;
                slot_number = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
                CceIndexCandidate = findPdcchSS(coresetIdx, slot_number, rnti, N_CCE, aggrL, M_candidate, isCSS);
                findCceIdx = 0;
                for idxCandidate = 1:length(CceIndexCandidate)
                    startCCE = CceIndexCandidate(idxCandidate);
                    occupied = sum(OccupiedCCE(startCCE+1:startCCE + aggrL));
                    if ~ occupied
                        OccupiedCCE(startCCE+1:startCCE + aggrL) = 1;
                        pduDci.CceIndex = startCCE;
                        findCceIdx = 1;
                        break;
                    end
                end
                if findCceIdx == 0
                    error('DCI can not be allocated ...\n');
                end
            end
            pduDci.AggregationLevel = aggrL;
            [enablePrcdBf, PM_W] = loadPrecodingMatrix(DCI.prcdBf, table);
            pduDci.enablePrcdBf = enablePrcdBf;
            pduDci.PM_W = PM_W;

            % TX precoding and beaforming PDU SCF-FAPI table 3-43
            pduDci.numPRGs = 1; % only support 1
            pduDci.prgSize = aggrL * 6 / pdu.DurationSymbols; %
            pduDci.digBFInterfaces = DCI.digBFInterfaces;
            pduDci.PMidx = DCI.prcdBf;
            pduDci.beamIdx = DCI.beamIdx;
            pduDci.beta_PDCCH_1_0 = DCI.beta_PDCCH_1_0; % SCF FAPIv2
            pduDci.powerControlOffsetSS = DCI.powerControlOffsetSS; % SCF FAPIv2
            pduDci.powerControlOffsetSSProfileNR = DCI.powerControlOffsetSSProfileNR; % SCF FAPIv4
            pduDci.PayloadSizeBits = DCI.PayloadSizeBits;
            
            if pdu.testModel
                % for test only. L2 should generate PN23 sequnce and provide to L1
                nBits = 6*9*2;
                DCI.PayloadSizeBits = nBits;
                rng(DCI.seed);
                DCI.Payload = round(rand(1, DCI.PayloadSizeBits));
            else
                if (DCI.seed > 0) || isempty(DCI.Payload)
                    rng(DCI.seed);
                    DCI.Payload = round(rand(1, DCI.PayloadSizeBits));
                end
            end
            pduDci.Payload = DCI.Payload;
            pduDci.PayloadSizeBits = DCI.PayloadSizeBits;
            if length(pduDci.Payload) ~= DCI.PayloadSizeBits
                error('DCI payload size is not correct ...');
            end
            pdu.DCI{idxDCI} = pduDci;
        end
    case 'pdsch'
        pdsch = Mac.Config.pdsch{allocIdx};
        pdu.type = 'pdsch';
        pdu.pduBitmap = pdsch.pduBitmap;
        pdu.RNTI = pdsch.RNTI;
        pdu.pduIndex = 0;
        pdu.BWPSize = pdsch.BWPSize;
        pdu.BWPStart = pdsch.BWPStart;
        pdu.SubcarrierSpacing = carrier.mu;
        pdu.CyclicPrefix = carrier.CpType;
        pdu.NrOfCodewords = pdsch.NrOfCodewords;
        pdschTable = table;
        mcs = pdsch.mcsIndex;
        mcsTable = pdsch.mcsTable;
        switch(mcsTable)
            case 0
                mcs_table = pdschTable.McsTable1;
            case 1
                mcs_table = pdschTable.McsTable2;
            case 2
                mcs_table = pdschTable.McsTable3;
        end
        if mcs == 100
            qam = 2;
            codeRate = 1;
        elseif (ismember(mcsTable, [1, 3, 4]) && mcs > 27) || ...
                (ismember(mcsTable, [0, 2]) && mcs > 28)
            qam = pdsch.qamModOrder;
            codeRate = pdsch.targetCodeRate/10/1024;
        else
            qam = mcs_table(mcs+1, 2);
            codeRate = mcs_table(mcs+1, 3)/1024;
        end
        pdu.targetCodeRate = codeRate*1024*10; % per SCF-FAPI spec
        pdu.qamModOrder = qam;
        pdu.mcsIndex = mcs;
        pdu.mcsTable = mcsTable;
        pdu.rvIndex = pdsch.rvIndex;

        DmrsSymbPos = pdsch.DmrsSymbPos;
        nDmrsSymb = sum(DmrsSymbPos(pdsch.StartSymbolIndex + 1:...
            pdsch.StartSymbolIndex + pdsch.NrOfSymbols));
        if pdsch.numDmrsCdmGrpsNoData == 1
            nDmrsSymb = nDmrsSymb/2;
        end
        nDataSymb = pdsch.NrOfSymbols - nDmrsSymb;
        if pdsch.resourceAlloc == 0
%             Nrb = sum(reshape(flipud(int2bit(pdsch.rbBitmap,8,true)),[],1));
            Nrb = sum(reshape(flipud(dec2bin(pdsch.rbBitmap,8)')-'0',[],1));
            Ninfo = Nrb*min(156,12*nDataSymb)*pdsch.nrOfLayers*codeRate*qam;
        else
            Ninfo = pdsch.rbSize*min(156,12*nDataSymb)*pdsch.nrOfLayers*codeRate*qam;
        end
        TBS_table = pdschTable.TBS_table;
        if Ninfo <= 3824
            %for "small" sizes, look up TBS in a table. First round the
            %number of information bits.
            n = max(3,(floor(log2(Ninfo)) - 6));
            Ninfo_prime = max(24, 2^n*floor(Ninfo / 2^n));

            %next lookup in table closest TBS (without going over).
            compare = Ninfo_prime - TBS_table;
            compare(compare > 0) = -100000;
            [~,max_index] = max(compare);
            TBS = TBS_table(max_index);
            C = 1;
        else
            %for "large" sizes, compute TBS. First round the number of
            %information bits to a power of two.
             n = floor(log2(Ninfo-24)) - 5;
             Ninfo_prime = max(3840, 2^n*round((Ninfo-24)/2^n));

            %Next, compute the number of code words. For large code rates,
            %use base-graph 1. For small code rate use base-graph 2.
            if codeRate < 1/4
                %display('Case coderate < 1/4')
                C = ceil( (Ninfo + 24) / 3816);
                TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
            else
                if Ninfo_prime > 8424
                    %display('Case Ninfo_prime > 8424')
                    C = ceil( (Ninfo_prime + 24) / 8424);
                    TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
                else
                    %display('Case 3824<Ninfo<=8424')
                    C = 1;
                    TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
                end
            end
        end
        pdu.TBSize = ceil(TBS/8);
        if (ismember(mcsTable, [1, 3, 4]) && mcs > 27) || ...
                (ismember(mcsTable, [0, 2]) && mcs > 28)
            pdu.TBSize = pdsch.TBSize;
        end
        pdu.dataScramblingId = pdsch.dataScramblingId;
        pdu.nrOfLayers = pdsch.nrOfLayers;
        if(isfield(pdsch,'nlAbove16'))
            pdu.nlAbove16 = pdsch.nlAbove16;
        else
            pdu.nlAbove16 = 0;
        end
        pdu.transmissionScheme = pdsch.transmissionScheme;
        pdu.refPoint = pdsch.refPoint;
        pdu.DmrsSymbPos = pdsch.DmrsSymbPos;
        pdu.DmrsMappingType = pdsch.DmrsMappingType; % (not defined in FAPI PDU, for compliance test only)
        pdu.dmrsConfigType = pdsch.dmrsConfigType;
        pdu.DmrsScramblingId = pdsch.DmrsScramblingId;
        pdu.SCID = pdsch.SCID;
        pdu.numDmrsCdmGrpsNoData = pdsch.numDmrsCdmGrpsNoData;
        dmrsPorts = zeros(1, 16);
        portIdx = pdsch.portIdx;
        nL = pdu.nrOfLayers;
        if length(portIdx) > nL
            portIdx = portIdx(1:nL);
        elseif length(portIdx) < nL
            portIdx = [portIdx, portIdx(end)+[1:(nL-length(portIdx))]];
        end
        dmrsPorts(portIdx+1) = 1;
        % flip based on SCF FAPI spec. (first port -> LSB)
        pdu.dmrsPorts = flip(dmrsPorts);
        pdu.resourceAlloc = pdsch.resourceAlloc;
        pdu.rbBitmap = pdsch.rbBitmap;
        pdu.rbStart = pdsch.rbStart;
        pdu.rbSize = pdsch.rbSize;
        pdu.VRBtoPRBMapping = pdsch.VRBtoPRBMapping;
        pdu.StartSymbolIndex = pdsch.StartSymbolIndex;
        pdu.NrOfSymbols = pdsch.NrOfSymbols;
        [enablePrcdBf, PM_W] = loadPrecodingMatrix(pdsch.prcdBf, pdschTable);
        pdu.enablePrcdBf = enablePrcdBf;
        pdu.PM_W = PM_W;

        % TX precoding and beamforming PDU SCF-FAPI v10.02 table 3-43
        pdu.numPRGs         = 1;
        pdu.prgSize         = pdsch.rbSize;
        if(isfield(pdsch,'numPRGs'))
            pdu.numPRGs         = pdsch.numPRGs;
        end
        if(isfield(pdsch,'prgSize'))
            pdu.prgSize         = pdsch.prgSize;
        end

        % static beamforming
        if SimCtrl.enable_static_dynamic_beamforming && pdsch.digBFInterfaces
            pdu.numPRGs         = 1;
            pdu.prgSize         = pdsch.rbSize;
        end

        pdu.digBFInterfaces = pdsch.digBFInterfaces;
        pdu.PMidx           = pdsch.prcdBf;
        pdu.beamIdx         = pdsch.beamIdx;

        pdu.powerControlOffset = pdsch.powerControlOffset;
        pdu.powerControlOffsetSS = pdsch.powerControlOffsetSS;
        pdu.IsLastCbPresent = pdsch.IsLastCbPresent;
        pdu.isInlineTbCrc = pdsch.isInlineTbCrc;
        pdu.dlTbCrc = pdsch.dlTbCrc;
        % for LBRM
        pdu.I_LBRM = pdsch.I_LBRM;
        pdu.maxLayers = pdsch.maxLayers;
        if mcsTable == 1
            pdu.maxQm = 8;
        else
            pdu.maxQm = 6;
        end
        pdu.n_PRB_LBRM = pdsch.n_PRB_LBRM;
        % for test model
        pdu.testModel = pdsch.testModel;
        
        if pdsch.testModel
            % for test only. L2 should generate PN23 sequnce and provide to L1
            rng(pdsch.seed); 
            nBits = Ninfo/codeRate;
            pdu.payload = round(rand(1, nBits));
            pdu.TBSize = ceil(nBits/8);
        else
            if (pdsch.seed > 0) || isempty(pdsch.payload)
                rng(pdsch.seed);
                pdu.payload = round(rand(1, pdu.TBSize*8));
            else
                if (length(pdsch.payload) == pdu.TBSize*8)
                    pdu.payload = pdsch.payload;
                else
                    error('pdsch payload size mismatch ...\n');
                end
            end
        end
        pdu.idxUE = pdsch.idxUE;
        pdu.idxUeg = pdsch.idxUeg;
    case 'csirs'
        csirs = Mac.Config.csirs{allocIdx};
        pdu.type = 'csirs';
        pdu.BWPSize = csirs.BWPSize;
        pdu.BWPStart = csirs.BWPStart;
        pdu.SubcarrierSpacing = carrier.mu;
        pdu.CyclicPrefix = carrier.CpType;
        pdu.StartRB = csirs.StartRB;
        pdu.NrOfRBs = csirs.NrOfRBs;
        pdu.CSIType = csirs.CSIType;
        pdu.Row = csirs.Row;
        pdu.FreqDomain = bin2dec(num2str(csirs.FreqDomain));
        pdu.SymbL0 = csirs.SymbL0;
        pdu.SymbL1 = csirs.SymbL1;
        pdu.CDMType = csirs.CDMType;
        pdu.FreqDensity = csirs.FreqDensity;
        pdu.ScrambId = csirs.ScrambId;
        [enablePrcdBf, PM_W] = loadPrecodingMatrix(csirs.prcdBf, table);
        pdu.enablePrcdBf = enablePrcdBf;
        pdu.PM_W = PM_W;
        pdu.idxUE = csirs.idxUE;

        % TX precoding and beaforming PDU SCF-FAPI table 3-43
        pdu.numPRGs = 1;
        if(isfield(csirs,'numPRGs'))
            pdu.numPRGs = csirs.numPRGs;
        end
        pdu.prgSize = csirs.NrOfRBs;
        if(isfield(csirs,'prgSize'))
            pdu.prgSize = csirs.prgSize;
        end

        % static beamforming
        if SimCtrl.enable_static_dynamic_beamforming && csirs.digBFInterfaces
            pdu.numPRGs         = 1;
            pdu.prgSize         = csirs.NrOfRBs;
        end

        pdu.digBFInterfaces = csirs.digBFInterfaces;
        pdu.PMidx = csirs.prcdBf;
        pdu.beamIdx = csirs.beamIdx;

        pdu.powerControlOffset = csirs.powerControlOffset;
        pdu.powerControlOffsetSS = csirs.powerControlOffsetSS;
    case 'bfw'
        bfw       = Mac.Config.bfw{allocIdx};
        pdu.type  = 'bfw';
        pdu.bfwUL = bfw.bfwUL;
        
        % convert unique ue Idxs to RNTI:
        [pdu.RNTI, ~, layerIdxToUeIdxMap] = unique(bfw.ueIdxs,'stable');
        pdu.srsChestBufferIndex = pdu.RNTI;
        pdu.RNTI = pdu.RNTI + 1; % increment to ensure non-zero values
        layerIdxToUeIdxMap = layerIdxToUeIdxMap - 1; % convert to C 0 indexing
        pdu.nUes           = length(pdu.RNTI);
        
        % for now, set a trivial pdu indicies:
        pdu.pduIndex = 0 : (pdu.nUes - 1);
        
        % gnbAntIdxs:
        pdu.gnbAntIdxStart = zeros(pdu.nUes, 1);
        pdu.gnbAntIdxEnd   = bfw.nRxAnt * ones(pdu.nUes, 1) - 1;
       
        % Compute number of antennas per user, and their indicies into SRS buffer:
        maxNumUeAnts   = 4;
        pdu.numOfUeAnt = zeros(pdu.nUes, 1);
        ueAntIdxs      = zeros(maxNumUeAnts, pdu.nUes);
        
        for layerIdx = 0 : (bfw.nBfLayers - 1)
            ueIdx    = layerIdxToUeIdxMap(layerIdx + 1);
            ueAntIdx = bfw.ueLayersIdxs(layerIdx + 1);
            
            ueAntIdxs(pdu.numOfUeAnt(ueIdx + 1) + 1, ueIdx + 1) = ueAntIdx;
            pdu.numOfUeAnt(ueIdx + 1) = pdu.numOfUeAnt(ueIdx + 1) + 1;
        end
        pdu.ueAntIdx0 = ueAntIdxs(1,:);
        pdu.ueAntIdx1 = ueAntIdxs(2,:);
        pdu.ueAntIdx2 = ueAntIdxs(3,:);
        pdu.ueAntIdx3 = ueAntIdxs(4,:);
        
        % common user group paramaters:
        pdu.prgSize = 2;
        if(isfield(bfw,'prgSize'))
            pdu.prgSize = bfw.prgSize;
        end
        pdu.bfwPrbGrpSize = bfw.prbGrpSize;

        pdu.rbStart  = bfw.startPrb;
        pdu.rbSize   = bfw.nPrb;
        pdu.numPRGs  = bfw.nPrbGrp;
        pdu.startPrb = bfw.startPrb;
        
        % BFW compression format
        pdu.compressBitWidth = SimCtrl.bfw.compressBitWidth;
        pdu.beta = SimCtrl.bfw.beta;
            
    otherwise
        pdu = [];
end

return


function Phy = gNbDlPhyGenSig(Phy, FAPIpdu)

table = Phy.Config.table;
carrier = Phy.Config.carrier;
Chan_DL = Phy.Chan_DL;
global SimCtrl
if SimCtrl.genTV.forceSlotIdxFlag
    carrier.idxSlotInFrame = SimCtrl.genTV.slotIdx(1);
end
NsymPerSlot = carrier.N_symb_slot;
idxSlot = carrier.idxSlot;
idxSubframe = carrier.idxSubframe;
idxFrame = carrier.idxFrame;
idxSlotInFrame = carrier.idxSlotInFrame;
Nsubcarrier = carrier.N_grid_size_mu*carrier.N_sc_RB;
Nant = carrier.numTxPort;

if SimCtrl.genTV.fhMsgMode
    [fhMsgParams, modCompRbParam, modComplist] = genFhMsgParams(FAPIpdu, table, carrier);
end

Xtf = zeros(Nsubcarrier, NsymPerSlot, Nant);

if idxSubframe == 0 && idxSlot == 0
    Phy.tx.Xtf_frame = [];
end
Xtf_remap = [];
Xtf_remap_trsnzp = [];

ssbPduList = [];
pdcchPduList = [];
pdschPduList = [];
csirsPduList = [];
bfwPduList   = [];

idxSsb = 1;
idxPdcch = 1;
idxPdsch = 1;
idxCsirs = 1;
idxBfw   = 1;
nPdu = length(FAPIpdu);

for idxPdu = 1:nPdu
    pdu = FAPIpdu{idxPdu};
    switch pdu.type
        case 'ssb'
            ssbPduList{idxSsb} = pdu;
            idxSsb = idxSsb + 1;
            ssb = Phy.Config.ssb;
            if pdu.lastSsbPdu
                Xtf = genSsb(ssbPduList, ssb, carrier, Xtf);
            end
            Phy.Config.ssb.pduList = ssbPduList;
        case 'pdcch'
            pdcchPduList{idxPdcch} = pdu;
            idxPdcch = idxPdcch + 1;
            pdcch = Phy.Config.pdcch;
            if pdu.lastPdcchPdu
                Xtf = genPdcch(pdcchPduList, table, carrier, Xtf);
            end
            Phy.Config.pdcch.pduList = pdcchPduList;
        case 'pdsch'
            pdschPduList{idxPdsch} = pdu;
            idxPdsch = idxPdsch + 1;
            pdsch = Phy.Config.pdsch;
            if pdu.lastPdschPdu
                [Xtf, Xtf_remap, Xtf_remap_trsnzp] = genPdsch(pdschPduList, table, carrier, Xtf, csirsPduList, Chan_DL);
                Phy.Config.pdsch.Xtf_remap = Xtf_remap;
                Phy.Config.pdsch.Xtf_remap_trsnzp = Xtf_remap_trsnzp;
            end
            Phy.Config.pdsch.pduList = pdschPduList;
        case 'csirs'
            csirsPduList{idxCsirs} = pdu;
            idxCsirs = idxCsirs + 1;
            csirs = Phy.Config.csirs;
            if pdu.lastCsirsPdu
                [Xtf, Xtf_remap, Xtf_remap_trsnzp] = genCsirs(csirsPduList, table, carrier, Xtf);
            end
            Phy.Config.csirs.Xtf_remap = Xtf_remap;
            Phy.Config.csirs.Xtf_remap_trsnzp = Xtf_remap_trsnzp;
            Phy.Config.csirs.pduList = csirsPduList;
        case 'bfw'
            bfwPduList{idxBfw} = pdu;
            idxBfw = idxBfw + 1;
            bfw    = Phy.Config.bfw;
            if pdu.lastBfwPdu
                [bfwBuf, bfwCompBuf] = genBfw(bfwPduList, carrier, Phy.srsChEstDatabase, SimCtrl.alg.bfwPowerNormAlg_selector);
                Phy.Config.bfw.bfwBuf  = bfwBuf;
                Phy.Config.bfw.bfwCompBuf  = bfwCompBuf;
                Phy.Config.bfw.pduList = bfwPduList;
            end
        otherwise
            error('pdu type is not supported ...\n');
    end
end

if SimCtrl.genTV.fhMsgMode
    Xtf0 = Xtf;
    [fhMsg, modCompMsg] = txFhMsg(fhMsgParams, Xtf, modComplist);
    if SimCtrl.genTV.fhMsgMode == 1 % modulation compression SE4 only
        Xtf = rxFhMsg(fhMsg, carrier);
    elseif SimCtrl.genTV.fhMsgMode == 2 % modulation compression SE4/5
        Xtf = rxFhMsg(modCompMsg, carrier);
    end
    err = sum(sum(sum(abs(Xtf0-Xtf).^2)));
    if err > 1
        warning('\n modComp err = %f\n', err);
    end
else
    fhMsg = [];
    modCompMsg = [];
end

if SimCtrl.capSamp.enable && SimCtrl.genTV.slotIdx == idxSlotInFrame
    [Nf,Nt,Na] = size(Xtf);
    iqWidth = 16;
    if (SimCtrl.capSamp.isComp == 1)
        iqWidth = SimCtrl.oranComp.iqWidth(1);
        FSOffset = SimCtrl.oranComp.FSOffset(1);
        Ref_c = SimCtrl.oranComp.Ref_c(1);
        Nre_max = SimCtrl.oranComp.Nre_max;
        max_amp_ul = SimCtrl.oranComp.max_amp_ul;
    end
    Xtf = loadCapSamp(SimCtrl.capSamp.fileName, 0, iqWidth, FSOffset, Ref_c, Nre_max, max_amp_ul, Nf/12, Nt, Na);
end

if SimCtrl.genTV.enable && SimCtrl.genTV.FAPI ...
        && ismember(idxSlotInFrame, SimCtrl.genTV.slotIdx)
    node = 'gNB';
    Xtf_prach = [];
    Xtf_srs = [];
    Xtf_uncomp = Xtf;
    Xtf_prach_uncomp = Xtf_prach;
    Xtf_srs_uncomp = Xtf_srs;
    saveTV_FAPI(SimCtrl.genTV, FAPIpdu, Phy.Config, Xtf_uncomp, Xtf, Xtf_prach_uncomp, Xtf_prach, Xtf_srs_uncomp, Xtf_srs, Xtf_remap, Xtf_remap_trsnzp, Phy.srsChEstDatabase, fhMsg, modCompMsg, idxSlotInFrame, node, table, 0);
end

if SimCtrl.enable_dynamic_BF
    % simple BFW: only send signal to the first numTxPort of TX antennas
    BFW = [diag(ones(1, carrier.numTxPort)), zeros(carrier.numTxPort, ...
        carrier.numTxAnt - carrier.numTxPort)];
    [d1, d2, d3] = size(Xtf);
    Xtf = reshape(reshape(Xtf, d1*d2, d3) * BFW, d1, d2, carrier.numTxAnt);
end


if SimCtrl.enableDlTxBf  % apply tx beamforming after compression
    Xtf = dlTxBf(Xtf, FAPIpdu, Phy.srsState, table, Chan_DL, SimCtrl);
end

Phy.tx.Xtf = Xtf;
Phy.tx.Xtf_frame = [Phy.tx.Xtf_frame, Xtf];

idxSlot = idxSlot + 1;

carrier = updateTimeIndex(carrier, idxSlot, SimCtrl.genTV.forceSlotIdxFlag);

Phy.Config.carrier = carrier;

return

function Phy = gNbDlGenTimeDomainSig(Phy)

carrier = Phy.Config.carrier;
NsymPerSlot = carrier.N_symb_slot;
idxSlot = carrier.idxSlot;
idxSubframe = carrier.idxSubframe;

global SimCtrl
if SimCtrl.enable_dynamic_BF
    Nant = carrier.numTxAnt;
else
    Nant = carrier.numTxPort;
end

Nfft = carrier. Nfft;
mu = carrier.mu;
N_sc = carrier.N_sc;
N_samp_slot = carrier.N_samp_slot;

if (mu == 1 && idxSubframe == 0 && idxSlot == 1) || ...
        (mu == 0 && idxSubframe == 1 && idxSlot == 0)
    Phy.tx.Xt_frame = [];
end
Xt_frame = Phy.tx.Xt_frame;
Xt = zeros(N_samp_slot, Nant);

Xtf = Phy.tx.Xtf;

% for non-PRACH
kappa = 64; % constants defined in 38.211
lenCP0 = (144*2^(-carrier.mu)+16)*kappa*carrier.T_c/carrier.T_samp;%(144+16)/2048*Nfft;
lenCP1 = (144*2^(-carrier.mu))*kappa*carrier.T_c/carrier.T_samp;%144/2048*Nfft;
for idxAnt = 1:Nant
    slotTD = [];
    for idxSym = 1:NsymPerSlot
        idxSymInOneSubframe = idxSlot*carrier.N_symb_slot + idxSym;
        % IFFT
        symFD = Xtf(:, idxSym, idxAnt);
        ifft_in = zeros(1, Nfft);
        ifft_in(1:N_sc/2) = symFD(N_sc/2+1:end);
        ifft_in(end-N_sc/2+1:end) = symFD(1:N_sc/2);
        ifft_out = ifft(ifft_in, Nfft)*sqrt(Nfft);
        % Add CP
        if idxSymInOneSubframe == 1 || idxSymInOneSubframe == 1+7*2^carrier.mu
            symTD = [ifft_out(end-lenCP0+1:end), ifft_out];
        else
            symTD = [ifft_out(end-lenCP1+1:end), ifft_out];
        end
        slotTD = [slotTD, symTD];
    end
    Xt(:, idxAnt) = slotTD;
end

Phy.tx.Xt = Xt;
Phy.tx.Xt_frame = [Xt_frame, Xt];

return