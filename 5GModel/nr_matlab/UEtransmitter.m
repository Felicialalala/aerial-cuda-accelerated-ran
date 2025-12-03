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

function UE = UEtransmitter(UE, idxUE)
% function UE = UEtransmitter(UE)
%
% This function simulates UE functionality
%
% Input:    UE: structure for a single UE
%
% Output:   UE: structure for a single UE
%

% Now we generate time domain samples slot by slot, which significantly
% increases the process latency and memeoy size.  We will need to generate
% time domain samples symbol by symbol in later version.
%

Mac = UE.Mac;
Phy = UE.Phy;
idxUE = UE.idxUE;

[FAPIpdu, Mac] = UeUlMacSendPduToPhy(Mac, idxUE);
Phy = UeUlGenFreqDomainSig(Phy, FAPIpdu, idxUE);

global SimCtrl
if SimCtrl.timeDomainSim
    Phy = UeUlGenTimeDomainSig(Phy);
end

UE.Mac = Mac;
UE.FAPIpdu = FAPIpdu;
UE.Phy = Phy;

return;


function [FAPIpdu, Mac] = UeUlMacSendPduToPhy(Mac, idxUE)

carrier = Mac.Config.carrier;
global SimCtrl
if SimCtrl.genTV.forceSlotIdxFlag
    carrier.idxSlotInFrame = SimCtrl.genTV.slotIdx(1);
end
NsymPerSlot = carrier.N_symb_slot;
N_slot_frame = carrier.N_slot_frame_mu;
idxSlot = carrier.idxSlot;
idxSubframe = carrier.idxSubframe;
idxFrame = carrier.idxFrame;
idxSlotInFrame = carrier.idxSlotInFrame;
idxPdu = 1;
prachIdx = 1;
pucchIdx = 1;
puschIdx = 1;
srsIdx = 1;
FAPIpdu = [];

alloc = Mac.tx.alloc;
N_alloc = length(alloc);

lastPuschIdx = 0;
for idxAlloc = 1:N_alloc
    thisAlloc = alloc{idxAlloc};
    allocType = thisAlloc.type;
    if strcmp(allocType, 'pusch')
        lastPuschIdx = idxAlloc;
    end
end

for idxAlloc = 1:N_alloc
    thisAlloc = alloc{idxAlloc};
    allocType = thisAlloc.type;
    switch allocType
        case 'prach'
            prach = Mac.Config.prach;
            carrier = Mac.Config.carrier;
            % find if it is right frame for PRACH
            SFN_x = prach{1}.SFN_x;
            SFN_y = prach{1}.SFN_y;
            isRaFrame = (mod(idxFrame, SFN_x) == SFN_y);
            % find if it is right subframe for PRACH
            raSubframeNum = prach{1}.subframeNum;
            isRaSubframe = ismember(idxSubframe, raSubframeNum);
            raSlotNum = prach{1}.n_slot_RA_sel;
            isRaSlot = ismember(idxSlot, raSlotNum);
            global SimCtrl
            forceSlotIdxFlag = SimCtrl.genTV.forceSlotIdxFlag;
            if (((isRaFrame && isRaSubframe) || prach{1}.allSubframes)&&...
                    isRaSlot) || forceSlotIdxFlag
                payload = 0;
                [pdu, Mac] = UeUlGenMac2PhyPdu(Mac, 'prach', prachIdx, payload);
                pdu.prachPduIdx = prachIdx;
                prachIdx = prachIdx + 1;
                FAPIpdu{idxPdu} = pdu;
                idxPdu = idxPdu + 1;
            end
        case 'pucch'
            payload = [];
            [pdu, Mac] = UeUlGenMac2PhyPdu(Mac, 'pucch', pucchIdx, payload);
            pdu.pucchPduIdx = pucchIdx;
            pucchIdx = pucchIdx + 1;
            FAPIpdu{idxPdu} = pdu;
            idxPdu = idxPdu + 1;
        case 'pusch'
            payload = [];
            [pdu, Mac] = UeUlGenMac2PhyPdu(Mac, 'pusch', puschIdx, payload);
            Mac.tx.alloc{idxAlloc}.payload = pdu.payload;
            pdu.puschPduIdx = puschIdx;
            puschIdx = puschIdx + 1;
            if idxAlloc == lastPuschIdx
                pdu.lastPuschPdu = 1;
            else
                pdu.lastPuschPdu = 0;
            end
            FAPIpdu{idxPdu} = pdu;
            idxPdu = idxPdu + 1;
        case 'srs'
            payload = [];
            srs = Mac.Config.srs;
            if (mod(N_slot_frame * idxFrame + mod(idxSlotInFrame, carrier.N_slot_frame_mu) - srs{1}.Toffset, srs{1}.Tsrs) == 0)
                is_SRS_slot  = 1;
            else
                is_SRS_slot = 0;
            end
            if (srs{1}.resourceType > 0 && is_SRS_slot) || (srs{1}.resourceType == 0)
                [pdu, Mac] = UeUlGenMac2PhyPdu(Mac, 'srs', srsIdx, payload);
                pdu.srsPduIdx = srsIdx;
                srsIdx = srsIdx + 1;
                FAPIpdu{idxPdu} = pdu;
                idxPdu = idxPdu + 1;
            end
        otherwise

    end
end

if SimCtrl.enableUlDlCoSim == 0
    idxSlot = idxSlot + 1;
end

carrier = updateTimeIndex(carrier, idxSlot, SimCtrl.genTV.forceSlotIdxFlag);

Mac.Config.carrier = carrier;

return


function [pdu, Mac] = UeUlGenMac2PhyPdu(Mac, pduType, idx, payload)

carrier = Mac.Config.carrier;
global SimCtrl
if SimCtrl.genTV.forceSlotIdxFlag
    carrier.idxSlotInFrame = SimCtrl.genTV.slotIdx(1);
end
table = Mac.Config.table;
idxSlotInFrame = carrier.idxSlotInFrame;

switch pduType
    case 'prach'
        prach = Mac.Config.prach{idx};
        pdu.type = 'prach';
        pdu.physCellID = carrier.N_ID_CELL;
        pdu.NumPrachOcas = prach.N_t_RA_slot;
        pdu.prachFormat = prach.preambleFormat;
        pdu.numRa = prach.msg1_FDM;
        pdu.prachStartSymbol = prach.startingSym;
        pdu.numCs = prach.N_CS;
        pdu.prmbIdx = prach.prmbIdx;
        pdu.configurationIndex = prach.configurationIndex;
        pdu.restrictedSet = prach.restrictedSet;
        pdu.rootSequenceIndex = prach.rootSequenceIndex;
    case 'pucch'
        pucch = Mac.Config.pucch{idx};
        pdu.type = 'pucch';
        pdu.RNTI = pucch.RNTI;
        pdu.BWPSize = pucch.BWPSize;
        pdu.BWPStart = pucch.BWPStart;
        pdu.SubcarrierSpacing = carrier.mu;
        pdu.CyclicPrefix = carrier.CpType;
        pdu.FormatType = pucch.FormatType;
        pdu.multiSlotTxIndicator = pucch.multiSlotTxIndicator;
        pdu.pi2Bpsk = pucch.pi2Bpsk;
        pdu.prbStart = pucch.prbStart;
        pdu.prbSize = pucch.prbSize;
        pdu.StartSymbolIndex = pucch.startSym;
        pdu.NrOfSymbols = pucch.nSym;
        pdu.freqHopFlag = pucch.freqHopFlag;
        pdu.secondHopPRB = pucch.secondHopPRB;
        pdu.groupHopFlag = pucch.groupHopFlag;
        pdu.sequenceHopFlag = pucch.sequenceHopFlag;
        pdu.hoppingId = pucch.hoppingId;
        pdu.InitialCyclicShift = pucch.cs0;
        pdu.dataScramblingId = pucch.dataScramblingId;
        pdu.TimeDomainOccIdx = pucch.tOCCidx;
        pdu.PreDftOccIdx = pucch.PreDftOccIdx;
        pdu.PreDftOccLen = pucch.PreDftOccLen;
        pdu.AddDmrsFlag = pucch.AddDmrsFlag;
        pdu.DmrsScramblingId = pucch.DmrsScramblingId;
        pdu.DMRScyclicshift = pucch.DMRScyclicshift;
        pdu.positiveSR = pucch.positiveSR;
        pdu.DTX = pucch.DTX;
        pdu.BitLenHarq = pucch.BitLenHarq;
        pdu.BitLenSr = pucch.BitLenSr;

        switch pdu.FormatType
            case {0, 1}
                pdu.SRFlag = pucch.SRFlag; % to do: SRFlag has been removed since FAPIv10. BitLenSr (SrBitLen) is used to indicate SR occasion
                if pdu.BitLenHarq>0
                    if pdu.DTX
                        pdu.payload = 2*ones(1, pdu.BitLenHarq);
                    else
                        if strcmp(SimCtrl.force_pucch_payload_as, 'random') % this is the default setting
                            pdu.payload = round(rand(1, pdu.BitLenHarq));
                        elseif strcmp(SimCtrl.force_pucch_payload_as, 'zeros')
                            pdu.payload = zeros(1, pdu.BitLenHarq);
                        elseif strcmp(SimCtrl.force_pucch_payload_as, 'ones')
                            pdu.payload = ones(1, pdu.BitLenHarq);    
                        end
                    end
                else
                    pdu.payload = [];
                end
            case {2, 3}
                pdu.maxCodeRate    = pucch.maxCodeRate;
                pdu.BitLenCsiPart1 = pucch.BitLenCsiPart1;
                pdu.UciP1ToP2Crpd.numPart2s = pucch.UciP1ToP2Crpd.numPart2s;

                if pdu.UciP1ToP2Crpd.numPart2s > 0
                    pdu.BitLenCsiPart2 = csiP2PayloadSizeCalc(pucch.rank);
                else
                    pdu.BitLenCsiPart2 = 0;
                end

                LenSeq1 = pdu.BitLenHarq + pdu.BitLenSr + pdu.BitLenCsiPart1;
                LenSeq2 = pdu.BitLenCsiPart2;

                if LenSeq1>0
                    if pdu.DTX
                        pdu.payloadSeq1 = 2*ones(LenSeq1, 1);
                    else
                        if strcmp(SimCtrl.force_pucch_payload_as, 'random') % this is the default setting
                            pdu.payloadSeq1 = round(rand(LenSeq1, 1));
                        elseif strcmp(SimCtrl.force_pucch_payload_as, 'zeros')
                            pdu.payloadSeq1 = zeros(LenSeq1, 1);
                        elseif strcmp(SimCtrl.force_pucch_payload_as, 'ones')
                            pdu.payloadSeq1 = ones(LenSeq1, 1);     
                        end

                        if pdu.BitLenCsiPart1>0 && pdu.UciP1ToP2Crpd.numPart2s > 0
                            if (pucch.rankBitOffset < (pdu.BitLenHarq + pdu.BitLenSr))
                                error('Error! rank bits must be within CSI-P1 payload');
                            end

                            if (pucch.rankBitOffset + pucch.rankBitSize) > LenSeq1
                                error('Error! Cannot place rank bits within CSI-P1 payload');
                            end

                            rankBits = flip(dec2bin(pucch.rank, pucch.rankBitSize));
                            for i = 0:(pucch.rankBitSize - 1)
                                pdu.payloadSeq1(pucch.rankBitOffset + 1 + i) = str2num(rankBits(i + 1));
                            end

                        end
                    end
                else
                    pdu.payloadSeq1 = [];
                end
                %% fix me: if the number of CSI-part-2 bits is less than 3, zeros shall be appended to the second UCI sequence until its length equals 3
                if LenSeq2>0
                    if pdu.DTX
                        pdu.payloadSeq2 = 2*ones(LenSeq2, 1);
                    else
                        pdu.payloadSeq2 = round(rand(LenSeq2, 1));
                    end
                else
                    pdu.payloadSeq2 = [];
                end
            otherwise
                error('PUCCH format is not supported.');
        end

%%Yan: initialize the pseudo-random sequence for group and sequence
%%hopping at the beginning of each frame, Sec. 6.3.2.2.1, TS 38.211
        if mod(idxSlotInFrame, carrier.N_slot_frame_mu) == 0 || 1
            [pucch] = initCSequences(pucch);
        end
        pdu.cSequenceGH = pucch.cSequenceGH;
    case 'pusch'
        pusch = Mac.Config.pusch{idx};
        pdu.type = 'pusch';
        pdu.pduBitmap = pusch.pduBitmap;
        pdu.RNTI = pusch.RNTI;
        pdu.pduIndex = 0;
        pdu.BWPSize = pusch.BWPSize;
        pdu.BWPStart = pusch.BWPStart;
        pdu.SubcarrierSpacing = carrier.mu;
        pdu.CyclicPrefix = carrier.CpType;
        pdu.NrOfCodewords = pusch.NrOfCodewords;
        puschTable = table;
        mcs = pusch.mcsIndex;
        mcsTable = pusch.mcsTable;
        switch(mcsTable)
            case 0
                mcs_table = puschTable.McsTable1;
            case 1
                mcs_table = puschTable.McsTable2;
            case 2
                mcs_table = puschTable.McsTable3;
            case 3
                mcs_table = puschTable.McsTable4;
            case 4
                mcs_table = puschTable.McsTable5;
        end
        if mcs == 100
            qam = 2;
            codeRate = 1024;
        elseif (ismember(mcsTable, [1, 3, 4]) && mcs > 27) || ...
                (ismember(mcsTable, [0, 2]) && mcs > 28)
            qam = pusch.qamModOrder;
            codeRate = pusch.targetCodeRate/10;
        else
            if pusch.pi2BPSK && ((mcsTable == 3 && mcs <= 1) || ...
                    (mcsTable == 4 && mcs <= 5))
                factor = 2;
            else
                factor = 1;
            end
            qam = mcs_table(mcs+1, 2)/factor;
            codeRate = mcs_table(mcs+1, 3)*factor;
        end
        pdu.targetCodeRate = codeRate*10; % per SCF-FAPI spec
        codeRate = codeRate/1024;
        pdu.qamModOrder = qam;
        pdu.mcsIndex = mcs;
        pdu.mcsTable = mcsTable;
        pdu.TransformPrecoding = pusch.TransformPrecoding;
        DmrsSymbPos = pusch.DmrsSymbPos;
        nDmrsSymb = sum(DmrsSymbPos(pusch.StartSymbolIndex + 1:...
            pusch.StartSymbolIndex + pusch.NrOfSymbols));
        % half of REs on DMRS symbols can be used for data
        isDataPresent = bitand(uint16(pdu.pduBitmap),uint16(2^0));
        if pusch.numDmrsCdmGrpsNoData == 1 && isDataPresent
            nDmrsSymb = nDmrsSymb/2;
        end
        nDataSymb = pusch.NrOfSymbols - nDmrsSymb;
        Ninfo = pusch.rbSize*min(156,12*nDataSymb)*pusch.nrOfLayers*codeRate*qam;
        TBS_table = puschTable.TBS_table;
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
                C = ceil( (Ninfo + 24) / 3816);
                TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
            else
                if Ninfo_prime > 8424
                    C = ceil( (Ninfo_prime + 24) / 8424);
                    TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
                else
                    C = 1;
                    TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
                end
            end
        end
        pdu.TBSize = ceil(TBS/8);
        if (ismember(mcsTable, [1, 3, 4]) && mcs > 27) || ...
                (ismember(mcsTable, [0, 2]) && mcs > 28)
            pdu.TBSize = pusch.TBSize;
        end
        pdu.dataScramblingId = pusch.dataScramblingId;
        pdu.nrOfLayers = pusch.nrOfLayers;
        pdu.DmrsSymbPos = pusch.DmrsSymbPos;
        pdu.DmrsMappingType = pusch.DmrsMappingType; % (not defined in FAPI PDU, for compliance test only)
        pdu.dmrsConfigType = pusch.dmrsConfigType;
        pdu.DmrsScramblingId = pusch.DmrsScramblingId;
        pdu.puschIdentity = pusch.puschIdentity;
        pdu.groupOrSequenceHopping = pusch.groupOrSequenceHopping;
        pdu.SCID = pusch.SCID;
        pdu.numDmrsCdmGrpsNoData = pusch.numDmrsCdmGrpsNoData;
        dmrsPorts = zeros(1, 16);
        portIdx = pusch.portIdx;
        nL = pdu.nrOfLayers;
        if length(portIdx) > nL
            portIdx = portIdx(1:nL);
        elseif length(portIdx) < nL
            portIdx = [portIdx, portIdx(end)+[1:(nL-length(portIdx))]];
        end
        dmrsPorts(portIdx+1) = 1;
        % flip based on SCF FAPI spec. (first port -> LSB)
        pdu.dmrsPorts = flip(dmrsPorts);
        pdu.resourceAlloc = pusch.resourceAlloc;
%         pdu.rbBitmap = pusch.rbBitmap;
        pdu.rbStart = pusch.rbStart;
        pdu.rbSize = pusch.rbSize;
        pdu.VRBtoPRBMapping = pusch.VRBtoPRBMapping;
        pdu.FrequencyHopping = pusch.FrequencyHopping;
        pdu.txDirectCurrentLocation = pusch.txDirectCurrentLocation;
        pdu.uplinkFrequencyShift7p5khz = pusch.uplinkFrequencyShift7p5khz;
        pdu.StartSymbolIndex = pusch.StartSymbolIndex;
        pdu.NrOfSymbols = pusch.NrOfSymbols;
        [enablePrcdBf, PM_W] = loadPrecodingMatrix(pusch.prcdBf, puschTable);
        pdu.enablePrcdBf = enablePrcdBf;
        pdu.PM_W = PM_W;
        pdu.rvIndex = pusch.rvIndex;
        pdu.harqProcessID = pusch.harqProcessID;
        pdu.newDataIndicator = pusch.newDataIndicator;
        pdu.numCb = 0; % not used
        pdu.cbPresentAndPosition = 0; % not used
        pdu.idxUE = pusch.idxUE;
        pdu.idxUeg = pusch.idxUeg;
        % for LBRM
        pdu.I_LBRM = pusch.I_LBRM;
        pdu.maxLayers = pusch.maxLayers;
        if mcsTable == 1
            pdu.maxQm = 8;
        else
            pdu.maxQm = 6;
        end
        pdu.n_PRB_LBRM = pusch.n_PRB_LBRM;
        if (~isDataPresent)      % PUSCH without data (w/UCI)
            pdu.payload = [];
            pdu.TBSize = 0;
        else
            if (pusch.seed > 0) || isempty(pusch.payload)
                if SimCtrl.genTV.enable && (SimCtrl.enable_freeze_tx==0) && (SimCtrl.enable_freeze_tx_and_channel==0)
                    rng(pusch.seed); % random seed for input data
                end
                pdu.payload = round(rand(1, pdu.TBSize*8));
                if SimCtrl.puschHARQ.EnableAutoHARQ
                    rng(pusch.seed + idxSlotInFrame); % random seed for AWGN noise generation in HARQ test cases
                end
            else
                if (length(pusch.payload) == pdu.TBSize*8)
                    pdu.payload = pusch.payload;
                else
                    error('pusch payload size mismatch ...\n');
                end
            end
        end
%         if pusch.newDataIndicator
%            Mac.Config.pusch{idx}.payload = pdu.payload;
%         elseif ~(isempty(Mac.Config.pusch{idx}.payload))
%            pdu.payload = Mac.Config.pusch{idx}.payload;
%         end
        
        harqState = Mac.Config.pusch{idx}.harqState;
        harqStatePresent = isfield(harqState,['hpid_',num2str(pdu.harqProcessID),'_rnti_',num2str(pdu.RNTI)]);
        if (~pdu.newDataIndicator && ~harqStatePresent)
            warning(['newDataIndicator=0 and HARQ buffer not found for RNTI ',num2str(pdu.RNTI),' and harqProcessID ',num2str(pdu.harqProcessID)]);
        end
        if pdu.newDataIndicator == 0
            pdu.payload = harqState.(['hpid_',num2str(pdu.harqProcessID),'_rnti_',num2str(pdu.RNTI)]);
        else
            harqState.(['hpid_',num2str(pdu.harqProcessID),'_rnti_',num2str(pdu.RNTI)]) = pdu.payload;
        end
        Mac.Config.pusch{idx}.harqState = harqState;        
        
        enable_multi_csiP2_fapiv3 = SimCtrl.enable_multi_csiP2_fapiv3;
        if enable_multi_csiP2_fapiv3
            csiP2exist = (pusch.flagCsiPart2 > 0);
            pdu.flagCsiPart2 = pusch.flagCsiPart2;
        else
            csiP2exist = bitand(uint16(pusch.pduBitmap),uint16(2^5));
        end
        pdu.harqPayload = pusch.harqPayload;
        pdu.csiPart1Payload = pusch.csiPart1Payload;
        pdu.harqAckBitLength = pusch.harqAckBitLength;
        pdu.csiPart1BitLength = pusch.csiPart1BitLength;
        if enable_multi_csiP2_fapiv3 && pdu.flagCsiPart2
            pdu.nCsi2Reports = pusch.nCsi2Reports;
        end
        if(csiP2exist)
            if enable_multi_csiP2_fapiv3
                % constants:
                MAX_NUM_CSI1_PRM     = SimCtrl.MAX_NUM_CSI1_PRM;
                MAX_NUM_CSI2_REPORTS = SimCtrl.MAX_NUM_CSI2_REPORTS;

                % CSI2 Map paramaters:
                csi2Maps_bufferStartIdxs  = SimCtrl.csi2Maps_bufferStartIdxs;
                csi2Maps_buffer           = SimCtrl.csi2Maps_buffer;

                % calc CSI2 size paramaters:
                calcCsi2Size_prmValues  = reshape(pusch.calcCsi2Size_prmValues , MAX_NUM_CSI1_PRM , MAX_NUM_CSI2_REPORTS);
                calcCsi2Size_prmSizes   = reshape(pusch.calcCsi2Size_prmSizes  , MAX_NUM_CSI1_PRM , MAX_NUM_CSI2_REPORTS);
                calcCsi2Size_nPart1Prms = pusch.calcCsi2Size_nPart1Prms;
                calcCsi2Size_csi2MapIdx = pusch.calcCsi2Size_csi2MapIdx;

                for csi2ReportIdx = 0 : (pdu.nCsi2Reports - 1)

                    % compute aggregated CSI1 paramater:
                    nCsi1Prms         = calcCsi2Size_nPart1Prms(csi2ReportIdx + 1);
                    prm0Value         = calcCsi2Size_prmValues(1, csi2ReportIdx + 1);
                    prm0Size          = calcCsi2Size_prmSizes(1, csi2ReportIdx + 1);
                    aggregatedCsi1Prm = dec2bin(prm0Value, prm0Size);

                    for prmIdx = 1 : (nCsi1Prms - 1)
                        prmValue          = calcCsi2Size_prmValues(prmIdx + 1, csi2ReportIdx + 1);
                        prmSize           = calcCsi2Size_prmSizes(prmIdx + 1, csi2ReportIdx + 1);
                        aggregatedCsi1Prm = [aggregatedCsi1Prm dec2bin(prmValue, prmSize)];
                    end
                    aggregatedCsi1Prm = bin2dec(aggregatedCsi1Prm);
                    
                    % use CSI2 map and aggregated CSI1 paramater to compute
                    % size of CSI2 payload:
                    csi2MapIdx         = calcCsi2Size_csi2MapIdx(csi2ReportIdx + 1);
                    csi2BufferStartIdx = csi2Maps_bufferStartIdxs(csi2MapIdx + 1);

                    pdu.csiPart2BitLength(csi2ReportIdx + 1) = csi2Maps_buffer(csi2BufferStartIdx + aggregatedCsi1Prm + 1);
                end
            else
                pdu.csiPart2BitLength = csiP2PayloadSizeCalc(pusch.rank);
            end
        else
            pdu.csiPart2BitLength = 0;
        end

        pdu.alphaScaling = pusch.alphaScaling;
        pdu.betaOffsetHarqAck = pusch.betaOffsetHarqAck;
        pdu.betaOffsetCsi1 = pusch.betaOffsetCsi1;
        pdu.betaOffsetCsi2 = pusch.betaOffsetCsi2;
        if (~isDataPresent) && (pdu.csiPart1BitLength > 0 && csiP2exist==0)%Ref. Sec. 6.3.2.1.1 38.212
            if(pusch.harqAckBitLength == 0)
                pdu.harqPayload = [0 0];
            elseif(pusch.harqAckBitLength == 1)
                pdu.harqPayload = [round(rand(1,1)) 0];
            else
                pdu.harqPayload = round(rand(1, pdu.harqAckBitLength));
            end
        else
            pdu.harqPayload = round(rand(1, pdu.harqAckBitLength));
        end
        if pdu.csiPart1BitLength > 0 && isempty(pusch.csiPart1Payload)
            tmp_payload = rand(1, pdu.csiPart1BitLength);
            pdu.csiPart1Payload = round(tmp_payload);
            if(csiP2exist)
                if enable_multi_csiP2_fapiv3
                    % constants:
                    MAX_NUM_CSI1_PRM     = SimCtrl.MAX_NUM_CSI1_PRM;
                    MAX_NUM_CSI2_REPORTS = SimCtrl.MAX_NUM_CSI2_REPORTS;

                    % calc CSI2 size paramaters:
                    calcCsi2Size_prmSizes   = reshape(pusch.calcCsi2Size_prmSizes    , MAX_NUM_CSI1_PRM , MAX_NUM_CSI2_REPORTS);
                    calcCsi2Size_prmValues  = reshape(pusch.calcCsi2Size_prmValues   , MAX_NUM_CSI1_PRM , MAX_NUM_CSI2_REPORTS);
                    calcCsi2Size_prmOffsets = reshape(pusch.calcCsi2Size_prmOffsets  , MAX_NUM_CSI1_PRM , MAX_NUM_CSI2_REPORTS);
                    calcCsi2Size_nPart1Prms = pusch.calcCsi2Size_nPart1Prms;

                    for csi2ReportIdx = 0 : (pdu.nCsi2Reports - 1)
                        for csi1PrmIdx = 0 : (calcCsi2Size_nPart1Prms(csi2ReportIdx + 1) - 1)
                            % csi1 paramater size and offset:
                            csi1PrmSize = calcCsi2Size_prmSizes(csi1PrmIdx + 1, csi2ReportIdx + 1);
                            csi1Offset  = calcCsi2Size_prmOffsets(csi1PrmIdx + 1, csi2ReportIdx + 1);

                            % convert csi1 paramater to binary string (LSB first):
                            csi1Prm       = calcCsi2Size_prmValues(csi1PrmIdx + 1, csi2ReportIdx + 1);
                            csi1PrmBinary = dec2bin(csi1Prm, csi1PrmSize);

                            % embed cs1 paramater into csi1 payload:
                            for csi1PrmBitIdx = 0 : (csi1PrmSize - 1)
                                pdu.csiPart1Payload(csi1Offset + csi1PrmBitIdx + 1) = str2num(csi1PrmBinary(csi1PrmBitIdx + 1));
                            end
                        end
                    end
                else
                    if((pusch.rankBitOffset + pusch.rankBitSize) > pdu.csiPart1BitLength)
                        error('Error! Cannot place rank within CSI-P1 payload');
                    else
                        rankBits = dec2bin(pusch.rank - 1,pusch.rankBitSize);
                        for i = 0 : (pusch.rankBitSize - 1)
                            pdu.csiPart1Payload(pusch.rankBitOffset + i + 1) = str2num(rankBits(i + 1));
                        end
                    end
                end
            end
        else
            pdu.csiPart1Payload = pusch.csiPart1Payload;
        end
        if SimCtrl.forceCsiPart2Length
            pdu.csiPart2BitLength = SimCtrl.forceCsiPart2Length;
            %             pusch.csiPart2Payload = [];
        end
        if csiP2exist && isempty(pusch.csiPart2Payload)
            if enable_multi_csiP2_fapiv3
                pdu.csiPart2Payload = round(rand(1, sum(pdu.csiPart2BitLength)));
            else
                pdu.csiPart2Payload = round(rand(1, pdu.csiPart2BitLength));
            end
        else
            pdu.csiPart2Payload = pusch.csiPart2Payload;
        end
    case 'srs'
        srs = Mac.Config.srs{idx};
        pdu.type = 'srs';
        pdu.RNTI = srs.RNTI;
        pdu.BWPSize = srs.BWPSize;
        pdu.BWPStart = srs.BWPStart;
        switch srs.numAntPorts
            case 1
                pdu.numAntPorts = 0;
            case 2
                pdu.numAntPorts = 1;
            case 4
                pdu.numAntPorts = 2;
            otherwise
                error('numAntPorts is not supported ...\n');
        end
        switch srs.numSymbols
            case 1
                pdu.numSymbols = 0;
            case 2
                pdu.numSymbols = 1;
            case 4
                pdu.numSymbols = 2;
            otherwise
                error('numSymbols is not supported ...\n');
        end

        switch srs.numRepetitions
            case 1
                pdu.numRepetitions = 0;
            case 2
                pdu.numRepetitions = 1;
            case 4
                pdu.numRepetitions = 2;
            otherwise
                error('numRepetitions is not supported ...\n');
        end
        switch srs.combSize
            case 2
                pdu.combSize = 0;
            case 4
                pdu.combSize = 1;
            otherwise
                error('combSize is not supported ...\n');
        end
        pdu.timeStartPosition = srs.timeStartPosition;
        pdu.configIndex = srs.configIndex;
        pdu.sequenceId = srs.sequenceId;
        pdu.bandwidthIndex = srs.bandwidthIndex;
        pdu.combOffset = srs.combOffset;
        pdu.cyclicShift = srs.cyclicShift;
        pdu.frequencyPosition = srs.frequencyPosition;
        pdu.frequencyShift = srs.frequencyShift;
        pdu.frequencyHopping = srs.frequencyHopping;
        pdu.groupOrSequenceHopping = srs.groupOrSequenceHopping;
        pdu.resourceType = srs.resourceType;
        pdu.Tsrs = srs.Tsrs;
        pdu.Toffset = srs.Toffset;
        pdu.Beamforming = srs.Beamforming;
    otherwise
        pdu = [];
end

return


function Phy = UeUlGenFreqDomainSig(Phy, FAPIpdu, idxUE)

carrier = Phy.Config.carrier;
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
Nant = carrier.numTxAnt;
table = Phy.Config.table;
Xtf = zeros(Nsubcarrier, NsymPerSlot, Nant);
prach = Phy.Config.prach;
Xtf_prach = zeros(prach.L_RA*prach.N_dur_RA, Nant);

if idxSubframe == 0 && idxSlot == 0
    Phy.tx.Xtf_frame = [];
    Phy.tx.Xtf_prach_frame = [];
end

Phy.tx.isFirstPrachSlot = 0;
puschPduList = [];
idxPusch = 1;

nPdu = length(FAPIpdu);
for idxPdu = 1:nPdu
    pdu = FAPIpdu{idxPdu};
    switch pdu.type
        case 'prach'
            prach = Phy.Config.prach;
            Xtf_prach = genPrach(pdu, table, carrier, Xtf_prach);
            Phy.Config.prach.pdu = pdu;
            Phy.tx.isFirstPrachSlot = 1;
        case 'pucch'
            pucch = Phy.Config.pucch;
            Xtf = genPucch(pdu, table, carrier, Xtf);
            Phy.Config.pucch.pdu = pdu;
        case 'pusch'
            puschPduList{idxPusch} = pdu;
            idxPusch = idxPusch + 1;
            pusch = Phy.Config.pusch;
            if pdu.lastPuschPdu
                [Xtf, Gtruth] = genPusch(puschPduList, table, carrier, Xtf);
                Phy.Gtruth.pusch = Gtruth;
            end
            Phy.Config.pusch.pduList = puschPduList;
        case 'srs'
            srs = Phy.Config.srs;
            Xtf = genSrs(pdu, table, carrier, Xtf, idxUE);
            Phy.Config.pusch.pdu = pdu;
    end
end

if SimCtrl.genTV.enableUE && SimCtrl.genTV.FAPI ...
        && ismember(idxSlotInFrame, SimCtrl.genTV.slotIdx)
    node = 'UE';
    Xtf_remap = [];
    fhMsg = [];
    Xtf_srs = [];
    % Xtf_uncomp = Xtf;
    % Xtf_prach_uncomp = Xtf_prach;
    % Xtf_srs_uncomp = Xtf_srs;
    % saveTV_FAPI(SimCtrl.genTV, FAPIpdu, Phy.Config, Xtf_uncomp, Xtf, Xtf_prach_uncomp, Xtf_prach, Xtf_srs_uncomp, Xtf_srs, Xtf_remap, Xtf_remap_trsnzp, Phy.srsChEstDatabase, fhMsg, modCompMsg, idxSlotInFrame, node, table, 1);
end

Phy.tx.Xtf = Xtf;
if isfield(Phy.tx, 'Xtf_frame')
    Phy.tx.Xtf_frame = [Phy.tx.Xtf_frame, Xtf];
else
    Phy.tx.Xtf_frame = Xtf;
end

Phy.tx.Xtf_prach = Xtf_prach;
if isfield(Phy.tx, 'Xtf_prach_frame')
    Phy.tx.Xtf_prach_frame = [Phy.tx.Xtf_prach_frame, Xtf_prach];
else
    Phy.tx.Xtf_prach_frame = Xtf_prach;
end

if SimCtrl.enableUlDlCoSim == 0
    idxSlot = idxSlot + 1;
end

carrier = updateTimeIndex(carrier, idxSlot, SimCtrl.genTV.forceSlotIdxFlag);

Phy.Config.carrier = carrier;

return



function Phy = UeUlGenTimeDomainSig(Phy)

carrier = Phy.Config.carrier;
NsymPerSlot = carrier.N_symb_slot;
idxSlot = carrier.idxSlot;
idxSubframe = carrier.idxSubframe;
Nant = carrier.numTxAnt;
Nfft = carrier. Nfft;
mu = carrier.mu;
N_sc = carrier.N_sc;
N_samp_slot = carrier.N_samp_slot;
isFirstPrachSlot = Phy.tx.isFirstPrachSlot;

if (mu == 1 && idxSubframe == 0 && idxSlot == 1) || ...
        (mu == 0 && idxSubframe == 1 && idxSlot == 0)
    Phy.tx.Xt_frame = [];
end

if isfield(Phy.tx, 'Xt_frame')
    Xt_frame = Phy.tx.Xt_frame;
else
    Xt_frame = [];
end
    
Xt = zeros(N_samp_slot, Nant);

Xtf = Phy.tx.Xtf;
Xtf_prach = Phy.tx.Xtf_prach;

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

% for PRACH
if isFirstPrachSlot == 0 % not first prach slot
    if Phy.Config.preambleBufferValid == 1
        % load the remaining preamble samples left over from last slot
        preambleBuffer = Phy.tx.preambleBuffer;
        preambleBufferLen = length(preambleBuffer);
        if N_samp_slot >= preambleBufferLen
            Xt(1:preambleBufferLen, 1) = preambleBuffer + Xt(1:preambleBufferLen, 1);
            Phy.Config.preambleBufferValid = 0;
            Phy.tx.preambleBuffer = [];
        else
            Xt(1:N_samp_slot, 1) = preambleBuffer(1:N_samp_slot) + Xt(1:N_samp_slot, 1);
            Phy.tx.preambleBuffer(1:N_samp_slot) = [];
        end
    end
else % first Prach slot
    prach = Phy.Config.prach;
    prach.y_uv = Xtf_prach(1:prach.L_RA);
    prach = convertPreamble(prach, carrier, 'UE');
    preambleSamp = prach.preambleSamp.';
    preambleSampStart = prach.preambleSampStart;
    preambleEnd = preambleSampStart + ...
        length(preambleSamp);
    if preambleEnd <= N_samp_slot % preamble <= 1 slot
        Xt(preambleSampStart+1:preambleEnd, 1) = preambleSamp + ...
            Xt(preambleSampStart+1:preambleEnd, 1);
    else  % preamble length > 1 slot
        preambleBufferLen = preambleEnd - N_samp_slot;
        Xt(preambleSampStart+1:N_samp_slot, 1) = preambleSamp(1:end-preambleBufferLen) + ...
            Xt(preambleSampStart+1:N_samp_slot, 1);
        % Save the remaining preamble samples to be loaded in next slot
        preambleBuffer = preambleSamp(end-preambleBufferLen+1:end);
        Phy.Config.preambleBufferValid = 1;
        Phy.tx.preambleBuffer = preambleBuffer;
    end
end

Phy.tx.Xt = Xt;
Phy.tx.Xt_frame = [Xt_frame, Xt];

return
