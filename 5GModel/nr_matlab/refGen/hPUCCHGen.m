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

function [sym_5g, sym_dmrs] = hPUCCHGen(pucch, carrier)

pdu = pucch.pdu;
sym_dmrs = [];

global SimCtrl
if ~SimCtrl.genTV.forceSlotIdxFlag
    nslot = carrier.idxSlotInFrame-1;
else
    nslot = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
end

switch carrier.mu
    case 0
        scs = 15;
    case 1
        scs = 30;
    case 2
        scs = 60;
    case 3
        scs = 120;
    case 4
        scs = 240;
end

switch pdu.FormatType
    case 0
        ack = pdu.payload;
        ack=ack';
        sr = [];
        if pdu.SRFlag && pdu.positiveSR
            sr = 1;
        end
        symAllocation = [pdu.StartSymbolIndex, pdu.NrOfSymbols];
        cp = 'normal';
        nid = pdu.hoppingId;
        if ~pdu.groupHopFlag && ~pdu.sequenceHopFlag
            groupHopping = 'neither';
        elseif pdu.groupHopFlag && ~pdu.sequenceHopFlag
            groupHopping = 'enable';
        elseif ~pdu.groupHopFlag && pdu.sequenceHopFlag
            groupHopping = 'disable';
        end
        initialCS = pdu.InitialCyclicShift;
        if pdu.freqHopFlag
            freqHopping = 'enabled';
        else
            freqHopping = 'disabled';
        end
        sym_5g = nrPUCCH0(ack,sr,symAllocation,cp,nslot,nid,groupHopping,initialCS,freqHopping);
    case 1
        ack = pdu.payload;
        sr = [];
        if pdu.SRFlag && pdu.positiveSR
            sr = 1;
        end
        
        symAllocation = [pdu.StartSymbolIndex, pdu.NrOfSymbols];
        nid = pdu.hoppingId;
        initialCS = pdu.InitialCyclicShift;
        cp = 'normal';
        if pdu.freqHopFlag
            freqHopping = 'enabled';
        else
            freqHopping = 'disabled';
        end
        
        if ~pdu.groupHopFlag && ~pdu.sequenceHopFlag
            groupHopping = 'neither';
        elseif pdu.groupHopFlag && ~pdu.sequenceHopFlag
            groupHopping = 'enable';
        elseif ~pdu.groupHopFlag && pdu.sequenceHopFlag
            groupHopping = 'disable';
        end
        
        occi = pdu.TimeDomainOccIdx;
        ack=ack';
        sym_5g = nrPUCCH1(ack,sr,symAllocation,cp,nslot, ...
            nid,groupHopping,initialCS,freqHopping,occi);
    case 2
        b = pdu.payloadSeq1;
        Kinfo = length(b);
        Nencoded = pdu.NrOfSymbols*pdu.prbSize*16;
        if(length(b) < 12)
            fecEnc = FecRmObj(1,Nencoded,Kinfo);
            uciCW = fecEnc(b);
        else
            uciCW = uciSegPolarEncode(length(b), Nencoded, b);
        end
        nid = pdu.dataScramblingId;
        rnti = pdu.RNTI;
        sym_5g = nrPUCCH2(uciCW,nid,rnti);  
        
        % DMRS
        carr = nrCarrierConfig;
        carr.NCellID = carrier.N_ID_CELL;
        carr.NSizeGrid = carrier.N_grid_size_mu;
        carr.NStartGrid = carrier.N_grid_start_mu;
        carr.NSlot = nslot;
        carr.NFrame = carrier.idxFrame;
        carr.SubcarrierSpacing = scs;
        
        pf2 = nrPUCCH2Config;
        pf2.NStartBWP = pdu.BWPStart;
        pf2.NSizeBWP = carr.NSizeGrid-pdu.BWPStart;
        pf2.SymbolAllocation = [pdu.StartSymbolIndex pdu.NrOfSymbols];
        pf2.PRBSet = pdu.prbStart:(pdu.prbStart+pdu.prbSize-1);
        if pdu.freqHopFlag
            pf2.FrequencyHopping = 'intraSlot';
        else
            pf2.FrequencyHopping = 'neither';
        end
        pf2.SecondHopStartPRB = pdu.secondHopPRB;
        pf2.NID0 = pdu.DmrsScramblingId;
        sym_dmrs = nrPUCCHDMRS(carr,pf2);
        
    case 3
        payloadSeq1 = pdu.payloadSeq1;
        payloadSeq2 = pdu.payloadSeq2;
        
        [F3Para] = deriveF3UciSeqTxSize(pdu.BitLenSr, pdu.BitLenHarq, pdu.BitLenCsiPart1, pdu.BitLenCsiPart2, pdu.freqHopFlag, pdu.AddDmrsFlag, pdu.maxCodeRate, pdu.pi2Bpsk, pdu.NrOfSymbols, pdu.prbSize);
        
        uciCwSeq1 = [];
        
        if F3Para.A_seg(1)<12
            fecEnc = FecRmObj(1, F3Para.E_seq(1), F3Para.A_seg(1));
            uciCwSeq1 = fecEnc(payloadSeq1);
        else
            uciCwSeq1 = uciSegPolarEncode(F3Para.A_seg(1), F3Para.E_seq(1), payloadSeq1);
        end
        
        uciCwSeq2 = [];
        
        if F3Para.A_seg(2)<12
            fecEnc = FecRmObj(1, F3Para.E_seq(2), F3Para.A_seg(2));
            uciCwSeq2 = fecEnc(payloadSeq2); 
        else
            uciCwSeq2 = uciSegPolarEncode(F3Para.A_seg(2), F3Para.E_seq(2), payloadSeq2);
        end
        
        [uciCW] = PucchF34UciMultiplexing(pdu.NrOfSymbols, pdu.NrOfSymbols - F3Para.nSymData, F3Para.pi2Bpsk, F3Para.nSC, F3Para.E_seq(1), F3Para.E_seq(2), uciCwSeq1, uciCwSeq2);
           
        if F3Para.pi2Bpsk
            modu = 'pi/2-BPSK';
        else % QPSK
            modu = 'QPSK';
        end
        
        nid = pdu.dataScramblingId;
        rnti = pdu.RNTI;
        mrb = pdu.prbSize;
        
        sym_5g = nrPUCCH3(uciCW,modu,nid,rnti,mrb);
    otherwise
        error('PUCCH format is not supported ...\n');
end
