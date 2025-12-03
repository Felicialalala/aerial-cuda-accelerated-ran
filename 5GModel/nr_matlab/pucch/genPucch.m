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

function Xtf = genPucch(pdu, table, carrier, Xtf)

%function applies full pucch transmit pipeline for a user.
% NOTE: currently fully supports pucch format 0 and 1

%outputs:
% txData.data --> transmited qpsk symbol
% txData.Xtf  --> transmited time frequency data. Dim: Nf x Nt

if pdu.DTX
    return;
end

pucchTable = table;

Xtf0 = Xtf; % for generating test vectors

Xtf = genPucch_cuphy(pdu, carrier, pucchTable, Xtf);

Xtf1 = Xtf - Xtf0; % for generating test vectors

global SimCtrl;
idxSlot = carrier.idxSlotInFrame;
idxPdu = pdu.pucchPduIdx-1;

if SimCtrl.genTV.enableUE && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)
    TVname = [SimCtrl.genTV.TVname, '_UE_CUPHY_s', num2str(idxSlot),...
        'p', num2str(idxPdu)];
    saveTV_pucch(SimCtrl.genTV.tvDirName, TVname, pucch_payload, pdu, carrier, Xtf1);
    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

return;

function Xtf = genPucch_cuphy(pdu, carrier, pucchTable, Xtf)

%%%%%%%%%%%%% derive pucch parameters%%%%%%%%%%%%%%
formatType = pdu.FormatType;

startSym = pdu.StartSymbolIndex;   % staring pucch symbol (1-10)
prbStart = pdu.prbStart;   % index of pucch prb before hopping
BWPStart = pdu.BWPStart;
crbStart = prbStart + BWPStart;
nSym = pdu.NrOfSymbols;           % number of pucch symbols: 1-2 for format 0, 2; 4-14 for format 1, 3, 4
nPRBs = pdu.prbSize;
nRNTI = pdu.RNTI;
dataScramblingId = pdu.dataScramblingId;
slotNumber = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
Nsymbols_per_slot = carrier.N_symb_slot;
IntraSlotfreqHop = pdu.freqHopFlag;
if IntraSlotfreqHop && nSym == 1
    error('Error. \nIntra-slot frequency hopping is invalid for single symbol duration.')
end
secondHopPRB = pdu.secondHopPRB; % second-hop PRB indexes for frequency hopping
secondHopCRB = secondHopPRB + BWPStart;
if formatType ~= 2
    nBits = pdu.BitLenHarq;
    positiveSR = pdu.positiveSR;
    SRFlag = 0;
    
    if formatType == 0 || formatType == 1
        SRFlag = pdu.SRFlag;
        if nBits == 0
            if ~SRFlag
                return;
            else
                if ~positiveSR
                    return;
                end
            end
        end
    end
    hop_id = pdu.hoppingId; % hopping id
    n_ID = hop_id; % hopping id
    tOCCidx = pdu.TimeDomainOccIdx; % index of time covering code
    groupHopping = pdu.groupHopFlag;
    sequenceHopping = pdu.sequenceHopFlag;
    
    if formatType == 3
        cs0 = 0; %% assume interlaced mapping is not enabled
    else
        cs0 = pdu.InitialCyclicShift; % initial cyclic shift (0-11)
    end
    
    cSequence = pdu.cSequenceGH;
    
    % group and sequence hopping
    v=0;
    u=0;
    
    if ~groupHopping && ~sequenceHopping
        f_gh = 0;
        f_ss = mod(n_ID, 30);
        u = mod(f_gh+f_ss, 30);
        v = 0;
    elseif groupHopping && ~sequenceHopping
        f_ss = mod(n_ID, 30);
        temp = 0;
        for m=0:7
            temp = temp + 2^m*cSequence(16*slotNumber + m + 1);
        end
        f_gh = mod(temp, 30);
        u = mod(f_gh+f_ss, 30);
        v = 0;
        if IntraSlotfreqHop
            temp1 = 0;
            for m=0:7
                temp1 = temp1 + 2^m*cSequence(16*slotNumber + 8 + m + 1);
            end
            
            f_gh = mod(temp1, 30);
            u_temp = mod(f_gh+f_ss, 30);
            u = [u u_temp];
        end
    elseif ~groupHopping && sequenceHopping
        f_gh = 0;
        f_ss = mod(n_ID, 30);
        u = mod(f_gh+f_ss, 30);
        v = cSequence(2*slotNumber+1);
        
        if IntraSlotfreqHop
            v_temp = cSequence(2*slotNumber + 1+1);
            v = [v v_temp];
        end
    else
        error('Error. \nGroup hopping and sequence hopping cannot be both enabled.')
    end
    
    
    %%Cyclic shift hopping
    c_init = n_ID;
    c = build_Gold_sequence(c_init,14*8*slotNumber + 8*(startSym + nSym - 1) + 8);
    
    m_cs = 0; %% TS 38.211, 6.3.2.2.2, m_cs = 0 except for PUCCH format 0
    
    if formatType == 0
        if nBits>0
            pucch_payload = pdu.payload;
            
            if SRFlag && positiveSR
                if length(pucch_payload) == 1
                    if pucch_payload == 0
                        m_cs = 3;
                    else
                        m_cs = 9;
                    end
                else
                    temp = pucch_payload(1)*2+pucch_payload(2);
                    switch temp
                        case 0
                            m_cs = 1;
                        case 1
                            m_cs = 4;
                        case 3
                            m_cs = 7;
                        case 2
                            m_cs = 10;
                    end
                end
            else
                if length(pucch_payload) == 1
                    if pucch_payload == 1
                        m_cs = 6;
                    end
                else
                    temp = pucch_payload(1)*2+pucch_payload(2);
                    switch temp
                        case 0
                            m_cs = 0;
                        case 1
                            m_cs = 3;
                        case 3
                            m_cs = 6;
                        case 2
                            m_cs = 9;
                    end
                end
            end
        end
    end
    
    cs = zeros(nSym,1);
    for i = 1 : nSym
        for m = 0 : 7
            cs(i) = cs(i) + 2^m * c(14*8*slotNumber + 8*(startSym + i  - 1) + m + 1);
        end
        cs(i) = mod(cs(i) + cs0 + m_cs,12);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if formatType == 0
        cs_freq = derive_cs_freq;
        r_base = pucchTable.r;
        r = r_base(:,u(1)+1);
        % r = LowPaprSeqGen(12, u(1), 0);
        
        %%%%%%%%%%%%
        
        %pucch frequency idx:
        freqIdx = (12*crbStart + 1) : 12*(crbStart+1);
        
        if IntraSlotfreqHop %for PUCCH format 0, intra-slot frequency hopping is only supported for nSym=2;
            %%first hop:
            Xtf(freqIdx, startSym + 1) = Xtf(freqIdx, startSym + 1) + r .* cs_freq(:,cs(1)+1);
            
            %%second hop:
            if groupHopping
                r = r_base(:,u(2)+1);
                % r = LowPaprSeqGen(12, u(2), 0);
            end
            
            freqIdx = (12*secondHopCRB + 1) : 12*(secondHopCRB+1);
            
            Xtf(freqIdx, startSym + 2) = Xtf(freqIdx, startSym + 2) + r .* cs_freq(:,cs(2)+1);
        else
            for i = 1 : nSym
                Xtf(freqIdx, i + startSym) = Xtf(freqIdx, i + startSym) + r .* cs_freq(:,cs(i)+1);
            end
        end
        
    elseif formatType == 1
        cs_freq = derive_cs_freq;
        
        nSym_data = floor(nSym/2); % number of data symbols
        nSym_dmrs = nSym - nSym_data; % number of dmrs symbols
        b = pdu.payload;
        
        %modulate bits to complex symbol:
        if nBits == 0
            x = sqrt(1 / 2) * (1 + 1i); % TS 38.213, positive SR corresponds to b(0) = 0
        elseif nBits == 1
            x = sqrt(1 / 2) * ((1 - 2*b(1)) + 1i*(1 - 2*b(1)));
        else % nBits == 2
            x = sqrt(1 / 2) * ((1 - 2*b(1)) + 1i*(1 - 2*b(2)));
        end
        
        %load base sequence:
        r_base = pucchTable.r;
        r = r_base(:,u(1)+1);
        % r = LowPaprSeqGen(12, u(1), 0);
        
        %load time codes:
        tOCC = pucchTable.tOCC;
        
        %pucch frequency idx:
        freqIdx = (12*crbStart + 1) : 12*(crbStart+1);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if IntraSlotfreqHop
            nSymFirstHop = floor(nSym/2);
            
            nSymDataFirstHop = floor(nSym_data/2);
            nSymDMRSFirstHop = nSymFirstHop - nSymDataFirstHop;
            nSymDataSecondHop = nSym_data - nSymDataFirstHop;
            nSymDMRSSecondHop = nSym_dmrs - nSymDMRSFirstHop;
            
            %%first hop:
            tOCC_data = tOCC{nSymDataFirstHop}{tOCCidx+1};
            tOCC_dmrs = tOCC{nSymDMRSFirstHop}{tOCCidx+1};
            %data:
            for i = 1 : nSymDataFirstHop
                symIdx = 2*(i-1) + 1;
                Xtf(freqIdx, symIdx + startSym + 1) = Xtf(freqIdx, symIdx + startSym + 1) + ...
                    x * tOCC_data(i) * r .* cs_freq(:,cs(symIdx+1)+1);
            end
            %dmrs:
            for i = 1 : nSymDMRSFirstHop
                symIdx = 2*(i-1);
                Xtf(freqIdx, symIdx + startSym + 1) = Xtf(freqIdx, symIdx + startSym + 1) + ...
                    tOCC_dmrs(i) * r .* cs_freq(:,cs(symIdx+1)+1);
            end
            
            %%second hop:
            if groupHopping
                r = r_base(:,u(2)+1);
                % r = LowPaprSeqGen(12, u(2), 0);
            end
            
            freqIdx = (12*secondHopCRB + 1) : 12*(secondHopCRB+1);
            
            tOCC_data = tOCC{nSymDataSecondHop}{tOCCidx+1};
            tOCC_dmrs = tOCC{nSymDMRSSecondHop}{tOCCidx+1};
            
            for i = nSymDataFirstHop+1 : nSym_data
                symIdx = 2*(i-1) + 1;
                Xtf(freqIdx, symIdx + startSym + 1) = Xtf(freqIdx, symIdx + startSym + 1) + ...
                    x * tOCC_data(i-nSymDataFirstHop) * r .* cs_freq(:,cs(symIdx+1)+1);
            end
            %dmrs:
            for i = nSymDMRSFirstHop+1 : nSym_dmrs
                symIdx = 2*(i-1);
                Xtf(freqIdx, symIdx + startSym + 1) = Xtf(freqIdx, symIdx + startSym + 1) + ...
                    tOCC_dmrs(i-nSymDMRSFirstHop) * r .* cs_freq(:,cs(symIdx+1)+1);
            end
            
        else
            tOCC_data = tOCC{nSym_data}{tOCCidx+1};
            tOCC_dmrs = tOCC{nSym_dmrs}{tOCCidx+1};
            
            %data:
            for i = 1 : nSym_data
                symIdx = 2*(i-1) + 1;
                Xtf(freqIdx, symIdx + startSym + 1) = Xtf(freqIdx, symIdx + startSym + 1) + ...
                    x * tOCC_data(i) * r .* cs_freq(:,cs(symIdx+1)+1);
            end
            
            %dmrs:
            for i = 1 : nSym_dmrs
                symIdx = 2*(i-1);
                Xtf(freqIdx, symIdx + startSym + 1) = Xtf(freqIdx, symIdx + startSym + 1) + ...
                    tOCC_dmrs(i) * r .* cs_freq(:,cs(symIdx+1)+1);
            end
        end
    elseif formatType == 3 % currently only support zero CSI part-2 bit
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% generate UCI sequences
        payloadSeq1 = pdu.payloadSeq1;
        payloadSeq2 = pdu.payloadSeq2;
        
        F3Para = deriveF3UciSeqTxSize(pdu.BitLenSr, pdu.BitLenHarq, pdu.BitLenCsiPart1, ...
            pdu.BitLenCsiPart2, pdu.freqHopFlag, pdu.AddDmrsFlag, pdu.maxCodeRate, pdu.pi2Bpsk, pdu.NrOfSymbols, pdu.prbSize);
        
        codedUciSeq1 = [];
        
        if F3Para.A_seg(1)<12
            fecEnc = FecRmObj(1, F3Para.E_seq(1), F3Para.A_seg(1));
            codedUciSeq1 = fecEnc(payloadSeq1);
        else
            codedUciSeq1 = uciSegPolarEncode(F3Para.A_seg(1), F3Para.E_seq(1), payloadSeq1);
        end
    
        codedUciSeq2 = [];
        
        if F3Para.A_seg(2)<12
           fecEnc = FecRmObj(1,  F3Para.E_seq(2), F3Para.A_seg(2));
           codedUciSeq2 = fecEnc(payloadSeq2);
        else
           codedUciSeq2 = uciSegPolarEncode(F3Para.A_seg(2), F3Para.E_seq(2), payloadSeq2);
        end
        
        [codedUCI] = PucchF34UciMultiplexing(nSym, nSym - F3Para.nSymData, F3Para.pi2Bpsk, F3Para.nSC, F3Para.E_seq(1), F3Para.E_seq(2), codedUciSeq1, codedUciSeq2);
        
        %% scrambling
        cInitScrm = nRNTI*2^15 + dataScramblingId;
        randomSeqScrm = build_Gold_sequence(cInitScrm, F3Para.E_seq(1) + F3Para.E_seq(2));
        scrmUCI = mod(codedUCI + randomSeqScrm, 2);
        
        %% modulation
        numSymbols = 0;
        
        if F3Para.pi2Bpsk
            
            numSymbols = F3Para.E_seq(1) + F3Para.E_seq(2);
            d = zeros(numSymbols, 1);
            for sIdx = 1:numSymbols
                d(sIdx) = sqrt(1 / 2) * exp(1i*pi/2*mod(sIdx-1, 2))*((1 - 2*scrmUCI(sIdx))+ 1i*(1-2*scrmUCI(sIdx)));
            end
        else % QPSK
            numSymbols = (F3Para.E_seq(1) + F3Para.E_seq(2))/2;
            d = zeros(numSymbols, 1);
            for sIdx = 1:numSymbols
                d(sIdx) = sqrt(1 / 2) * ((1 - 2*scrmUCI(2*sIdx-1)) + 1i*(1 - 2*scrmUCI(2*sIdx)));
            end
        end
        
        %% if interlaced mapping is not configured, no block-wise spreading is applied
        y = d;
        
        %% transform precoding
        
        z = zeros(numSymbols, 1);
        
        for l = 0:(F3Para.nSymData - 1)
            for k = 0:(F3Para.nSC - 1)
                zTemp = 0;
                for n = 0:(F3Para.nSC - 1)
                    zTemp = zTemp + y(l*F3Para.nSC + n + 1)*exp(-1i*2*pi*n*k/F3Para.nSC);
                end
                
                z(l*F3Para.nSC+k+1) = zTemp/sqrt(F3Para.nSC);
            end
        end
        
        %% embed into frequency-time grid
        freqIdx = (12*crbStart + 1) : 12*(crbStart+F3Para.prbSize);
        SetSymData = F3Para.SetSymData;
        SetSymDmrs = F3Para.SetSymDmrs;
        
        r = LowPaprSeqGen(F3Para.nSC, u(1), v(1));
        
        CsIdx = (0:(F3Para.nSC-1))*2*pi/12;
        CsIdx = transpose(CsIdx);
        if IntraSlotfreqHop
            nSymFirstHop = floor(nSym/2);
            %% first hop
            %% data
            lData = 1;
            
            while SetSymData(lData)<nSymFirstHop
                Xtf(freqIdx, SetSymData(lData)+startSym + 1) = Xtf(freqIdx, SetSymData(lData)+startSym + 1) + z(((lData-1)*F3Para.nSC+1):(lData*F3Para.nSC));
                
                lData = lData+1;
            end
            %% DMRS
            lDmrs = 1;
            
            while SetSymDmrs(lDmrs)<nSymFirstHop
                Xtf(freqIdx, SetSymDmrs(lDmrs)+startSym + 1) = Xtf(freqIdx, SetSymDmrs(lDmrs)+startSym + 1) + r.*exp(1i* cs(SetSymDmrs(lDmrs)+1) .* CsIdx);
                
                lDmrs = lDmrs+1;
            end
            
            %% second hop
            freqIdx = (12*secondHopCRB + 1) : 12*(secondHopCRB+F3Para.prbSize);
            
            %% data
            while lData<=F3Para.nSymData
                Xtf(freqIdx, SetSymData(lData)+startSym + 1) = Xtf(freqIdx, SetSymData(lData)+startSym + 1) + z(((lData-1)*F3Para.nSC+1):(lData*F3Para.nSC));
                
                lData = lData+1;
            end
            %% DMRS
            if groupHopping && ~sequenceHopping
                r = LowPaprSeqGen(F3Para.nSC, u(2), v(1));
            elseif ~groupHopping && sequenceHopping
                r = LowPaprSeqGen(F3Para.nSC, u(1), v(2));
            end
            
            while lDmrs<=length(SetSymDmrs)
                Xtf(freqIdx, SetSymDmrs(lDmrs)+startSym + 1) = Xtf(freqIdx, SetSymDmrs(lDmrs)+startSym + 1) + r.*exp(1i* cs(SetSymDmrs(lDmrs)+1) .* CsIdx);
                
                lDmrs = lDmrs+1;
            end
        else
            %% data
            for lData = 1:F3Para.nSymData
                Xtf(freqIdx, SetSymData(lData)+startSym + 1) = Xtf(freqIdx, SetSymData(lData)+startSym + 1) + z(((lData-1)*F3Para.nSC+1):(lData*F3Para.nSC));
            end
            
            %% DMRS
            for lDmrs = 1:length(SetSymDmrs)
                Xtf(freqIdx, SetSymDmrs(lDmrs)+startSym + 1) = Xtf(freqIdx, SetSymDmrs(lDmrs)+startSym + 1) + r.*exp(1i* cs(SetSymDmrs(lDmrs)+1) .* CsIdx);
            end
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else % formatType == 2
    %% generate UCI sequence
    %% currently not supporting CSI part 2
    payloadSeq1 = pdu.payloadSeq1;
    nBits = pdu.BitLenHarq + pdu.BitLenCsiPart1 + pdu.BitLenSr;
    
    DmrsScramblingId = pdu.DmrsScramblingId;
    
    startSym = startSym + 1;
    startCRB = crbStart + 1;
    
    shCRB = secondHopCRB + 1;
    
    Kinfo = nBits;
    Nencoded = nSym*nPRBs*16;
    if (nBits >=3 && nBits <= 11)
        % FEC encode using RM code
        fecEnc = FecRmObj(1,Nencoded,Kinfo);
        y = fecEnc(payloadSeq1);
    else
        y = uciSegPolarEncode(Kinfo, Nencoded, payloadSeq1);
    end
    
    c_init = nRNTI*2^15 + dataScramblingId;
    c = build_Gold_sequence(c_init,Nencoded);
    y_scramble = mod(y+c,2);
    
    % QPSK modulate the encoded symbols
    x = 1/sqrt(2) * (1-2*y_scramble(1:2:end) + 1i*(1-2*y_scramble(2:2:end)));
    
    x = reshape(x, nPRBs*8, nSym);
    
    % dmrs signal
    mStartFirstHop  = 4*crbStart;
    
    if IntraSlotfreqHop
        mStartSecondHop = 4*secondHopCRB;
    else
        mStartSecondHop = mStartFirstHop;
    end
    len1 = nPRBs*8 + 2*mStartFirstHop;
    len2 = nPRBs*8 + 2*mStartSecondHop;
    Ndmrs_bits_per_symbol = max(len1, len2);
    
    x_dmrs = [];
    for k = 1:nSym
        l = startSym + k - 2; % 5G is zero-based, but startSym is one-based at this point TODO FIXME VERIFY
        c_init = mod(2^17 * (Nsymbols_per_slot*slotNumber + l + 1) * (2*DmrsScramblingId + 1) + 2*DmrsScramblingId, 2^31);
        c = build_Gold_sequence(c_init, Ndmrs_bits_per_symbol);
        if k == 1
            x_dmrs = [1/sqrt(2) * (1-2*c((2*mStartFirstHop+1):2:(nPRBs*8 + 2*mStartFirstHop)) + ...
                1i*(1-2*c((2*mStartFirstHop+2):2:(nPRBs*8 + 2*mStartFirstHop))))];
        else
            x_dmrs = [x_dmrs; 1/sqrt(2) * (1-2*c((2*mStartSecondHop+1):2:(nPRBs*8 + 2*mStartSecondHop)) + ...
                1i*(1-2*c((2*mStartSecondHop+2):2:(nPRBs*8 + 2*mStartSecondHop))))];
        end
    end
    
    x_dmrs = reshape(x_dmrs, nPRBs*4, nSym);
    % dmrs
    dmrs_fIdx_base = transpose([1 4 7 10]);
    dmrs_fIdx_offset = kron(transpose(startCRB:startCRB+nPRBs-1),ones(4,1))*12 - 11; % 1-based indexing
    dmrs_fIdx = repmat(dmrs_fIdx_base,nPRBs,1) + dmrs_fIdx_offset;
    Xtf(dmrs_fIdx,startSym) = Xtf(dmrs_fIdx,startSym) + x_dmrs(:, 1);

    % data
    data_fIdx_base = transpose([0 2 3 5 6 8 9 11]);
    data_fIdx_offset = kron(transpose(startCRB:startCRB+nPRBs-1),ones(8,1))*12 - 11; % 1-based indexing
    data_fIdx = repmat(data_fIdx_base,nPRBs,1) + data_fIdx_offset;
    Xtf(data_fIdx,startSym) = Xtf(data_fIdx,startSym) + x(:, 1);
    
    if nSym > 1 % nSym == 2
        if IntraSlotfreqHop
            dmrs_fIdx_offset = kron(transpose(shCRB:shCRB+nPRBs-1),ones(4,1))*12 - 11; % 1-based indexing
            dmrs_fIdx = repmat(dmrs_fIdx_base,nPRBs,1) + dmrs_fIdx_offset;
            
            data_fIdx_offset = kron(transpose(shCRB:shCRB+nPRBs-1),ones(8,1))*12 - 11; % 1-based indexing
            data_fIdx = repmat(data_fIdx_base,nPRBs,1) + data_fIdx_offset;
        end
        
        % dmrs
        Xtf(dmrs_fIdx,startSym + 1) = Xtf(dmrs_fIdx,startSym + 1) + x_dmrs(:, 2);
        
        %data
        Xtf(data_fIdx,startSym + 1) = Xtf(data_fIdx,startSym + 1) + x(:, 2);
    end
end
return


function saveTV_pucch(tvDirName, TVname, pucch_payload, pdu, carrier, Xtf)

[status,msg] = mkdir(tvDirName);

PucchParams.BWPStart = uint32(pdu.BWPStart);
PucchParams.tOCCidx = uint32(pdu.TimeDomainOccIdx);     % index of time covering code
PucchParams.startSym = uint32(pdu.StartSymbolIndex);   % staring pucch symbol (1-10)
PucchParams.prbIdx = uint32(pdu.prbStart);       % index of pucch prb
PucchParams.formatType = uint32(pdu.FormatType); % format type
PucchParams.nPRBs = uint32(pdu.prbSize);
PucchParams.nRNTI = uint32(pdu.RNTI);
PucchParams.nID = uint32(pdu.dataScramblingId);
PucchParams.nSym = uint32(pdu.NrOfSymbols);
PucchParams.slotNumber = uint32(mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu));
PucchParams.Nsymbols_per_slot = uint32(carrier.N_symb_slot);
PucchParams.IntraSlotfreqHop = uint32(pdu.freqHopFlag);
PucchParams.GroupHopping = uint32(pdu.groupHopFlag);
PucchParams.sequenceHopFlag = uint32(pdu.sequenceHopFlag);
PucchParams.hop_id = uint32(pdu.hoppingId);                        % hopping id
PucchParams.prbStart = uint32(pdu.prbStart);   % index of pucch prb before hopping
PucchParams.dataScramblingId = uint32(pdu.dataScramblingId);

PucchParams.secondHopPRB = uint32(pdu.secondHopPRB); % second-hop PRB indexes for frequency hopping
PucchParams.nBits = uint32(pdu.nBits);
PucchParams.cs0 = uint32(pdu.InitialCyclicShift);                % initial cyclic shift (0-11)

h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_nv(h5File, 'PucchParams', PucchParams);
hdf5_write_nv(h5File, 'x_uci', uint32(pucch_payload));
hdf5_write_nv(h5File, 'X_tf', single(Xtf));
H5F.close(h5File);

return
