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

function [harq_LLR_descr, csi1_LLR_descr, ulsch_LLR_descr, info]= uciUlschDemuxDescram(G_harq,G_harq_rvd,G_csi1,G,...
                                     LLRseq,nPuschSym,startSym,dataSymLoc_array,dmrsSymLoc_array,...
                                     nPrb,Nl,Qm,nBitsHarq, N_id, n_rnti)

% The function performs UCI demultiplexing on PUSCH, with or without UL-SCH
% and resturns descrambled LLR seq.s 

% Input parameters

% G_harq: rate matched sequence length for HARQ-ACK 
% G_harq_rvd: rate matched sequence length for HARQ reserved resources (nBitsHarq <=2)
% G_csi1: rate matched sequence length for CSI part 1
% G: rate matched sequence length for UL-SCH
% LLRseq: size (Qm*nL*nPrb*12*nDataSymbols)
% nPuschSym: number of symbols allocated for PUSCH (including DMRS)
% startSym: Starting symbol index of PUSCH (MATLAB 1 indexing)
% dataSymLoc_array: symbol index for PUSCH data (MATLAB 1 indexing)
% dmrsSymLoc_array: symbol index for PUSCH DMRS (MATLAB 1 indexing)
% nPrb: Number of PRBs allocated for PUSCH
% Nl: Number of layers per UE
% Qm: Modulation order 
% nBitsHarq: number of HARQ bits
% N_id: dataScramblingId or N_ID^cell as described in Sec. 6.3.1.1, TS38.211
% n_rnti: RNTI as described in Sec. 6.3.1.1, TS38.211

% Output 

% harq_LLR_descr: Decsrambled HARQ LLRs
% csi1_LLR_descr: Descrambled CSI-1 LLRS
% ulsch_LLR_descr: Decsrambled UL-SCH LLRs

% Reference : TS 38.212 Sec. 6.2.7

% Assumptions:

% No DFT-s-OFDM for UL-SCH
% No frequency hopping
% No PT-RS
% No data REs at DMRS symbols (i.e. nDmrsCdmGroupsWithoutData = 2)
% No CSI-2 bits are present

%=========================================================================
% l = OFDM symbol index of scheduled PUSCH, range = 0:nSymAllPusch-1 
% k = subcarrier index of scheduled PUSCH, range = 0: (mScPusch-1) (273 PRBs
% allocated for PUSCH)

%% PUSCH derived parameters

mScPusch = nPrb*12;         % number of subcarriers of the scheduled PUSCH
NlQm = Nl*Qm;                           % Product of number of layers and Modulation order

%% UL-SCH and UCI mapping grid
% NOTE1: phi_l_Ulsch = phiUlsch(l,:) and m_sc_Ulsch_l = mscUlsch(l,:)
% NOTE2 : phi_l_Uci = phiUci(l,:) and m_sc_Uci_l = mscUci(l,:)

phiUlschGrid = zeros (nPuschSym,mScPusch);                  % Resource grid for PUSCH 
mScUlschGrid = repmat(numel(phiUlschGrid(1,:)),nPuschSym,1); % array of number of REs on each OFDM symbol in the resource grid  

mScUci = mScUlschGrid;  
dmrsSymLoc_array_from_startSym = dmrsSymLoc_array - startSym + 1;          % Converting to MATLAB 1 indexing with +1
dataSymLoc_array_from_startSym = dataSymLoc_array - startSym + 1;          % Converting to MATLAB 1 indexing with +1

mScUci(dmrsSymLoc_array_from_startSym,:)=0;   % mSc_l_Uci = 0 if l = DMRS symbol
mScUlsch = mScUci;         % #NOTE: Assuming nDmrsCdmGroupWithoutData = 2, no REs on DMRS symbols available for PUSCH data

% Input vaildation
LLRSeqLen = sum(mScUlsch(:))*NlQm;
if LLRSeqLen ~= numel(LLRseq)
    error('Invalid input: the length of input LLR sequence does not match with the received cw length(LLRSeqLen)');
end

% Set of resource elements available for transmission of data

phiUlsch = cell(1,nPuschSym);
for j=1:nPuschSym
    phiUlsch{j} = (0:mScUlsch(j,1)-1)';
end

%  Set of resource elements available for transmission of UCI

phiUci = phiUlsch; %Same set of resource elements as UL-SCH, since UL-SCH already precluded DMRS symbol REs for nDmrsCdmGroupsWithoutData = 2

%% Parameters for Step 1-6 for frequency hopping disabled PUSCH

if ~isempty(dataSymLoc_array_from_startSym)
    l1Csi = dataSymLoc_array_from_startSym(1);                                % OFDM symbol index (MATLAB indexing)of the first OFDM symbol that does not carry DMRS
end
if~isempty(dmrsSymLoc_array_from_startSym)&& any(dmrsSymLoc_array_from_startSym(1)<dataSymLoc_array_from_startSym)
    l1 = dataSymLoc_array_from_startSym(find(dmrsSymLoc_array_from_startSym(1)<dataSymLoc_array_from_startSym,1)); % OFDM symbol index (MATLAB indexing) of the first OFDM symbol after the first set of consecutive OFDM symbol(s) carrying DMRS
else
    l1 = l1Csi;
end

GAck1 = G_harq;             % G_ACK (number of coded HARQ-ACK bits)
GCsiPart11= G_csi1;         % G_CSI-Part1 (number of coded CSI part 1 bits)
nHopPusch = 1;              % No freq. hopping is supported

GAck2 = 0;               % No freq. hopping
GCsiPart12 = 0;          % No freq. hopping
l2 = 0;                  % No freq. hopping
l2Csi = 0;               % No freq. hopping

gBar = zeros(nPuschSym,mScPusch,NlQm);

%% Step 1 (HARQ Reserved bits)

phiBarUlsch = phiUlsch;
mBarScUlsch = mScUlsch;
phiBarUci = phiUci;
mBarScUci = mScUci;

%Initialize phiBarRvd as the reserved resource elements for potential HARQ-ACK transmission
phiBarRvd = cell(nPuschSym,1); 
for j = 1:nPuschSym
    phiBarRvd{j} = zeros(0,1); % <nSymAllPuschx1> cell arrray of empty elements {0 x1 double}
end

if G_harq_rvd
    GAckRvd1 = G_harq_rvd; %Assumption: frequency hopping is not configured for PUSCH
    GAckRvd2 = 0;
    GAckRvdVal = [GAckRvd1 GAckRvd2];
    lPrime = [l1 l2];
    mCountAck = [0 0];

    for i = 1:nHopPusch
        l = lPrime(i);
        while mCountAck(i)< GAckRvdVal(i)
            if l >nPuschSym
                break;              % Check for error: symbol index cannot be more than number of PUSCh symbols
            end
            if mBarScUci(l)>0
                GAckRvdDiff = GAckRvdVal(i)-mCountAck(i);  %Number of remaining reserved elements
                if GAckRvdDiff >= mBarScUci(l)*NlQm
                    d = 1;
                    mReCount = mBarScUlsch(l);
                else
                    d = floor (mBarScUci(l)*NlQm/GAckRvdDiff);
                    mReCount = ceil(GAckRvdDiff/NlQm);
                end
                for j =0:mReCount-1
                    phiBarRvd{l} = union(phiBarRvd{l},phiBarUlsch{l}((0:mReCount-1)*d+1)); % MATLAB indexing
                    mCountAck(i) = mCountAck(i) + NlQm;
                end
            end
            l = l+1;
        end 
    end
end  

% Number of reserved elements in each OFDM symbol
mBarPhiBarScRvd = zeros(nPuschSym,1);
for i = 1:nPuschSym
    mBarPhiBarScRvd(i) = numel(phiBarRvd{i});
end

%% Step 2 (HARQ bits > 2)
% If HARQ-ACK is present for transmission on the PUSCH and number of
% bits > 2
if ~G_harq_rvd && G_harq 
    mCountAck = [0 0];
    mCountAckAll = 0;
    lPrime = [l1 l2];
    GHarqAck = [GAck1 GAck2];
    gAckMap = ones(G_harq,1)*(-1);

    for i = 1:nHopPusch
        l = lPrime(i);
        while mCountAck(i) < GHarqAck(i)
            if l >nPuschSym
                break;              % Check for error: symbol index cannot be more than number of PUSCh symbols
            end
            if mBarScUci(l) >0
                GAckDiff = GHarqAck(i)-mCountAck(i); % Number of remaining HARQ-ACK bits to be multiplexed on PUSCH
                if GAckDiff >= mBarScUci(l)*NlQm
                    d=1;
                    mReCount = mBarScUci(l);
                else
                    d = floor (mBarScUci(l)*NlQm/GAckDiff);
                    mReCount = ceil(GAckDiff/NlQm);
                end
                % Placing coded HARQ-ACK bits in the gBar sequence at right
                % positions (multiplexing with data and other UCIs)
                for j = 0:mReCount-1
                    k = phiBarUci{l}(j*d+1);    %MATLAB indexing
                    for nu = 0:NlQm-1
                        gBar(l,k+1,nu+1) = gAckMap(mCountAckAll+1); % +1 for MATLAB indexing
                        mCountAckAll = mCountAckAll+1;
                        mCountAck(i) = mCountAck(i)+1;
                    end
                end
                phiBarUciTmp = zeros(0,1);
                phiBarUciTmp = union(phiBarUciTmp,phiBarUci{l}((0:mReCount-1)*d+1));
                phiBarUci{l} = setdiff(phiBarUci{l},phiBarUciTmp);
                phiBarUlsch{l} = setdiff(phiBarUlsch{l},phiBarUciTmp);
                
                mBarScUci(l) = numel(phiBarUci{l});
                mBarScUlsch(l) = numel(phiBarUlsch{l});
            end
            l = l+1;
        end 
    end
end

%% Step 3 (CSI Part 1 bits)

if G_csi1
    mCountCsiPart1 = [0 0];
    mCountCsiPart1All = 0;
    lPrimeCsi = [l1Csi l2Csi];
    GCsiPart1 = [GCsiPart11 GCsiPart12];
    gCsiPart1Map = ones(G_csi1,1)*-2;

    for i = 1:nHopPusch
        l = lPrimeCsi(i);
        while mBarScUci(l) - mBarPhiBarScRvd(l) <= 0
            l = l+1;
            if l > nPuschSym
                break;              % Check for error: symbol index cannot be more than number of PUSCh symbols
            end
        end
        while mCountCsiPart1(i) < GCsiPart1(i)
            if l > nPuschSym
                break;              % Check for error: symbol index cannot be more than number of PUSCh symbols
            end
            mBarDiff = mBarScUci(l)-mBarPhiBarScRvd(l);
            GCsi1Diff = GCsiPart1(i) - mCountCsiPart1(i);
            if mBarDiff > 0
                if GCsi1Diff >= mBarDiff*NlQm
                    d = 1;
                    mReCount = mBarDiff;
                else
                    d = floor(mBarDiff*NlQm/GCsi1Diff);
                    mReCount = ceil(GCsi1Diff/NlQm);
                end
                phiBarTmp = setdiff(phiBarUci{l}, phiBarRvd{l});
                % Placing coded CSI Part1 bits in the gBar sequence at right
                % positions (multiplexing with data and other UCIs)
                for j = 0:mReCount-1
                    k = phiBarTmp(j*d+1);   %MATLAB indexing
                    for nu = 0:NlQm-1
                        gBar(l,k+1,nu+1) = gCsiPart1Map(mCountCsiPart1All+1); %+1 for Matlab indexing
                        mCountCsiPart1All = mCountCsiPart1All+1;
                        mCountCsiPart1(i)= mCountCsiPart1(i)+1;
                    end
                end                
                phiBarUciTmp = zeros(0,1);                
                phiBarUciTmp = union(phiBarUciTmp,phiBarTmp((0:mReCount-1)*d+1));
                
                phiBarUci{l} = setdiff(phiBarUci{l}, phiBarUciTmp);
                phiBarUlsch{l} = setdiff(phiBarUlsch{l}, phiBarUciTmp);
                mBarScUci(l) = numel(phiBarUci{l});
                mBarScUlsch(l) = numel(phiBarUlsch{l});
            end
            l = l+1;
        end
    end
end

%% Step 4 (UL-SCH data bits) 

if G
    mCountUlsch =0;
    gUlschMap = ones(G,1)*-4;
    
    for l = 0:nPuschSym-1
        if mBarScUlsch(l+1)>0  %MATLAB indexing
            %Placing coded UL-SCH data bits in gBar sequence at right
            %positions (multiplexing with UCI)
            for j = 0:(mBarScUlsch(l+1)-1)
                k = phiBarUlsch{l+1}(j+1);           %MATLAB indexing
                for nu = 0:NlQm-1
                    gBar(l+1,k+1,nu+1) = gUlschMap(mCountUlsch+1);       %MATLAB indexing
                    mCountUlsch = mCountUlsch+1; %MATLAB indexing
                end
            end
        end
    end
end

%% Step 5(HARQ-ACK bits <=2)

if G_harq_rvd && G_harq
    mCountAck = [0 0];
    mCountAckAll = 0;
    lPrime = [l1 l2];
    GHarqAck = [GAck1 GAck2];
    
    for i = 1:nHopPusch
        l = lPrime(i);
        while mCountAck(i) <GHarqAck(i)
            if l > nPuschSym
                break;              % Check for error: symbol index cannot be more than number of PUSCh symbols
            end
            if mBarPhiBarScRvd(l) > 0
                GAckDiff = GHarqAck(i) - mCountAck(i);
                if GAckDiff >= mBarPhiBarScRvd(l)*NlQm
                    d = 1;
                    mReCount = mBarPhiBarScRvd(l);
                else
                    d = floor(mBarPhiBarScRvd(l)*NlQm/GAckDiff);
                    mReCount = ceil(GAckDiff/NlQm);
                end
                % Placing coded HARQ-ACK bits (<=2)in gBar sequence at right
                % positions (at HARQ ACK reserved bit locations)
                for j = 0:mReCount-1
                    k = phiBarRvd{l}(j*d+1); % MATLAB indexing
                    for nu = 0:NlQm-1
                        gBar(l,k+1,nu+1) = gBar(l,k+1,nu+1)+5; % MATLAB indexing (value =2 if CSI-2 punctured and =1 if UL-SCH punctured)
                        mCountAckAll = mCountAckAll+1;
                        mCountAck(i) = mCountAck(i)+1;
                    end
                end
            end
            l = l+1;
        end
    end
end

%% STEP 6  (extraction of indexes for Ack, CSI-1, and ULSCH)

harqInd = zeros(0,1);
csi1Ind = zeros(0,1);
ulschInd = zeros(0,1);
harqRvdUlschInd = zeros(0,1);

if G || G_csi1 || G_harq
    t = 0;
    cw = zeros(LLRSeqLen,1);
    for l = 0:nPuschSym-1
        for j = 0:mScUlsch(l+1)-1
            k = phiUlsch{l+1}(j+1);
            for nu = 0:NlQm-1                           
                cw(t+1) = gBar(l+1,k+1,nu+1);
                t = t+1;
            end
        end
    end
    harqInd = sort([find(cw==-1); find(cw>0)]);
    csi1Ind = find(cw==-2);
    harqRvdUlschInd = find(cw==1);
    ulschInd = sort([find(cw==-4);harqRvdUlschInd]);
    
    info = struct;
    info.harqInd = harqInd;
    info.csi1Ind = csi1Ind;
    info.harqRvdUlschInd = harqRvdUlschInd;
    info.ulschInd = ulschInd;
end

ackLlrSeq = LLRseq(harqInd);
csi1LlrSeq = LLRseq(csi1Ind);
ulschLlrSeq = LLRseq(ulschInd);

%% Descrambling 

% Generation of Gold sequence of length = LLRSeq

c_init = n_rnti*2^15 + N_id;
c = build_Gold_sequence(c_init, numel(LLRseq)); 

%% Decrambling of ack LLRs 

if G_harq
    harq_LLR_descr = zeros(G_harq,1);
    if nBitsHarq ==1 % Simplex descrambling
        if Qm == 1
            for bitIdx = 1:G_harq
                harq_LLR_descr(bitIdx) = (1 - 2*c(harqInd(bitIdx))) * ackLlrSeq(bitIdx); % flip LLR if c(bitIdx) == 1
            end
        else
            for bitIdx = 1:G_harq
                temp = mod(bitIdx - 1, Qm) + 1;
                if temp == 1
                    harq_LLR_descr(bitIdx) = (1 - 2*c(harqInd(bitIdx))) * ackLlrSeq(bitIdx); % flip LLR if c(bitIdx) == 1
                elseif temp == 2
                    harq_LLR_descr(bitIdx) = (1- 2*c(harqInd(bitIdx - 1))) * ackLlrSeq(bitIdx); % flip LLR if c(bitIdx - 1) == 1
                end
            end
        end
     elseif nBitsHarq ==2 % Simplex descrambling
        if Qm == 1 || Qm == 2
            for bitIdx = 1:G_harq
                harq_LLR_descr(bitIdx) = (1 - 2*c(harqInd(bitIdx))) * ackLlrSeq(bitIdx); % flip LLR if c(bitIdx) == 1
            end
        else
            for bitIdx = 1:G_harq
                temp = mod(bitIdx, Qm);
                if temp == 1 || temp == 2
                    harq_LLR_descr(bitIdx) = (1 - 2*c(harqInd(bitIdx))) * ackLlrSeq(bitIdx); % flip LLR if c(bitIdx) == 1
                end 
            end
        end
     else % More than 2 HARQ bits
         harq_LLR_descr =  (1 - 2 * c(harqInd(:))) .*ackLlrSeq;
     end 
else
    harq_LLR_descr = [];
end
 
%% Descrambling of CSI-1 LLRs

if G_csi1
    csi1_LLR_descr = (1 - 2 * c(csi1Ind(:))) .*csi1LlrSeq;
else
    csi1_LLR_descr = [];
end

%% Descrambling of UL-SCH LLRs and zeroing out of punctured LLRs (if any)

if G
    ulsch_LLR_descr = (1 - 2 * c(ulschInd(:))) .*ulschLlrSeq;
    if harqRvdUlschInd
        ulsch_punctured = zeros(0,1);
        for idx = 1:numel(harqRvdUlschInd)
            ulsch_punctured(idx) = find(ulschInd == harqRvdUlschInd(idx));
        end   
        ulsch_LLR_descr(ulsch_punctured) = 0;
    end
else
   ulsch_LLR_descr = [];
end

end
