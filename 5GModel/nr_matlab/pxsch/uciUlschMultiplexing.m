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

function [cw, idxinfo]=uciUlschMultiplexing(G_harq,G_harq_rvd,G_csi1,G_csi2,G,...
                                     g_harq,g_csi1,g_csi2,g_ulsch,...
                                     nPuschSym,dataSymLoc_array,dmrsSymLoc_array,...
                                     nPrb,Nl,Qm,startSym, numDmrsCdmGrpsNoData,...
                                     isDataPresent)

% The function performs UCI multiplexing on PUSCH, with or without UL-SCH

% Input parameters

% G_harq: rate matched sequence length for HARQ-ACK 
% G_harq_rvd: rate matched sequence length for HARQ reserved resources (nHarqBits <=2)
% G_csi1: rate matched sequence length for CSI part 1
% G_csi2: rate matched sequence length for CSI part 2
% G: rate matched sequence length for UL-SCH
% g_harq: rate matched sequence for HARQ
% g_csi1: rate matched sequence for CSI-1
% g_csi2: rate matched sequence for CSI-2
% g_ulsch: rate matched sequence for UL-SCH
% nPuschSym: number of symbols allocated for PUSCH (including DMRS)
% dataSymLoc_array: symbol index for PUSCH data (MATLAB 1 indexing)
% dmrsSymLoc_array: symbol index for PUSCH DMRS (MATLAB 1 indexing)
% nPrb: Number of PRBs allocated for PUSCH
% nL: Number of layers per UE
% Qm: Modulation order
% startSym: Starting symbol index of PUSCH (0 indexing)


% Output 

% cw: the rate matched sequence multiplexing UCI and ULSCH bits
% idxInfo: Indices of HARQ, CSI-1, CSI-2 and UL-SCH bits within the codeword cw

% Reference : TS 38.212 Sec. 6.2.7

% Assumptions:

% No DFT-s-OFDM for UL-SCH
% No frequency hopping
% No PT-RS
% No data REs at DMRS symbols (i.e. nDmrsCdmGroupsWithoutData = 2)

%=========================================================================
% l = OFDM symbol index of scheduled PUSCH, range = 0:nSymAllPusch-1 
% k = subcarrier index of scheduled PUSCH, range = 0: (mScPusch-1) (273 PRBs
% allocated for PUSCH)

%% PUSCH derived parameters

mScPusch = nPrb*12;         % number of subcarriers of the scheduled PUSCH
NlQm = Nl*Qm;               % Product of number of layers and Modulation order

%% UL-SCH and UCI mapping grid
% NOTE1: phi_l_Ulsch (in 3GPP Sepc.)= phiUlsch(l,:) and m_sc_Ulsch_l = mscUlsch(l,:)
% NOTE2 : phi_l_Uci (in 3GPP Spec.) = phiUci(l,:) and m_sc_Uci_l = mscUci(l,:)

phiUlschGrid = zeros (nPuschSym,mScPusch);                  % Resource grid for PUSCH 
mScUlschGrid = repmat(numel(phiUlschGrid(1,:)),nPuschSym,1); % array of number of REs on each OFDM symbol in the resource grid  

mScUlsch = mScUlschGrid;
dmrsSymLoc_array_from_startSym = dmrsSymLoc_array - startSym + 1;          % Converting to MATLAB 1 indexing with +1
dataSymLoc_array_from_startSym = dataSymLoc_array - startSym + 1;          % Converting to MATLAB 1 indexing with +1

if numDmrsCdmGrpsNoData == 1
    mScUlsch(dmrsSymLoc_array_from_startSym,:) = 0.5 * mScUlsch(dmrsSymLoc_array_from_startSym,:); 
else
    mScUlsch(dmrsSymLoc_array_from_startSym,:) = 0;                              % No REs are available for UCI on DMRS symbols
end
mScUci = mScUlschGrid;              % #NOTE: Assuming nDmrsCdmGroupWithoutData = 2, no REs on DMRS symbols available for PUSCH data
mScUci(dmrsSymLoc_array_from_startSym,:) = 0;   

% Set of resource elements available for transmission of data

phiUlsch = cell(1,nPuschSym);
for j=1:nPuschSym
    phiUlsch{j} = (0:mScUlsch(j,1)-1)';
end

%  Set of resource elements available for transmission of UCI

phiUci = cell(1,nPuschSym);
for j=1:nPuschSym
    phiUci{j} = (0:mScUci(j,1)-1)';
end

% phiUci = phiUlsch; %Same set of resource elements as UL-SCH, since UL-SCH already precluded DMRS symbol REs for nDmrsCdmGroupsWithoutData = 2

%% Parameters for Step 1-6 for frequency hopping disabled PUSCH

if ~isempty(dataSymLoc_array_from_startSym)
    l1Csi = dataSymLoc_array_from_startSym(1);                                % OFDM symbol index (MATLAB indexing)of the first OFDM symbol that does not carry DMRS
end
if~isempty(dmrsSymLoc_array_from_startSym)&& any(dmrsSymLoc_array_from_startSym(1)<dataSymLoc_array_from_startSym)
    l1 = dataSymLoc_array_from_startSym(find(dmrsSymLoc_array_from_startSym(1)<dataSymLoc_array_from_startSym,1)); % OFDM symbol index (MATLAB indexing) of the first OFDM symbol after the first set of consecutive OFDM symbol(s) carrying DMRS
else
    l1 = l1Csi;
end

GAck1 = G_harq;           % G_ACK (number of coded HARQ-ACK bits)
GCsiPart11= G_csi1;       % G_CSI-Part1 (number of coded CSI part 1 bits)
GCsiPart21= G_csi2;       % G_CSI-Part2 (number of coded CSI part 2 bits)
nHopPusch = 1;            % No freq. hopping is supported

GAck2 = 0;               % No freq. hopping
GCsiPart12 = 0;          % No freq. hopping
GCsiPart22 = 0;          % No freq. hopping
l2 = 0;                  % No freq. hopping
l2Csi = 0;               % No freq. hopping

gBar = zeros(nPuschSym,mScPusch,NlQm);
gBarMap = zeros(nPuschSym,mScPusch,NlQm);
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
                break;              % Check for error: symbol index cannot be more than number of PUSCH symbols
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
                        gBar(l,k+1,nu+1) = g_harq(mCountAckAll+1); % +1 for MATLAB indexing
                        gBarMap(l,k+1,nu+1) = gAckMap(mCountAckAll+1);
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

%% Step 3a (CSI Part 1 bits)

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
                        gBar(l,k+1,nu+1) = g_csi1(mCountCsiPart1All+1); %+1 for Matlab indexing
                        gBarMap(l,k+1,nu+1) = gCsiPart1Map(mCountCsiPart1All+1);
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

%% Step 3b (CSI Part-2 bits)

if G_csi2
    mCountCsiPart2 = [0 0];
    mCountCsiPart2All = 0;
    lPrimeCsi = [l1Csi l2Csi];
    GCsiPart2 = [GCsiPart21 GCsiPart22];
    gCsiPart2Map = ones(G_csi2,1)*-3;
    
    for i = 1:nHopPusch
        l = lPrimeCsi(i);
        while mBarScUci(l) <= 0
            l = l+1;
            if l > nPuschSym
                break;              % Check for error: symbol index cannot be more than number of PUSCh symbols
            end
        end
        while mCountCsiPart2(i) < GCsiPart2(i)
            if l > nPuschSym
                break;              % Check for error: symbol index cannot be more than number of PUSCh symbols
            end
            if mBarScUci(l) > 0
                GCsi2Diff = GCsiPart2(i) - mCountCsiPart2(i);
                if GCsi2Diff >= mBarScUci(l)*NlQm
                    d = 1;
                    mReCount = mBarScUci(l);
                else
                    d = floor(mBarScUci(l)*NlQm/GCsi2Diff);
                    mReCount = ceil(GCsi2Diff/NlQm);
                end
                % Placing coded CSI Part2 bits in the gBar sequence at right
                % positions (multiplexing with data and other UCIs)
                for j = 0:mReCount-1
                    k = phiBarUci{l}(j*d+1);    %MATLAB indexing
                    for nu = 0:NlQm-1
                        gBar(l,k+1,nu+1) = g_csi2(mCountCsiPart2All+1); %+1 for Matlab indexing
                        gBarMap(l,k+1,nu+1) = gCsiPart2Map(mCountCsiPart2All+1);
                        mCountCsiPart2All = mCountCsiPart2All+1;
                        mCountCsiPart2(i)= mCountCsiPart2(i)+1;
                    end
                end                
                phiBarUciTmp = zeros(0,1);                
                phiBarUciTmp = union(phiBarUciTmp,phiBarUci{l}((0:mReCount-1)*d+1));
                
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
                    gBar(l+1,k+1,nu+1) = g_ulsch(mCountUlsch+1);       %MATLAB indexing
                    gBarMap(l+1,k+1,nu+1) = gUlschMap(mCountUlsch+1); 
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
                        gBar(l,k+1,nu+1) = g_harq(mCountAckAll+1); % MATLAB indexing (value =2 if CSI-2 punctured and =1 if UL-SCH punctured)
                        gBarMap(l,k+1,nu+1) = gBarMap(l,k+1,nu+1)+5;
                        mCountAckAll = mCountAckAll+1;
                        mCountAck(i) = mCountAck(i)+1;
                    end
                end
            end
            l = l+1;
        end
    end
end

%% STEP 6  (data and control indexes in the multiplexed sequence)

if G || G_csi1 || G_csi2 ||G_harq
    t = 0;
    cwMapLen = sum(mScUlsch(:))*NlQm;
    cw = zeros(cwMapLen,1);
    cwMap = zeros(cwMapLen,1);
    for l = 0:nPuschSym-1
        for j = 0:mScUlsch(l+1)-1
            k = phiUlsch{l+1}(j+1);
            for nu = 0:NlQm-1
                cw(t+1) = gBar(l+1,k+1,nu+1);
                cwMap(t+1) = gBarMap(l+1,k+1,nu+1);
                t = t+1;
            end
        end
    end
end
ackInd = sort([find(cwMap==-1); find(cwMap>0)]);
csi1Ind = find(cwMap==-2);
ackRvdCsi2Ind = find(cwMap==2);
csi2Ind = sort([find(cwMap==-3);ackRvdCsi2Ind]);
ackRvdUlschInd = find(cwMap==1);
ulschInd = sort([find(cwMap==-4);ackRvdUlschInd]);

idxinfo = struct;
idxinfo.ackInd = ackInd;
idxinfo.csi1Ind = csi1Ind;
idxinfo.ackRvdCsi2Ind = ackRvdCsi2Ind;
idxinfo.csi2Ind = csi2Ind;
idxinfo.ackRvdUlschInd = ackRvdUlschInd;
idxinfo.ulschInd = ulschInd;
return
