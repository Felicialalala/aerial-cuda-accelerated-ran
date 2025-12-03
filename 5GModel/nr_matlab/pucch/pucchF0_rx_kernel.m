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

function [pF0UcisOut] = pucchF0_rx_kernel(nRxAnt, pucchF0tables, pucchF0dynamicDesc, pF0UcisOut, grpIdx)
%% load group common parameters
rBase               = pucchF0tables.rBase;
csPhaseRamp         = pucchF0tables.csPhaseRamp;
Xtf                 = pucchF0dynamicDesc.Xtf;
BWPStart            = pucchF0dynamicDesc.BWPStart{grpIdx};
freqHopFlag         = pucchF0dynamicDesc.freqHopFlag{grpIdx};
startSym            = pucchF0dynamicDesc.startSym{grpIdx};
startPrb            = pucchF0dynamicDesc.startPrb{grpIdx};
nSym                = pucchF0dynamicDesc.nSym{grpIdx};
groupHopFlag        = pucchF0dynamicDesc.groupHopFlag{grpIdx};
secondHopPrb        = pucchF0dynamicDesc.secondHopPrb{grpIdx};
u                   = pucchF0dynamicDesc.u{grpIdx};
csCommon            = pucchF0dynamicDesc.csCommon{grpIdx};
nUci                = pucchF0dynamicDesc.nUciInGrp{grpIdx};
N_TONES_PER_PRB     = 12;

%% Threshold for determining confidence levels of SR and HARQ values
%% According to SCF FAPI, Table 3-68, 0 stands for "Good" and 1 stands for "Bad"
confidenceThr = 0.1; 

%% Scaling parameter for DTXthreshold
beta = 7.7/sqrt(nRxAnt*nSym);

startCrb            = startPrb + BWPStart;
secondHopCrb        = secondHopPrb + BWPStart;

%% load data from Xtf
Y_pucch = [];
freq_idx = (startCrb*N_TONES_PER_PRB + 1) : N_TONES_PER_PRB*(startCrb + 1);
if freqHopFlag %nSym == 2
    % first hop
    Y_pucch = Xtf(freq_idx,startSym + 1,1:nRxAnt);
    % second hop
    freq_idx = (secondHopCrb*N_TONES_PER_PRB + 1) : N_TONES_PER_PRB*(secondHopCrb + 1);
    Y_pucch = [Y_pucch Xtf(freq_idx,startSym + 2,1:nRxAnt)];
else
    time_idx = (startSym + 1) : (startSym + nSym);
    Y_pucch = Xtf(freq_idx,time_idx,1:nRxAnt);
end

%% compute RSSI
rssiTemp = sum(abs(Y_pucch).^2, 1);
rssiTemp = reshape(rssiTemp, [size(rssiTemp, 2), size(rssiTemp, 3)]); % remove the first dimension which is a frequency dimension
rssi = sum(mean(rssiTemp,1),2);% average across all symbols and sum across all Rx antenna                               % 5G FAPI Table 3-16: "RSSI reported will be total received power summed across all antennas"
rssi_dB = 10*log10(rssi);

%% compute all correlations between low PAPR sequences and received signal
corArray = zeros(N_TONES_PER_PRB, nSym);

if freqHopFlag
    r1stHop = rBase(:,u(1)+1);
    if groupHopFlag
        r2ndHop = rBase(:,u(2)+1);
    else
        r2ndHop = r1stHop;
    end
    
    for i = 1:N_TONES_PER_PRB
        corArray(i,1) = 0;
        corArray(i,2) = 0;
        for k = 1:nRxAnt
            corArray(i,1) = corArray(i,1) + abs((r1stHop .* csPhaseRamp(:,i))' * Y_pucch(:,1,k))^2;
            corArray(i,2) = corArray(i,2) + abs((r2ndHop .* csPhaseRamp(:,i))' * Y_pucch(:,2,k))^2;
        end
    end
else
    r = rBase(:,u(1)+1);
    for i = 1:N_TONES_PER_PRB
        for j = 1:nSym
            corArray(i,j) = 0;
            for k = 1:nRxAnt
                corArray(i,j) = corArray(i,j) + abs((r .* csPhaseRamp(:,i))' * Y_pucch(:,j,k))^2;
            end
        end
    end      
end

noiseCorr = sum(corArray(:));

%% %%%%%%%%%%%%%%%%%%% per UCI processing %%%%%%%%%%%%%%%%%%%%%%%
bitLenHarq      = zeros(1, nUci);
srFlag          = zeros(1, nUci);
uciOutputIdx    = zeros(1, nUci);
extDTXthreshold = zeros(1, nUci);
maxCorr         = zeros(1, nUci);
maxCorrIndex    = zeros(1, nUci);

for UciIdx = 1:nUci %% iterate over PUCCH UCIs within a group
    %% load per-UCI parameters
    bitLenHarq(UciIdx)      = pucchF0dynamicDesc.bitLenHarq{grpIdx}(UciIdx);
    cs0                     = pucchF0dynamicDesc.cs0{grpIdx}(UciIdx);                % initial cyclic shift (0-11)
    uciOutputIdx(UciIdx)    = pucchF0dynamicDesc.uciOutputIdx{grpIdx}(UciIdx);
    extDTXthreshold(UciIdx) = pucchF0dynamicDesc.DTXthreshold{grpIdx}(UciIdx);
    srFlag(UciIdx)          = pucchF0dynamicDesc.srFlag{grpIdx}(UciIdx);
   
    %% compute csArray array of all possible cyclic shifts for the UCI    
    if bitLenHarq(UciIdx) == 0
        if srFlag(UciIdx)
            m_csArray = [0];
        else % Null condition
            pF0UcisOut{uciOutputIdx(UciIdx)}.SinrDB   = -99; % default value for SINR in dB
            pF0UcisOut{uciOutputIdx(UciIdx)}.InterfDB =  99; % default value for interference plus noise in dB
            pF0UcisOut{uciOutputIdx(UciIdx)}.RSRP     = -99;   % default value for RSRP in dB
            pF0UcisOut{uciOutputIdx(UciIdx)}.RSSI     = -99;   % default value for RSSI in dB
            pF0UcisOut{uciOutputIdx(UciIdx)}.HarqValues = zeros(1,2);
            pF0UcisOut{uciOutputIdx(UciIdx)}.NumHarq = 0;
            pF0UcisOut{uciOutputIdx(UciIdx)}.SRindication = 0;
            pF0UcisOut{uciOutputIdx(UciIdx)}.SRconfidenceLevel = 1;
            pF0UcisOut{uciOutputIdx(UciIdx)}.HarqconfidenceLevel = 1;
            continue;
        end
    elseif bitLenHarq(UciIdx) == 1
        if srFlag(UciIdx)
            m_csArray = [0 3 6 9];
        else
            m_csArray = [0 6];
        end
    else %n Bits == 2
        if srFlag(UciIdx)
            m_csArray = [0 1 3 4 6 7 9 10];
        else
            m_csArray = [0 3 6 9];
        end
    end
    
    nMcs = length(m_csArray);
    csArray = zeros(nMcs, nSym);

    for cs_i = 1:nMcs
        m_cs = m_csArray(cs_i);
        for i = 1 : nSym
            csArray(cs_i, i) = mod(csCommon(i) + cs0 + m_cs, N_TONES_PER_PRB);
        end
    end

    %% extract correlations corresonding to csArray
    
    cor = zeros(1,nMcs);
    for n = 1:nMcs        
        cs = csArray(n, :);
        
        for j = 1:nSym
            cor(n) = cor(n) + corArray(cs(j)+1,j);
        end
    end
    
    %% compute largest correlation and corresponding correlation (NOTE, index is 1-matlab based!!)

    [largestCor, maxCorrIndex(UciIdx)] = max(cor);

    noiseCorr = noiseCorr - largestCor;
    maxCorr(UciIdx) = largestCor;
end

for UciIdx = 1:nUci     
    %% calculate RSRP
    rsrpTemp = maxCorr(UciIdx)/nSym/N_TONES_PER_PRB^2/nRxAnt;
    rsrp_dB = 10*log10(rsrpTemp);

    if bitLenHarq(UciIdx) == 0 && srFlag(UciIdx) == 0
        continue;
    end

    %% Determine DTXthreshold based on the provided external threshold
    DTXthreshold = noiseCorr*beta*extDTXthreshold(UciIdx)/(12-nUci);

    %% Determine SR/HARQ confidence levels
    SRconfidenceLevel = 0;
    HarqconfidenceLevel = 0;
    
    gapPercDtx = (maxCorr(UciIdx) - DTXthreshold)/maxCorr(UciIdx);
    
    if gapPercDtx < confidenceThr
        SRconfidenceLevel = 1;
        HarqconfidenceLevel = 1;
    end
     
    %% check detection/extract UCI payload
    SR = 0;
    pucch_payload = [];

    if (maxCorr(UciIdx) > DTXthreshold)
        if bitLenHarq(UciIdx) == 0
            SR = 1;
        else
            pucch_payload = zeros(1, bitLenHarq(UciIdx));
            
            if bitLenHarq(UciIdx) == 1
                if srFlag(UciIdx)
                    switch maxCorrIndex(UciIdx)
                        case 1
                            pucch_payload(1) = 0;
                        case 2
                            pucch_payload(1) = 0;
                            SR = 1;
                        case 3
                            pucch_payload(1) = 1;
                        case 4
                            pucch_payload(1) = 1;
                            SR = 1;
                    end
                else
                    switch maxCorrIndex(UciIdx)
                        case 1
                            pucch_payload(1) = 0;
                        case 2
                            pucch_payload(1) = 1;
                    end
                end
            else % bitLenHarq(UciIdx) == 2
                if srFlag(UciIdx)
                    switch maxCorrIndex(UciIdx)
                        case 1
                            pucch_payload(1) = 0;
                            pucch_payload(2) = 0;
                        case 2
                            pucch_payload(1) = 0;
                            pucch_payload(2) = 0;
                            SR = 1;
                        case 3
                            pucch_payload(1) = 0;
                            pucch_payload(2) = 1;
                        case 4
                            pucch_payload(1) = 0;
                            pucch_payload(2) = 1;
                            SR = 1;
                        case 5
                            pucch_payload(1) = 1;
                            pucch_payload(2) = 1;
                        case 6
                            pucch_payload(1) = 1;
                            pucch_payload(2) = 1;
                            SR = 1;
                        case 7
                            pucch_payload(1) = 1;
                            pucch_payload(2) = 0;
                        case 8
                            pucch_payload(1) = 1;
                            pucch_payload(2) = 0;
                            SR = 1;
                    end
                else
                    switch maxCorrIndex(UciIdx)
                        case 1
                            pucch_payload(1) = 0;
                            pucch_payload(2) = 0;
                        case 2
                            pucch_payload(1) = 0;
                            pucch_payload(2) = 1;
                        case 3
                            pucch_payload(1) = 1;
                            pucch_payload(2) = 1;
                        case 4
                            pucch_payload(1) = 1;
                            pucch_payload(2) = 0;
                    end
                end
            end
        end
    else %% DTX detected
        if bitLenHarq(UciIdx) > 0
            pucch_payload = 2*ones(1, bitLenHarq(UciIdx));
        end
    end
    
    %% write to output buffer
    pF0UcisOut{uciOutputIdx(UciIdx)}.taEstMicroSec = 0; % not supported yet
    pF0UcisOut{uciOutputIdx(UciIdx)}.SinrDB   = 0; % placeholder for SINR in dB
    pF0UcisOut{uciOutputIdx(UciIdx)}.InterfDB = 0; % placeholder for interference plus noise in dB
    pF0UcisOut{uciOutputIdx(UciIdx)}.RSRP     = rsrp_dB;
    pF0UcisOut{uciOutputIdx(UciIdx)}.RSSI     = rssi_dB;
    pF0UcisOut{uciOutputIdx(UciIdx)}.NumHarq = bitLenHarq(UciIdx);  
    pF0UcisOut{uciOutputIdx(UciIdx)}.SRindication = SR;
    pF0UcisOut{uciOutputIdx(UciIdx)}.SRconfidenceLevel = SRconfidenceLevel;
    pF0UcisOut{uciOutputIdx(UciIdx)}.HarqValues = zeros(1,2);
    pF0UcisOut{uciOutputIdx(UciIdx)}.HarqValues(1 : bitLenHarq(UciIdx)) = pucch_payload;
    pF0UcisOut{uciOutputIdx(UciIdx)}.HarqconfidenceLevel = HarqconfidenceLevel;
end

return
