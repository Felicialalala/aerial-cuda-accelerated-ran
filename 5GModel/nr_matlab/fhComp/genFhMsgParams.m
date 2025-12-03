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

function [fhMsgParams, modCompRbParam, modCompList] = genFhMsgParams(FAPIpdu, table, carrier)

global SimCtrl;
nPort_enable_csirs_compression = SimCtrl.nPort_enable_csirs_compression;
[row2nPort, ~] = getCsirsConfig();

nPdu = length(FAPIpdu);
idxFhMsgParams = 0;
fhMsgParams = [];
Xtf_remap = zeros(12*carrier.N_grid_size_mu, 14);
qamScaler = [2/sqrt(2), 4/sqrt(10), 8/sqrt(42), 16/sqrt(170)]/sqrt(2);

% Determine maximum number of ports needed (including CSI-RS ports)
maxPorts = carrier.numTxPort;
for idxPdu = 1:nPdu
    pdu = FAPIpdu{idxPdu};
    if strcmp(pdu.type, 'csirs')
        Row = pdu.Row;
        maxPorts = max(maxPorts, row2nPort(Row));
    end
end

for idxPrb = 1:carrier.N_grid_size_mu
    for idxSym = 1:14
        for idxPort = 1:maxPorts
            modCompRbParam(idxPrb, idxSym, idxPort).nMask = 0;
            modCompRbParam(idxPrb, idxSym, idxPort).udIqWidth = 0;
            modCompRbParam(idxPrb, idxSym, idxPort).reMask = zeros(2, 12);
            modCompRbParam(idxPrb, idxSym, idxPort).scaler = [1 1];
            modCompRbParam(idxPrb, idxSym, idxPort).csf = [1 1];
            modCompRbParam(idxPrb, idxSym, idxPort).chanType = [0 0];
            modCompRbParam(idxPrb, idxSym, idxPort).pduIdx = 0; % initialize PDU identifier
        end
    end
end

for idxPdu = 1:nPdu
    pdu = FAPIpdu{idxPdu};
    switch pdu.type
        case 'ssb'
            startSymbol = pdu.nSSBStartSymbol;
            nSymbol = 4;
            startPrb = pdu.SsbOffsetPointA/2^carrier.mu;
            startPrb_floor = floor(startPrb);
            prbFrac = startPrb - startPrb_floor;
            nPrb = 20;
            nPrb = nPrb + ceil(pdu.ssbSubcarrierOffset/2^carrier.mu/12 + prbFrac);
            startPrb = startPrb_floor;
            ssbIdx = genSsbIdx(carrier.N_ID_CELL);
            nSbc = 20*12;
            nSym = 4;
            ssbMap = zeros(nSbc, nSym);
            ssbMap(ssbIdx.pss_idx+1) = 1;
            ssbMap(ssbIdx.sss_idx+1) = 2;
            ssbMap(ssbIdx.dmrs_idx+1) = 3;
            ssbMap(ssbIdx.qam_idx+1) = 3;
            scOffset = pdu.ssbSubcarrierOffset/2^carrier.mu + prbFrac*12;
            if scOffset > 0
                ssbMap = [zeros(scOffset, 4); ssbMap; ...
                    zeros(mod(12-scOffset,12), 4)];
            end

            if (pdu.betaPss == 0)
                beta_pss = 1;
            elseif (pdu.betaPss == 1)
                beta_pss = 10^(3/20); % linear scaler of 3 dB
            else
                error(['Unknown value for betaPss: ',num2str(pdu.betaPss)]);
            end
            beta_sss = 1;

            apply_modComp = 1;
            if apply_modComp == 0
                for idxSym = startSymbol:startSymbol+nSymbol-1
                    sectionType = 1;
                    extType = 0; % do NOT apply modulation compression
                    fhMsg = initFhMsgParams(sectionType, extType);
                    fhMsg.startSymbolid = idxSym;
                    fhMsg.startPrbc = startPrb;
                    fhMsg.numPrbc = nPrb;
                    fhMsg.portIdx = 0;
                    idxFhMsgParams = idxFhMsgParams + 1;
                    fhMsgParams{idxFhMsgParams} = fhMsg;
                end
            else % modulation compression SE4
                idxFhMsgRb = 1;
                fhMsgRb = [];
                for idxSym =  1:4
                    for idxPrb = 1:nPrb
                        reMask = ssbMap(12*(idxPrb-1)+1:12*idxPrb, idxSym)';
                        if sum(reMask) > 0
                            findPSS = ~isempty(find(reMask == 1, 1));
                            findSSS = ~isempty(find(reMask == 2, 1));
                            findQPSK = ~isempty(find(reMask == 3, 1));
                            if  findPSS
                                fhMsgRb(idxFhMsgRb).startSymbolid = idxSym + startSymbol - 1;
                                fhMsgRb(idxFhMsgRb).startPrbc = idxPrb + startPrb - 1;
                                fhMsgRb(idxFhMsgRb).reMask = double(reMask == 1);
                                fhMsgRb(idxFhMsgRb).modCompScaler = sqrt(2) * beta_pss;
                                fhMsgRb(idxFhMsgRb).udIqWidth = 2;
                                fhMsgRb(idxFhMsgRb).csf = 0;
                                idxFhMsgRb = idxFhMsgRb + 1;
                            elseif findSSS && ~findQPSK
                                fhMsgRb(idxFhMsgRb).startSymbolid = idxSym + startSymbol - 1;
                                fhMsgRb(idxFhMsgRb).startPrbc = idxPrb + startPrb - 1;
                                fhMsgRb(idxFhMsgRb).reMask = double(reMask == 2);
                                fhMsgRb(idxFhMsgRb).modCompScaler= sqrt(2) * beta_sss;
                                fhMsgRb(idxFhMsgRb).udIqWidth = 2;
                                fhMsgRb(idxFhMsgRb).csf = 0;
                                idxFhMsgRb = idxFhMsgRb + 1;
                            elseif findSSS && findQPSK
                                if find(reMask == 2, 1) > find(reMask == 3, 1) % QPSK before SSS
                                    fhMsgRb(idxFhMsgRb).startSymbolid = idxSym + startSymbol - 1;
                                    fhMsgRb(idxFhMsgRb).startPrbc = idxPrb + startPrb - 1;
                                    fhMsgRb(idxFhMsgRb).reMask = double(reMask == 3);
                                    fhMsgRb(idxFhMsgRb).modCompScaler = beta_sss * qamScaler(1);
                                    fhMsgRb(idxFhMsgRb).udIqWidth = 1;
                                    fhMsgRb(idxFhMsgRb).csf = 1;
                                    idxFhMsgRb = idxFhMsgRb + 1;
                                    fhMsgRb(idxFhMsgRb).startSymbolid = idxSym + startSymbol - 1;
                                    fhMsgRb(idxFhMsgRb).startPrbc = idxPrb + startPrb - 1;
                                    fhMsgRb(idxFhMsgRb).reMask = double(reMask == 2);
                                    fhMsgRb(idxFhMsgRb).modCompScaler = sqrt(2) * beta_sss;
                                    fhMsgRb(idxFhMsgRb).udIqWidth = 2;
                                    fhMsgRb(idxFhMsgRb).csf = 0;
                                    idxFhMsgRb = idxFhMsgRb + 1;
                                else % SSS before QPSK
                                    fhMsgRb(idxFhMsgRb).startSymbolid = idxSym + startSymbol - 1;
                                    fhMsgRb(idxFhMsgRb).startPrbc = idxPrb + startPrb - 1;
                                    fhMsgRb(idxFhMsgRb).reMask = double(reMask == 2);
                                    fhMsgRb(idxFhMsgRb).modCompScaler = sqrt(2) * beta_sss;
                                    fhMsgRb(idxFhMsgRb).udIqWidth = 2;
                                    fhMsgRb(idxFhMsgRb).csf = 0;
                                    idxFhMsgRb = idxFhMsgRb + 1;
                                    fhMsgRb(idxFhMsgRb).startSymbolid = idxSym + startSymbol - 1;
                                    fhMsgRb(idxFhMsgRb).startPrbc = idxPrb + startPrb - 1;
                                    fhMsgRb(idxFhMsgRb).reMask = double(reMask == 3);
                                    fhMsgRb(idxFhMsgRb).modCompScaler = beta_sss * qamScaler(1);
                                    fhMsgRb(idxFhMsgRb).udIqWidth = 1;
                                    fhMsgRb(idxFhMsgRb).csf = 1;
                                    idxFhMsgRb = idxFhMsgRb + 1;
                                end
                            elseif findQPSK
                                fhMsgRb(idxFhMsgRb).startSymbolid = idxSym + startSymbol - 1;
                                fhMsgRb(idxFhMsgRb).startPrbc = idxPrb + startPrb - 1;
                                fhMsgRb(idxFhMsgRb).reMask = double(reMask == 3);
                                fhMsgRb(idxFhMsgRb).modCompScaler = beta_sss * qamScaler(1);
                                fhMsgRb(idxFhMsgRb).udIqWidth = 1;
                                fhMsgRb(idxFhMsgRb).csf = 1;
                                idxFhMsgRb = idxFhMsgRb + 1;
                            else
                                error('error in fhMsgRb SSB ...\n');
                            end
                        end
                    end
                end

                nFhMsgRb = length(fhMsgRb);
                preFhMsgRb = fhMsgRb(1);
                idxList = 1;
                fhMsgList = [];
                sectionType = 1;
                extType = 4;
                fhMsgList{idxList} = initFhMsgParams(sectionType, extType);
                fhMsgList{idxList}.startSymbolid = preFhMsgRb.startSymbolid;
                fhMsgList{idxList}.portIdx = 0;
                fhMsgList{idxList}.startPrbc = preFhMsgRb.startPrbc;
                fhMsgList{idxList}.numPrbc = 0;
                fhMsgList{idxList}.udIqWidth = preFhMsgRb.udIqWidth;
                fhMsgList{idxList}.reMask = preFhMsgRb.reMask;
                fhMsgList{idxList}.modCompScaler = preFhMsgRb.modCompScaler;
                fhMsgList{idxList}.csf = preFhMsgRb.csf;

                for idxFhMsgRb = 1:nFhMsgRb
                    thisFhMsgRb = fhMsgRb(idxFhMsgRb);

                    msgMatch = 1;
                    if thisFhMsgRb.startSymbolid ~= preFhMsgRb.startSymbolid
                        msgMatch = 0;
                    end
                    if thisFhMsgRb.udIqWidth ~= preFhMsgRb.udIqWidth
                        msgMatch = 0;
                    end
                    if sum(sum(abs(thisFhMsgRb.reMask - preFhMsgRb.reMask))) > 1e-5
                        msgMatch = 0;
                    end
                    if sum(abs(thisFhMsgRb.modCompScaler - preFhMsgRb.modCompScaler)) > 1e-5
                        msgMatch = 0;
                    end
                    if sum(abs(thisFhMsgRb.csf - preFhMsgRb.csf)) > 1e-5
                        msgMatch = 0;
                    end

                    if msgMatch
                        fhMsgList{idxList}.numPrbc = fhMsgList{idxList}.numPrbc + 1;
                    else
                        idxList = idxList + 1;
                        sectionType = 1;
                        extType = 4;
                        fhMsgList{idxList} = initFhMsgParams(sectionType, extType);
                        fhMsgList{idxList}.startSymbolid = thisFhMsgRb.startSymbolid;
                        fhMsgList{idxList}.portIdx = 0;
                        fhMsgList{idxList}.startPrbc = thisFhMsgRb.startPrbc;
                        fhMsgList{idxList}.numPrbc = 1;
                        fhMsgList{idxList}.udIqWidth = thisFhMsgRb.udIqWidth;
                        fhMsgList{idxList}.reMask = thisFhMsgRb.reMask;
                        fhMsgList{idxList}.modCompScaler = thisFhMsgRb.modCompScaler;
                        fhMsgList{idxList}.csf = thisFhMsgRb.csf;
                        preFhMsgRb = thisFhMsgRb;
                    end
                end

                nFhMsgList = length(fhMsgList);
                for idxList = 1:nFhMsgList
                    idxFhMsgParams = idxFhMsgParams + 1;
                    fhMsgParams{idxFhMsgParams} = fhMsgList{idxList};
                end
            end

            % populate modCompRbParam SE4/5
            
            for idxSym =  1:4
                for idxPrb = 1:nPrb
                    reMask = ssbMap(12*(idxPrb-1)+1:12*idxPrb, idxSym)';
                    if sum(reMask) > 0
                        findPSS = ~isempty(find(reMask == 1, 1));
                        findSSS = ~isempty(find(reMask == 2, 1));
                        findQPSK = ~isempty(find(reMask == 3, 1));
                        if  findPSS
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).nMask = 1;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).reMask(1,:) = double(reMask == 1);
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).scaler(1) = sqrt(2) * beta_pss;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).udIqWidth = 2;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).csf(1) = 0;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).chanType(1) = 1;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).pduIdx = idxPdu;
                        elseif findSSS && ~findQPSK
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).nMask = 1;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).reMask(1,:) = double(reMask == 2);
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).scaler(1) = sqrt(2) * beta_sss;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).udIqWidth = 2;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).csf(1) = 0;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).chanType(1) = 1;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).pduIdx = idxPdu;
                        elseif findSSS && findQPSK
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).nMask = 2;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).reMask(1,:) = double(reMask == 2);
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).scaler(1) = sqrt(2) * beta_sss;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).udIqWidth = 2;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).csf(1) = 0;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).chanType(1) = 1;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).reMask(2,:) = double(reMask == 3);
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).scaler(2) = beta_sss * qamScaler(1);
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).csf(2) = 0;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).chanType(2) = 1;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).pduIdx = idxPdu;
                        elseif findQPSK
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).nMask = 1;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).reMask(1,:) = double(reMask == 3);
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).scaler(1) = beta_sss * qamScaler(1);
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).udIqWidth = 1;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).csf(1) = 1;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).chanType(1) = 1;
                            modCompRbParam(idxPrb+startPrb, idxSym+startSymbol, 1).pduIdx = idxPdu;
                        else
                            error('error in modComp SSB ...\n');
                        end
                    end
                end
            end
        case 'pdcch'
            startRb = pdu.BWPStart;      % pdcch tx starting RB (0 indexing)
            startSym = pdu.StartSymbolIndex;    % starting symbol pdcch tx (0 indexing)
            nSym = pdu.DurationSymbols;            % number of pdcch tx symbols (1-3)
            bundleSize = pdu.RegBundleSize;
            interleaveSize = pdu.InterleaverSize;
            shiftIdx = pdu.ShiftIndex;
            interleaved = pdu.CceRegMappingType;
            FreqDomainResource0 = dec2bin(pdu.FreqDomainResource0, 32)-'0';
            FreqDomainResource1 = dec2bin(pdu.FreqDomainResource1, 32)-'0';
            FreqDomainResource = [FreqDomainResource0, FreqDomainResource1];
            lenCoresetMap = find(FreqDomainResource, 1, 'last');
            coresetMap = FreqDomainResource(1:lenCoresetMap);
            numDlDci = pdu.numDlDci;
            CoreSetType = pdu.CoreSetType;
            
            N_CCE = sum(coresetMap)*nSym;
            N_REG =  N_CCE*6;
            C = N_REG/(bundleSize*max(1,interleaveSize));
            if abs(round(C)-C) > 0.001 && interleaved
                error('PDCCH: C is not an integer ... \n');
            end
            
            if ~ interleaved
                bundleSize = 6;
            end
            
            N_bundle = N_CCE*6/bundleSize;
            N_bundle_phy = lenCoresetMap*nSym*6/bundleSize;
            bundleTable = zeros(1, N_bundle_phy);
            idxLogBundle = 0;
            idxPhyBundle = 0;
            nBundlePerRb6 = 6*nSym/bundleSize;
            
            % find mapping table for contiguous or non-contiguous REG allocation
            for rb6Idx = 1:lenCoresetMap
                if coresetMap(rb6Idx)
                    bundleTable(idxLogBundle + [1:nBundlePerRb6]) = ...
                        idxPhyBundle + [1:nBundlePerRb6];
                    idxLogBundle = idxLogBundle + nBundlePerRb6;
                end
                idxPhyBundle = idxPhyBundle + nBundlePerRb6;
            end
            
            % find mapping table for non-interleaved or interleaved cce-reg mapping
            for idxBundle = 0:N_bundle-1
                if interleaved
                    c = floor(idxBundle/interleaveSize);
                    r = mod(idxBundle, interleaveSize);
                    bundleMap(idxBundle+1) = mod(r*C + c + shiftIdx, N_bundle);
                else
                    bundleMap(idxBundle+1) = idxBundle;
                end
            end
            
            dmrsPerBundle = bundleSize/nSym*3;
            rbPerBundle = bundleSize/nSym;
            if CoreSetType == 0
                endRb = lenCoresetMap*6;
            else
                endRb = startRb + lenCoresetMap*6;
            end
            N_qpskPerBundle = rbPerBundle*9;
            qamLoc_base = [0 2 3 4 6 7 8 10 11];
            qamLoc = [];
            for i = 1:rbPerBundle
                qamLoc = [qamLoc, qamLoc_base + (i-1) * 12];
            end
            startFreq = startRb*12;
            
            for idxDCI = 1:numDlDci
                DCI = pdu.DCI{idxDCI};
                Npayload = DCI.PayloadSizeBits;
                Payload = DCI.Payload(1:Npayload);
                rntiCrc = DCI.RNTI;
                rntiBits = DCI.ScramblingRNTI;
                dmrsId = DCI.ScramblingId;
                aggrL = DCI.AggregationLevel;
                cceIdx = DCI.CceIndex;
                
                qam_dB = DCI.powerControlOffsetSSProfileNR;
                beta_qam = 10^(qam_dB/20);
                dmrs_dB = DCI.powerControlOffsetSSProfileNR;
                beta_dmrs = 10^(dmrs_dB/20);
                    
                % mark used bundle based on cce index and aggrL
                usedBundleMap = zeros(1, N_bundle_phy);
                for usedCceIdx = cceIdx:cceIdx+aggrL-1
                    usedBundleIdx = 6*usedCceIdx/bundleSize:6*(usedCceIdx+1)/bundleSize-1;
                    logBundleIdx = bundleMap(usedBundleIdx+1)+1;
                    usedBundleMap(bundleTable(logBundleIdx)) = 1;
                end
                
                % Map to RE
                for i = 0 : (nSym - 1)
                    idxSym = startSym + i;
                    rbMap = zeros(1, carrier.N_grid_size_mu);
                    for bundleIdx = 0:N_bundle_phy-1
                        if usedBundleMap(bundleIdx+1)
                            idxRb = startFreq/12+bundleIdx*rbPerBundle;
                            rbMap(idxRb+1:idxRb+rbPerBundle) = 1;
                            
                            % populate modCompRbParam                            
                            for idxRbinBundle = 1:rbPerBundle
                                modCompRbParam(idxRb+idxRbinBundle, idxSym+1, 1).nMask = 2;
                                modCompRbParam(idxRb+idxRbinBundle, idxSym+1, 1).udIqWidth = 1;
                                modCompRbParam(idxRb+idxRbinBundle, idxSym+1, 1).reMask = [0 1 0 0 0 1 0 0 0 1 0 0; 1 0 1 1 1 0 1 1 1 0 1 1];
                                modCompRbParam(idxRb+idxRbinBundle, idxSym+1, 1).scaler = [beta_dmrs, beta_qam] * qamScaler(1); % QPSK
                                modCompRbParam(idxRb+idxRbinBundle, idxSym+1, 1).csf = [1, 1];
                                modCompRbParam(idxRb+idxRbinBundle, idxSym+1, 1).chanType = [2 2];
                                modCompRbParam(idxRb+idxRbinBundle, idxSym+1, 1).pduIdx = idxPdu;
                            end
                        end
                    end
                    
                    nContBlock = 0;
                    contBlock = [];
                    preMap = 0;
                    lenRbMap = length(rbMap);
                    for idxRbMap = 1:lenRbMap
                        currMap = rbMap(idxRbMap);
                        if preMap == 0 && currMap == 1
                            nContBlock = nContBlock + 1;
                            contBlock(nContBlock).start = idxRbMap-1;
                            contBlock(nContBlock).end = 0;
                        elseif preMap == 1 && currMap == 0
                            contBlock(nContBlock).end = idxRbMap-2;
                        end
                        preMap = currMap;
                    end
                    if contBlock(nContBlock).end == 0
                        contBlock(nContBlock).end = lenRbMap-1;
                    end
                    
                    for idxBlock = 1:nContBlock
                        % DMRS
                        sectionType = 1;
                        extType = 4; % apply modulation compression
                        fhMsg = initFhMsgParams(sectionType, extType);
                        fhMsg.startSymbolid = idxSym;
                        fhMsg.startPrbc = contBlock(idxBlock).start;
                        fhMsg.numPrbc = contBlock(idxBlock).end - contBlock(idxBlock).start + 1;
                        fhMsg.udIqWidth = 1; % QPSK
                        fhMsg.udCompMeth = 4; % modulation compression
                        fhMsg.reMask = [0 1 0 0 0 1 0 0 0 1 0 0];
                        fhMsg.modCompScaler = beta_dmrs;
                        fhMsg.portIdx = 0;
                        idxFhMsgParams = idxFhMsgParams + 1;
                        fhMsgParams{idxFhMsgParams} = fhMsg;
                        
                        % Data
                        sectionType = 1;
                        extType = 4; % apply modulation compression
                        fhMsg = initFhMsgParams(sectionType, extType);
                        fhMsg.startSymbolid = idxSym;
                        fhMsg.startPrbc = contBlock(idxBlock).start;
                        fhMsg.numPrbc = contBlock(idxBlock).end - contBlock(idxBlock).start + 1;
                        fhMsg.udIqWidth = 1; % QPSK
                        fhMsg.udCompMeth = 4; % modulation compression
                        fhMsg.reMask = [1 0 1 1 1 0 1 1 1 0 1 1];
                        fhMsg.modCompScaler = beta_qam;
                        fhMsg.portIdx = 0;
                        idxFhMsgParams = idxFhMsgParams + 1;
                        fhMsgParams{idxFhMsgParams} = fhMsg;
                    end
                end
            end            
        case 'pdsch'
            resourceAlloc   = pdu.resourceAlloc;
            tmp_rbBitmap    = [zeros(1,pdu.BWPStart),reshape(flipud(dec2bin(pdu.rbBitmap,8)')-'0',1,[])];
            rbBitmap        = tmp_rbBitmap(1:273);
            rbStart = pdu.rbStart + pdu.BWPStart;
            tbSize = pdu.rbSize;
            StartSymbolIndex = pdu.StartSymbolIndex;
            NrOfSymbols = pdu.NrOfSymbols;
            DmrsSymbPos = pdu.DmrsSymbPos;
            qamModOrder = pdu.qamModOrder;
            nCdm = pdu.numDmrsCdmGrpsNoData;
                        
            pdsch2csirs = pdu.powerControlOffset - 8;
            csirs2ssb = (pdu.powerControlOffsetSS-1) * 3;
            pdsch2ssb = pdsch2csirs + csirs2ssb;
            beta_qam = 10^(pdsch2ssb/20);
            beta_dmrs = 10^(pdsch2ssb/20);
        
            for symIdx = StartSymbolIndex:StartSymbolIndex + NrOfSymbols - 1
                for idxPort = find(fliplr(pdu.dmrsPorts))
                    if DmrsSymbPos(symIdx+1) % dmrs symbol
                        
                        % populate modCompRbParam
                        if resourceAlloc == 0
                            list_valid_RBs = find(rbBitmap==1);
                        else
                            list_valid_RBs = rbStart+1:rbStart + tbSize;
                        end
                        nContBlock = 0;
                        preIdxRb = -1;
                        contBlock = [];
                        for idxRb = list_valid_RBs
                            currIdxRb = idxRb;
                            if (currIdxRb - preIdxRb) > 1
                                nContBlock = nContBlock + 1;
                                contBlock(nContBlock).start = idxRb-1;
                                contBlock(nContBlock).end = idxRb-1;
                            else
                                contBlock(nContBlock).end = idxRb-1;
                            end
                            preIdxRb = currIdxRb;

                            if pdu.SCID
                                portIdx = idxPort + 8;
                            else
                                portIdx = idxPort;
                            end
                            if pdu.nlAbove16
                                portIdx = portIdx + 16;
                            end
                            modCompRbParam(idxRb, symIdx+1, portIdx).nMask = 1;
                            modCompRbParam(idxRb, symIdx+1, portIdx).udIqWidth = 1; % QPSK
                            dmrsOffset = [0 0 1 1 0 0 1 1];
                            if dmrsOffset(idxPort) == 0
                                modCompRbParam(idxRb, symIdx+1, portIdx).reMask(1,:) = [1 0 1 0 1 0 1 0 1 0 1 0];
                            else
                                modCompRbParam(idxRb, symIdx+1, portIdx).reMask(1,:) = [0 1 0 1 0 1 0 1 0 1 0 1];
                            end
                            modCompRbParam(idxRb, symIdx+1, portIdx).scaler(1) = sqrt(2)*beta_dmrs * qamScaler(1); % QPSK
                            modCompRbParam(idxRb, symIdx+1, portIdx).csf(1) = 1;
                            modCompRbParam(idxRb, symIdx+1, portIdx).chanType(1) = 3;
                            modCompRbParam(idxRb, symIdx+1, portIdx).pduIdx = idxPdu;
                            if nCdm == 1 % data RE on dmrs symbol
                                modCompRbParam(idxRb, symIdx+1, portIdx).nMask = 2;
                                modCompRbParam(idxRb, symIdx+1, portIdx).udIqWidth = qamModOrder/2; % QPSK
                                reMask = 1 - modCompRbParam(idxRb, symIdx+1, portIdx).reMask(1,:);
                                modCompRbParam(idxRb, symIdx+1, portIdx).reMask(2,:) = reMask;
                                modCompRbParam(idxRb, symIdx+1, portIdx).scaler(2) = beta_qam * qamScaler(qamModOrder/2); 
                                % reduce DMRS power by 3dB
                                modCompRbParam(idxRb, symIdx+1, portIdx).scaler(1) = sqrt(0.5)* modCompRbParam(idxRb, symIdx+1, portIdx).scaler(1);
                                modCompRbParam(idxRb, symIdx+1, portIdx).csf(2) = 1;
                                if qamModOrder/2 > 1
                                    modCompRbParam(idxRb, symIdx+1, portIdx).csf(1) = 0; % no shift for DMRS QPSK
                                end
                                modCompRbParam(idxRb, symIdx+1, portIdx).chanType(2) = 3;
                                % PDU identifier already set above, no need to set again
                            end
                        end
                        for idxBlock = 1:nContBlock
                            sectionType = 1;
                            extType = 4; % apply modulation compression
                            fhMsg = initFhMsgParams(sectionType, extType);
                            fhMsg.startSymbolid = symIdx;
                            fhMsg.startPrbc = contBlock(idxBlock).start;
                            fhMsg.numPrbc = contBlock(idxBlock).end - contBlock(idxBlock).start + 1;
                            fhMsg.udIqWidth = 1; % QPSK
                            fhMsg.udCompMeth = 4; % modulation compression
                            fhMsg.portIdx = idxPort-1;
                            if pdu.SCID
                                fhMsg.portIdx = fhMsg.portIdx + 8;
                            end
                            if pdu.nlAbove16
                                fhMsg.portIdx = fhMsg.portIdx + 16;
                            end
                            dmrsOffset = [0 0 1 1 0 0 1 1];
                            if dmrsOffset(idxPort) == 0
                                fhMsg.reMask = [1 0 1 0 1 0 1 0 1 0 1 0];
                            else
                                fhMsg.reMask = [0 1 0 1 0 1 0 1 0 1 0 1];
                            end
                            fhMsg.modCompScaler = sqrt(2)*beta_dmrs;
                            idxFhMsgParams = idxFhMsgParams + 1;
                            fhMsgParams{idxFhMsgParams} = fhMsg;
                            if nCdm == 1 % data RE on dmrs symbol
                                fhMsgParams{idxFhMsgParams}.modCompScaler = sqrt(0.5)*fhMsgParams{idxFhMsgParams}.modCompScaler;
                                sectionType = 1;
                                extType = 4; % apply modulation compression
                                fhMsg = initFhMsgParams(sectionType, extType);
                                fhMsg.startSymbolid = symIdx;
                                fhMsg.startPrbc = contBlock(idxBlock).start;
                                fhMsg.numPrbc = contBlock(idxBlock).end - contBlock(idxBlock).start + 1;
                                fhMsg.udIqWidth = qamModOrder/2;
                                fhMsg.udCompMeth = 4; % modulation compression
                                fhMsg.portIdx = idxPort-1;
                                if pdu.SCID
                                    fhMsg.portIdx = fhMsg.portIdx + 8;
                                end
                                if pdu.nlAbove16
                                    fhMsg.portIdx = fhMsg.portIdx + 16;
                                end
                                dmrsOffset = [0 0 1 1 0 0 1 1];
                                if dmrsOffset(idxPort) == 0
                                    fhMsg.reMask = 1-[1 0 1 0 1 0 1 0 1 0 1 0];
                                else
                                    fhMsg.reMask = 1-[0 1 0 1 0 1 0 1 0 1 0 1];
                                end
                                fhMsg.modCompScaler = beta_qam * qamScaler(qamModOrder/2);
                                idxFhMsgParams = idxFhMsgParams + 1;
                                fhMsgParams{idxFhMsgParams} = fhMsg;
                            end
                        end
                    else % data symbol
                        preReMap = zeros(12, 1);
                        preIdxRb = -2;
                        nContBlock = 0;
                        contBlock = [];
                        if resourceAlloc == 0
                            list_valid_RBs = find(rbBitmap==1) - 1;
                            rbStart = min(list_valid_RBs);
                        else
                            list_valid_RBs = rbStart:rbStart + tbSize - 1;
                        end                        
                        for rbIdx = list_valid_RBs
                            currReMap = Xtf_remap(rbIdx*12+1:(rbIdx+1)*12, symIdx+1);
                            currIdxRb = rbIdx;
                            if (sum(abs(preReMap - currReMap)) > 0) || ((currIdxRb - preIdxRb) > 1)
                                nContBlock = nContBlock + 1;
                                contBlock(nContBlock).start = rbIdx;
                                contBlock(nContBlock).end = rbIdx;
                            else
                                if rbIdx == rbStart
                                    nContBlock = nContBlock + 1;
                                    contBlock(nContBlock).start = rbIdx;
                                    contBlock(nContBlock).end = rbIdx;
                                end
                                contBlock(nContBlock).end = rbIdx;
                            end
                            preReMap = currReMap;
                            preIdxRb = currIdxRb;

                            % populate modCompRbParam
                            if pdu.SCID
                                portIdx = idxPort + 8;
                            else
                                portIdx = idxPort;
                            end
                            if pdu.nlAbove16
                                portIdx = portIdx + 16;
                            end
                            
                            if modCompRbParam(rbIdx+1, symIdx+1, portIdx).nMask == 0
                                idxMask = 1;
                                modCompRbParam(rbIdx+1, symIdx+1, portIdx).nMask = 1;
                            else
                                idxMask = 2;
                                modCompRbParam(rbIdx+1, symIdx+1, portIdx).nMask = 2;
                            end                            
                            modCompRbParam(rbIdx+1, symIdx+1, portIdx).udIqWidth = qamModOrder/2; % QPSK
                            reMask = (1-Xtf_remap(12*rbIdx+1:12*(rbIdx+1), symIdx+1))';
                            modCompRbParam(rbIdx+1, symIdx+1, portIdx).reMask(idxMask,:) = reMask;
                            modCompRbParam(rbIdx+1, symIdx+1, portIdx).scaler(idxMask) = beta_qam * qamScaler(qamModOrder/2); % QPSK
                            modCompRbParam(rbIdx+1, symIdx+1, portIdx).csf(idxMask) = 1;
                            if qamModOrder/2 > 1 && idxMask == 2 % overlap with CSIRS and PDSCH QAM > QPSK
                                modCompRbParam(rbIdx+1, symIdx+1, portIdx).csf(1) = 0; % no shift for CSIRS QPSK
                            end                            
                            modCompRbParam(rbIdx+1, symIdx+1, portIdx).chanType(idxMask) = 3;
                            modCompRbParam(rbIdx+1, symIdx+1, portIdx).pduIdx = idxPdu;
                        end
                        if contBlock(nContBlock).end == 0
                            contBlock(nContBlock).end = rbStart + tbSize - 1;
                        end
                        for idxBlock = 1:nContBlock
                            sectionType = 1;
                            extType = 4; % apply modulation compression
                            fhMsg = initFhMsgParams(sectionType, extType);
                            fhMsg.startSymbolid = symIdx;
                            fhMsg.startPrbc = contBlock(idxBlock).start;
                            fhMsg.numPrbc = contBlock(idxBlock).end - contBlock(idxBlock).start + 1;
                            fhMsg.udIqWidth = qamModOrder/2; 
                            fhMsg.udCompMeth = 4; % modulation compression
                            fhMsg.reMask = (1-Xtf_remap(12*fhMsg.startPrbc+1:12*(fhMsg.startPrbc+1), symIdx+1))';
                            fhMsg.portIdx = idxPort-1;
                            if pdu.SCID
                                fhMsg.portIdx = fhMsg.portIdx + 8;
                            end
                            if pdu.nlAbove16
                                    fhMsg.portIdx = fhMsg.portIdx + 16;
                            end
                            fhMsg.modCompScaler = beta_qam * qamScaler(qamModOrder/2);
                            idxFhMsgParams = idxFhMsgParams + 1;
                            fhMsgParams{idxFhMsgParams} = fhMsg;
                        end
                    end
                end
            end
        case 'csirs'
            Row = pdu.Row;
            CDMType = pdu.CDMType;
            FreqDensity = pdu.FreqDensity;
            StartRB = pdu.StartRB;
            NrOfRBs = pdu.NrOfRBs;
            SymbL0 = pdu.SymbL0;
            SymbL1 = pdu.SymbL1;
            FreqDomain_bin = dec2bin(pdu.FreqDomain, 12) - '0';
            SymbL = [SymbL0, SymbL1];
            
            % read table 7.4.1.5.3-1
            nPort = row2nPort(Row);
            if nPort >= nPort_enable_csirs_compression
                enable_csirs_compression = 1;
            else
                enable_csirs_compression = 0;
            end

            KBarLBar = table.csirsLocTable.KBarLBar{Row};
            CDMGroupIndices = table.csirsLocTable.CDMGroupIndices{Row};
            KPrime = table.csirsLocTable.KPrime{Row};
            LPrime = table.csirsLocTable.LPrime{Row};
            
            beta_db = (pdu.powerControlOffsetSS - 1)*3;
            beta = 10^(beta_db/20);
            
            % 0: noCDM, 1: fd-CDM2, 2: cdm4-FD2-TD2, 3: cdm8-FD2-TD4
            switch CDMType
                case 0
                    seqTable = table.noCdmTable;
                    LL = 1;
                case 1
                    seqTable = table.fdCdm2Table;
                    LL = 2;
                case 2
                    seqTable = table.cdm4Table;
                    LL = 4;
                case 3
                    seqTable = table.cdm8Table;
                    LL = 8;
                otherwise
                    error('CDMType is not supported...\n');
            end
            
            switch FreqDensity
                case 0
                    rho = 0.5;
                    genEvenRB = 1;
                case 1
                    rho = 0.5;
                    genEvenRB = 0;
                case 2
                    rho = 1;
                case 3
                    rho = 3;
            end
            
            if nPort == 1
                alpha = rho;
            else
                alpha = 2*rho;
            end
            
            % FreqDomain_bin = dec2bin(FreqDomain, 12) - '0';
            FreqDomain_flip = fliplr(FreqDomain_bin);
            switch Row
                case 1
                    idxOne = find(FreqDomain_flip(1:4));
                    ki =idxOne - 1;
                case 2
                    idxOne = find(FreqDomain_flip(1:12));
                    ki = idxOne - 1;
                case 4
                    idxOne = find(FreqDomain_flip(1:3));
                    ki = 4*(idxOne - 1);
                otherwise
                    idxOne = find(FreqDomain_flip(1:6));
                    ki = 2*(idxOne - 1);
            end
            
            lenKBarLBar = length(KBarLBar);
            lenLPrime = length(LPrime);
            lenKPrime = length(KPrime);
            
            hasTwoSyms = ismember(Row, [13 14 16 17]);
            
            if rho == 0.5
                rb = 1;
                isEvenRB = (mod(StartRB, 2) == 0);
                if (genEvenRB && isEvenRB) || (~genEvenRB && ~isEvenRB )
                    startPrbc = StartRB;
                    numPrbc = ceil(NrOfRBs/2);
                else
                    startPrbc = StartRB+1;
                    numPrbc = floor(NrOfRBs/2);
                end
            else
                rb = 0;
                startPrbc = StartRB;
                numPrbc = NrOfRBs;
            end
            
            remap = zeros(12, 14);
            remap_port = zeros(12, 14, max([carrier.numTxPort, length(pdu.beamIdx)]));
            
            for idxKBarLBar = 1:lenKBarLBar
                kl_BarPair = KBarLBar{idxKBarLBar};
                if Row == 1 || Row == 4
                    k_bar = ki(1) + kl_BarPair(1);
                else
                    k_bar = ki(kl_BarPair(1)+1);
                end
                if hasTwoSyms && idxKBarLBar > lenKBarLBar/2
                    l_bar = SymbL(2) + kl_BarPair(2);
                else
                    l_bar = SymbL(1) + kl_BarPair(2);
                end
                for idxLPrime = 1:lenLPrime
                    for idxKPrime = 1:lenKPrime
                        k_prime = KPrime(idxKPrime);
                        k = k_bar + k_prime;
                        l_prime = LPrime(idxLPrime);
                        ll = l_bar + l_prime;
                        remap(k+1, ll+1) = 1;
                        for s = 1:LL
                            jj = CDMGroupIndices(idxKBarLBar);
                            p = jj*LL+s;
                            remap_port(k+1, ll+1, p) = 1;
                        end
                    end
                end
            end
            
            for idxRB = StartRB:StartRB + NrOfRBs - 1
                isEvenRB = (mod(idxRB, 2) == 0);
                if (rho == 0.5)
                    if (genEvenRB && ~isEvenRB) || (~ genEvenRB && isEvenRB)
                        continue;
                    end
                end
                Xtf_remap(12*idxRB+1:12*(idxRB+1), :) = remap + Xtf_remap(12*idxRB+1:12*(idxRB+1), :);
            end
            
            if pdu.CSIType ~= 2 % Not ZP-CSI-RS
                for idxSym = 1:14
                    for idxPort = 1:nPort
                        if sum(remap_port(:, idxSym, idxPort)) > 0
                            sectionType = 1;
                            extType = 4; % apply modulation compression
                            fhMsg = initFhMsgParams(sectionType, extType);
                            fhMsg.startSymbolid = idxSym-1;
                            fhMsg.startPrbc = startPrbc;
                            fhMsg.numPrbc = numPrbc;
                            fhMsg.udIqWidth = 1; % QPSK
                            fhMsg.udCompMeth = 4; % modulation compression
                            fhMsg.modCompScaler = beta;
                            fhMsg.reMask = remap_port(:, idxSym, idxPort)';
                            fhMsg.rb = rb;
                            fhMsg.portIdx = idxPort - 1;
                            fhMsg.modCompScaler = beta;
                            if enable_csirs_compression
                                fhMsg.csiRsCompression = 1;
                            end
                            idxFhMsgParams = idxFhMsgParams + 1;
                            fhMsgParams{idxFhMsgParams} = fhMsg;
                            
                            % populate modCompRbParam
                            for prbIdx = 1:numPrbc
                                if rb == 1
                                    idxPrb = startPrbc + 2*(prbIdx-1) + 1;
                                else
                                    idxPrb = startPrbc + prbIdx;
				end
                                flowIdx = idxPort;
                                if (fhMsg.csiRsCompression)
                                    flowIdx = mod(idxPort-1, carrier.numTxPort)+1;
                                end
                                currMap = modCompRbParam(idxPrb, idxSym, flowIdx);
                                reMask = remap_port(:, idxSym, idxPort)';
                                if currMap.chanType(1) == 4 && currMap.scaler(1) - beta * qamScaler(1) < 1e-5
                                    reMask = reMask | currMap.reMask(1,:);
                                elseif currMap.nMask > 0
                                    warning('Existing PRB is occupied by data incompatible with CSIRS compression');
                                end
                                modCompRbParam(idxPrb, idxSym, flowIdx).nMask = 1;
                                modCompRbParam(idxPrb, idxSym, flowIdx).udIqWidth = 1;
                                modCompRbParam(idxPrb, idxSym, flowIdx).reMask(1,:) = reMask;
                                modCompRbParam(idxPrb, idxSym, flowIdx).scaler(1) = beta * qamScaler(1); % QPSK
                                modCompRbParam(idxPrb, idxSym, flowIdx).csf(1) = [1];
                                modCompRbParam(idxPrb, idxSym, flowIdx).chanType(1) = [4];
                                modCompRbParam(idxPrb, idxSym, flowIdx).pduIdx = idxPdu;
                            end                                
                            
                        end
                    end
                end
            end
        otherwise
            error('pdu type is not supported ...\n');
    end
end

idxList = 0;
for idxSym = 1:size(modCompRbParam,2)
    for idxPort = 1:size(modCompRbParam,3)
        idxList = idxList + 1;
        preMap = modCompRbParam(1, idxSym, idxPort);
        modCompList_all(idxList).idxSym = idxSym;
        modCompList_all(idxList).idxPort = idxPort;
        modCompList_all(idxList).idxPrb = 1;
        modCompList_all(idxList).nPrb = 0;
        modCompList_all(idxList).nMask = preMap.nMask;
        modCompList_all(idxList).udIqWidth = preMap.udIqWidth;
        modCompList_all(idxList).reMask = preMap.reMask;
        modCompList_all(idxList).scaler = preMap.scaler;
        modCompList_all(idxList).csf = preMap.csf;
        modCompList_all(idxList).chanType = preMap.chanType;
        
        for idxPrb = 1:carrier.N_grid_size_mu
            thisMap = modCompRbParam(idxPrb, idxSym, idxPort);
            
            mapMatch = 1;
            if thisMap.nMask ~= preMap.nMask
                mapMatch = 0;
            end
            if thisMap.udIqWidth ~= preMap.udIqWidth
                mapMatch = 0;
            end            
            if sum(sum(abs(thisMap.reMask - preMap.reMask))) > 1e-5
                mapMatch = 0;
            end
            if sum(abs(thisMap.scaler - preMap.scaler)) > 1e-5
                mapMatch = 0;
            end
            if sum(abs(thisMap.csf - preMap.csf)) > 1e-5
                mapMatch = 0;
            end
            if sum(abs(thisMap.chanType - preMap.chanType)) > 1e-5
                mapMatch = 0;
            end            
            if thisMap.pduIdx ~= preMap.pduIdx && thisMap.pduIdx ~= 0
                mapMatch = 0;
            end
            
            if mapMatch
                modCompList_all(idxList).nPrb = modCompList_all(idxList).nPrb + 1;
            else
                idxList = idxList + 1;
                modCompList_all(idxList).idxSym = idxSym;
                modCompList_all(idxList).idxPort = idxPort;
                modCompList_all(idxList).idxPrb = idxPrb;
                modCompList_all(idxList).nPrb = 1;
                modCompList_all(idxList).nMask = thisMap.nMask;
                modCompList_all(idxList).udIqWidth = thisMap.udIqWidth;
                modCompList_all(idxList).reMask = thisMap.reMask;
                modCompList_all(idxList).scaler = thisMap.scaler;
                modCompList_all(idxList).csf = thisMap.csf;
                modCompList_all(idxList).chanType = thisMap.chanType;
                preMap = thisMap;
            end
        end
    end
end


idxNewList = 1;
nList = idxList;

% modcomp needs to be scaled by full carrier BW to normalize for 1.0 amplitude being full carrier at 0dBFS. See NVbug 4728999
bw_scale = 1/sqrt(carrier.N_sc); 
for idxList = 1:nList
    if modCompList_all(idxList).nMask > 0
        modCompList(idxNewList) = modCompList_all(idxList);
        modCompList(idxNewList).scaler = modCompList_all(idxList).scaler*bw_scale;
        idxNewList = idxNewList + 1;
    end
end

if idxNewList == 1
    modCompList = [];
end
        
return
