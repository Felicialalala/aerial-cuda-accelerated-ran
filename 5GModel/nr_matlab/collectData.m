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

function SimCtrl = collectData(SimCtrl, UE, gNB, SysPar, Chan_DL, Chan_UL)
% SimCtrl = collectData(SimCtrl, UE, gNB, Chan)
%
% This function collect data for UE and gNB at the end of each frame for
% performance analysis
%

carrier = SysPar.carrier;
% Chan = SysPar.Chan{1};
% Chan_DL = SysPar.Chan_DL;
% Chan_UL = SysPar.Chan_UL;

idxSlot = SimCtrl.idxSlot;
idxFrame = SimCtrl.idxFrame;
idxSubframe = SimCtrl.idxSubframe;

N_UE = length(UE);
N_pdu = length(gNB.FAPIpdu);
idxPrachPdu = 1;
idxPdcchUeFind = [];
for idxPdu = 1:N_pdu
    pduType = gNB.FAPIpdu{idxPdu}.type;
    switch pduType
        case 'ssb'
            if SimCtrl.enableUeRx
                payload_tx = dec2bin(gNB.FAPIpdu{idxPdu}.bchPayload, 24)-'0';
                for idxUE = 0:N_UE-1
                    UePdu = UE{idxUE+1}.FAPIpdu;
                    findPdu = 0;
                    for idxUePdu = 1:length(UePdu)
                        if strcmp(UePdu{idxUePdu}.type, 'ssb')
                            findPdu = findPdu + 1;
                            idxUePduFind = idxUePdu;
                        end
                    end
                    if findPdu == 0
                        error('No SSB PDU is found...\n');
                    end
                    payload_rx = UE{idxUE+1}.FAPIpdu{idxUePduFind}.pbch_payload;
                    errFlag = sum(abs(payload_tx(:) - payload_rx(:)));
                    if SimCtrl.capSamp.enable
                        errFlag = UE{idxUE+1}.FAPIpdu{idxUePduFind}.errFlag;
                    end
                    rslt = SimCtrl.results.ssb{idxUE+1};
                    rslt.totalCnt = rslt.totalCnt + 1;
                    if errFlag
                        rslt.errCnt = rslt.errCnt + 1;
                    end
                    SimCtrl.results.ssb{idxUE+1} = rslt;
                end
            end
        case 'pdcch'
            if SimCtrl.enableUeRx
                for idxUE = gNB.FAPIpdu{idxPdu}.idxUE
                    if ~ismember(idxUE, idxPdcchUeFind)
                        UePdu = UE{idxUE+1}.FAPIpdu;
                        findPdu = 0;
                        idxUePduFind = [];
                        for idxUePdu = 1:length(UePdu)
                            if strcmp(UePdu{idxUePdu}.type, 'pdcch')
                                findPdu = findPdu + 1;
                                idxUePduFind = [idxUePduFind, idxUePdu];
                            end
                        end
                        if findPdu == 0
                            error('No PDCCH PDU is found...\n');
                        else
                            rslt = SimCtrl.results.pdcch{idxUE+1};
                            for idxUePdu = idxUePduFind
                                numDci = UE{idxUE+1}.FAPIpdu{idxUePdu}.numDlDci;
                                for idxDci = 1:numDci
                                    errFlag = UE{idxUE+1}.FAPIpdu{idxUePdu}.pdcchOutput.err{idxDci};
                                    rslt.totalCnt = rslt.totalCnt + 1;
                                    if errFlag
                                        rslt.errCnt = rslt.errCnt + 1;
                                    end
                                end
                            end
                            SimCtrl.results.pdcch{idxUE+1} = rslt;
                        end
                        idxPdcchUeFind = [idxPdcchUeFind, idxUE];
                    end
                            
%                     numDci = UE{idxUE+1}.FAPIpdu{idxUePdu}.numDlDci;
%                     for idxDci = 1:numDci
%                         payload_tx = gNB.FAPIpdu{idxPdu}.DCI{idxDci}.Payload;
%                         payload_rx = UE{idxUE+1}.FAPIpdu{idxUePduFind}.pdcchOutput.DCI{idxDci};
%                         errFlag = sum(abs(payload_tx(:) - payload_rx(:)));
%                         if SimCtrl.capSamp.enable
%                             errFlag = UE{idxUE+1}.FAPIpdu{idxUePduFind}.pdcchOutput.err{idxDci};
%                         end
%                         rslt = SimCtrl.results.pdcch{idxUE+1};
%                         rslt.totalCnt = rslt.totalCnt + 1;
%                         if errFlag
%                             rslt.errCnt = rslt.errCnt + 1;
%                         end
%                     end
%                     SimCtrl.results.pdcch{idxUE+1} = rslt;
                end
            end
        case 'pdsch'
            if SimCtrl.enableUeRx
                for idxUE = gNB.FAPIpdu{idxPdu}.idxUE
                    UePdu = UE{idxUE+1}.FAPIpdu;
                    findPdu = 0;
                    for idxUePdu = 1:length(UePdu)
                        if strcmp(UePdu{idxUePdu}.type, 'pdsch')
                            findPdu = findPdu + 1;
                            idxUePduFind = idxUePdu;
                        end
                    end
                    if findPdu == 0
                        error('No PDSCH PDU is found...\n');
                    end
                    rslt = SimCtrl.results.pdsch{idxUE+1};
                    tbErr = UE{idxUE+1}.FAPIpdu{idxUePduFind}.pdschOutput.tbErr;
                    cbErr = UE{idxUE+1}.FAPIpdu{idxUePduFind}.pdschOutput.cbErr;                    
                    rslt.tbCnt = rslt.tbCnt + length(tbErr);
                    rslt.tbErrorCnt = rslt.tbErrorCnt + sum(tbErr);
                    rslt.cbCnt = rslt.cbCnt + length(cbErr);
                    rslt.cbErrorCnt = rslt.cbErrorCnt + sum(cbErr);
                    SimCtrl.results.pdsch{idxUE+1} = rslt;
                end
            end
        case 'csirs'
            if SimCtrl.enableUeRx
                for idxUE = gNB.FAPIpdu{idxPdu}.idxUE
                    UePdu = UE{idxUE+1}.FAPIpdu;
                    findPdu = 0;
                    for idxUePdu = 1:length(UePdu)
                        if strcmp(UePdu{idxUePdu}.type, 'csirs')
                            findPdu = findPdu + 1;
                            idxUePduFind = idxUePdu;
                        end
                    end
                    if findPdu == 0
                        error('No CSIRS PDU is found...\n');
                    end
                    rslt = SimCtrl.results.csirs{idxUE+1};
                    Hest = UE{idxUE+1}.FAPIpdu{idxUePduFind}.csirsOutput.Hest;
                    if ~isempty(Hest)
                        Htrue = Chan_DL{idxUE+1}.chanMatrix;
                        [~, nPort, ~] = size(Hest);
                        nPort = min([nPort, size(Htrue,1)]);
                        Hest = Hest(1,1:nPort,:);
                        Hest = Hest(:);
                        Htrue = Htrue(1:nPort, :);
                        Htrue = Htrue(:);
                        errDB = 10*log10(mean(abs(Hest-Htrue).^2)/mean(abs(Htrue).^2));
                        rslt.totalCnt = rslt.totalCnt + 1;
                        if errDB > -30
                            rslt.errCnt = rslt.errCnt + 1;
                        end
                        SimCtrl.results.csirs{idxUE+1} = rslt;
                    end
                end
            end
        case 'prach'
            gNBprach = gNB.Mac.Config.prach{idxPrachPdu};
            if strcmp(Chan_UL{1}.type, 'AWGN')
%                 if strcmp(gNBprach.preambleFormat, '0')
                if gNBprach.L_RA == 839
                    timingErrorThreshold = 1.04e-6;
                elseif carrier.delta_f == 15e3
                    timingErrorThreshold = 0.52e-6;
                else
                    timingErrorThreshold = 0.26e-6;
                end
            else
%                 if strcmp(gNBprach.preambleFormat, '0')
                if gNBprach.L_RA == 839
                    timingErrorThreshold = 2.55e-6;
                elseif carrier.delta_f == 15e3
                    timingErrorThreshold = 2.03e-6;
                else
                    timingErrorThreshold = 1.77e-6;
                end
            end
            
            raSubframeNum = gNBprach.subframeNum;
            raSlotNum = gNBprach.n_slot_RA_sel;
            isRaSlot = ismember(idxSlot-1, raSlotNum);
            N_slot_subframe = carrier.N_slot_subframe_mu;
            idxSlotInFrame = mod(gNB.Mac.Config.carrier.idxSlotInFrame-1, carrier.N_slot_frame_mu);
            slotSpan = findPrachSlotSpan(gNBprach.preambleFormat,carrier.mu);
            isLastPrachSlot = ismember(idxSlotInFrame, raSlotNum + raSubframeNum * N_slot_subframe + [slotSpan-1]);
            
            if gNBprach.allSubframes % may apply for format 0 and B4, but not format 1
                if (slotSpan == 1)
                    if isRaSlot
                        isLastPrachSlot = 1;
                    else
                        isLastPrachSlot = 0;
                    end
                elseif (slotSpan == 2)
                    if isRaSlot
                        isLastPrachSlot = 0;
                    else
                        isLastPrachSlot = 1;
                    end
                elseif (slotSpan > 2)
                    error('prach.allSubframes = 1 is not supported for this config ... \n');
                end
            end
            
            if SimCtrl.genTV.forceSlotIdxFlag
                isLastPrachSlot = 1;
            end

            if (isLastPrachSlot)
                rxPrach = gNB.FAPIpdu{idxPdu}.payload;
                if SimCtrl.capSamp.enable
                    if rxPrach.detIdx > 0
                        fprintf('\n==> nPrmb = %d, prmbID = %d, peak = %5.3f, delay = %5.3f us, rssi = %5.3f dB\n', ...
                            rxPrach.detIdx, rxPrach.prmbIdx_det, rxPrach.peak_det, rxPrach.delay_time_det * 1e6, rxPrach.rssi_det);
                    else
                        fprintf('\n==> No preamble detetced\n');
                    end
                end
                rslt = SimCtrl.results.prach{idxPrachPdu};
                rslt.totalCnt = rslt.totalCnt + 1;
                if SimCtrl.prachFalseAlarmTest
                    if rxPrach.detIdx > 0
                        rslt.falseCnt = rslt.falseCnt + 1;
                    end
                else
                    if rxPrach.detIdx < 1
                        rslt.missCnt = rslt.missCnt + 1;
                    else
                        txPrmbIdx = [];
                        for idxUe = 1:length(UE)
                            txPrmbIdx = [txPrmbIdx, UE{idxUe}.Mac.Config.prach{1}.prmbIdx];
                        end
                        prmbFound = 0;
                        for n = 1:length(rxPrach.prmbIdx_det)
                            txPrmbFound = 0;
                            for ii = 1:length(txPrmbIdx)
                                if rxPrach.prmbIdx_det(n) == txPrmbIdx(ii)
                                    txPrmbFound = 1;
                                    prmbFound = prmbFound + 1;
                                    break;
                                end
                            end
                            if txPrmbFound
                                timingError = rxPrach.delay_time_det(n) - Chan_UL{ii}.delay;
                                if abs(timingError) > timingErrorThreshold
                                    rslt.missCnt = rslt.missCnt + 1;
                                end
                            % else
                            %    rslt.falseCnt = rslt.falseCnt + 1;
                            end
                        end
                        rslt.prmbCnt = rslt.prmbCnt + prmbFound;
                        if prmbFound == 0
                            rslt.missCnt = rslt.missCnt + 1;
                        end
                    end
                end
                SimCtrl.results.prach{idxPrachPdu} = rslt;
            end
            idxPrachPdu = idxPrachPdu + 1;
        case 'srs'
            srsIdx = gNB.FAPIpdu{idxPdu}.srsPduIdx;
            wideSnr = gNB.FAPIpdu{idxPdu}.srsOutput.widebandSnr;
            toEstMicroSec = gNB.FAPIpdu{idxPdu}.srsOutput.toEstMicroSec;
            hestErr = gNB.FAPIpdu{idxPdu}.srsOutput.hestErr;
            rslt = SimCtrl.results.srs{srsIdx};
            rslt.totalCnt = rslt.totalCnt + 1;
            rslt.snrErr = (wideSnr - Chan_UL{1}.SNR - 20*log10(Chan_UL{1}.gain))^2 + rslt.snrErr;
            rslt.toErr = (toEstMicroSec - Chan_UL{1}.delay*1e6)^2 + rslt.toErr;
            rslt.hestErr = hestErr + rslt.hestErr;
            SimCtrl.results.srs{srsIdx} = rslt;
        case 'bfw'
            bfwIdx           = gNB.FAPIpdu{idxPdu}.bfwPduIdx;
            minBfSinr = computeBfSinr(gNB.FAPIpdu{idxPdu}, gNB.Phy.Config.bfw.bfwBuf, SysPar.chan_BF, -Chan_UL{1}.SNR, idxPdu, SysPar.srsChEstBuff);
            rslt.minBfSinr              = minBfSinr;
            SimCtrl.results.bfw{bfwIdx} = rslt;
        case 'pucch'
                payload_rx = gNB.FAPIpdu{idxPdu}.payload;
                pucchPduIdx =  gNB.FAPIpdu{idxPdu}.pucchPduIdx;
                idxUE = gNB.FAPIpdu{idxPdu}.idxUE;
                rslt = SimCtrl.results.pucch{pucchPduIdx};
                UePdu = UE{idxUE+1}.FAPIpdu;
                findPdu = 0;
                for idxUePdu = 1:length(UePdu)
                    if strcmp(UePdu{idxUePdu}.type, 'pucch')
                        findPdu = findPdu + 1;
                        idxUePduFind = idxUePdu;
                    end
                end
                if findPdu == 0
                    error('No PUCCH PDU is found...\n');
                elseif findPdu > 1
                    % special case for PUCCH performance test which allows one UE
                    % to transmit multiple UCIs to save simulation time
                    idxUePduFind = idxPdu;
                end
                FormatType = gNB.FAPIpdu{idxPdu}.FormatType;
                DTX = UePdu{idxUePduFind}.DTX;
                rslt.detStatError = 0;
                if FormatType == 1 || FormatType == 0
                    payload_tx = UePdu{idxUePduFind}.payload;
                    nPayload = length(payload_tx);
                    rslt.totalCnt = rslt.totalCnt + nPayload;
                    for idx = 1:nPayload
                        if DTX
                            if payload_rx(idx) == 0
                                rslt.falseCnt = rslt.falseCnt + 1;
                            end
                        else
                            if payload_tx(idx) == 0 && payload_rx(idx) == 2
                                rslt.missack = rslt.missack + 1;
                            elseif payload_tx(idx) == 1 && payload_rx(idx) == 0
                                rslt.nack2ack = rslt.nack2ack + 1;
                            elseif payload_tx(idx) == 0 && payload_rx(idx) == 1
                                rslt.missack = rslt.missack + 1;
                            end
                        end
                    end
                    rslt.errorCnt = rslt.nack2ack + rslt.missack + rslt.falseCnt;
                    
                    if payload_rx(1) ~= 2
                       rslt.taEstMicroSec = [rslt.taEstMicroSec, gNB.FAPIpdu{idxPdu}.taEstMicroSec]; 
                    end
                elseif FormatType == 2 || FormatType == 3
                    payload_tx = UePdu{idxUePduFind}.payloadSeq1;
                    nPayload = length(payload_tx);
                    rslt.totalUciCnt = rslt.totalUciCnt + 1;
                    errFlag = 0;
                    for idx = 1:nPayload
                        if payload_tx(idx) ~= payload_rx(idx)
                            errFlag = 1;
                        end
                    end
                    rslt.errorCnt = rslt.errorCnt + errFlag;
                    
                    BitLenHarq     = UePdu{idxUePduFind}.BitLenHarq;
                    rslt.totalCnt = rslt.totalCnt + BitLenHarq;

                    HarqDetectionStatus = gNB.FAPIpdu{idxPdu}.HarqDetectionStatus;
                    for idx = 1:BitLenHarq
                        if DTX
                            if payload_rx(idx) == 0 && (HarqDetectionStatus == 1 || HarqDetectionStatus == 4)
                                rslt.falseCnt = rslt.falseCnt + 1;
                            end
                        else
                            if HarqDetectionStatus == 2 || HarqDetectionStatus == 3
                                rslt.missack = rslt.missack + 1;
                            else
                                if payload_tx(idx) == 0
                                    if payload_rx(idx) ~= 0
                                        rslt.missack = rslt.missack + 1;
                                    end
                                end
                            end
                        end
                    end
                    
                    %% determine detectionStatus error
                    CsiPart1DetectionStatus = gNB.FAPIpdu{idxPdu}.CsiPart1DetectionStatus;
                    CsiPart2DetectionStatus = gNB.FAPIpdu{idxPdu}.CsiPart2DetectionStatus;
                    
                    % CSI part 2
                    nPayloadCsiP2 = UePdu{idxUePduFind}.BitLenCsiPart2;
                    if nPayloadCsiP2 >= 12
                        if DTX
                            if CsiPart2DetectionStatus ~= 2
                                rslt.detStatError = 1;
                            end
                        else
                            if CsiPart2DetectionStatus ~= 1
                                rslt.detStatError = 1;
                            end
                        end
                    else
                        if nPayload < 12
                            if DTX
                                if CsiPart2DetectionStatus ~= 3
                                    rslt.detStatError = 1;
                                end
                            else
                                if CsiPart2DetectionStatus ~= 4
                                    rslt.detStatError = 1;
                                end
                            end
                        else
                            if CsiPart2DetectionStatus ~= 5
                                rslt.detStatError = 1;
                            end
                        end
                    end
                    
                    % HARQ & CSI part 1
                    
                    BitLenCsiPart1 = UePdu{idxUePduFind}.BitLenCsiPart1;
                    if nPayload < 12
                        if DTX
                            if HarqDetectionStatus ~= 3 || CsiPart1DetectionStatus ~= 3
                                rslt.detStatError = 1;
                            end
                        else
                            if HarqDetectionStatus ~= 4 || CsiPart1DetectionStatus ~= 4
                                rslt.detStatError = 1;
                            end
                        end
                        
                        if HarqDetectionStatus ~= 3 && CsiPart1DetectionStatus ~= 3
                            rslt.taEstMicroSec = [rslt.taEstMicroSec, gNB.FAPIpdu{idxPdu}.taEstMicroSec];
                        end
                    else
                        if DTX
                            if HarqDetectionStatus ~= 2 || CsiPart1DetectionStatus ~= 2
                                rslt.detStatError = 1;
                            end
                        else
                            if BitLenHarq > 0
                                if HarqDetectionStatus ~= 1
                                    rslt.detStatError = 1;
                                end
                            else
                                if HarqDetectionStatus ~= 2
                                    rslt.detStatError = 1;
                                end 
                            end
                            
                            if BitLenCsiPart1 > 0
                                if CsiPart1DetectionStatus ~= 1
                                    rslt.detStatError = 1;
                                end
                            else
                                if CsiPart1DetectionStatus ~= 2
                                    rslt.detStatError = 1;
                                end
                            end
                        end
                        
                        if HarqDetectionStatus ~= 2 || CsiPart1DetectionStatus ~= 2
                            rslt.taEstMicroSec = [rslt.taEstMicroSec, gNB.FAPIpdu{idxPdu}.taEstMicroSec];
                        end
                    end
                end
                
                %% determine TA estimation error
%                 taErrorThr = 0;
%                 realSNR = SysPar.Chan{1}.SNR + 20*log10(SysPar.Chan{1}.gain);
%                 
%                 switch FormatType
%                     case 1
%                         if realSNR <= 0
%                             taErrorThr = 4.9;
%                         elseif realSNR >0 && realSNR <= 3
%                             taErrorThr = 0.3;
%                         else
%                             taErrorThr = 0.2;
%                         end
%                     case 2
%                         if realSNR <= 0
%                             taErrorThr = 0.6;
%                         elseif realSNR > 0 && realSNR<= 3
%                             taErrorThr = 0.5;
%                         else
%                             taErrorThr = 0.2;
%                         end
%                     case 3
%                         if realSNR <= 0
%                             taErrorThr = 0.6;
%                         elseif realSNR > 0 && realSNR <= 3
%                             taErrorThr = 0.3;
%                         else
%                             taErrorThr = 0.2;
%                         end
%                 end
%                 
%                 if FormatType ~= 0
%                     if N_pdu > 1 && FormatType == 1 %% TA estimation for PF1 UCI multiplexing is not supported yet
%                         rslt.taError = 0;
%                         rslt.taDiff = 0;
%                     else
%                         taDiff = abs(gNB.FAPIpdu{idxPdu}.taEstMicroSec - Chan.delay*1e6);
%                         rslt.taDiff = taDiff; % difference in microsecond
%                         if taDiff > taErrorThr
%                             rslt.taError = 1;
%                             rslt.errorCnt = rslt.errorCnt + 1;
%                         else
%                             rslt.taError = 0;
%                         end
%                     end
%                 else
%                     rslt.taError = 0;
%                     rslt.taDiff = 0;
%                 end
                rslt.snrdB = [rslt.snrdB, gNB.FAPIpdu{idxPdu}.snrdB];
                SimCtrl.results.pucch{pucchPduIdx} = rslt;
        case 'pusch'
            tbErr = gNB.FAPIpdu{idxPdu}.tbErr;
            cbErr = gNB.FAPIpdu{idxPdu}.cbErr;
            cfoEstHz = gNB.FAPIpdu{idxPdu}.cfoEstHz;
            toEstMicroSec = gNB.FAPIpdu{idxPdu}.toEstMicroSec;
            sinrdB = gNB.FAPIpdu{idxPdu}.sinrdB;
            postEqSinrdB = gNB.FAPIpdu{idxPdu}.postEqSinrdB;
            puschIdx = gNB.FAPIpdu{idxPdu}.puschPduIdx;
            rslt = SimCtrl.results.pusch{puschIdx};
            rslt.tbCnt = rslt.tbCnt + length(tbErr);
            rslt.tbErrorCnt = rslt.tbErrorCnt + sum(tbErr);
            rslt.cbCnt = rslt.cbCnt + length(cbErr);
            rslt.cbErrorCnt = rslt.cbErrorCnt + sum(cbErr);
            rslt.cfoEstHz = [rslt.cfoEstHz, cfoEstHz];
            rslt.toEstMicroSec = [rslt.toEstMicroSec, toEstMicroSec];
            rslt.sinrdB = [rslt.sinrdB, sinrdB];
            rslt.postEqSinrdB = [rslt.postEqSinrdB, postEqSinrdB];
            idxUE = gNB.FAPIpdu{idxPdu}.idxUE;
            % check UCI on PUSCH
            UePdu = UE{idxUE+1}.FAPIpdu;
            findPdu = 0;
            for idxUePdu = 1:length(UePdu)
                if strcmp(UePdu{idxUePdu}.type, 'pusch')
                    findPdu = findPdu + 1;
                    idxUePduFind = idxUePdu;
                end
            end            
            if findPdu == 0
                error('No PUSCH PDU is found...\n');
            end            
            if SimCtrl.capSamp.enable == 0 % bypass UCI check for captured data analysis
                if UePdu{idxUePduFind}.harqAckBitLength
                    if sum(abs(UePdu{idxUePduFind}.harqPayload(:) - gNB.FAPIpdu{idxPdu}.harqUci(:)))
                        rslt.harqErrCnt = rslt.harqErrCnt + 1;
                    end
                end
                if UePdu{idxUePduFind}.csiPart1BitLength
                    if sum(abs(UePdu{idxUePduFind}.csiPart1Payload(:) - gNB.FAPIpdu{idxPdu}.csi1Uci(:)))
                        rslt.csi1ErrCnt = rslt.csi1ErrCnt + 1;
                    end
                end
                if UePdu{idxUePduFind}.csiPart2BitLength
                    if length(UePdu{idxUePduFind}.csiPart2Payload(:)) ~= length(gNB.FAPIpdu{idxPdu}.csi2Uci(:))
                        rslt.csi2ErrCnt = rslt.csi2ErrCnt + 1;
                    elseif sum(abs(UePdu{idxUePduFind}.csiPart2Payload(:) - gNB.FAPIpdu{idxPdu}.csi2Uci(:)))
                        rslt.csi2ErrCnt = rslt.csi2ErrCnt + 1;
                    end
                end
            end
            SimCtrl.results.pusch{puschIdx} = rslt;
        otherwise
            %             error('alloc type is not supported ... \n');
    end
end

return

