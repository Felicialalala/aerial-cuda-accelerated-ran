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

function [pdcch_payload] = detPdcch(pduList, table, carrier, Xtf)

nPdu = length(pduList);
PdcchParamsList = [];

for idxPdu = 1:nPdu
    pdu = pduList{idxPdu};
    % load Coreset parameters
    PdcchParamsList(idxPdu).slotNumber = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
    PdcchParamsList(idxPdu).start_rb = pdu.BWPStart;      % pdcch tx starting RB (0 indexing)
    PdcchParamsList(idxPdu).start_sym = pdu.StartSymbolIndex;    % starting symbol pdcch tx (0 indexing)
    PdcchParamsList(idxPdu).n_sym = pdu.DurationSymbols;            % number of pdcch tx symbols (1-3)
    PdcchParamsList(idxPdu).bundleSize = pdu.RegBundleSize;
    PdcchParamsList(idxPdu).interleaveSize = pdu.InterleaverSize;
    PdcchParamsList(idxPdu).shiftIdx = pdu.ShiftIndex ;
    PdcchParamsList(idxPdu).interleaved = pdu.CceRegMappingType;
    PdcchParamsList(idxPdu).FreqDomainResource0 = pdu.FreqDomainResource0;
    PdcchParamsList(idxPdu).FreqDomainResource1 = pdu.FreqDomainResource1;
    PdcchParamsList(idxPdu).numDlDci = pdu.numDlDci;
    PdcchParamsList(idxPdu).testModel = pdu.testModel;
    PdcchParamsList(idxPdu).CoreSetType = pdu.CoreSetType;
    % load DCI parameters
    DciParams = [];
    for idxDCI = 1:pdu.numDlDci
        DCI = pdu.DCI{idxDCI};
        Npayload = DCI.PayloadSizeBits;
%         pdcch_payload = DCI.Payload;
        DciParams{idxDCI}.Npayload = Npayload;
%         DciParams{idxDCI}.Payload = pdcch_payload';
        DciParams{idxDCI}.rntiCrc = DCI.RNTI;
        DciParams{idxDCI}.rntiBits = DCI.ScramblingRNTI;
        DciParams{idxDCI}.dmrsId = DCI.ScramblingId;
        DciParams{idxDCI}.aggrL = DCI.AggregationLevel;
        DciParams{idxDCI}.cceIdx = DCI.CceIndex;
%         qam_dB = (DCI.powerControlOffsetSS - 1) * 3;
        qam_dB = DCI.powerControlOffsetSSProfileNR;
        DciParams{idxDCI}.beta_qam = 10^(qam_dB/20);
%         dmrs_dB = (DCI.powerControlOffsetSS - 1) * 3;
        dmrs_dB = DCI.powerControlOffsetSSProfileNR;
        DciParams{idxDCI}.beta_dmrs = 10^(dmrs_dB/20);
        DciParams{idxDCI}.enablePrcdBf = DCI.enablePrcdBf;
        DciParams{idxDCI}.PM_W = DCI.PM_W;
    end
    PdcchParamsList(idxPdu).DciParams = DciParams;
end

[pdcch_payload] = detPdcch_cuphy(PdcchParamsList, Xtf);

global SimCtrl;
idxSlot = carrier.idxSlotInFrame;
idxPdu = pdu.pdcchPduIdx-1;
if SimCtrl.genTV.enableUE && SimCtrl.genTV.enable && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)
    TVname = [SimCtrl.genTV.TVname, '_PDCCH_UE_CUPHY_s', num2str(idxSlot),...
        'p', num2str(idxPdu)];
    saveTV_pdcch_multiDci(SimCtrl.genTV.tvDirName, TVname, PdcchParamsList, Xtf, pdcch_payload);
    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

return


function [pdcch_payload] = detPdcch_cuphy(PdcchParamsList, Xtf)

nAnt = size(Xtf, 3);
nPdu = length(PdcchParamsList);

for idxPdu = 1:nPdu
    % load parameters
    PdcchParams = PdcchParamsList(idxPdu);
    slotNumber = PdcchParams.slotNumber;
    startRb = PdcchParams.start_rb;      % pdcch tx starting RB (0 indexing)
    startSym = PdcchParams.start_sym;    % starting symbol pdcch tx (0 indexing)
    nSym = PdcchParams.n_sym;            % number of pdcch tx symbols (1-3)
    bundleSize = PdcchParams.bundleSize;
    interleaveSize = PdcchParams.interleaveSize;
    shiftIdx = PdcchParams.shiftIdx;
    interleaved = PdcchParams.interleaved;
    FreqDomainResource0 = dec2bin(PdcchParams.FreqDomainResource0, 32)-'0';
    FreqDomainResource1 = dec2bin(PdcchParams.FreqDomainResource1, 32)-'0';
    FreqDomainResource = [FreqDomainResource0, FreqDomainResource1];
    lenCoresetMap = find(FreqDomainResource, 1, 'last');
    coresetMap = FreqDomainResource(1:lenCoresetMap);
    numDlDci = PdcchParams.numDlDci;
    testModel = PdcchParams.testModel;
    CoreSetType = PdcchParams.CoreSetType;

    N_CCE = sum(coresetMap)*nSym;
    N_REG =  N_CCE*6;
    C = N_REG/(bundleSize*interleaveSize);
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
        DCI = PdcchParams.DciParams{idxDCI};
        Npayload = DCI.Npayload;
%         Payload = DCI.Payload(1:Npayload);
        rntiCrc = DCI.rntiCrc;
        rntiBits = DCI.rntiBits;
        dmrsId = DCI.dmrsId;
        aggrL = DCI.aggrL;
        cceIdx = DCI.cceIdx;
        beta_qam = DCI.beta_qam;
        beta_dmrs = DCI.beta_dmrs;
        enablePrcdBf = DCI.enablePrcdBf;
        PM_W = DCI.PM_W;

        % mark used bundle based on cce index and aggrL
        usedBundleMap = zeros(1, N_bundle_phy);
        for usedCceIdx = cceIdx:cceIdx+aggrL-1
            usedBundleIdx = 6*usedCceIdx/bundleSize:6*(usedCceIdx+1)/bundleSize-1;
            logBundleIdx = bundleMap(usedBundleIdx+1)+1;
            usedBundleMap(bundleTable(logBundleIdx)) = 1;
        end

        % DCI channel coding and modutlation
%         x = genPdcchQpsk(Payload, Npayload, rntiBits, rntiCrc, dmrsId, aggrL);
        qpsk_ptr = 0;
        x_allAnt = [];
        % Map to RE
        for i = 0 : (nSym - 1)
            % compute seed:
            t = startSym + i;
            c_init =  mod(2^17*(14*slotNumber+t+1)*(2*dmrsId+1)+2*dmrsId,2^31);

            % compute Gold sequence:
            c = build_Gold_sequence(c_init,2*3*endRb);
            if CoreSetType > 0
                c = c(startRb*6+1:end);
            end
            for bundleIdx = 0:N_bundle_phy-1
                if usedBundleMap(bundleIdx+1)
                    for idxAnt = 1:nAnt
                        % estimate channel from DMRS
                        dmrs_bit = c(2*bundleIdx*dmrsPerBundle+1:2*(bundleIdx+1)*dmrsPerBundle);
                        r = qpsk_modulate(dmrs_bit, 2*dmrsPerBundle);
                        dmrs_rx = Xtf(startFreq+bundleIdx*rbPerBundle*12+2: 4: startFreq+(bundleIdx+1)*rbPerBundle*12, startSym+i+1, idxAnt);
                        dmrs_ch = dmrs_rx(:) .* conj(r(:));
                        dmrs_ch = reshape(dmrs_ch, [3, length(dmrs_ch)/3]);
                        dmrs_ch = mean(dmrs_ch, 1);
                        dmrs_ch = repmat(dmrs_ch, 9, 1);
                        % channel equalization on data RE
                        qam_rx = Xtf(startFreq+bundleIdx*rbPerBundle*12+qamLoc+1, startSym+i+1, idxAnt);
                        qam_rx = reshape(qam_rx, [9, length(qam_rx)/9]);
                        qam_eq = qam_rx .* conj(dmrs_ch);
                        x_allAnt(qpsk_ptr+1:qpsk_ptr+N_qpskPerBundle, idxAnt) = qam_eq(:);
                    end
                    qpsk_ptr = qpsk_ptr + N_qpskPerBundle;
                end
            end
        end
        x = mean(x_allAnt, 2);

        % TBD: Need to replace 5GToolbox functions nrPDCCHDecode and nrDCIDecode
        dcicw = nrPDCCHDecode(x(:),dmrsId,rntiBits);
        if testModel
            dcibits = (1-sign(dcicw))/2;
            mask = [];
        else
            L = 8;
            [dcibits,mask] = nrDCIDecode(dcicw,Npayload,L,rntiCrc);
        end
        pdcch_payload{idxPdu}.DCI{idxDCI} = double(dcibits)';
        pdcch_payload{idxPdu}.err{idxDCI} = double(mask);
    end
end

return



function saveTV_pdcch_multiDci(tvDirName, TVname, PdcchParamsList, Xtf, pdcch_payload)

[status,msg] = mkdir(tvDirName);
h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

nPdu = length(PdcchParamsList);
[n_f, n_t, ~] = size(Xtf);

for idxPdu = 1:nPdu
    PdcchParams = PdcchParamsList(idxPdu);
    PdcchParams.n_f = n_f;
    PdcchParams.n_t = n_t;
    DCI = PdcchParams.DciParams;
    nDCI = length(DCI);
    singleField = {'beta_dmrs', 'beta_qam'};
    for n = 1:nDCI
        thisDCI = DCI{n};
        PM_W = thisDCI.PM_W;
        thisDCI = rmfield(thisDCI, 'PM_W');
        Payload = pdcch_payload{idxPdu}.DCI{n};
        thisDCI = formatU32Struct(thisDCI, singleField);
        dateSetName = ['DciParams_coreset_', num2str(idxPdu-1), '_dci_', num2str(n-1)];
        hdf5_write_nv(h5File, dateSetName, thisDCI);
        dateSetName = ['DciPayload_coreset_', num2str(idxPdu-1), '_dci_', num2str(n-1)];
        hdf5_write_nv(h5File, dateSetName, uint8(uint8_convert(Payload, 0)));
        dateSetName = ['DciPmW_coreset_', num2str(idxPdu-1), '_dci_', num2str(n-1)];
        hdf5_write_nv(h5File, dateSetName, single(PM_W));
    end
    PdcchParams = rmfield(PdcchParams, 'DciParams');
    PdcchParamsNew(idxPdu) = formatU32Struct(PdcchParams);
end

dateSetName = 'PdcchParams';
hdf5_write_nv(h5File, dateSetName, PdcchParamsNew);

global SimCtrl
bypassComp = SimCtrl.genTV.bypassComp;
if ~bypassComp
    [cSamples_uint8, X_tf_fp16] = oranCompress(Xtf, 0); % generate ORAN compressed samples
    hdf5_write_nv(h5File, 'X_tf_fp16', X_tf_fp16, 'fp16');
    for k=1:length(SimCtrl.oranComp.iqWidth)
        hdf5_write_nv(h5File, ['X_tf_cSamples_bfp',num2str(SimCtrl.oranComp.iqWidth(k))], uint8(cSamples_uint8{k}));
    end
end

H5F.close(h5File);

return
