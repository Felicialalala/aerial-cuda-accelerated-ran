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

function [Xtf, Xtf_remap, Xtf_remap_trsnzp] = genCsirs(pduList, table, carrier, Xtf)

global SimCtrl;
nPort_enable_csirs_compression = SimCtrl.nPort_enable_csirs_compression;
[row2nPort, ~] = getCsirsConfig();

[nSc, nSym, Nant] = size(Xtf);
Xtf_remap = zeros(nSc, nSym);
Xtf_remap_trsnzp = zeros(nSc, nSym);
[Xtf_remap, Xtf_remap_trsnzp] = gen_csirs_remap(pduList, Xtf_remap, Xtf_remap_trsnzp, table);

csirsTable = table;
nPdu = length(pduList);
CsirsParamsList = [];
idxNZPpdu = 0;

for idxPdu = 1:nPdu
    pdu = pduList{idxPdu};
    % If the number of antennas is less than the number of interfaces in the pdu, expand the Xtf
    if (nPort_enable_csirs_compression > pdu.digBFInterfaces) && (pdu.digBFInterfaces > Nant)
        Xtf_expand = zeros(nSc, nSym, pdu.digBFInterfaces);
        Xtf_expand(:,:,1:Nant) = Xtf;
        Nant = pdu.digBFInterfaces;
        Xtf = Xtf_expand;
    end
    if pdu.CSIType < 2 % only pass TRS/NZP-CSIRS params to cuPHY
        idxNZPpdu = idxNZPpdu + 1;
        % load parameters
        CsirsParamsList(idxNZPpdu).StartRB = pdu.StartRB;
        CsirsParamsList(idxNZPpdu).NrOfRBs = pdu.NrOfRBs;
        CsirsParamsList(idxNZPpdu).CSIType = pdu.CSIType;
        CsirsParamsList(idxNZPpdu).Row = pdu.Row;
        CsirsParamsList(idxNZPpdu).FreqDomain = pdu.FreqDomain;
        CsirsParamsList(idxNZPpdu).SymbL0 = pdu.SymbL0;
        CsirsParamsList(idxNZPpdu).SymbL1 = pdu.SymbL1;
        CsirsParamsList(idxNZPpdu).CDMType = pdu.CDMType;
        CsirsParamsList(idxNZPpdu).FreqDensity = pdu.FreqDensity;
        CsirsParamsList(idxNZPpdu).ScrambId = pdu.ScrambId;
        CsirsParamsList(idxNZPpdu).enablePrcdBf = pdu.enablePrcdBf;
        CsirsParamsList(idxNZPpdu).PM_W = pdu.PM_W;
        beta_db = (pdu.powerControlOffsetSS - 1)*3;
        CsirsParamsList(idxNZPpdu).beta = 10^(beta_db/20);
        CsirsParamsList(idxNZPpdu).idxSlotInFrame = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
        CsirsParamsList(idxNZPpdu).N_symb_slot = carrier.N_symb_slot;
        nPort = row2nPort(pdu.Row);
        if nPort >= nPort_enable_csirs_compression
            CsirsParamsList(idxNZPpdu).enable_csirs_compression = 1;
        else
            CsirsParamsList(idxNZPpdu).enable_csirs_compression = 0;
        end
    end
end

if idxNZPpdu > 0
    Xtf0 = Xtf; % for generating test vectors
    
%     [Xtf, Xtf_remap] = genCsirs_cuphy(Xtf, CsirsParamsList, csirsTable, Xtf_remap);
    Xtf = genCsirs_cuphy(Xtf, CsirsParamsList, csirsTable, Xtf_remap);
    
    Xtf1 = Xtf - Xtf0; % for generating test vectors
    
    global SimCtrl;
    idxSlot = carrier.idxSlotInFrame;
    idxPdu = idxNZPpdu-1;
    
    if SimCtrl.genTV.enable && SimCtrl.genTV.cuPHY && ...
            ismember(idxSlot,SimCtrl.genTV.slotIdx)
        TVname = [SimCtrl.genTV.TVname, '_CSIRS_gNB_CUPHY_s', num2str(idxSlot),...
            'p', num2str(idxPdu)];
        saveTV_csirs(SimCtrl.genTV.tvDirName, TVname, CsirsParamsList, Xtf1, Xtf_remap);
        SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
    end
end

return


function [Xtf, Xtf_remap] = genCsirs_cuphy(Xtf, CsirsParamsList, csirsTable, Xtf_remap)

nPdu = length(CsirsParamsList);
[~, ~, nAnt] = size(Xtf);

for idxPdu = 1:nPdu

    % load parameters
    CsirsParams = CsirsParamsList(idxPdu);
    StartRB = CsirsParams.StartRB;
    NrOfRBs = CsirsParams.NrOfRBs;
    CSIType = CsirsParams.CSIType;
    Row = CsirsParams.Row;
    FreqDomain = CsirsParams.FreqDomain;
    SymbL0 = CsirsParams.SymbL0;
    SymbL1 = CsirsParams.SymbL1;
    CDMType = CsirsParams.CDMType;
    FreqDensity = CsirsParams.FreqDensity;
    ScrambId = CsirsParams.ScrambId;
    enablePrcdBf = CsirsParams.enablePrcdBf;
    PM_W = CsirsParams.PM_W;
    beta = CsirsParams.beta;
    idxSlotInFrame = CsirsParams.idxSlotInFrame;
    N_symb_slot = CsirsParams.N_symb_slot;
    enable_csirs_compression = CsirsParams.enable_csirs_compression;

    SymbL = [SymbL0, SymbL1];

    % read table 7.4.1.5.3-1
    X = csirsTable.csirsLocTable.Ports{Row};
    KBarLBar = csirsTable.csirsLocTable.KBarLBar{Row};
    CDMGroupIndices = csirsTable.csirsLocTable.CDMGroupIndices{Row};
    KPrime = csirsTable.csirsLocTable.KPrime{Row};
    LPrime = csirsTable.csirsLocTable.LPrime{Row};

    % 0: noCDM, 1: fd-CDM2, 2: cdm4-FD2-TD2, 3: cdm8-FD2-TD4
    switch CDMType
        case 0
            seqTable = csirsTable.noCdmTable;
            LL = 1;
        case 1
            seqTable = csirsTable.fdCdm2Table;
            LL = 2;
        case 2
            seqTable = csirsTable.cdm4Table;
            LL = 4;
        case 3
            seqTable = csirsTable.cdm8Table;
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

    if X == 1
        alpha = rho;
    else
        alpha = 2*rho;
    end

    if CSIType == 2 % CSI-RS ZP
        beta  = 0;
    end

    FreqDomain_bin = dec2bin(FreqDomain, 12) - '0';
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

    for idxSym = 0:N_symb_slot-1
        c_init = mod((2^10*(N_symb_slot*idxSlotInFrame+idxSym+1)*(2*ScrambId+1)+...
            ScrambId), 2^31);
        c_seq(idxSym+1, :) = build_Gold_sequence(c_init,2*273*3)';
    end

    lenKBarLBar = length(KBarLBar);
    lenLPrime = length(LPrime);
    lenKPrime = length(KPrime);

    hasTwoSyms = ismember(Row, [13 14 16 17]);

    for idxRB = StartRB:StartRB + NrOfRBs - 1
        isEvenRB = (mod(idxRB, 2) == 0);
        if (rho == 0.5)
            if (genEvenRB && ~isEvenRB) || (~ genEvenRB && isEvenRB)
                continue;
            end
        end

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
                    k = k_bar + k_prime + idxRB*12;
                    l_prime = LPrime(idxLPrime);
                    ll = l_bar + l_prime;
                    m_prime = floor(idxRB*alpha) + k_prime + ...
                        floor(k_bar*rho/12);
                    Xtf_remap(k+1, ll+1) = 1;
                    if beta > 0 % NZP-CSI-RS
                        for s = 1:LL
                            wf = seqTable{s}{1}(k_prime+1);
                            wt = seqTable{s}{2}(l_prime+1);
                            r = sqrt(0.5)*(1-2*c_seq(ll+1, 2*m_prime+1))+j*sqrt(0.5)*(1-2*c_seq(ll+1,2*m_prime+2));
                            jj = CDMGroupIndices(idxKBarLBar);
                            p = jj*LL+s;
                            a = beta * wf * wt * r;
                            if enablePrcdBf
                                Xtf(k+1, ll+1, :) = Xtf(k+1, ll+1, :) + a * reshape(PM_W(:, p), 1, 1, nAnt);
                            else
                                Xtf(k+1, ll+1, mod(p-1, nAnt)+1) = a;
                            end
                        end
                    end
                end
            end
        end
    end
end

return


function saveTV_csirs(tvDirName, TVname, CsirsParamsList, Xtf, Xtf_remap)

[status,msg] = mkdir(tvDirName);

nPdu = length(CsirsParamsList);

NZPexist = 0;

for idxPdu = 1:nPdu
    CsirsParams(idxPdu).StartRB = uint32(CsirsParamsList(idxPdu).StartRB);
    CsirsParams(idxPdu).NrOfRBs = uint32(CsirsParamsList(idxPdu).NrOfRBs);
    CsirsParams(idxPdu).CSIType = uint32(CsirsParamsList(idxPdu).CSIType);
    if CsirsParamsList(idxPdu).CSIType < 2 % not ZP CSI-RS
        NZPexist = 1;
    end
    CsirsParams(idxPdu).Row = uint32(CsirsParamsList(idxPdu).Row);
    CsirsParams(idxPdu).FreqDomain = uint32(CsirsParamsList(idxPdu).FreqDomain);
    CsirsParams(idxPdu).SymbL0 = uint32(CsirsParamsList(idxPdu).SymbL0);
    CsirsParams(idxPdu).SymbL1 = uint32(CsirsParamsList(idxPdu).SymbL1);
    CsirsParams(idxPdu).CDMType = uint32(CsirsParamsList(idxPdu).CDMType);
    CsirsParams(idxPdu).FreqDensity = uint32(CsirsParamsList(idxPdu).FreqDensity);
    CsirsParams(idxPdu).ScrambId = uint32(CsirsParamsList(idxPdu).ScrambId);
    CsirsParams(idxPdu).beta = single(CsirsParamsList(idxPdu).beta);
    CsirsParams(idxPdu).idxSlotInFrame = uint32(CsirsParamsList(idxPdu).idxSlotInFrame);
    CsirsParams(idxPdu).N_symb_slot = uint32(CsirsParamsList(idxPdu).N_symb_slot);
    CsirsParams(idxPdu).enablePrcdBf = uint32(CsirsParamsList(idxPdu).enablePrcdBf);
    Csirs_PM_W{idxPdu} = single(CsirsParamsList(idxPdu).PM_W);
end

if NZPexist % at least one NZP CSI-RS or TRS exists
    h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

    hdf5_write_nv(h5File, 'nCsirs', uint32(nPdu));
    hdf5_write_nv(h5File, 'CsirsParamsList', CsirsParams);
    for idxPdu = 1:nPdu
        dataSetName = ['Csirs_PM_W', num2str(idxPdu-1)];
        hdf5_write_nv(h5File, dataSetName, complex(Csirs_PM_W{idxPdu}));
    end
    hdf5_write_nv(h5File, 'X_tf', single(Xtf));
    hdf5_write_nv(h5File, 'X_tf_remap', uint32(Xtf_remap));

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
end

return
