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

function SrsOutputList = detSrs(pduList, table, carrier, Xtf)

srsTable = table;

nPdu = length(pduList);
SrsParamsList = [];

for idxPdu = 1:nPdu
    pdu = pduList{idxPdu};
    numAntPorts_mapping = [1 2 4];
    SrsParamsList(idxPdu).N_ap_SRS = numAntPorts_mapping(pdu.numAntPorts+1);
    numSymbols_mapping = [1 2 4];
    SrsParamsList(idxPdu).N_symb_SRS = numSymbols_mapping(pdu.numSymbols+1);
    numRepetitions_mapping = [1 2 4];
    SrsParamsList(idxPdu).R = numRepetitions_mapping(pdu.numRepetitions+1);
    combSize_mapping = [2 4];
    SrsParamsList(idxPdu).K_TC = combSize_mapping(pdu.combSize+1);

    SrsParamsList(idxPdu).l0 = pdu.timeStartPosition;
    SrsParamsList(idxPdu).n_ID_SRS = pdu.sequenceId;
    SrsParamsList(idxPdu).C_SRS = pdu.configIndex;
    SrsParamsList(idxPdu).B_SRS = pdu.bandwidthIndex;
    SrsParamsList(idxPdu).k_TC_bar = pdu.combOffset;
    SrsParamsList(idxPdu).n_SRS_cs = pdu.cyclicShift;
    SrsParamsList(idxPdu).n_RRC = pdu.frequencyPosition;
    SrsParamsList(idxPdu).n_shift = pdu.frequencyShift;
    SrsParamsList(idxPdu).b_hop = pdu.frequencyHopping;
    SrsParamsList(idxPdu).resourceType = pdu.resourceType;
    SrsParamsList(idxPdu).Tsrs = pdu.Tsrs;
    SrsParamsList(idxPdu).Toffset = pdu.Toffset;
    SrsParamsList(idxPdu).groupOrSequenceHopping = pdu.groupOrSequenceHopping;

    SrsParamsList(idxPdu).N_slot_frame = carrier.N_slot_frame_mu;
    SrsParamsList(idxPdu).N_symb_slot = carrier.N_symb_slot;
    SrsParamsList(idxPdu).idxSlotInFrame = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
    SrsParamsList(idxPdu).idxFrame = carrier.idxFrame;
    SrsParamsList(idxPdu).delta_f = carrier.delta_f;
end

SrsOutputList = detSrs_cuphy(Xtf, SrsParamsList, srsTable);

global SimCtrl;
idxSlot = carrier.idxSlotInFrame;
idxPdu = pdu.srsPduIdx-1;

if SimCtrl.genTV.enable && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)
    TVname = [SimCtrl.genTV.TVname, '_SRS_gNB_CUPHY_s', num2str(idxSlot),...
        'p', num2str(idxPdu)];
    saveTV_srs(SimCtrl.genTV.tvDirName, TVname, SrsParamsList, Xtf, SrsOutputList);
    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

return;


function SrsOutputList = detSrs_cuphy(Xtf, SrsParamsList, srsTable)

[~, ~, nAnt] = size(Xtf);

nPdu = length(SrsParamsList);

for idxPdu = 1:nPdu

    % load parameters
    SrsParams = SrsParamsList(idxPdu);
    N_ap_SRS = SrsParams.N_ap_SRS;
    N_symb_SRS = SrsParams.N_symb_SRS;
    R = SrsParams.R;
    l0 = SrsParams.l0;
    n_ID_SRS = SrsParams.n_ID_SRS;
    C_SRS = SrsParams.C_SRS;
    B_SRS = SrsParams.B_SRS;
    K_TC = SrsParams.K_TC;
    k_TC_bar = SrsParams.k_TC_bar;
    n_SRS_cs = SrsParams.n_SRS_cs;
    n_RRC = SrsParams.n_RRC;
    n_shift = SrsParams.n_shift;
    b_hop = SrsParams.b_hop;
    resourceType = SrsParams.resourceType;
    Tsrs = SrsParams.Tsrs;
    Toffset = SrsParams.Toffset;
    groupOrSequenceHopping = SrsParams.groupOrSequenceHopping;
    delta_f = SrsParams.delta_f;

    N_slot_frame = SrsParams.N_slot_frame;
    N_symb_slot = SrsParams.N_symb_slot;
    idxSlotInFrame = SrsParams.idxSlotInFrame;
    idxFrame = SrsParams.idxFrame;
    N_sc_RB = 12;

    srs_BW_table = srsTable.srs_BW_table;

    m_SRS_b = srs_BW_table(C_SRS+1,2*B_SRS+1);
    M_sc_b_SRS = m_SRS_b*N_sc_RB/K_TC;
    if K_TC == 4
        n_SRS_cs_max = 12;
    elseif K_TC == 2
        n_SRS_cs_max = 8;
    else
        error('K_TC is not supported ...\n');
    end

    % compute phase shift alpha
    alpha = [];
    for p = 0:N_ap_SRS-1
        n_SRS_cs_i = mod(n_SRS_cs + (n_SRS_cs_max * p)/N_ap_SRS,  n_SRS_cs_max);
        alpha(p+1) = 2 * pi * n_SRS_cs_i/n_SRS_cs_max;
    end

    % compute SRS sequence group u and sequence number v
    c = build_Gold_sequence(n_ID_SRS, 10 * N_slot_frame * N_symb_slot);
    u = [];
    v = [];
    for l_prime = 0:N_symb_SRS-1
        if groupOrSequenceHopping == 0
            f_gh = 0;
            v(l_prime + 1) = 0;
        elseif groupOrSequenceHopping == 1
            f_gh = 0;
            for m = 0:7
                idxSeq = 8 * (idxSlotInFrame * N_symb_slot + l0 + l_prime) + m;
                f_gh = f_gh + c(idxSeq + 1) * 2^m;
            end
            f_gh = mod(f_gh, 30);
            v(l_prime + 1) = 0;
        elseif groupOrSequenceHopping == 2
            f_gh = 0;
            if M_sc_b_SRS >= 6 * N_sc_RB
                idxSeq = idxSlotInFrame * N_symb_slot + l0 + l_prime;
                v(l_prime + 1) = c(idxSeq + 1);
            else
                v(l_prime + 1) = 0;
            end
        else
            error('groupOrSequenceHopping is not supported ...\n');
        end
        u(l_prime + 1) = mod(f_gh + n_ID_SRS, 30);
    end

    % compute r_bar
    r_bar = [];
    for l_prime = 0:N_symb_SRS-1
        r_bar(l_prime+1,:) = LowPaprSeqGen(M_sc_b_SRS, u(l_prime+1), v(l_prime+1));
    end

    % compute freq domain starting position k0
    k0 = [];
    for l_prime = 0:N_symb_SRS-1
        for p = 0:N_ap_SRS-1
            if (n_SRS_cs >= n_SRS_cs_max/2) && (N_ap_SRS == 4) && (p == 1 || p == 3)
                k_TC = mod(k_TC_bar + K_TC/2, K_TC);
            else
                k_TC = k_TC_bar;
            end
            k0_bar = n_shift * N_sc_RB + k_TC;
            k0(l_prime+1, p+1) = k0_bar;
            for b = 0:B_SRS
                if b_hop >= B_SRS
                    Nb = srs_BW_table(C_SRS+1,2*b+2);
                    m_SRS_b = srs_BW_table(C_SRS+1,2*b+1);
                    nb = mod(floor(4*n_RRC/m_SRS_b), Nb);
                else
                    Nb = srs_BW_table(C_SRS+1,2*b+2);
                    m_SRS_b = srs_BW_table(C_SRS+1,2*b+1);
                    if b <= b_hop
                        nb = mod(floor(4*n_RRC/m_SRS_b), Nb);
                    else
                        if resourceType == 0
                            n_SRS = floor(l_prime/R);
                        else
                            slotIdx = N_slot_frame * idxFrame + idxSlotInFrame - Toffset;
                            if mod(slotIdx, Tsrs) == 0
                                n_SRS = (slotIdx/Tsrs) * (N_symb_SRS/R) + floor(l_prime/R);
                            else
                                warning('Not a SRS slot ...\n');
                                n_SRS = 0;
                            end
                        end
                        PI_bm1 = 1;
                        for b_prime = b_hop+1:b-1
                            PI_bm1 = PI_bm1*srs_BW_table(C_SRS+1,2*b_prime+2);
                        end
                        PI_b = PI_bm1 * Nb;
                        if mod(Nb, 2) == 0
                            Fb = (Nb/2)*floor(mod(n_SRS, PI_b)/PI_bm1) + floor(mod(n_SRS, PI_b)/(2*PI_bm1));
                        else
                            Fb = floor(Nb/2)*floor(n_SRS/PI_bm1);
                        end
                        nb = mod(Fb + floor(4*n_RRC/m_SRS_b), Nb);
                    end
                end
                M_sc_b_SRS = m_SRS_b*N_sc_RB/K_TC;
                k0(l_prime+1, p+1) = k0(l_prime+1, p+1) + K_TC * M_sc_b_SRS * nb;
            end
        end
    end

    xcor = [];
    % map ZC sequence to REs and do cross-correlation
    for l_prime = 0:N_symb_SRS-1
        for p = 0:N_ap_SRS-1
            freq_idx = k0(l_prime+1, p+1) + [0:K_TC:(M_sc_b_SRS-1)*K_TC];
            sym_idx = l_prime + l0;
            r = r_bar(l_prime+1,:) .* exp(1i*[0:M_sc_b_SRS-1]*alpha(p+1)); % add cyclic shift
            for idxAnt = 1:nAnt
                xcor(:, l_prime+1, idxAnt, p+1) = conj(r(:)) .* Xtf(freq_idx+1, sym_idx+1, idxAnt);
                %             Hest(freq_idx+1, sym_idx+1, idxAnt, p+1) = conj(r(:)) .* Xtf(freq_idx+1, sym_idx+1, idxAnt);
            end
        end
    end

    ant_rssi_lin = [];
    ant_rssi_dB = [];
    for idxAnt = 1:nAnt
        temp = xcor(:, :, idxAnt, :);
        ant_rssi_lin(idxAnt) = mean(abs(temp(:).^2));
        if ant_rssi_lin(idxAnt) == 0
            ant_rssi_dB(idxAnt) = -100;
        else
            ant_rssi_dB = 10*log10(abs(ant_rssi_lin(idxAnt)));
        end
    end
    rssi_lin = mean(ant_rssi_lin);
    if rssi_lin == 0
        rssi_dB = -100;
    else
        rssi_dB = 10*log10(rssi_lin);
    end

    Hest1 = [];
    Hest2 = [];
    Hest3 = [];
    Hest_port = [];
    xcor_sum = [];

    % estimate timing offset
    for idxSym = 1:N_symb_SRS
        for idxAnt = 1:nAnt
            Hest1 = xcor(:,idxSym,idxAnt,:);
            Hest1 = Hest1(:);
            len_Hest1 = length(Hest1);
            Hest1 = reshape(Hest1, [N_ap_SRS, len_Hest1/N_ap_SRS]);
            Hest2 = mean(Hest1, 1); % average over N_ap_SRS REs
            Hest2 = reshape(Hest2, [M_sc_b_SRS/N_ap_SRS, N_ap_SRS]);
            for idxPort = 1:N_ap_SRS
                Hest_port = Hest2(:,idxPort);
                xcor_sum(idxSym, idxAnt, idxPort) = sum(Hest_port(2:end) .* conj(Hest_port(1:end-1)));
                Hest3(:, idxSym, idxAnt, idxPort) = Hest_port;
            end
        end
    end

    phaRot = angle(sum(sum(sum(xcor_sum))));

    nReTotal = M_sc_b_SRS/N_ap_SRS;
    reDist = K_TC*N_ap_SRS;
    if reDist >= 16
        nRbEst = 4;
    else
        nRbEst = 2;
    end
    nRePerEst = nRbEst*N_sc_RB/K_TC/N_ap_SRS;
    nEst = nReTotal/nRePerEst;
    estFactor1 = 3/2;

    Hest4 = [];
    Hest5 = [];
    Hest = [];
    Ps = [];
    Pn = [];

    % compensate timing offset and estimate SNR
    for idxSym = 1:N_symb_SRS
        for idxAnt = 1:nAnt
            for idxPort = 1:N_ap_SRS
                % compensate timing offset
                Hest4 = Hest3(:, idxSym, idxAnt, idxPort) .* exp(-1j*phaRot*([0:M_sc_b_SRS/N_ap_SRS-1])');

                % estimate SNR
                for idxEst = 1:nEst
                    Hest5 = Hest4((idxEst-1)*nRePerEst+1:idxEst*nRePerEst);
                    % report Hest avareged over nRbEst RBs
%                     Hest(nRbEst*(idxEst-1)+1:nRbEst*idxEst, idxSym, idxAnt, idxPort) = mean(Hest5);
                    Hest(idxEst, idxSym, idxAnt, idxPort) = mean(Hest5);
                    algSel = 1;
                    if algSel == 0  % assume channel is flat over nRePerEst poinits
                        Havg = mean(Hest5);
                        Hdiff = Hest5 - Havg;
                        Ps(idxEst, idxSym, idxAnt, idxPort) = abs(Havg)^2;
                        Pn(idxEst, idxSym, idxAnt, idxPort) = mean(abs(Hdiff).^2)*estFactor0;
                    elseif algSel == 1 % assume channel is linear over 3 points
                        for idxRe = 2:nRePerEst-1
                            Havg(idxRe-1) = (Hest5(idxRe-1) + Hest5(idxRe) + Hest5(idxRe+1))/3;
                            Hdiff(idxRe-1) = Havg(idxRe-1)-Hest5(idxRe);
                        end
                        Ps(idxEst, idxSym, idxAnt, idxPort) = mean(abs(Havg).^2);
                        Pn(idxEst, idxSym, idxAnt, idxPort) = mean(abs(Hdiff).^2) * estFactor1;
                    end
                end
            end
        end
    end

    % average SNR
    Ps_wide = mean(Ps, [1,2,3,4]);
    Pn_wide = mean(Pn, [1,2,3,4]);

    Ps_rb = mean(Ps, [3,4]);
    Pn_rb = mean(Pn, [3,4]);

    SNR_wide = 10*log10(Ps_wide/Pn_wide)  - 10*log10(N_ap_SRS);
    SNR_rb = 10*log10(Ps_rb./Pn_rb)  - 10*log10(N_ap_SRS);
    SNR_rb = reshape(repmat(SNR_rb(:), [1, nRbEst]).', [m_SRS_b, N_symb_SRS]);

    to_est = -1/2/pi*phaRot;
    to_est_sec = to_est/(delta_f * N_ap_SRS * K_TC);

    if rssi_dB == -100
        SrsOutputList{idxPdu}.to_est_ms = 0;
        SrsOutputList{idxPdu}.Hest = Hest*0;
        SrsOutputList{idxPdu}.nRbHest = 0;
        SrsOutputList{idxPdu}.wideSnr = -100 * ones(size(SNR_wide));
        SrsOutputList{idxPdu}.rbSnr = -100 * ones(size(SNR_rb));
    else
        SrsOutputList{idxPdu}.to_est_ms = to_est_sec*1e6;
        SrsOutputList{idxPdu}.Hest = Hest;
        SrsOutputList{idxPdu}.nRbHest = nRbEst;
        SrsOutputList{idxPdu}.wideSnr = SNR_wide;
        SrsOutputList{idxPdu}.rbSnr = SNR_rb;
    end
    SrsOutputList{idxPdu}.rssi = rssi_dB;
    SrsOutputList{idxPdu}.ant_rssi = ant_rssi_dB;

end

return


function saveTV_srs(tvDirName, TVname, SrsParamsList, Xtf, SrsOutputList)

[status,msg] = mkdir(tvDirName);
h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

nPdu = length(SrsParamsList);

for idxPdu = 1:nPdu

    SrsParams = SrsParamsList(idxPdu);
    SrsParams = formatU32Struct(SrsParams);

    hdf5_write_nv(h5File, ['SrsParams_', num2str(idxPdu-1)], SrsParams);
    hdf5_write_nv(h5File, ['to_est_ms_', num2str(idxPdu-1)], single(SrsOutputList{idxPdu}.to_est_ms));
    hdf5_write_nv(h5File, ['Hest', num2str(idxPdu-1)], single(SrsOutputList{idxPdu}.Hest));
    hdf5_write_nv(h5File, ['nRbHest', num2str(idxPdu-1)], uint32(SrsOutputList{idxPdu}.nRbHest));
    hdf5_write_nv(h5File, ['wideSnr', num2str(idxPdu-1)], single(SrsOutputList{idxPdu}.wideSnr));
    hdf5_write_nv(h5File, ['rbSnr', num2str(idxPdu-1)], single(SrsOutputList{idxPdu}.rbSnr));
end


global SimCtrl
bypassComp = SimCtrl.genTV.bypassComp;

if ~bypassComp
    [cSamples_uint8, X_tf_fp16] = oranCompress(Xtf, 1); % generate ORAN compressed samples
    hdf5_write_nv(h5File, 'X_tf_fp16', X_tf_fp16, 'fp16');
    for k=1:length(SimCtrl.oranComp.iqWidth)
        hdf5_write_nv(h5File, ['X_tf_cSamples_bfp',num2str(SimCtrl.oranComp.iqWidth(k))], uint8(cSamples_uint8{k}));
    end
end

H5F.close(h5File);

return
