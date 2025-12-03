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

function prach_payload = detPrach(pduList, configList, table, carrier, Xtf)
% function prach = genPreambleSeq(prach, carrier, nodeType)
%
% This function generates PRACH paramters, ZC sequence and preamble in
% freq domain.
%
% Input:    prach: prach related configuration
%           carrier: carrier related configuration
%           nodeType: 'UE' or 'gNB'
%
% Output:   prach: add fields for PRACH paramters, ZC sequence and
%           preamble in freq domain
%

global SimCtrl

% Read parameters from pdu, prachConfig and carrier to PrachCuphyParams
nPdu = length(pduList);

for idxPdu = 1:nPdu
    pdu = pduList{idxPdu};
    prachConfig = configList{idxPdu};

    % Read parameters from pdu, prachConfig and carrier to PrachCuphyParams
    PrachCuphyParamsList(idxPdu).configurationIndex = prachConfig.configurationIndex;
    PrachCuphyParamsList(idxPdu).restrictedSet = prachConfig.restrictedSet;
    PrachCuphyParamsList(idxPdu).prachRootSequenceIndex = prachConfig.prachRootSequenceIndex;
    PrachCuphyParamsList(idxPdu).prachZeroCorrConf = prachConfig.zeroCorrelationZone;
    PrachCuphyParamsList(idxPdu).startRaSym = prachConfig.startRaSym;
    PrachCuphyParamsList(idxPdu).N_CP_RA = prachConfig.N_CP_RA;
    PrachCuphyParamsList(idxPdu).K = prachConfig.K;
    PrachCuphyParamsList(idxPdu).k1 = prachConfig.k1;
    PrachCuphyParamsList(idxPdu).N_u = prachConfig.N_u;
    PrachCuphyParamsList(idxPdu).n_slot_RA_sel = prachConfig.n_slot_RA_sel;
    PrachCuphyParamsList(idxPdu).mu = carrier.mu;
    if SimCtrl.enable_static_dynamic_beamforming % 64TR
        PrachCuphyParamsList(idxPdu).N_ant = pdu.digBFInterfaces;
    else
        PrachCuphyParamsList(idxPdu).N_ant = carrier.numRxPort;
    end
    PrachCuphyParamsList(idxPdu).FR = carrier.FR;
    PrachCuphyParamsList(idxPdu).duplex = carrier.duplex;
    % thr0 is the preamble detection threshold.
    % If force_thr0 = 0, use the default thr0 calculated by cuPHY internally.
    % Otherwise, use this force_thr0 value for thr0.
    PrachCuphyParamsList(idxPdu).force_thr0 = prachConfig.force_thr0;
end

for idxPdu = 1:nPdu
    PrachParamsList(idxPdu) = derive_prach_params(PrachCuphyParamsList(idxPdu), table);
    y_uv_rx{idxPdu} = Xtf{idxPdu}.';
end

prach_payload = detPrach_cuphy(PrachParamsList, y_uv_rx);

idxSlot = carrier.idxSlotInFrame;
idxPdu = pdu.prachPduIdx-1;

global SimCtrl;

% if SimCtrl.timeDomainSim && PrachCuphyParams.L_RA == 839 && PrachCuphyParams.mu == 1
%     % In this case, preamble spans over two slots. PRACH receiver detects
%     % preamble at the second slot while FAPI PDU is issued at the first slot
%     idxSlot = idxSlot - 1;
% end

if SimCtrl.genTV.enable && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)
    TVname = [SimCtrl.genTV.TVname, '_PRACH_gNB_CUPHY_s', num2str(idxSlot),...
        'p', num2str(idxPdu)];
    saveTV_prach(SimCtrl.genTV.tvDirName, TVname,  PrachCuphyParamsList, PrachParamsList, y_uv_rx, prach_payload, carrier);
    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

return

function payloadList = detPrach_cuphy(prachList, y_uv_rx_List)

nPdu = length(prachList);

for idxPdu = 1:nPdu
    prach = prachList(idxPdu);
    y_uv_rx = y_uv_rx_List{idxPdu};

    N_CS = prach.N_CS;
    uCount = prach.uCount;
    L_RA = prach.L_RA;
    N_rep = prach.N_rep;
    mu = prach.mu;
    delta_f_RA = prach.delta_f_RA;
    Nfft = prach.Nfft;
    N_nc = prach.N_nc;
    thr0 = prach.thr0;
    kBar = prach.kBar;
    N_ant = prach.N_ant;

    y_u_ref = prach.y_u_ref;
    u_ref = prach.u_ref;
    C_v_ref = prach.C_v_ref;

    % compute RSSI
    for idxAnt = 1:N_ant
        antRssiLin(idxAnt) = mean(abs(y_uv_rx(idxAnt, :)).^2); % averaged power on each antenna
        if antRssiLin(idxAnt) == 0
            antRssiDb(idxAnt) = -100;
        else
            antRssiDb(idxAnt) = 10*log10(antRssiLin(idxAnt));
        end
    end
    rssiLin = mean(antRssiLin); % averaged power over all antennas
    if rssiLin == 0
        rssiDb = -100;
    else
        rssiDb = 10*log10(rssiLin);
    end

    % One sample in L_RA scale equals to Nfft/L_RA samples in Nfft scale
    zoneSearchGap = 1*floor(Nfft/L_RA);

    prmbCount = 0;

    for idxU = 0:uCount-1
        % load reference preamble in freq domain for one u
        y_u0_tx = y_u_ref(idxU+1, :);
        u1 = u_ref(prmbCount+1);
        Nzone = sum(u1 == u_ref);
        [~, Nsamp] = size(y_uv_rx);
        Nsamp = Nsamp/N_rep;
        for idxAnt = 1:N_ant
            y_uv_rx_rep = y_uv_rx(idxAnt,:);
            % average for repeatitive preamble
            rep_start = 1;
            for idxNc = 1:N_nc
                y_uv_rx_mean = zeros(1, L_RA);
                if idxNc < N_nc
                    step = floor(N_rep/N_nc);
                else
                    step = N_rep - (N_nc-1)*floor(N_rep/N_nc);
                end
                for idxRep = rep_start:rep_start+step-1
                    samp_start = Nsamp*(idxRep-1)+1+kBar;
                    samp_end = samp_start + L_RA - 1;
                    y_uv_rx_mean = y_uv_rx_mean + ...
                        y_uv_rx_rep(samp_start:samp_end);
                end
                y_uv_rx_mean = y_uv_rx_mean/step;
                rep_start = rep_start + step;
                % multiplication in freq domain
                z_u = y_uv_rx_mean.*conj(y_u0_tx);

                % convert to time domain and calculate power of each sample
                pdp_nc(idxNc, :) = abs(ifft(z_u, Nfft)*(Nfft/L_RA)).^2;
            end
            pdp = mean(pdp_nc, 1);
            % Right shift zoneSearchGap samples to avoid misdetection of the
            % strongest path at beginning of zone(k) to zone(k-1)
            pdp= [pdp(end-zoneSearchGap+1:end), pdp(1:end-zoneSearchGap)];
            for idxZone = 0:Nzone-1

                % find each zone's location
                zone_start = mod(Nfft-ceil(C_v_ref(prmbCount+idxZone+1)...
                    *Nfft/L_RA), Nfft)+1;
                zone_end = zone_start + floor(N_CS*Nfft/L_RA);

                % calculate each zone's mean/max power and stronges path location
                pdp_zone = pdp(zone_start:zone_end);
                pdp_zone_power(idxAnt, idxZone+1) = mean(pdp_zone);
                [pdp_zone_max(idxAnt, idxZone+1), ...
                    pdp_max_loc(idxAnt, idxZone+1)] = max(pdp_zone);
                pdp_max_loc(idxAnt, idxZone+1) = ...
                    pdp_max_loc(idxAnt, idxZone+1)-zoneSearchGap-1;
            end % idxZone = 0:Nzone-1

        end % idxAnt = 1:N_ant

        for idxZone = 0:Nzone-1
            pdp_zone_power_mean(prmbCount+idxZone+1) = ...
                mean(pdp_zone_power(:, idxZone+1));
            pdp_zone_max_mean(prmbCount+idxZone+1) = ...
                mean(pdp_zone_max(:, idxZone+1));
            [~, maxIdx] = max(pdp_zone_max(:, idxZone+1));
            pdp_max_loc_mean(prmbCount+idxZone+1) = ...
                pdp_max_loc(maxIdx, idxZone+1);
        end

        prmbCount = prmbCount + Nzone;
        if prmbCount >= 64
            break;
        end
    end

    % estimate noise floor
    np1 = mean(pdp_zone_power_mean);
    thr1 = thr0*np1;
    idxNoiseZone = find(pdp_zone_max_mean < thr1);
    np2 = mean(pdp_zone_power_mean(idxNoiseZone));
    if np2 == 0 || np1 == 0 % handle forceRxZero case
        np2dB = -100;
    else
        np2dB = 10*log10(np2);
    end
    thr2 = max(np2*thr0, min(rssiLin, 1) * 1e-2); % to reduce false alarm

    detIdx = 0;
    Nprmb = 64;
    peak_det = [];
    prmbIdx_det = [];
    delay_samp_det = [];
    for prmbIdx = 0:Nprmb-1
        if pdp_zone_max_mean(prmbIdx+1) > thr2
            detIdx = detIdx + 1;
            peak_det(detIdx) = pdp_zone_max_mean(prmbIdx+1);
            prmbIdx_det(detIdx) = prmbIdx;
            delay_samp_det(detIdx) = max(0, pdp_max_loc_mean(prmbIdx+1));
        end
    end

    prach.pdp_zone_max_mean = pdp_zone_max_mean;
    prach.pdp_max_loc_mean = pdp_max_loc_mean;
    prach.thr2 = thr2;
    prach.thr0 = thr0;
    prach.detIdx = detIdx;

    if detIdx > 0
        snr_det = 10*log10(peak_det/np2)-10*log10(L_RA);
        prach.peak_det = peak_det;
        prach.prmbIdx_det = prmbIdx_det;
        prach.delay_samp_det = delay_samp_det;
        delay_time_det = delay_samp_det/(Nfft*delta_f_RA);
        prach.delay_time_det = delay_time_det;
        prach.snr_det = snr_det;
    else
        prach.peak_det = 0;
        prach.prmbIdx_det = -1;
        prach.delay_samp_det = 0;
        prach.delay_time_det = 0;
        prach.snr_det = 100;

    end

    payload.detIdx = prach.detIdx;
    payload.peak_det = prach.peak_det;
    payload.prmbIdx_det = prach.prmbIdx_det;
    payload.delay_samp_det = prach.delay_samp_det;
    payload.delay_time_det = prach.delay_time_det;
    payload.snr_det = prach.snr_det;
    payload.rssi_det = rssiDb;
    payload.antRssi_det = antRssiDb;
    payload.noise_det = np2dB;
    payloadList{idxPdu} = payload;

end

return


function prach = derive_prach_params(PrachCuphyParams, table)

prachTable = table;

% read input from carrier and prach
restrictedSet = PrachCuphyParams.restrictedSet;
rootSequenceIndex = PrachCuphyParams.prachRootSequenceIndex;
zeroCorrelationZone = PrachCuphyParams.prachZeroCorrConf;
prachCfgIdx = PrachCuphyParams.configurationIndex;
mu = PrachCuphyParams.mu;
FR = PrachCuphyParams.FR;
duplex = PrachCuphyParams.duplex;
delta_f = 15000 * 2^mu;
N_ant = PrachCuphyParams.N_ant;
force_thr0 = PrachCuphyParams.force_thr0;

[preambleFormat] = readPrachCfgTable(prachCfgIdx, FR, duplex, prachTable);

% find dela_f_RA and L_RA
switch preambleFormat
    case {'0', '1', '2'}
        delta_f_RA = 1250;
        L_RA = 839;
    case '3'
        delta_f_RA = 5000;
        L_RA = 839;
    otherwise
        delta_f_RA = 15000*2^mu;
        L_RA = 139;
end

if L_RA == 839
    Nfft = 1024;
else
    Nfft = 256;
end

switch preambleFormat
    case '0'
        N_rep = 1;
    case 'B4'
        N_rep = 12;
    case '1'
        N_rep = 2;
    otherwise
        error('preambleFormat is not supported ...\n');
end

% load logIdx2u table for logical root mapping
if L_RA == 839
    % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-3
    logIdx2u_table = prachTable.table_logIdx2u_839;
elseif L_RA == 139
    % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-4
    logIdx2u_table = prachTable.table_logIdx2u_139;
else
    error('L_RA length is not supported ... \n');
end

% find N_CS from zeroCorrelationZone
switch delta_f_RA
    case 1250
        switch restrictedSet
            case 0
                typeIdx = 1;
            case 1
                typeIdx = 2;
            case 2
                typeIdx = 3;
            otherwise
                error('restrictedSet is not supported ...\n');
        end
        % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-5
        N_CS_table = prachTable.table_NCS_1p25k;
        N_CS = N_CS_table(zeroCorrelationZone + 1, typeIdx);
    case 5000
        switch restrictedSet
            case 0
                typeIdx = 1;
            case 1
                typeIdx = 2;
            case 2
                typeIdx = 3;
            otherwise
                error('restrictedSet is not supported ...\n');
        end
        % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-6
        N_CS_table = prachTable.table_NCS_5k;
        N_CS = N_CS_table(zeroCorrelationZone + 1, typeIdx);
    otherwise
        % 3GPP TS 38.211 V15.4.0 Table 6.3.3.1-7
        N_CS_table = prachTable.table_NCS_15kplus;
        N_CS = N_CS_table(zeroCorrelationZone + 1);
end


% generate ZC sequence and preamble

% calculate u and C_v

Nprmb = 64;
y_u_ref = [];
u_ref = [];
C_v_ref = [];
for prmbIdx = 0:Nprmb-1
    [u, C_v] = findZcPar(prmbIdx, rootSequenceIndex, ...
        L_RA, restrictedSet, N_CS, logIdx2u_table);
    if prmbIdx == 0
        uList = u;
        uCount = 1;
    else
        if u ~= uList(uCount)
            uCount = uCount + 1;
            uList(uCount) = u;
        end
    end
    y_u_ref(uCount, :) = genZcPreamble(L_RA, 0, u);
    u_ref(prmbIdx+1) = u;
    C_v_ref(prmbIdx+1) = C_v;
end

% Set preamble detection SNR threshold
% Test with preamble format 0 and B4, mu = 0 and 1.
% May need optimization for other combinations.

N_nc = 1;
if mu == 1 && L_RA == 839
    switch N_ant
        case {1, 2}
            thr0 = 10;
        case {3, 4, 5}
            thr0 = 8.5;
        otherwise
            thr0 = 6.0;
    end
elseif mu == 1 && L_RA == 139
    switch N_ant
        case {1, 2}
            thr0 = 12.0;
        case {3, 4, 5}
            thr0 = 9.5;
        otherwise
            thr0 = 7.5;
    end
elseif mu == 0 && L_RA == 839
    switch N_ant
        case {1, 2}
            thr0 = 10;
        case {3, 4, 5}
            thr0 = 8;
        otherwise
            thr0 = 6;
    end
elseif mu == 0 && L_RA == 139
    switch N_ant
        case {1, 2}
            thr0 = 11;
        case {3, 4, 5}
            thr0 = 8.5;
        otherwise
            thr0 = 7;
    end
else
    error('thr0 is not defined...\n');
end

if force_thr0 > 0
    thr0 = force_thr0;
end

% find kBar for subcarrier offset in unit of delta_f_RA
% 3GPP 38.211 (V15.4) Table 6.3.3.2-1
kBar_table = prachTable.kBar_table;
[M, ~] = size(kBar_table);
find_flag = 0;
for m = 1:M
    if (L_RA == kBar_table(m, 1)) && (delta_f_RA == kBar_table(m, 2)*1000) ...
            && (delta_f == kBar_table(m, 3)*1000)
        N_RB_RA = kBar_table(m, 4);
        kBar = kBar_table(m, 5); % preamble first subcarrier shift
        find_flag = 1;
        break;
    end
end
if find_flag == 0
    error('kBar table error ... \n');
end

% Add derived paramters into PrachParams
prach.L_RA = L_RA;
prach.y_u_ref = y_u_ref;
prach.u_ref = u_ref;
prach.C_v_ref = C_v_ref;
prach.uCount = uCount;
prach.N_rep = N_rep;
prach.Nfft = Nfft;
prach.thr0 = thr0;
prach.N_nc = N_nc;
prach.kBar = kBar;
prach.delta_f_RA = delta_f_RA;
prach.N_CS = N_CS;
prach.mu = mu;
prach.N_ant = N_ant;
prach.startRaSym = PrachCuphyParams.startRaSym;
prach.N_CP_RA = PrachCuphyParams.N_CP_RA;
prach.K = PrachCuphyParams.K;
prach.k1 = PrachCuphyParams.k1;
prach.N_u = PrachCuphyParams.N_u;
prach.n_slot_RA_sel = PrachCuphyParams.n_slot_RA_sel;

return


function saveTV_prach(tvDirName, TVname, PrachCuphyParamsList, PrachParamsList, XtfList, prach_payload_list, carrier)

[status,msg] = mkdir(tvDirName);
nPdu = length(PrachCuphyParamsList);
h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_nv(h5File, 'nPrach', uint32(nPdu));

PrachCellStatPrms.N_ant = PrachCuphyParamsList(1).N_ant;
PrachCellStatPrms.FR = PrachCuphyParamsList(1).FR;
PrachCellStatPrms.duplex = PrachCuphyParamsList(1).duplex;
PrachCellStatPrms.mu = PrachCuphyParamsList(1).mu;
PrachCellStatPrms.configurationIndex = PrachCuphyParamsList(1).configurationIndex;
PrachCellStatPrms.restrictedSet = PrachCuphyParamsList(1).restrictedSet;
PrachCellStatPrms = formatU32Struct(PrachCellStatPrms);
hdf5_write_nv(h5File, 'PrachCellStatPrms', PrachCellStatPrms);

for idxPdu = 1:nPdu
    PrachCuphyParams = PrachCuphyParamsList(idxPdu);
    PrachParams = PrachParamsList(idxPdu);
    Xtf = XtfList{idxPdu};
    prach_payload = prach_payload_list{idxPdu};

    PrachCuphyParams = rmfield(PrachCuphyParams, 'N_ant');
    PrachCuphyParams = rmfield(PrachCuphyParams, 'FR');
    PrachCuphyParams = rmfield(PrachCuphyParams, 'duplex');
    PrachCuphyParams = rmfield(PrachCuphyParams, 'mu');
    PrachCuphyParams = rmfield(PrachCuphyParams, 'configurationIndex');
    PrachCuphyParams = rmfield(PrachCuphyParams, 'restrictedSet');
    PrachCuphyParams = formatU32Struct(PrachCuphyParams, {'force_thr0'});

    y_uv_rx = complex(single(Xtf.'));
    y_u_ref = single(PrachParams.y_u_ref.');
    PrachParams = rmfield(PrachParams, 'y_u_ref');
    u_ref = uint32(PrachParams.u_ref);
    PrachParams = rmfield(PrachParams, 'u_ref');
    C_v_ref = uint32(PrachParams.C_v_ref);
    PrachParams = rmfield(PrachParams, 'C_v_ref');
    k1_temp = PrachParams.k1;
    PrachParams = formatU32Struct(PrachParams, {'thr0'});
    PrachParams.k1 = int32(k1_temp);
    
    idxStr = ['_', num2str(idxPdu-1)];

    hdf5_write_nv(h5File, ['prachCuphyParams', idxStr], PrachCuphyParams);
    hdf5_write_nv(h5File, ['prachParams', idxStr], PrachParams);
    hdf5_write_nv(h5File, ['y_uv_rx', idxStr], y_uv_rx);
    hdf5_write_nv(h5File, ['y_u_ref', idxStr], y_u_ref);
    hdf5_write_nv(h5File, ['u_ref', idxStr], u_ref);
    hdf5_write_nv(h5File, ['C_v_ref', idxStr], C_v_ref);

    % dump Tx X_tf
    global SimCtrl
    if SimCtrl.enable_snapshot_gNB_UE_into_SimCtrl && SimCtrl.genTV.enable_logging_tx_Xtf
        X_tf_transmitted_from_UE = SimCtrl.gNBUE_snapshot.UE{idxPdu}.Phy.tx.Xtf;
        hdf5_write_nv(h5File, ['X_tf_transmitted_from_UE', idxStr], X_tf_transmitted_from_UE);
    end

    %dump time-domain TX signal Xt
    if SimCtrl.enable_snapshot_gNB_UE_into_SimCtrl
        X_t_transmitted_from_UE = SimCtrl.gNBUE_snapshot.UE{idxPdu}.Phy.tx.Xt;
        hdf5_write_nv(h5File, ['X_t_transmitted_from_UE', idxStr], X_t_transmitted_from_UE);

        % ground-truth info 
        prmbIdx = SimCtrl.gNBUE_snapshot.UE{idxPdu}.Mac.Config.prach{1}.prmbIdx;
        hdf5_write_nv(h5File, ['UE_prmbIdx', idxStr], uint32(prmbIdx));
    end

    if SimCtrl.genTV.enable_logging_carrier_and_channel_info
        saveCarrierChanPars(h5File, SimCtrl, carrier, 1);
    end
    % save output
    prmbIdx_det = prach_payload.prmbIdx_det;
    delay_time_det = prach_payload.delay_time_det;
    detIdx = prach_payload.detIdx;
    peak_det = prach_payload.peak_det;
    rssi_det = prach_payload.rssi_det;
    antRssi_det = prach_payload.antRssi_det;
    noise_det = prach_payload.noise_det;

    hdf5_write_nv(h5File, ['detIdx', idxStr], uint32(detIdx));
    hdf5_write_nv(h5File, ['prmbIdx_det', idxStr], uint32(prmbIdx_det));
    hdf5_write_nv(h5File, ['delay_time_det', idxStr], single(delay_time_det));
    hdf5_write_nv(h5File, ['peak_det', idxStr], single(peak_det));
    hdf5_write_nv(h5File, ['rssi_det', idxStr], single(rssi_det));
    hdf5_write_nv(h5File, ['numAnt', idxStr], uint32(length(antRssi_det)));
    hdf5_write_nv(h5File, ['antRssi_det', idxStr], single(antRssi_det));
    hdf5_write_nv(h5File, ['noise_det', idxStr], single(noise_det));
    
    % ground-truth info 
    prachFalseAlarmTest = SimCtrl.prachFalseAlarmTest;
    hdf5_write_nv(h5File, ['prachFalseAlarmTest', idxStr], uint32(prachFalseAlarmTest));

    global SimCtrl
    bypassComp = SimCtrl.genTV.bypassComp;
    if ~bypassComp
        [cSamples_uint8, X_tf_fp16] = oranCompress(Xtf.', 1); % generate ORAN compressed samples
        hdf5_write_nv(h5File, ['X_tf', idxStr, '_fp16'], X_tf_fp16, 'fp16');
        for k=1:length(SimCtrl.oranComp.iqWidth)
            hdf5_write_nv(h5File, ['X_tf', idxStr, '_cSamples_bfp', num2str(SimCtrl.oranComp.iqWidth(k))], uint8(cSamples_uint8{k}));
        end
    end
end

H5F.close(h5File);

return
