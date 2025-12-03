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

function Xtf = genSsb(pduList, ssb, carrier, Xtf)

% Function generates SS block with a random pbch payload.

%inputs:
% ss      --> synchronization signal paramaters
% gnb     --> gnb paramaters

%outputs:
% Xtf_ss  --> time-frequency SS signal. Dim: 240 x 4

nPdu = length(pduList);
SSTxParamsList = [];
pbch_payload_list = [];

for idxPdu = 1:nPdu
    pdu = pduList{idxPdu};
    pbch_payload_list(idxPdu) = pdu.bchPayload;

    SSTxParamsList(idxPdu).NID = carrier.N_ID_CELL;             % physical cell id
    SSTxParamsList(idxPdu).SFN = carrier.idxFrame;
    SSTxParamsList(idxPdu).nHF = ssb.n_hf;            % half frame index (0 or 1)
    SSTxParamsList(idxPdu).Lmax = ssb.L_max;          % max number of ss blocks in pbch period (4,8,or 64)

    % Interpret pdu.betaPss according to SCS FAPI Table 3-40, 0=0 dB, 1=3 dB
    % power increase of PSS over SSS
    % SSTxParams.beta is used internally as a linear (amplitude) scaler
    if (pdu.betaPss == 0)
        SSTxParamsList(idxPdu).beta_pss = 1;
    elseif (pdu.betaPss == 1)
        SSTxParamsList(idxPdu).beta_pss = 10^(3/20); % linear scaler of 3 dB
    else
        error(['Unknown value for betaPss: ',num2str(pdu.betaPss)]);
    end
    SSTxParamsList(idxPdu).beta_sss = 1;

    k_SSB = pdu.ssbSubcarrierOffset;
    offsetPointA = pdu.SsbOffsetPointA;
    SSTxParamsList(idxPdu).k_SSB = k_SSB;
    SSTxParamsList(idxPdu).blockIndex = pdu.ssbBlockIndex;
    SSTxParamsList(idxPdu).t0 = pdu.nSSBStartSymbol;
    SSTxParamsList(idxPdu).f0 = (k_SSB + offsetPointA*carrier.N_sc_RB)/2^carrier.mu;
    SSTxParamsList(idxPdu).enablePrcdBf = pdu.enablePrcdBf;
    SSTxParamsList(idxPdu).PM_W = pdu.PM_W;
end

Xtf0 = Xtf; % for generating test vectors

Xtf = genSsb_cuphy(pbch_payload_list, SSTxParamsList, Xtf);

Xtf1 = Xtf - Xtf0; % for generating test vectors

global SimCtrl;
idxSlot = carrier.idxSlotInFrame;
idxPdu = pdu.ssbPduIdx-1;
if SimCtrl.genTV.enable && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)
    TVname = [SimCtrl.genTV.TVname, '_SSB_gNB_CUPHY_s', num2str(idxSlot),...
        'p', num2str(idxPdu)];
    saveTV_ssb(SimCtrl.genTV.tvDirName, TVname, pbch_payload_list, SSTxParamsList, Xtf1);
    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

return


function Xtf = genSsb_cuphy(pbch_payload_list, SSTxParamsList, Xtf)

nPdu = length(SSTxParamsList);

for idxPdu = 1:nPdu
    N_id = SSTxParamsList(idxPdu).NID;
    SFN = SSTxParamsList(idxPdu).SFN;
    n_hf = SSTxParamsList(idxPdu).nHF;
    L_max = SSTxParamsList(idxPdu).Lmax;
    block_idx = SSTxParamsList(idxPdu).blockIndex;
    beta_pss = SSTxParamsList(idxPdu).beta_pss;
    beta_sss = SSTxParamsList(idxPdu).beta_sss;
    k_SSB = SSTxParamsList(idxPdu).k_SSB;
    firstSymIdx = SSTxParamsList(idxPdu).t0;
    firstSbcIdx = SSTxParamsList(idxPdu).f0;
    enablePrcdBf = SSTxParamsList(idxPdu).enablePrcdBf;
    PM_W = SSTxParamsList(idxPdu).PM_W;
    pbch_payload = pbch_payload_list(idxPdu);

    ssbIdx = genSsbIdx(N_id);
    dmrs_idx = ssbIdx.dmrs_idx;
    qam_idx = ssbIdx.qam_idx;
    pss_idx = ssbIdx.pss_idx;
    sss_idx = ssbIdx.sss_idx;

    K = 56;          % number of pbch payload + crc bits
    E = 864;         % number of pbch bits always 864

    % payload generation
    x = pbch_payload_coding(pbch_payload, SFN, n_hf, block_idx, N_id, L_max, k_SSB)';

    % initialize:
    nSym = 4;       % number of symbols for SS block
    nSbc = 240;     % number of subcarriers for SS block
    Xtf_ss = zeros(nSbc, nSym);

    % step 1:
    x_crc = add_CRC_LUT(x,'24C');

    % step 2:
    [x_encoded,N] = polar_encode(x_crc,K,E);

    % step 3:
    x_rm = polar_rate_match(x_encoded,N,K,E);

    % step 4:
    x_scram = pbch_scrambling(x_rm,E,N_id,L_max,block_idx);

    % step 5:
    x_qam = qpsk_modulate(x_scram,E);
    Xtf_ss(qam_idx + 1) = beta_sss * x_qam;

    % step 6:
    dmrs = build_pbch_dmrs(L_max,block_idx,n_hf,N_id);
    Xtf_ss(dmrs_idx + 1) = beta_sss * dmrs;

    % step 7:
    d_pss = build_pss(N_id);
    Xtf_ss(pss_idx + 1) = beta_pss * d_pss;

    % step 8:
    d_sss = build_sss(N_id);
    Xtf_ss(sss_idx + 1) = beta_sss * d_sss;

    if enablePrcdBf
        [~, ~, nAnt] = size(Xtf);
        Xtf_ss = reshape(Xtf_ss, nSbc*nSym, 1);
        [nAnt_Prcd, ~] = size(PM_W);
        if nAnt_Prcd < nAnt
            PM_W = [PM_W; zeros(nAnt-nAnt_Prcd, 1)];
        elseif nAnt_Prcd > nAnt
                warning('The number of ports for SSB precoding matrix is larger than the number of antennas');
        end
        Xtf_ss = Xtf_ss * PM_W.';
        Xtf_ss = reshape(Xtf_ss, nSbc, nSym, nAnt);
        Xtf(firstSbcIdx+1:firstSbcIdx+nSbc, firstSymIdx+1:firstSymIdx+nSym, :) = Xtf_ss;
    else
        Xtf(firstSbcIdx+1:firstSbcIdx+nSbc, firstSymIdx+1:firstSymIdx+nSym) = Xtf_ss;
    end
end

return


function saveTV_ssb(tvDirName, TVname, pbch_payload_list, SSTxParamsList, Xtf)

[status,msg] = mkdir(tvDirName);

nPdu = length(SSTxParamsList);

for idxPdu = 1:nPdu
    SSTxParams = SSTxParamsList(idxPdu);

    PM_W_list{idxPdu} = SSTxParams.PM_W;
    SSTxParams = rmfield(SSTxParams, 'PM_W');
    singleField = {'beta_pss', 'beta_sss'};
    SSTxParams = formatU32Struct(SSTxParams, singleField);

    SSTxParamsNew(idxPdu) = SSTxParams;
end

h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_nv(h5File, 'nSsb', uint32(nPdu));
hdf5_write_nv(h5File, 'SSTxParams', SSTxParamsNew);
hdf5_write_nv(h5File, 'x_mib', uint32(pbch_payload_list));
hdf5_write_nv(h5File, 'X_tf', single(Xtf));

for idxPdu = 1:nPdu
    hdf5_write_nv(h5File, ['Ssb_PM_W',num2str(idxPdu-1)], complex(single(PM_W_list{idxPdu})));
end

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
