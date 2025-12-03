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

function [pbch_payload, errFlag] = detSsb(pdu, ssb, carrier, Xtf)

% Function generates SS block with a random pbch payload.

%inputs:
% ss      --> synchronization signal paramaters
% gnb     --> gnb paramaters

%outputs:
% Xtf_ss  --> time-frequency SS signal. Dim: 240 x 4

% pbch_payload = dec2bin(pdu.bchPayload, 24) - '0';

SSTxParams.NID = carrier.N_ID_CELL;             % physical cell id
SSTxParams.SFN = carrier.idxFrame;
SSTxParams.nHF = ssb.n_hf;            % half frame index (0 or 1)
SSTxParams.Lmax = ssb.L_max;          % max number of ss blocks in pbch period (4,8,or 64)

% Interpret pdu.betaPss according to SCS FAPI Table 3-40, 0=0 dB, 1=3 dB
% power increase of PSS over SSS
% SSTxParams.beta is used internally as a linear (amplitude) scaler
if (pdu.betaPss == 0)
    SSTxParams.beta_pss = 1;
elseif (pdu.betaPss == 1)
    SSTxParams.beta_pss = 10^(3/20); % linear scaler of 3 dB
else
    error(['Unknown value for betaPss: ',num2str(pdu.betaPss)]);
end
SSTxParams.beta_sss = 1;

k_SSB = pdu.ssbSubcarrierOffset;
offsetPointA = pdu.SsbOffsetPointA;
SSTxParams.k_SSB = k_SSB;
SSTxParams.blockIndex = pdu.ssbBlockIndex;
SSTxParams.t0 = pdu.nSSBStartSymbol;
SSTxParams.f0 = (k_SSB + offsetPointA*carrier.N_sc_RB)/2^carrier.mu;
SSTxParams.enablePrcdBf = pdu.enablePrcdBf;
SSTxParams.PM_W = pdu.PM_W;

[pbch_payload, errFlag] = detSsb_cuphy(SSTxParams, Xtf);

global SimCtrl;
idxSlot = carrier.idxSlotInFrame;
idxPdu = pdu.ssbPduIdx-1;
if SimCtrl.genTV.enableUE && SimCtrl.genTV.enable && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)
    TVname = [SimCtrl.genTV.TVname, '_SSB_UE_CUPHY_s', num2str(idxSlot),...
        'p', num2str(idxPdu)];
    saveTV_ssb(SimCtrl.genTV.tvDirName, TVname, pbch_payload, SSTxParams, Xtf);
    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

return


function  [pbch_payload, errFlag] = detSsb_cuphy(SSTxParams, Xtf)

nAnt = size(Xtf, 3);
L_max = SSTxParams.Lmax; % based on fc

% Assume T/F sync has been done in time domain
firstSymIdx = SSTxParams.t0;
firstSbcIdx = SSTxParams.f0;

% define constants
K = 56;          % number of pbch payload + crc bits
E = 864;         % number of pbch bits always 864
N = 512;
nSym = 4;       % number of symbols for SS block
nSbc = 240;     % number of subcarriers for SS block
pss_idx = 56 : 182;
sss_idx = (56 : 182) + 240*2;

% grab SSB from Xtf
Xtf_ss_allAnt = Xtf(firstSbcIdx+1:firstSbcIdx+nSbc, firstSymIdx+1:firstSymIdx+nSym, :);

% detect N_id2 based on pss
pss_cor_allAnt = [];
for idxAnt = 1:nAnt
    Xtf_ss = Xtf_ss_allAnt(:,:,idxAnt);
    pss_rx = Xtf_ss(pss_idx + 1);
    for N_id2 = 0:2
        pss_tx = gen_pss(N_id2) ;
        pss_cor_allAnt(N_id2 + 1, idxAnt) = abs(sum(pss_tx(:)' * pss_rx(:)));
    end
end
pss_cor = mean(pss_cor_allAnt, 2);
[max_pss_cor, idx_pss_cor] = max(pss_cor);
N_id2_det = idx_pss_cor - 1;

% detect N_id1 based on sss
sss_cor_allAnt = [];
for idxAnt = 1:nAnt
    Xtf_ss = Xtf_ss_allAnt(:,:,idxAnt);
    sss_rx = Xtf_ss(sss_idx + 1);
    for N_id1 = 0:335
        sss_tx = gen_sss(N_id1, N_id2_det) ;
        sss_cor_allAnt(N_id1 + 1, idxAnt) = abs(sum(sss_tx(:)' * sss_rx(:)));
    end
end
sss_cor = mean(sss_cor_allAnt, 2);
[max_sss_cor, idx_sss_cor] = max(sss_cor);
N_id1_det = idx_sss_cor - 1;

% derive N_id from N_id1 and N_id2
N_id = 3 * N_id1_det + N_id2_det;

ssbIdx = genSsbIdx(N_id);
dmrs_idx = ssbIdx.dmrs_idx;
qam_idx = ssbIdx.qam_idx;

% detect i_bar_ssb
dmrs_cor_allAnt = [];
for idxAnt = 1:nAnt
    Xtf_ss = Xtf_ss_allAnt(:,:,idxAnt);
    dmrs_rx = Xtf_ss(dmrs_idx + 1);
    for i_bar_ssb = 0:7
        dmrs_tx = gen_pbch_dmrs(i_bar_ssb, N_id);
        dmrs_cor_allAnt(i_bar_ssb + 1, idxAnt) = abs(sum(dmrs_tx(:)' * dmrs_rx(:)));
    end
end
dmrs_cor = mean(dmrs_cor_allAnt, 2);
[max_dmrs_cor, idx_dmrs_cor] = max(dmrs_cor);
i_bar_ssb_det = idx_dmrs_cor - 1;

if L_max == 4
    n_hf = floor(i_bar_ssb_det/4);
    block_idx = i_bar_ssb_det - 4*n_hf;
elseif L_max == 8 || L_max == 64
    block_idx = i_bar_ssb_det;
    n_hf = 0;
else
    error('L_max is not supported ...\n');
end

pbch_sd_allAnt = [];
dmrs_tx = gen_pbch_dmrs(i_bar_ssb_det, N_id);
for idxAnt = 1:nAnt
    Xtf_ss = Xtf_ss_allAnt(:,:,idxAnt);

    dmrs_des = Xtf_ss(dmrs_idx + 1) .* conj(dmrs_tx);

    % equalize qam
    qam = Xtf_ss(qam_idx + 1);
    qam_eq = zeros(size(qam));
    nPrb = length(qam)/9;

    for idxPrb = 1:nPrb
        chest = mean(dmrs_des((idxPrb-1)*3+1:idxPrb*3));
        qam_eq((idxPrb-1)*9+1:idxPrb*9) = qam((idxPrb-1)*9+1:idxPrb*9) .* conj(chest);
    end

    % soft demapper
    pbch_sd = [real(qam_eq(:)), imag(qam_eq(:))]';
    pbch_sd_allAnt(:, idxAnt) = pbch_sd(:);
end
pbch_sb = mean(pbch_sd_allAnt, 2);

% descramble
pbch_des = pbch_descrambling(pbch_sb,E,N_id,L_max,block_idx);

% TBD: Need to replace 5GToolBox function nrBCHDecode
L = 8;
[scrblk,errFlag,rxtrblk,rxSFN4lsb,rxHRF,rxKssb] = nrBCHDecode(pbch_des, L, L_max, N_id);

pbch_payload = rxtrblk';

return


function saveTV_ssb(tvDirName, TVname, pbch_payload, SSTxParams, Xtf)

[status,msg] = mkdir(tvDirName);

PM_W = SSTxParams.PM_W;
SSTxParams = rmfield(SSTxParams, 'PM_W');
singleField = {'beta_pss', 'beta_sss'};
SSTxParams = formatU32Struct(SSTxParams, singleField);

h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_nv(h5File, 'SSTxParams', SSTxParams);
hdf5_write_nv(h5File, 'x_mib', uint32(pbch_payload'));
hdf5_write_nv(h5File, 'X_tf', single(Xtf));
hdf5_write_nv(h5File, 'PM_W', single(PM_W));

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
