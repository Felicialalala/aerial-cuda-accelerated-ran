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

function [Xtf,Gtruth] = genPusch(pduList, table, carrier, Xtf)

global SimCtrl
Gtruth = {};

Xtf0 = Xtf; % for generating test vectors

puschTable = table;
nPdu = length(pduList);

PuschParamsList = [];
pusch_payload_list = [];

for idxPdu = 1:nPdu
    pdu = pduList{idxPdu};

    pusch_payload = pdu.payload';

    % derive configuration
    dmrs    = derive_dmrs_main(pdu, puschTable);
    alloc   = derive_alloc_main(pdu, dmrs);
    coding  = derive_coding_main(pdu, alloc, puschTable);

    % load parameters
    PuschParams.CRC = coding.CRC;
    PuschParams.C = coding.C;
    PuschParams.K = coding.K;
    PuschParams.F = coding.F;
    PuschParams.K_prime = coding.K_prime;
    PuschParams.BGN = coding.BGN;   %1 or 2. Indicates which base graph used
    PuschParams.i_LS = coding.i_LS; %lifting set index
    PuschParams.Zc = coding.Zc;     %lifting size
    PuschParams.qam = coding.qam;
    PuschParams.rvIdx = coding.rvIdx;
    PuschParams.nl = alloc.nl;
    PuschParams.N_data = alloc.N_data;
    PuschParams.N_id = alloc.dataScramblingId;
    PuschParams.n_rnti = alloc.RNTI;
    PuschParams.qamstr = coding.qamstr;
    PuschParams.portIdx = alloc.portIdx;          % user's antenna ports (matlab 1 indexing). Dim: nl x 1
    PuschParams.n_scid = alloc.SCID;            % 0 or 1. User's dmrs scrambling id
    PuschParams.nPrb = alloc.nPrb;               % number of prbs in allocation
    PuschParams.startPrb = alloc.startPrb;       % starting prb of allocation
    PuschParams.Nf_data = alloc.Nf_data;         % number of data subcarriers in allocation
    PuschParams.Nt_data = alloc.Nt_data;         % number of data symbols in allocation
    PuschParams.symIdx_data = alloc.symIdx_data; % indicies of data symbols. Dim: Nt_data x 1
    PuschParams.slotNumber = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
    if isfield(SimCtrl.genTV,'fakeSlotNumber')
        PuschParams.slotNumber = SimCtrl.genTV.fakeSlotNumber;
    end
    PuschParams.Nf = carrier.N_sc;
    PuschParams.Nt = carrier.N_symb_slot;
    PuschParams.N_dmrs_id = dmrs.DmrsScramblingId;
    PuschParams.symIdx_dmrs = dmrs.symIdx_dmrs;  % Indicies of dmrs symbols (matlab 1 indexing). Dim: Nt_dmrs x 1
    PuschParams.Nt_dmrs = dmrs.Nt_dmrs;          % number of dmrs symbols
    PuschParams.energy = dmrs.energy;            % dmrs energy
    PuschParams.Nref = coding.Nref;
    PuschParams.I_LBRM = coding.I_LBRM;
    PuschParams.maxLayers = coding.maxLayers;
    PuschParams.maxQm = coding.maxQm;
    PuschParams.n_PRB_LBRM = coding.n_PRB_LBRM;
    PuschParams.enablePrcdBf = pdu.enablePrcdBf;
    PuschParams.PM_W = pdu.PM_W;
    PuschParams.numDmrsCdmGrpsNoData = pdu.numDmrsCdmGrpsNoData;
    PuschParams.enableTfPrcd = (pdu.TransformPrecoding == 0);   
    PuschParams.puschIdentity = pdu.puschIdentity;
    PuschParams.N_slot_frame = carrier.N_slot_frame_mu;
    PuschParams.N_symb_slot = carrier.N_symb_slot;
    PuschParams.groupOrSequenceHopping = pdu.groupOrSequenceHopping;
    PuschParams.idxUE = pdu.idxUE;
    PuschParams.idxUeg = pdu.idxUeg;
    %PUSCH parameters for UCI on PUSCH
    PuschParams.pduBitmap = pdu.pduBitmap;
    PuschParams.codeRate = coding.codeRate;
    PuschParams.nSym = alloc.nSym;                  % Total number of PUSCH symbols (data + DMRS)
    PuschParams.StartSymbolIndex = alloc.startSym;  % MATLAB 1 indexing
    PuschParams.alphaScaling = pdu.alphaScaling;
    PuschParams.betaOffsetHarqAck = pdu.betaOffsetHarqAck;
    PuschParams.betaOffsetCsi1 = pdu.betaOffsetCsi1;
    PuschParams.betaOffsetCsi2 = pdu.betaOffsetCsi2;
    % UCI payloads on PUSCH
    pusch_harqPayload = pdu.harqPayload';
    pusch_csiPart1Payload = pdu.csiPart1Payload';
    pusch_csiPart2Payload = pdu.csiPart2Payload';
    % Xtf0 = Xtf; % for generating test vectors

    [Xtf, Gtruth] = genPusch_cuphy(Xtf, Gtruth, pusch_payload, pusch_harqPayload,...
               pusch_csiPart1Payload, pusch_csiPart2Payload,PuschParams, puschTable);

    PuschParamsList{idxPdu} = PuschParams;
    pusch_payload_list{idxPdu} = pusch_payload;
    pusch_harqPayload_list{idxPdu} = pusch_harqPayload;
    pusch_csiPart1Payload_list{idxPdu} = pusch_csiPart1Payload;
    pusch_csiPart2Payload_list{idxPdu} = pusch_csiPart2Payload;

end

Xtf1 = Xtf - Xtf0; % for generating test vectors

idxSlot = carrier.idxSlotInFrame;
idxPdu = pdu.puschPduIdx-1;

if SimCtrl.genTV.enableUE && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)
    TVname = [SimCtrl.genTV.TVname, '_UE_CUPHY_s', num2str(idxSlot),...
        'p', num2str(idxPdu)];
    saveTV_pusch(SimCtrl.genTV.tvDirName, TVname, pusch_payload_list, PuschParamsList, Xtf1);
    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

return


function [Xtf, Gtruth] = genPusch_cuphy(Xtf, Gtruth, pusch_payload, pusch_harqPayload,...
               pusch_csiPart1Payload, pusch_csiPart2Payload,PuschParams, puschTable)

% load parameters
CRC = PuschParams.CRC;
C = PuschParams.C;
K = PuschParams.K;
F = PuschParams.F;
K_prime = PuschParams.K_prime;
BGN = PuschParams.BGN;   %1 or 2. Indicates which base graph used
i_LS = PuschParams.i_LS; %lifting set index
Zc = PuschParams.Zc;     %lifting size
qam = PuschParams.qam;
rvIdx = PuschParams.rvIdx;
nl = PuschParams.nl;
N_data = PuschParams.N_data;
N_id = PuschParams.N_id;
n_rnti = PuschParams.n_rnti;
qamstr = PuschParams.qamstr;
portIdx = PuschParams.portIdx;          % user's antenna ports (matlab 1 indexing). Dim: nl x 1
n_scid = PuschParams.n_scid;            % 0 or 1. User's dmrs scrambling id
nPrb = PuschParams.nPrb;               % number of prbs in allocation
startPrb = PuschParams.startPrb;       % starting prb of allocation
Nf_data = PuschParams.Nf_data;         % number of data subcarriers in allocation
Nt_data = PuschParams.Nt_data;         % number of data symbols in allocation
symIdx_data = PuschParams.symIdx_data; % indicies of data symbols. Dim: Nt_data x 1
slotNumber = PuschParams.slotNumber;
Nf = PuschParams.Nf;
Nt = PuschParams.Nt;
N_dmrs_id = PuschParams.N_dmrs_id;
symIdx_dmrs = PuschParams.symIdx_dmrs;  % Indicies of dmrs symbols (matlab 1 indexing). Dim: Nt_dmrs x 1
Nt_dmrs = PuschParams.Nt_dmrs;          % number of dmrs symbols
energy = PuschParams.energy;            % dmrs energy
Nref = PuschParams.Nref;
numDmrsCdmGrpsNoData = PuschParams.numDmrsCdmGrpsNoData;
enableTfPrcd = PuschParams.enableTfPrcd;
puschIdentity = PuschParams.puschIdentity;
N_slot_frame = PuschParams.N_slot_frame;
N_symb_slot = PuschParams.N_symb_slot;
groupOrSequenceHopping = PuschParams.groupOrSequenceHopping;    
enablePrcdBf = PuschParams.enablePrcdBf;
PM_W = PuschParams.PM_W;

% Parameters for UCI on PUSCH
codeRate = PuschParams.codeRate / 1024;
nPuschSym  = PuschParams.nSym;                  % total number of symbols (data + DMRS) allocated for PUSCH
startSym = PuschParams.StartSymbolIndex;        % MATLAB 1 indexing
pduBitmap = PuschParams.pduBitmap;              % SCF Table 3-46, supported values: 1,2,3-->UL data, UCI, UCI+ data
alphaScaling = PuschParams.alphaScaling;
betaOffsetHarqAck = PuschParams.betaOffsetHarqAck;
betaOffsetCsi1 = PuschParams.betaOffsetCsi1;
betaOffsetCsi2 = PuschParams.betaOffsetCsi2;
nBitsHarq = numel(pusch_harqPayload);          % Number of HARQ-ACK payload bits (to be encoded)
PuschParams.harqAckBitLength = nBitsHarq;
nBitsCsi1 = numel(pusch_csiPart1Payload);
PuschParams.csiPart1BitLength = nBitsCsi1;
nBitsCsi2 = numel(pusch_csiPart2Payload);
PuschParams.csiPart2BitLength = nBitsCsi2;
dataPayloadSize = numel(pusch_payload);
PuschParams.nDataBits = dataPayloadSize;

Gtruth_local = struct;

%% PUSCH Tx pipeline for various payloads
% Opt. 1: ULSCH only (pduBitmap = 1)
% Opt. 2: UCI only (pduBitmap = 2)
% Opt. 3: UCI + UL-SCH) (pduBitmap = 3)

% Estimation of rate matched sequence lengths for PUSCH payload bits
% (HARQ/CSI-1/UL-SCH)
% Assumption: no PTRS in TS 38.212 Sec. 6.3.2.4

% Rate matched sequence lengths for ULSCH and UCI (if present)

[G_harq, G_harq_rvd, G_csi1, G_csi2, G]= rateMatchSeqLenTx(nBitsHarq,nBitsCsi1,nBitsCsi2,pduBitmap,alphaScaling,...
                                         betaOffsetHarqAck,betaOffsetCsi1,betaOffsetCsi2,...
                                         nPuschSym,symIdx_data,symIdx_dmrs,nPrb,...
                                         nl,qam,C,K,codeRate,startSym,numDmrsCdmGrpsNoData);

isDataPresent = bitand(uint16(pduBitmap),uint16(2^0));
isUciPresent = bitand(uint16(pduBitmap),uint16(2^1));

if isUciPresent
% Rate matched sequence for HARQ (UCI)
    if nBitsHarq
        if nBitsHarq >11
            g_harq = uciSegPolarEncode(nBitsHarq, G_harq, pusch_harqPayload);
        else
            g_harq = uciSegRMSimplexEncode(nBitsHarq, G_harq, pusch_harqPayload, qam);
        end
    else
        g_harq = [];
    end

% Rate matched sequence for CSI Part 1 (UCI)
    if nBitsCsi1
        if nBitsCsi1 >11
            g_csi1 = uciSegPolarEncode(nBitsCsi1,G_csi1,pusch_csiPart1Payload);
        else
            g_csi1 = uciSegRMSimplexEncode(nBitsCsi1,G_csi1,pusch_csiPart1Payload,qam);
        end
    else
        g_csi1 = [];
    end

    % Rate matched sequence for CSI Part 1 (UCI)
    if nBitsCsi2
        if nBitsCsi2 >11
            g_csi2 = uciSegPolarEncode(nBitsCsi2,G_csi2,pusch_csiPart2Payload);
        else
            g_csi2 = uciSegRMSimplexEncode(nBitsCsi2,G_csi2,pusch_csiPart2Payload,qam);
        end
    else
        g_csi2 = [];
    end

    if isDataPresent
        Tb              = pusch_payload;
        TbCrc           = add_CRC_LUT(Tb, CRC, puschTable);
        TbCbs           = code_block_segment(TbCrc, C, K, F, K_prime, puschTable);
        TbCodedCbs      = LDPC_encode(TbCbs, C, K, F, BGN, i_LS, Zc, puschTable);
        g_ulsch         = rate_match(TbCodedCbs, C, qam, nl, N_data, rvIdx, BGN, Zc, Nref,G);
    else
        g_ulsch = [];
    end

    [TbRateMatCbs,~] = uciUlschMultiplexing(G_harq,G_harq_rvd,G_csi1,G_csi2,G,...
                                         g_harq,g_csi1,g_csi2,g_ulsch,...
                                         nPuschSym,symIdx_data,symIdx_dmrs,...
                                         nPrb,nl,qam,startSym,numDmrsCdmGrpsNoData,...
                                         isDataPresent);
    E = numel(TbRateMatCbs);
    [TbScramCbs]  = scramble_uci_pusch(TbRateMatCbs,E,n_rnti, N_id);
    Gtruth_local.TbScramCbs = TbScramCbs;
else
    % pipeline for payload
    Tb              = pusch_payload;
    TbCrc           = add_CRC_LUT(Tb, CRC, puschTable);
    TbCbs           = code_block_segment(TbCrc, C, K, F, K_prime, puschTable);
    TbCodedCbs      = LDPC_encode(TbCbs, C, K, F, BGN, i_LS, Zc, puschTable);
    TbRateMatCbs    = rate_match(TbCodedCbs, C, qam, nl, N_data, rvIdx, BGN, Zc, Nref,G);
    [~,TbScramCbs]  = scramble_bits(TbRateMatCbs, N_id, n_rnti);
    Gtruth_local.TbScramCbs = TbScramCbs;
end

TbLayerMapped   = layer_mapping_nr(TbScramCbs, nl, qam);
Qams            = modulate_bits(TbLayerMapped, qamstr, puschTable);
Gtruth_local.Qams = Qams;
Xtf             = embed_qams_UL(Xtf,Qams,0, nl, portIdx, n_scid,...
    nPrb, startPrb,Nf_data,Nt_data,N_data,symIdx_data, symIdx_dmrs, ...
    numDmrsCdmGrpsNoData, isDataPresent, enableTfPrcd, enablePrcdBf, PM_W);

% pipeline for DMRS
if enableTfPrcd == 1
    [r_dmrs, ~, ~] = gen_dmrs_dftsofdm(Nf, startPrb, nPrb, N_slot_frame, N_symb_slot, symIdx_dmrs, ...
    slotNumber, puschIdentity, groupOrSequenceHopping);
else
    r_dmrs = gen_dmrs_sym(slotNumber, Nf, Nt, N_dmrs_id);
end

Xtf    = embed_dmrs_UL(Xtf, r_dmrs,  nl, portIdx, n_scid, symIdx_dmrs, ...
    Nt_dmrs, energy, nPrb, startPrb, puschTable, enableTfPrcd,...
    enablePrcdBf, PM_W);

Gtruth_local.idxUE = PuschParams.idxUE;
Gtruth_local.idxUeg = PuschParams.idxUeg;
Gtruth{length(Gtruth)+1} = Gtruth_local;

return


function saveTV_pusch(tvDirName, TVname, pusch_payload_list, PuschParamsList, Xtf)

[status,msg] = mkdir(tvDirName);
h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

nPdu = length(PuschParamsList);

for idxPdu = 1:nPdu

    pusch_payload = pdsch_payload_list(idxPdu);
    pusch_payload = pusch_payload{1};
    PuschParams = PuschParamsList(idxPdu);
    PuschParams = PuschParams{1};

    switch PuschParams.CRC
        case '24A'
            PuschParams.CRC = uint32(0);
        case '24B'
            PuschParams.CRC = uint32(1);
        case '24C'
            PuschParams.CRC = uint32(2);
        case '16'
            PuschParams.CRC = uint32(3);
        case '11'
            PuschParams.CRC = uint32(4);
    end
    % PuschParams.CRC = PuschParams.CRC;
    PuschParams.C = uint32(PuschParams.C);
    PuschParams.K = uint32(PuschParams.K);
    PuschParams.F = uint32(PuschParams.F);
    PuschParams.K_prime = uint32(PuschParams.K_prime);
    PuschParams.BGN = uint32(PuschParams.BGN);   %1 or 2. Indicates which base graph used
    PuschParams.i_LS = uint32(PuschParams.i_LS); %lifting set index
    PuschParams.Zc = uint32(PuschParams.Zc);     %lifting size
    PuschParams.qam = uint32(PuschParams.qam);
    PuschParams.nl = uint32(PuschParams.nl);
    PuschParams.N_data = uint32(PuschParams.N_data);
    PuschParams.N_id = uint32(PuschParams.N_id);
    PuschParams.n_rnti = uint32(PuschParams.n_rnti);
    PuschParams.qamstr = uint32(0);

    portIdx = zeros(1, 8);
    portIdx(PuschParams.portIdx) = 1;
    portIdx = bin2dec(num2str(portIdx));
    PuschParams.portIdx = uint32(portIdx);

    PuschParams.n_scid = uint32(PuschParams.n_scid);            % 0 or 1. User's dmrs scrambling id
    PuschParams.nPrb = uint32(PuschParams.nPrb);               % number of prbs in allocation
    PuschParams.startPrb = uint32(PuschParams.startPrb);       % starting prb of allocation
    PuschParams.Nf_data = uint32(PuschParams.Nf_data);         % number of data subcarriers in allocation
    PuschParams.Nt_data = uint32(PuschParams.Nt_data);         % number of data symbols in allocation

    symIdx_data = zeros(1, 14);
    symIdx_data(PuschParams.symIdx_data) = 1;
    symIdx_data = bin2dec(num2str(symIdx_data));
    PuschParams.symIdx_data = uint32(symIdx_data);

    PuschParams.slotNumber = uint32(PuschParams.slotNumber);
    PuschParams.Nf = uint32(PuschParams.Nf);
    PuschParams.Nt = uint32(PuschParams.Nt);
    PuschParams.N_dmrs_id = uint32(PuschParams.N_dmrs_id);
    PuschParams.numDmrsCdmGrpsNoData = uint8(PuschParams.numDmrsCdmGrpsNoData);
    PuschParams.enableTfPrcd = uint32(PuschParams.enableTfPrcd);
    PuschParams.puschIdentity = uint32(PuschParams.puschIdentity);
    PuschParams.N_slot_frame = uint32(PuschParams.N_slot_frame_mu);
    PuschParams.N_symb_slot = uint32(PuschParams.N_symb_slot);
    PuschParams.groupOrSequenceHopping = uint32(PuschParams.groupOrSequenceHopping);

    symIdx_dmrs = zeros(1, 14);
    symIdx_dmrs(PuschParams.symIdx_dmrs) = 1;
    symIdx_dmrs = bin2dec(num2str(symIdx_dmrs));
    PuschParams.symIdx_dmrs = uint32(symIdx_dmrs);

    PuschParams.Nt_dmrs = uint32(PuschParams.Nt_dmrs);          % number of dmrs symbols
    PuschParams.energy = single(PuschParams.energy);            % dmrs energy
    PuschParams.Nref = uint32(PuschParams.Nref);
    PuschParams.I_LBRM = uint32(PuschParams.I_LBRM);
    PuschParams.maxLayers = uint32(PuschParams.maxLayers);
    PuschParams.maxQm = uint32(PuschParams.maxQm);
    PuschParams.n_PRB_LBRM = uint32(PuschParams.n_PRB_LBRM);
    PuschParams.enablePrcdBf = uint32(PuschParams.enablePrcdBf);
    PM_W = PuschParams.PM_W;
    PuschParams = rmfield(PuschParams, 'PM_W');

    str = strcat('PuschParams_tb',num2str(idxPdu-1));
    hdf5_write_nv(h5File, str, PuschParams);
    str = strcat('x_payload_tb',num2str(idxPdu-1));
    hdf5_write_nv(h5File, str, uint32(pusch_payload));
    str = strcat('PM_W_tb',num2str(idxPdu-1));
    hdf5_write_nv(h5File, str, single(PM_W));

end

hdf5_write_nv(h5File, 'X_tf', single(Xtf));
H5F.close(h5File);

return
