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

function [pdsch_payload_list, Xtf_remap] = detPdsch(pduList, table, carrier, Xtf, csirsPduList)

global SimCtrl;


[nSc, nSym, ~] = size(Xtf);
Xtf_remap = zeros(nSc, nSym);
Xtf_remap_trsnzp = zeros(nSc, nSym);
[Xtf_remap, Xtf_remap_trsnzp] = gen_csirs_remap(csirsPduList, Xtf_remap, Xtf_remap_trsnzp, table);

pdschTable = table;
nPdu = length(pduList);

PdschParamsList = [];
pdsch_payload_list = [];

lastPdu = pduList{nPdu};
nUeg = lastPdu.idxUeg + 1;
currentUeg = 0;
for idxUeg = 1:nUeg
    UegList{idxUeg}.idxPdu = [];
    UegList{idxUeg}.nlUeg = 0;
end
for idxPdu = 1:nPdu
    pdu = pduList{idxPdu};
    if pdu.idxUeg == currentUeg
        UegList{currentUeg+1}.idxPdu = [UegList{currentUeg+1}.idxPdu, idxPdu-1];
        UegList{currentUeg+1}.nlUeg = UegList{currentUeg+1}.nlUeg + pdu.nrOfLayers;
    else
        UegList{currentUeg+1+1}.idxPdu = [UegList{currentUeg+1+1}.idxPdu, idxPdu-1];
        UegList{currentUeg+1+1}.nlUeg = UegList{currentUeg+1+1}.nlUeg + pdu.nrOfLayers;
        currentUeg = currentUeg + 1;
    end
end

for idxUeg = 1:nUeg
    nUeInUeg = length(UegList{idxUeg}.idxPdu);
    PdschParamsUeg{idxUeg}.PdschParams = [];
    pdsch_payload_ueg = [];
    for idxUeInUeg = 1:nUeInUeg
        idxPdu = UegList{idxUeg}.idxPdu(idxUeInUeg);
        pdu = pduList{idxPdu+1};

%         pdsch_payload_ueg{idxUeInUeg} = pdu.payload';

        % derive configuration
        dmrs    = derive_dmrs_main(pdu, pdschTable);
        alloc   = derive_alloc_main(pdu, dmrs, Xtf_remap);
        coding  = derive_coding_main(pdu, alloc, pdschTable);

        % load parameters
        PdschParams.CRC = coding.CRC;
        PdschParams.C = coding.C;
        PdschParams.K = coding.K;
        PdschParams.F = coding.F;
        PdschParams.K_prime = coding.K_prime;
        PdschParams.BGN = coding.BGN;   %1 or 2. Indicates which base graph used
        PdschParams.i_LS = coding.i_LS; %lifting set index
        PdschParams.Zc = coding.Zc;     %lifting size
        PdschParams.qam = coding.qam;
        PdschParams.rvIdx = coding.rvIdx;
        PdschParams.nl = alloc.nl;
        PdschParams.N_data = alloc.N_data;
        PdschParams.N_data_used = alloc.N_data_used;
        PdschParams.N_id = alloc.dataScramblingId;
        PdschParams.n_rnti = alloc.RNTI;
        PdschParams.qamstr = coding.qamstr;
        PdschParams.portIdx = alloc.portIdx;          % user's antenna ports (matlab 1 indexing). Dim: nl x 1
        PdschParams.n_scid = alloc.SCID;            % 0 or 1. User's dmrs scrambling id
        PdschParams.nlAbove16 = pdu.nlAbove16;
        PdschParams.resourceAlloc = alloc.resourceAlloc;
        PdschParams.rbBitmap = alloc.rbBitmap;
        PdschParams.nPrb = alloc.nPrb;               % number of prbs in allocation
        PdschParams.startPrb = alloc.startPrb;       % starting prb of allocation
        PdschParams.BWPStart = alloc.BWPStart;
        PdschParams.refPoint = pdu.refPoint;
        PdschParams.Nf_data = alloc.Nf_data;         % number of data subcarriers in allocation
        PdschParams.Nt_data = alloc.Nt_data;         % number of data symbols in allocation
        PdschParams.symIdx_data = alloc.symIdx_data; % indicies of data symbols. Dim: Nt_data x 1
        PdschParams.slotNumber = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
        PdschParams.Nf = carrier.N_sc;
        PdschParams.Nt = carrier.N_symb_slot;
        PdschParams.N_dmrs_id = dmrs.DmrsScramblingId;
        PdschParams.symIdx_dmrs = dmrs.symIdx_dmrs;  % Indicies of dmrs symbols (matlab 1 indexing). Dim: Nt_dmrs x 1
        PdschParams.Nt_dmrs = dmrs.Nt_dmrs;          % number of dmrs symbols
        PdschParams.energy = dmrs.energy;            % dmrs energy

        % Tx power based on SCF-FAPIv2
        pdsch2csirs = pdu.powerControlOffset - 8;
        csirs2ssb = (pdu.powerControlOffsetSS-1) * 3;
        pdsch2ssb = pdsch2csirs + csirs2ssb;
        PdschParams.beta_qam = 10^(pdsch2ssb/20);
        PdschParams.beta_dmrs = 10^(pdsch2ssb/20);

        PdschParams.Nref = coding.Nref;
        PdschParams.I_LBRM = coding.I_LBRM;
        PdschParams.maxLayers = coding.maxLayers;
        PdschParams.maxQm = coding.maxQm;
        PdschParams.n_PRB_LBRM = coding.n_PRB_LBRM;
        PdschParams.testModel = pdu.testModel;
        PdschParams.enablePrcdBf = pdu.enablePrcdBf;
        PdschParams.PM_W = pdu.PM_W;
        PdschParams.idxUeg = idxUeg-1;

        % Add fields to generate TVs compatible to cuPHY
        PdschParams.mcsTable = pdu.mcsTable;
        PdschParams.mcsIndex = pdu.mcsIndex;
        PdschParams.startSym = alloc.startSym;
        PdschParams.nSym = alloc.nSym;
        PdschParams.N_ID_CELL = carrier.N_ID_CELL;
        PdschParams.numRxAnt = carrier.numRxAnt;
        PdschParams.numTxAnt = carrier.numTxAnt;
        PdschParams.N_grid_size_mu = carrier.N_grid_size_mu;
        PdschParams.N_grid_size_mu = carrier.N_grid_size_mu;
        PdschParams.mu = carrier.mu;
        PdschParams.numDmrsCdmGrpsNoData = pdu.numDmrsCdmGrpsNoData;
        PdschParams.AdditionalPosition = dmrs.AdditionalPosition;
        PdschParams.maxLength = dmrs.maxLength;
        PdschParamsUeg{idxUeg}.PdschParams = [PdschParamsUeg{idxUeg}.PdschParams, PdschParams];
    end

    [pdsch_payload_ueg] = detPdsch_cuphy(Xtf, PdschParamsUeg{idxUeg}, pdschTable, Xtf_remap);

    for idxUeInUeg = 1:nUeInUeg
        idxPdu = UegList{idxUeg}.idxPdu(idxUeInUeg)+1;
        PdschParamsList{idxPdu} = PdschParamsUeg{idxUeg}.PdschParams(idxUeInUeg);
        pdsch_payload_list{idxPdu} = pdsch_payload_ueg{idxUeInUeg};
    end
end

idxSlot = carrier.idxSlotInFrame;
idxPdu = pdu.pdschPduIdx-1;

if SimCtrl.genTV.enableUE && SimCtrl.genTV.enable && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)
    if contains(SimCtrl.genTV.TVname, 'TV_cuphy')
        TVname = SimCtrl.genTV.TVname;
    else
        TVname = [SimCtrl.genTV.TVname, '_PDSCH_UE_CUPHY_s', num2str(idxSlot),...
            'p', num2str(idxPdu)];
    end
    saveTV_pdsch_cuphy(SimCtrl.genTV.tvDirName, TVname, UegList, pdsch_payload_list, ...
        PdschParamsList, Xtf, Xtf_remap, csirsPduList);
    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

return


function [pdsch_payload_ueg] = detPdsch_cuphy(Xtf, PdschParamsUeg, pdschTable, Xtf_remap)

global SimCtrl;

nUe = length(PdschParamsUeg.PdschParams);

for idxUe = 1:nUe
    PdschParams = PdschParamsUeg.PdschParams(idxUe);

    % load parameters
    CRC_TYPE = PdschParams.CRC;
    C = PdschParams.C;
    K = PdschParams.K;
    F = PdschParams.F;
    K_prime = PdschParams.K_prime;
    BGN = PdschParams.BGN;   %1 or 2. Indicates which base graph used
    i_LS = PdschParams.i_LS; %lifting set index
    Zc = PdschParams.Zc;     %lifting size
    qam = PdschParams.qam;
    rvIdx = PdschParams.rvIdx;
    nl = PdschParams.nl;
    N_data = PdschParams.N_data;
    N_data_used = PdschParams.N_data_used;
    N_id = PdschParams.N_id;
    n_rnti = PdschParams.n_rnti;
    qamstr = PdschParams.qamstr;
    portIdx = PdschParams.portIdx;          % user's antenna ports (matlab 1 indexing). Dim: nl x 1
    n_scid = PdschParams.n_scid;            % 0 or 1. User's dmrs scrambling id
    nlAbove16 = PdschParams.nlAbove16;
    resourceAlloc = PdschParams.resourceAlloc;
    rbBitmap = PdschParams.rbBitmap;
    nPrb = PdschParams.nPrb;               % number of prbs in allocation
    startPrb = PdschParams.startPrb;       % starting prb of allocation
    BWPStart = PdschParams.BWPStart;
    refPoint = PdschParams.refPoint;
    Nf_data = PdschParams.Nf_data;         % number of data subcarriers in allocation
    Nt_data = PdschParams.Nt_data;         % number of data symbols in allocation
    symIdx_data = PdschParams.symIdx_data; % indicies of data symbols. Dim: Nt_data x 1
    slotNumber = PdschParams.slotNumber;
    Nf = PdschParams.Nf;
    Nt = PdschParams.Nt;
    N_dmrs_id = PdschParams.N_dmrs_id;
    symIdx_dmrs = PdschParams.symIdx_dmrs;  % Indicies of dmrs symbols (matlab 1 indexing). Dim: Nt_dmrs x 1
    Nt_dmrs = PdschParams.Nt_dmrs;          % number of dmrs symbols
    energy = PdschParams.energy;            % dmrs energy
    beta_qam = PdschParams.beta_qam;
    beta_dmrs = PdschParams.beta_dmrs;
    Nref = PdschParams.Nref;
    numDmrsCdmGrpsNoData = PdschParams.numDmrsCdmGrpsNoData;
    enablePrcdBf = PdschParams.enablePrcdBf;
    PM_W = PdschParams.PM_W;
    testModel = PdschParams.testModel;
    AdditionalPosition = PdschParams.AdditionalPosition;

    % channel estimation based on dmrs
    r_dmrs = gen_dmrs_sym(slotNumber, Nf, Nt, N_dmrs_id);
    r_dmrs_scaled = r_dmrs / beta_dmrs;
    dmrs_ch    = chest_dmrs_DL(Xtf, r_dmrs_scaled,  nl, portIdx, n_scid, symIdx_dmrs, ...
        Nt_dmrs, energy, nPrb, BWPStart, startPrb, pdschTable, refPoint, AdditionalPosition, resourceAlloc, rbBitmap);
    eq_coef = derive_DL_eq(dmrs_ch, symIdx_data, symIdx_dmrs, numDmrsCdmGrpsNoData);

    % channel equalization on data
    Qams_eq             = eq_qams_DL(Xtf, nl, portIdx, n_scid,...
        nPrb, BWPStart, startPrb,N_data_used,symIdx_data, symIdx_dmrs, ...
        numDmrsCdmGrpsNoData, Xtf_remap, eq_coef, resourceAlloc, rbBitmap);
    Qams_eq = Qams_eq /beta_qam;
    if SimCtrl.enable_snapshot_gNB_UE_into_SimCtrl && SimCtrl.enable_EVM_calculation
        tx_Qams = reshape(SimCtrl.gNBUE_snapshot.Phy.tx.Xtf(:,symIdx_data,:),[],1);
        evm = sqrt(sum(abs(Qams_eq-tx_Qams).^2))/sqrt(sum(abs(tx_Qams).^2));
        SimCtrl.results.pdsch{idxUe}.evm = [SimCtrl.results.pdsch{idxUe}.evm, evm];
    end
    Ree = 1e-4;
    LLRseq = softDemapper_DL(Qams_eq, Ree, pdschTable, nl, Nt_data, Nf, nPrb, startPrb, ...
        qam, n_scid);

    [~,LLR_descr] = descramble_bits(LLRseq, N_id, n_rnti);
    
    if testModel % bypass derateMatch/LDCP/CRC
        Tb_est = (1-sign(LLR_descr))/2;
        tbErr = 0;
        cbErr = 0;
    else        
        G = N_data_used * qam * nl;
        [derateCbs, nV_parity, derateCbsIndices, derateCbsIndicesSizes] = derate_match(LLR_descr, BGN, C, qam, K, F, K_prime, Zc, nl, N_data, rvIdx, Nref,G);
        
        [TbCbs_est, numItr]    = LDPC_decode(derateCbs,nV_parity,pdschTable, Zc, C, BGN, i_LS, K_prime, CRC_TYPE, SimCtrl.alg.LDPC_maxItr);
        [TbCrc_est, cbErr]     = block_desegment(TbCbs_est, C, K_prime, pdschTable);
        [Tb_est, tbErr]        = CRC_decode(TbCrc_est, CRC_TYPE, pdschTable);        
    end
    pdsch_payload_ueg{idxUe}.Tb = Tb_est;
    pdsch_payload_ueg{idxUe}.tbErr = tbErr;
    pdsch_payload_ueg{idxUe}.cbErr = cbErr;
end

return


function saveTV_pdsch_cuphy(tvDirName, TVname, UegList, pdsch_payload_list, ...
    PdschParamsList, Xtf, Xtf_remap, csirsPduList)
global SimCtrl;

[status,msg] = mkdir(tvDirName);
h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

nPdu = length(PdschParamsList);
nUeg = length(UegList);
InputData = [];
tbCrc_buffer = zeros(nPdu,1);
dmrs_pars = [];
ueGrp_pars = [];
ue_pars = [];
offset = uint32(0);

nCsirs = length(csirsPduList);
csirs_pars = [];
for idxCsirs = 1:nCsirs
    thisCsirs = csirsPduList{idxCsirs};
    csirs_pars(idxCsirs).Row = uint8(thisCsirs.Row);
    csirs_pars(idxCsirs).FreqDensity = uint8(thisCsirs.FreqDensity);
    csirs_pars(idxCsirs).StartRB = uint16(thisCsirs.StartRB);
    csirs_pars(idxCsirs).NrOfRBs = uint16(thisCsirs.NrOfRBs);
    csirs_pars(idxCsirs).SymbL0 = uint8(thisCsirs.SymbL0);
    csirs_pars(idxCsirs).SymbL1 = uint8(thisCsirs.SymbL1);
    csirs_pars(idxCsirs).FreqDomain = uint16(thisCsirs.FreqDomain);
end

for idxUeg = 1:nUeg
    ueGrp_pars(idxUeg).cellIdx = uint16(0);
    idxPdu = UegList{idxUeg}.idxPdu(1)+1;
    ueGrp_pars(idxUeg).dmrsIdx = idxUeg-1;
    startPrb = PdschParamsList{idxPdu}.startPrb;
    ueGrp_pars(idxUeg).startPrb = uint16(startPrb-1);
    nPrb = PdschParamsList{idxPdu}.nPrb;
    ueGrp_pars(idxUeg).nPrb = uint16(nPrb);
    ueGrp_pars(idxUeg).nUes = uint16(length(UegList{idxUeg}.idxPdu));
    ueGrp_pars(idxUeg).UePrmIdxs = uint16(UegList{idxUeg}.idxPdu);

    % add dmrsSymLocBmsk, pdschStartSym and nPdschSym to UE group level
    symIdx_dmrs = zeros(1, 14);
    symIdx_dmrs(PdschParamsList{idxPdu}.symIdx_dmrs) = 1;
    symIdx_dmrs = fliplr(symIdx_dmrs);
    symIdx_dmrs = bin2dec(num2str(symIdx_dmrs));
    ueGrp_pars(idxUeg).dmrsSymLocBmsk = uint16(symIdx_dmrs);
    ueGrp_pars(idxUeg).pdschStartSym = uint8(PdschParamsList{idxPdu}.startSym-1);
    ueGrp_pars(idxUeg).nPdschSym = uint8(PdschParamsList{idxPdu}.nSym);

    PdschParams = PdschParamsList(idxPdu);
    PdschParams = PdschParams{1};
    if SimCtrl.relNum == 2240
        dmrs_pars(idxUeg).dmrsAddlnPos = uint8(PdschParams.AdditionalPosition);
        dmrs_pars(idxUeg).dmrsMaxLen = uint8(PdschParams.maxLength);
    end
    dmrs_pars(idxUeg).nDmrsCdmGrpsNoData = uint8(PdschParams.numDmrsCdmGrpsNoData);
    dmrs_pars(idxUeg).dmrsScramId = uint16(PdschParams.N_dmrs_id);

end

for idxPdu = 1:nPdu
    pdsch_payload = pdsch_payload_list(idxPdu);
    pdsch_payload = pdsch_payload{1};
    PdschParams = PdschParamsList(idxPdu);
    PdschParams = PdschParams{1};

    ue_pars(idxPdu).ueGrpIdx = uint16(PdschParams.idxUeg);
    ue_pars(idxPdu).scid = uint8(PdschParams.n_scid);
    ue_pars(idxPdu).nUeLayers = uint8(PdschParams.nl);

    rel_22_4_compat_nPortIndex = dec2bin(0,32);
    dmrs_port_bitmask = 0;
    nl = PdschParams.nl;
    portIdx = PdschParams.portIdx;
    for i = 1 : nl
        b = portIdx(i) - 1;
        dmrs_port_bitmask = bitor(dmrs_port_bitmask, bitshift(1, b));
        rel_22_4_compat_b = dec2bin(portIdx(i) - 1,4);
        rel_22_4_compat_nPortIndex((i-1)*4 + 1 : i*4) = rel_22_4_compat_b;
    end
    %if SimCtrl.relNum == 2240
        rel_22_4_compat_nPortIndex = bin2dec(rel_22_4_compat_nPortIndex);
        ue_pars(idxPdu).nPortIndex = uint32(rel_22_4_compat_nPortIndex);
    %else
        ue_pars(idxPdu).dmrsPortBmsk = uint16(dmrs_port_bitmask);
    %end

    ue_pars(idxPdu).rnti = uint16(PdschParams.n_rnti);
    ue_pars(idxPdu).dataScramId = uint16(PdschParams.N_id);
    ue_pars(idxPdu).nCw = uint8(1); % only support 1 now
    ue_pars(idxPdu).CwIdxs = uint16(idxPdu-1);
    ue_pars(idxPdu).enablePrcdBf = uint16(PdschParams.enablePrcdBf);
    ue_pars(idxPdu).beta_qam = single(PdschParams.beta_qam);
    ue_pars(idxPdu).beta_dmrs = single(PdschParams.beta_dmrs);
    ue_pars(idxPdu).refPoint = uint8(PdschParams.refPoint);
    ue_pars(idxPdu).BWPStart = uint16(PdschParams.BWPStart);

    cw_pars(idxPdu).ueIdx = uint16(idxPdu-1);
    cw_pars(idxPdu).mcsTableIndex = uint8(PdschParams.mcsTable);
    cw_pars(idxPdu).mcsIndex = uint8(PdschParams.mcsIndex);
    cw_pars(idxPdu).rv = uint8(PdschParams.rvIdx);
    cw_pars(idxPdu).tbStartOffset = uint32(offset);
    cw_pars(idxPdu).tbSize = uint32(length(pdsch_payload)/8);
    cw_pars(idxPdu).I_LBRM = uint8(PdschParams.I_LBRM);
    cw_pars(idxPdu).maxLayers = uint8(PdschParams.maxLayers);
    cw_pars(idxPdu).maxQm = uint8(PdschParams.maxQm);
    cw_pars(idxPdu).n_PRB_LBRM = uint16(PdschParams.n_PRB_LBRM);

    tb_idx = idxPdu-1;
    str = strcat('tb',num2str(tb_idx),'_payload');
    hdf5_write_nv(h5File, str, uint8(uint8_convert(pdsch_payload_list{idxPdu}.Tb, 0)));
    str = strcat('tb',num2str(tb_idx),'_tbErr');
    hdf5_write_nv(h5File, str, uint8(pdsch_payload_list{idxPdu}.tbErr));
    str = strcat('tb',num2str(tb_idx),'_cbErr');
    hdf5_write_nv(h5File, str, uint8(pdsch_payload_list{idxPdu}.cbErr));
    str = strcat('tb',num2str(tb_idx),'_PM_W');
    hdf5_write_nv(h5File, str, single(PdschParams.PM_W), 'fp16');
end

cellStat_pars.phyCellId = uint16(PdschParams.N_ID_CELL);
cellStat_pars.nRxAnt = uint16(PdschParams.numRxPort);
cellStat_pars.nTxAnt = uint16(PdschParams.numTxPort);
cellStat_pars.nPrbUlBwp = uint16(PdschParams.N_grid_size_mu);
cellStat_pars.nPrbDlBwp = uint16(PdschParams.N_grid_size_mu);
cellStat_pars.mu = uint8(PdschParams.mu);
% The following beta_dmrs and beta_qam should be moved into ue_pars
cellStat_pars.beta_dmrs = single(1);
cellStat_pars.beta_qam = single(1);

cellDyn_pars.cellStatIdx = uint16(0);
cellDyn_pars.cellDynIdx = uint16(0);
cellDyn_pars.slotNum = uint16(PdschParams.slotNumber);
cellDyn_pars.testModel = uint8(PdschParams.testModel);

% dmrsSymLocBmsk, pdschStartSym and nPdschSym are also included in
% UE group level structure dmrs_pars.
% Only when these fields are zeros in cellDyn_pars, we use the
% fields in dmrs_pars
cellDyn_pars.pdschStartSym = uint8(0); % uint8(PdschParams.startSym-1);
cellDyn_pars.nPdschSym = uint8(0); % uint8(PdschParams.nSym);
symIdx_dmrs = zeros(1, 14);
symIdx_dmrs(PdschParams.symIdx_dmrs) = 1;
symIdx_dmrs = fliplr(symIdx_dmrs);
symIdx_dmrs = bin2dec(num2str(symIdx_dmrs));
cellDyn_pars.dmrsSymLocBmsk = uint16(0); % uint16(symIdx_dmrs);
cellDyn_pars.nCsirsPrms = uint16(nCsirs); % number of CSIRS params
cellDyn_pars.csirsPrmsOffset = uint16(0); % starting index of CSIRS params

cellGrpDyn_pars.nCells = uint16(1);
cellGrpDyn_pars.nUeGrps = uint16(nUeg);
cellGrpDyn_pars.nUes = uint16(nPdu);
cellGrpDyn_pars.nCws = uint16(nPdu);

hdf5_write_nv_exp(h5File, 'cellStat_pars', cellStat_pars);
hdf5_write_nv_exp(h5File, 'dmrs_pars', dmrs_pars);
hdf5_write_nv_exp(h5File, 'cellDyn_pars', cellDyn_pars);
hdf5_write_nv_exp(h5File, 'ueGrp_pars', ueGrp_pars);
hdf5_write_nv_exp(h5File, 'ue_pars', ue_pars);
hdf5_write_nv_exp(h5File, 'cw_pars', cw_pars);
hdf5_write_nv_exp(h5File, 'cellGrpDyn_pars', cellGrpDyn_pars);
hdf5_write_nv_exp(h5File, 'csirs_pars', csirs_pars);

hdf5_write_nv_exp(h5File, 'Xtf', Xtf);
hdf5_write_nv_exp(h5File, 'Xtf_remap', uint32(Xtf_remap));

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
