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

function Xtf = genSrs(pdu, table, carrier, Xtf, idxUE)

srsTable = table;

Xtf0 = Xtf; % for generating test vectors

% SrsParams = derive_srs_main(pdu, srsTable);
% Xtf = genSrs_cuphy_legacy(Xtf, SrsParams);

numAntPorts_mapping = [1 2 4];
SrsParams.N_ap_SRS = numAntPorts_mapping(pdu.numAntPorts+1);
numSymbols_mapping = [1 2 4];
SrsParams.N_symb_SRS = numSymbols_mapping(pdu.numSymbols+1);
numRepetitions_mapping = [1 2 4];
SrsParams.R = numRepetitions_mapping(pdu.numRepetitions+1);
combSize_mapping = [2 4];
SrsParams.K_TC = combSize_mapping(pdu.combSize+1);

SrsParams.l0 = pdu.timeStartPosition;
SrsParams.n_ID_SRS = pdu.sequenceId;
SrsParams.C_SRS = pdu.configIndex;
SrsParams.B_SRS = pdu.bandwidthIndex;
SrsParams.k_TC_bar = pdu.combOffset;
SrsParams.n_SRS_cs = pdu.cyclicShift;
SrsParams.n_RRC = pdu.frequencyPosition;
SrsParams.n_shift = pdu.frequencyShift;
SrsParams.b_hop = pdu.frequencyHopping;
SrsParams.resourceType = pdu.resourceType;
SrsParams.Tsrs = pdu.Tsrs;
SrsParams.Toffset = pdu.Toffset;
SrsParams.groupOrSequenceHopping = pdu.groupOrSequenceHopping;

SrsParams.N_slot_frame = carrier.N_slot_frame_mu;
SrsParams.N_symb_slot = carrier.N_symb_slot;
SrsParams.idxSlotInFrame = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
SrsParams.idxFrame = carrier.idxFrame;

Xtf = genSrs_cuphy(Xtf, SrsParams, srsTable);

Xtf1 = Xtf - Xtf0; % for generating test vectors

global SimCtrl;
idxSlot = carrier.idxSlotInFrame;
idxPdu = pdu.srsPduIdx-1;

if SimCtrl.genTV.enableUE && SimCtrl.genTV.cuPHY && ...
        ismember(idxSlot, SimCtrl.genTV.slotIdx)
    TVname = [SimCtrl.genTV.TVname, '_SRS_UE', num2str(idxUE-1), '_CUPHY_s', num2str(idxSlot),...
        'p', num2str(idxPdu)];
    saveTV_srs(SimCtrl.genTV.tvDirName, TVname, SrsParams, Xtf1);
    SimCtrl.genTV.idx = SimCtrl.genTV.idx + 1;
end

return;


function Xtf = genSrs_cuphy(Xtf, SrsParams, srsTable)

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

% map ZC sequence to REs
for l_prime = 0:N_symb_SRS-1
    for p = 0:N_ap_SRS-1
        freq_idx = k0(l_prime+1, p+1) + [0:K_TC:(M_sc_b_SRS-1)*K_TC];
        sym_idx = l_prime + l0;
        r = r_bar(l_prime+1,:) .* exp(1i*[0:M_sc_b_SRS-1]*alpha(p+1)); % add cyclic shift
        Xtf(freq_idx+1, sym_idx+1, p+1) = r;
    end
end


return


function saveTV_srs(tvDirName, TVname, SrsParams, Xtf)

[status,msg] = mkdir(tvDirName);

SrsParams = formatU32Struct(SrsParams);

h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_nv(h5File, 'SrsParams', SrsParams);
hdf5_write_nv(h5File, 'X_tf', single(Xtf));
hdf5_write_nv(h5File, 'X_tf_fp16', complex(single(Xtf)),'fp16');
H5F.close(h5File);

return


function Xtf = genSrs_cuphy_legacy(Xtf,SrsParams)

% function computes and embeds srs for a single comb.

%inputs:
% s_tx    --> srs transmitted by all users. Dim: 4 x Nf x nSym x nRes
% pdu --> pdu for current comb

%output:
% s_tx   --> updated srs signal

%PARAMATERS


%time/frequency allocation:
startPrb = SrsParams.startPrb;   % start srs prb. 0-272
nPrb =  SrsParams.nPrb;          % number of srs prbs. 0-272
startSym = SrsParams.startSym;   % starting srs symbol. 0-5. (Within all srs symbols! Not slot)
nSym = SrsParams.nSym;           % number of srs symbols. 1, 2, or 4

%comb structure:
combSize = SrsParams.combSize;      % scomb spacing. 2 or 4
nScPrb = SrsParams.nScPrb;          % number of srs subcarriers per Prb. 6 or 3
combOffset = SrsParams.combOffset;  % offset of comb from 0th subcarrier or 0th srs prb
% 0->1 for combSize 2. 0->3 for combSize 4.

%sequence:
N_zc = SrsParams.N_zc;                % Zadoff-Chu prime
q = SrsParams.q;                      % Zadoff-Chu sequence number
cyclicShift = SrsParams.cyclicShift;  % cyclic shift offset. 0-7 for combSize 2. 0-11 for combSize 4.
numAntPorts = SrsParams.numAntPorts;  % number of antenna ports muxed in comb. 1,2, or 4.

%storage:
portMapping = 0: (numAntPorts-1); % mapping from comb antenna ports to srs antenna ports.

%GENERATE ZC

n_srs = nPrb*nScPrb;

idx = 0 : (n_srs - 1);
idx = mod(idx,N_zc).';

ZC = exp(-1i*pi*q*idx.*(idx+1) / N_zc);


%GENERATE SRS SEQUENCE

s = zeros(n_srs,numAntPorts);

idx = 0 : (n_srs - 1);
idx = idx';

if combSize == 2
    cs_max = 8;
else
    cs_max = 12;
end

for p = 0 : (numAntPorts - 1)
    cs = mod(cyclicShift + cs_max*p/numAntPorts,cs_max);
    alpha = 2*pi*cs/cs_max;
    
    s(:,p+1) = exp(1i*alpha*idx).*ZC;
end

%EMBED SRS SEQUENCE

% freq indicies:
freqIdx = 0 : combSize : (12*nPrb-1);
freqIdx = 12*startPrb + freqIdx + combOffset;

% time indicies:
timeIdx = startSym : (startSym + nSym - 1);

% % repeate signal:
% s = permute(s,[2 1]);          % now: numAntPorts x Nf_srs
% s = repmat(s,nSym,1);        % now: numAntPorts x Nf_srs x nSym
%
% % embed:
% Xtf(freqIdx + 1,timeIdx + 1, portMapping + 1) = s.';


% repeate signal:
s1(:,1,:) = s;
s = repmat(s1, [1, nSym,1]);        % now: Nf_srs x nSym x numAntPorts

% embed:
Xtf(freqIdx + 1,timeIdx + 1, portMapping + 1) = s;

return


function saveTV_srs_legacy(tvDirName, TVname, SrsParams, Xtf)

[status,msg] = mkdir(tvDirName);

SrsParams.startPrb = uint32(SrsParams.startPrb);   % start srs prb. 0-272
SrsParams.nPrb = uint32(SrsParams.nPrb);          % number of srs prbs. 0-272
SrsParams.startSym = uint32(SrsParams.startSym);   % starting srs symbol. 0-5. (Within all srs symbols! Not slot)
SrsParams.nSym = uint32(SrsParams.nSym);           % number of srs symbols. 1, 2, or 4
SrsParams.combSize = uint32(SrsParams.combSize);      % scomb spacing. 2 or 4
SrsParams.nScPrb = uint32(SrsParams.nScPrb);          % number of srs subcarriers per Prb. 6 or 3
SrsParams.combOffset = uint32(SrsParams.combOffset);  % offset of comb from 0th subcarrier or 0th srs prb
SrsParams.N_zc = uint32(SrsParams.N_zc);                % Zadoff-Chu prime
SrsParams.q = uint32(SrsParams.q);                      % Zadoff-Chu sequence number
SrsParams.cyclicShift = uint32(SrsParams.cyclicShift);  % cyclic shift offset. 0-7 for combSize 2. 0-11 for combSize 4.
SrsParams.numAntPorts = uint32(SrsParams.numAntPorts);  % number of antenna ports muxed in comb. 1,2, or 4.
SrsParams = rmfield(SrsParams, 'portMapping'); % mapping from comb antenna ports to srs antenna ports.
SrsParams = rmfield(SrsParams, 'resIdx');

h5File  = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_nv(h5File, 'SrsParams', SrsParams);
hdf5_write_nv(h5File, 'X_tf', single(Xtf));
H5F.close(h5File);

return
