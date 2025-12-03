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

function coding = derive_coding_main(pdu, alloc, pdschTable)

% lookup_mcs_table
if pdu.qamModOrder == 1
    pi2BPSK = 1;
else
    pi2BPSK = 0;
end
mcs = pdu.mcsIndex;
mcsTable = pdu.mcsTable;

switch(mcsTable)
    case 0
        mcs_table = pdschTable.McsTable1;
    case 1
        mcs_table = pdschTable.McsTable2;
    case 2
        mcs_table = pdschTable.McsTable3;
    case 3
        mcs_table = pdschTable.McsTable4;
    case 4
        mcs_table = pdschTable.McsTable5;
end

if mcs == 100
    qam = 2;
    codeRate = 1024;
elseif (ismember(mcsTable, [1, 3, 4]) && mcs > 27) || ...
        (ismember(mcsTable, [0, 2]) && mcs > 28)
    qam = pdu.qamModOrder;
    codeRate = pdu.targetCodeRate/10;
else
    if pi2BPSK
        factor = 2;
    else
        factor = 1;
    end
    qam = mcs_table(mcs+1, 2)/factor;
    codeRate = mcs_table(mcs+1, 3)*factor;
end

switch qam
    case 1
        qamstr = 'pi/2-BPSK';
    case 2
        qamstr = 'QPSK';
    case 4
        qamstr = '16QAM';
    case 6
        qamstr = '64QAM';
    case 8
        qamstr = '256QAM';
end

coding.mcsTable = pdu.mcsTable;
coding.mcs = pdu.mcsIndex;
coding.qamstr = qamstr;
coding.qam = qam;
coding.codeRate = codeRate;
codeRate = codeRate/1024;
% derive_TB_size (TS 38.214 5.1.3.2)
nPrb = alloc.nPrb;
N_data = alloc.N_data;
nl = alloc.nl;

TBS_table = pdschTable.TBS_table;

%compute number of avaliable TF resources avaliable to the UE
Nre = min(156, N_data / nPrb) * nPrb;

%approximate number information bits (given code rate, qam, layers)
Ninfo = Nre * codeRate * qam * nl;

if Ninfo <= 3824
    %for "small" sizes, look up TBS in a table. First round the
    %number of information bits.
    n = max(3,(floor(log2(Ninfo)) - 6));
    Ninfo_prime = max(24, 2^n*floor(Ninfo / 2^n));

    %next lookup in table closest TBS (without going over).
    compare = Ninfo_prime - TBS_table;
    compare(compare > 0) = -100000;
    [~,max_index] = max(compare);
    TBS = TBS_table(max_index);
    C = 1;
else
    %for "large" sizes, compute TBS. First round the number of
    %information bits to a power of two.
     n = floor(log2(Ninfo-24)) - 5;
     Ninfo_prime = max(3840, 2^n*round((Ninfo-24)/2^n));

    %Next, compute the number of code words. For large code rates,
    %use base-graph 1. For small code rate use base-graph 2.
    if codeRate < 1/4
        C = ceil( (Ninfo + 24) / 3816);
        TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
    else
        if Ninfo_prime > 8424
            C = ceil( (Ninfo_prime + 24) / 8424);
            TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
        else
            C = 1;
            TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
        end
    end
end

if (ismember(pdu.mcsTable, [1, 3, 4]) && pdu.mcsIndex > 27) || ...
        (ismember(pdu.mcsTable, [0, 2]) && pdu.mcsIndex > 28)
    TBS = pdu.TBSize * 8;
end

coding.TBS = TBS;

% derive_BGN (TS 38.212 7.2.2)
if (TBS <= 292) || ((TBS <= 3824) && (codeRate <= 0.67)) || (codeRate <= 0.25)
    BGN = 2;
else
    BGN = 1;
end
coding.BGN = BGN;

% derive_CRC (TS 38.212 7.2.1)
if TBS > 3824
    CRC = '24A';
    B = TBS + 24;
else
    CRC = '16';
    B = TBS + 16;
end

coding.CRC = CRC;
coding.B = B;

% derive lifting

%DERIVE C

%derive max number of bits per codeblock (TS 38.212 5.2.2)
if BGN == 1
    K_cb = 8448;
else
    K_cb = 3840;
end

%derive number of codeblocks
if B <= K_cb
    C = 1;
    B_prime = B;
else
    L = 24; %CRC bits per code block
    C = ceil(B / (K_cb - L)); %number of code blocks
    B_prime = B + C*L; %total number of bits
end

%bits per code block:
K_prime = B_prime / C;

%number of systematic information bits:
if BGN == 1
    K_b = 22;
else
    if B > 640
        K_b = 10;
    else
        if B > 560
            K_b = 9;
        else
            if B > 192
                K_b = 8;
            else
                K_b = 6;
            end
        end
    end
end


%DERIVE Zc

Z = [2, 4, 8, 16, 32, 64, 128, 256,...
    3, 6, 12, 24, 48, 96, 192, 384,...
    5, 10, 20, 40, 80, 160, 320,...
    7, 14, 28, 56, 112, 224, ...
    9, 18, 36, 72, 144, 288,...
    11, 22, 44, 88, 176, 352,...
    13, 26, 52, 104, 208,...
    15, 30, 60, 120, 240];

%find smallest Z such that Z*K_b >= K_prime:
Diff = Z*K_b - K_prime;
Diff(Diff < 0) = Inf;

[~,index] = min(Diff);
Zc = Z(index);

global SimCov;
if (isstruct(SimCov) && SimCov.doCodeCoverage == 1)
    SimCov.pxsch.Zc_finder.count(index) = SimCov.pxsch.Zc_finder.count(index) + 1;
    % Log statistics of lifting sizes
    if (SimCov.pxsch.Zc_finder.count(index) == 1)
        SimCov.pxsch.Zc_finder.mcsTable(index) = pdu.mcsTable;
        SimCov.pxsch.Zc_finder.mcsIndex(index) = pdu.mcsIndex;
        SimCov.pxsch.Zc_finder.nrb(index) = pdu.rbSize;
        SimCov.pxsch.Zc_finder.nsym(index) = pdu.NrOfSymbols;
    end

    % Display lifting size stats if enabled
    if (0)
        display(['Index: ',num2str(index),'   Zc: ',num2str(Zc)]);
        display(['K_b: ',num2str(K_b)]);
        display(['Found:', num2str(sum(SimCov.pxsch.Zc_finder.nrb ~= 0))]);
        display(['Missing:', num2str(sum(SimCov.pxsch.Zc_finder.nrb == 0))]);
    end
end

%LIFTING SET

if (1 <= index) && (index <= 8)
    i_LS = 1;
end

if (9 <= index) && (index <= 16)
    i_LS = 2;
end

if (17 <= index) && (index <= 23)
    i_LS = 3;
end

if (24 <= index) && (index <= 29)
    i_LS = 4;
end

if (30 <= index) && (index <= 35)
    i_LS = 5;
end

if (36 <= index) && (index <= 41)
    i_LS = 6;
end

if (42 <= index) && (index <= 46)
    i_LS = 7;
end

if (47 <= index) && (index <= 51)
    i_LS = 8;
end

%number of systematic bits:
if BGN == 1
    K = Zc*22;
else
    K = Zc*10;
end

%number of filler bits:
F = K - K_prime;

% calculate Nref
maxLayers = min(pdu.maxLayers, 4);
maxQm = pdu.maxQm;
n_PRB_LBRM = pdu.n_PRB_LBRM;
if pdu.I_LBRM
    R_LBRM = 2/3;
    maxRate = 948/1024;
    N_RE_LBRM = 156*n_PRB_LBRM;
    Ninfo_LBRM = N_RE_LBRM*maxQm*maxRate*maxLayers;
    n_LBRM = floor(log2(Ninfo_LBRM-24)) - 5;
    Ninfo_prime_LBRM = max(3840, 2^n_LBRM*round((Ninfo_LBRM-24)/2^n_LBRM));
    C_LBRM = ceil( (Ninfo_prime_LBRM + 24) / 8424);
    TBS_LBRM = 8*C_LBRM*ceil( (Ninfo_prime_LBRM + 24) / (8*C_LBRM) ) - 24;
    Nref = floor(TBS_LBRM/(C*R_LBRM));
else
    Nref = 0;
end

coding.C = C;
coding.Zc = Zc;
coding.i_LS = i_LS;
coding.K = K;
coding.F = F;
coding.K_prime = K_prime;
coding.rvIdx = pdu.rvIndex;
coding.Nref = Nref;
coding.I_LBRM = pdu.I_LBRM;
coding.maxLayers = maxLayers;
coding.maxQm = maxQm;
coding.n_PRB_LBRM = n_PRB_LBRM;

return

