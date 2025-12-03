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

function [F3Para] = deriveF3UciSeqTxSize(BitLenSr, BitLenHarq, BitLenCsiPart1, BitLenCsiPart2, freqHopFlag, AddDmrsFlag, maxCodeRate, pi2Bpsk, nSym, nPrb)

% inptus:
% F3 PUCCH PDU parameters

% outputs:
% E_seq(1) --> number of Tx bits assigned to the first UCI segment (HARQ + SR + CSI1)
% E_seq(2) --> number of tx bits assigend to the second UCI segment (CSI2)

%%

%%
% TIME ALLOCATION (38.211 table 6.4.1.3.3.2-1)

SetSymData = [];
SetSymDmrs = [];

switch nSym
    case 4
        if(freqHopFlag == 1)
            nSymData = 2;
            SetSymData = [1 3];
            SetSymDmrs = [0 2];
        else
            nSymData = 3;
            SetSymData = [0 2 3];
            SetSymDmrs = [1];
        end
    case 5
        nSymData = nSym - 2;
        SetSymData = [1 2 4];
        SetSymDmrs = [0 3];
    case 6
        nSymData = nSym - 2;
        SetSymData = [0 2 3 5];
        SetSymDmrs = [1 4];
    case 7
        nSymData = nSym - 2;
        SetSymData = [0 2 3 5 6];
        SetSymDmrs = [1 4];
    case 8
        nSymData = nSym - 2;
        SetSymData = [0 2 3 4 6 7];
        SetSymDmrs = [1 5];
    case 9
        nSymData = nSym - 2;
        SetSymData = [0 2 3 4 5 7 8];
        SetSymDmrs = [1 6];
    case 10
        if(AddDmrsFlag == 1)
            nSymData = nSym - 4;
            SetSymData = [0 2 4 5 7 9];
            SetSymDmrs = [1 3 6 8];
        else
            nSymData = nSym - 2;
            SetSymData = [0 1 3 4 5 6 8 9];
            SetSymDmrs = [2 7];
        end
    case 11
        if(AddDmrsFlag == 1)
            nSymData = nSym - 4;
            SetSymData = [0 2 4 5 7 8 10];
            SetSymDmrs = [1 3 6 9];
        else
            nSymData = nSym - 2;
            SetSymData = [0 1 3 4 5 6 8 9 10];
            SetSymDmrs = [2 7];
        end
    case 12
        if(AddDmrsFlag == 1)
            nSymData = nSym - 4;
            SetSymData = [0 2 3 5 6 8 9 11];
            SetSymDmrs = [1 4 7 10];
        else
            nSymData = nSym - 2;
            SetSymData = [0 1 3 4 5 6 7 9 10 11];
            SetSymDmrs = [2 8];
        end
    case 13
        if(AddDmrsFlag == 1)
            nSymData = nSym - 4;
            SetSymData = [0 2 3 5 6 8 9 10 12];
            SetSymDmrs = [1 4 7 11];
        else
            nSymData = nSym - 2;
            SetSymData = [0 1 3 4 5 6 7 8 10 11 12];
            SetSymDmrs = [2 9];
        end
    case 14
        if(AddDmrsFlag == 1)
            nSymData = nSym - 4;
            SetSymData = [0 2 3 4 6 7 9 10 11 13];
            SetSymDmrs = [1 5 8 12];
        else
            nSymData = nSym - 2;
            SetSymData = [0 1 2 4 5 6 7 8 9 11 12 13];
            SetSymDmrs = [3 10];
        end   
end


%%
% SPLIT PAYLOAD (38.212 6.3.1.1)
% pucch payload split into two sequences: Harq + Sr + Csi1, and Csi2 

if(BitLenCsiPart2 > 0)
    nSeqs = 2;
else
    nSeqs = 1;
end

A_seg    = zeros(2,1);
A_seg(1) = BitLenHarq + BitLenSr + BitLenCsiPart1;
A_seg(2) = BitLenCsiPart2;

%%
% DETERMINE CODING (38.212 6.3.1.3.1)
% For first UCI segment, determine number CRC bits per codeblock. 

if(A_seg(1) <= 11)
    nCrcBitsPerCb = 0;
elseif(A_seg(1) <= 19)
    nCrcBitsPerCb = 6;
else
    nCrcBitsPerCb = 11;
end

%%
% DETERMINE E_seq (38.212 6.3.1.2)
% Determine number of tx bits assigned to each sequence

% first determine total number tx bits avaliable (38.212 table 6.3.1.4-1)
if(pi2Bpsk) 
    nBitsPerRe = 1;
else
    nBitsPerRe = 2;
end
E_tot = nBitsPerRe * 12 * nSymData * nPrb;

% assign tx bits (38.212 Table 6.3.1.4.1-1)
E_seq = zeros(2,1);

switch maxCodeRate
    case 0
        R_max = 0.08;
    case 1
        R_max = 0.15;
    case 2
        R_max = 0.25;
    case 3
        R_max = 0.35;
    case 4
        R_max = 0.45;
    case 5
        R_max = 0.60;
    case 6
        R_max = 0.80;
end
        
if (nSeqs == 1)
    E_seq(1) = E_tot;
else
    E_seq(1) = min(E_tot, ceil((A_seg(1) + nCrcBitsPerCb) / R_max / nBitsPerRe) * nBitsPerRe);
    E_seq(2) = E_tot - E_seq(1);
end

F3Para.A_seg = A_seg;
F3Para.E_seq = E_seq;
F3Para.pi2Bpsk = pi2Bpsk;
F3Para.nSymData = nSymData;
F3Para.nBitsPerRe = nBitsPerRe;
F3Para.nSC = 12*nPrb;
F3Para.prbSize = nPrb;
F3Para.AddDmrsFlag = AddDmrsFlag;
F3Para.freqHopFlag = freqHopFlag;
F3Para.pi2Bpsk = pi2Bpsk;
F3Para.SetSymData = SetSymData;
F3Para.SetSymDmrs = SetSymDmrs;
end