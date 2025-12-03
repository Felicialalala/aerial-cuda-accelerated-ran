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

function [descrmLLRSeq1, descrmLLRSeq2] = PucchF34UciDeMultiplexing(nSym, nSym_dmrs, pi2Bpsk, nSymUci, E_seq1, E_seq2, descrmLLRArr)
%%
% The function performs multiplexing of coded UCI bits for PUCCH format 3
% refer to 3GPP TS 38.212, Section 6.3.1.6

% inptus:
% nSym          --> number of PF3/PF4 OFDM symbols
% nSym_dmrs     --> number of DMRS OFDM symbols
% pi2Bpsk   --> indicator for pi2BPSK; 1 - pi2BPSK; 0 - QPSK
% nSymUci       --> number of resource elements per OFDM symbol in the PF3/PF4 (12*number of PRBs)
% E_seq1        --> length of sequence 1
% E_seq2        --> length of sequence 2
% descrmLLRArr  --> multiplexed LLR array

% outputs:
% descrmLLRSeq1 --> demultiplexed LLR array for sequence 1
% descrmLLRSeq2 --> demultiplexed LLR array for sequence 2
%%

descrmLLRSeq1 = zeros(E_seq1, 1);
descrmLLRSeq2 = zeros(E_seq2, 1);

Qm = 1;
if ~pi2Bpsk % QPSK
    Qm = 2;
end

nSetUci = 0;
SiUci = -1 * ones(3, 8);
NiUci = zeros(1, 3);
sumNiUci = zeros(1, 3);

nPucchSymUci = nSym-nSym_dmrs;

gBar_lkv = zeros(nPucchSymUci, nSymUci, Qm);

uciSymInd = -1 * ones(1, 14);

switch nSym
    case 4
        if nSym_dmrs == 1
            uciSymInd(1:3) = [0, 2, 3];
            nSetUci = 2;
            SiUci(1,1:2) = [0, 2];
            SiUci(2,1) = [3];
            NiUci(1) = 2;
            NiUci(2) = 1;
            sumNiUci(1:2) = [2, 3];
        else % nSym_dmrs == 2
            uciSymInd(1:2) = [1, 3];
            nSetUci = 1;
            SiUci(1,1:2) = [1,3];
            NiUci(1) = 2;
            sumNiUci(1) = [2];
        end
    case 5
        uciSymInd(1:3) = [1, 2, 4];
        nSetUci = 1;
        SiUci(1,1:3) = [1, 2, 4];
        NiUci(1) = 3;
        sumNiUci(1) = [3];
    case 6
        uciSymInd(1:4) = [0, 2, 3, 5];
        nSetUci = 1;
        SiUci(1,1:4) = [0,2,3,5];
        NiUci(1) = 4;
        sumNiUci(1) = [4];
    case 7
        uciSymInd(1:5) = [0, 2, 3, 5, 6];
        nSetUci = 2;
        SiUci(1,1:4) = [0,2,3,5];
        SiUci(2,1) = [6];
        NiUci(1) = 4;
        NiUci(2) = 1;
        sumNiUci(1:2) = [4, 5];
    case 8
        uciSymInd(1:6) = [0, 2, 3, 4, 6, 7];
        nSetUci = 2;
        SiUci(1,1:4) = [0,2,4,6];
        SiUci(2,1:2) = [3, 7];
        NiUci(1) = 4;
        NiUci(2) = 2;
        sumNiUci(1:2) = [4, 6];
    case 9
        uciSymInd(1:7) = [0, 2, 3, 4, 5, 7, 8];
        nSetUci = 2;
        SiUci(1,1:4) = [0,2,5,7];
        SiUci(2,1:3) = [3, 4, 8];
        NiUci(1) = 4;
        NiUci(2) = 3;
        sumNiUci(1:2) = [4, 7];
    case 10
        if nSym_dmrs == 2
            uciSymInd(1:8) = [0, 1, 3, 4, 5, 6, 8, 9];
            nSetUci = 2;
            SiUci(1,1:4) = [1,3,6,8];
            SiUci(2,1:4) = [0, 4, 5, 9];
            NiUci(1) = 4;
            NiUci(2) = 4;
            sumNiUci(1:2) = [4, 8];
        else % nSym_dmrs == 4
            uciSymInd(1:6) = [0,2,4,5,7,9];
            nSetUci = 1;
            SiUci(1,1:6) = [0,2,4,5,7,9];
            NiUci(1) = 6;
            sumNiUci(1) = [6];
        end
    case 11
        if nSym_dmrs == 2
            uciSymInd(1:9) = [0, 1, 3, 4, 5, 6, 8, 9, 10];
            nSetUci = 3;
            SiUci(1,1:4) = [1,3,6,8];
            SiUci(2,1:4) = [0,4,5,9];
            SiUci(3,1) = [10];
            NiUci(1) = 4;
            NiUci(2) = 4;
            NiUci(3) = 1;
            sumNiUci(1:3) = [4, 8, 9];
        else % nSym_dmrs == 4
            uciSymInd(1:7) = [0,2,4,5,7,8,10];
            nSetUci = 1;
            SiUci(1,1:7) = [0,2,4,5,7,8,10];
            NiUci(1) = 7;
            sumNiUci(1) = [7];
        end
    case 12
        if nSym_dmrs == 2
            uciSymInd(1:10) = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11];
            nSetUci = 3;
            SiUci(1,1:4) = [1,3,7,9];
            SiUci(2,1:4) = [0,4,6,10];
            SiUci(3,1:2) = [5, 11];
            NiUci(1) = 4;
            NiUci(2) = 4;
            NiUci(3) = 2;
            sumNiUci(1:3) = [4, 8, 10];
        else % nSym_dmrs == 4
            uciSymInd(1:8) = [0,2,3,5,6,8,9,11];
            nSetUci = 1;
            SiUci(1,1:8) = [0,2,3,5,6,8,9,11];
            NiUci(1) = 8;
            sumNiUci(1) = [8];
        end
    case 13
        if nSym_dmrs == 2
            uciSymInd(1:11) = [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12];
            nSetUci = 3;
            SiUci(1,1:4) = [1,3,8,10];
            SiUci(2,1:4) = [0,4,7,11];
            SiUci(3,1:3) = [5,6,12];
            NiUci(1) = 4;
            NiUci(2) = 4;
            NiUci(3) = 3; 
            sumNiUci(1:3) = [4, 8, 11];
        else % nSym_dmrs == 4
            uciSymInd(1:9) = [0,2,3,5,6,8, 9, 10,12];
            nSetUci = 2;
            SiUci(1,1:8) = [0,2,3,5,6,8,10,12];
            SiUci(2,1) = [9];
            NiUci(1) = 8;
            NiUci(2) = 1; 
            sumNiUci(1:2) = [8, 9];
        end
    case 14
        if nSym_dmrs == 2
            uciSymInd(1:12) = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13];
            nSetUci = 3;
            SiUci(1,1:4) = [2,4,9,11];
            SiUci(2,1:4) = [1,5,8,12];
            SiUci(3,1:4) = [0,6,7,13];
            NiUci(1) = 4;
            NiUci(2) = 4;
            NiUci(3) = 4;
            sumNiUci(1:3) = [4, 8, 12];
        else % nSym_dmrs == 4
            uciSymInd(1:10) = [0, 2, 3, 4, 6, 7, 9, 10, 11, 13];
            nSetUci = 2;
            SiUci(1,1:8) = [0,2,4,6,7,9,11,13];
            SiUci(2,1:2) = [3, 10];
            NiUci(1) = 8;
            NiUci(2) = 2;
            sumNiUci(1:2) = [8, 10];
        end
end

temp = find(sumNiUci*nSymUci*Qm >= E_seq1);
j = temp(1); % 1-based

comSiUciLessj = [];
if j == 2
    comSiUciLessj = SiUci(1,:);
elseif j == 3
    comSiUciLessj = [SiUci(1,:), SiUci(2,:)];
end

n1 = 1; % adjusted for Matlab 1-based
n2 = 1; % adjusted for Matlab 1-based

nBarSymUci = 0;
M = 0;
if j>1
    nBarSymUci = floor((E_seq1 - sumNiUci(j-1)*nSymUci*Qm)/(NiUci(j)*Qm));

    M = mod((E_seq1 - sumNiUci(j-1)*nSymUci*Qm)/Qm, NiUci(j));
else % j == 1
    nBarSymUci = floor(E_seq1/(NiUci(j)*Qm));

    M = mod(E_seq1/Qm, NiUci(j));
end

for l = 1:nPucchSymUci
    sl = uciSymInd(l);
    if ismember(sl, comSiUciLessj)
        for k = 1:nSymUci
           for v = 1:Qm
               descrmLLRSeq1(n1) = descrmLLRArr((l-1)*nSymUci*Qm + (k-1)*Qm + v);
               n1 = n1 + 1;
           end
        end
    elseif ismember(sl, SiUci(j,:))
        gamma = 0;
        if M>0
            gamma = 1;
        else
            gamma = 0;
        end
        M = M - 1;
        for k = 1:(nBarSymUci+gamma)
            for v = 1:Qm
                descrmLLRSeq1(n1) = descrmLLRArr((l-1)*nSymUci*Qm + (k-1)*Qm + v);
                n1 = n1 + 1;
            end
        end
        
        for k = (nBarSymUci+gamma+1):nSymUci
            for v = 1:Qm
                descrmLLRSeq2(n2) = descrmLLRArr((l-1)*nSymUci*Qm + (k-1)*Qm + v);
                n2 = n2 + 1;
            end
        end
    else
        for k = 1:nSymUci
           for v = 1:Qm
               descrmLLRSeq2(n2) = descrmLLRArr((l-1)*nSymUci*Qm + (k-1)*Qm + v);
               n2 = n2 + 1;
           end
        end
    end
end



end
