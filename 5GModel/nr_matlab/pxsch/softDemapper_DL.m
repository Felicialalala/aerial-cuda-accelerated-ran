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

function [LLRseq] = softDemapper_DL(Qams_eq, Ree, table, nl, Nt_data, Nf, nPrb, startPrb, ...
    qam, n_scid)

global SimCtrl

L_UE = nl;
nSym_data = Nt_data;

d_qpsk = table.d_qpsk;
d_qam16 = table.d_qam16;
d_qam64 = table.d_qam64;
d_qam256 = table.d_qam256;

QAM4_LLR = table.QAM4_LLR;
QAM16_LLR = table.QAM16_LLR;
QAM64_LLR = table.QAM64_LLR;
QAM256_LLR = table.QAM256_LLR;

%extract d:
switch qam
    case 2
        d = d_qpsk;
    case 4
        d = d_qam16;
    case 6
        d = d_qam64;
    case 8
        d = d_qam256;
end

if SimCtrl.useCuphySoftDemapper
    switch qam
        case 2
            T = QAM4_LLR;
        case 4
            T = QAM16_LLR;
        case 6
            T = QAM64_LLR;
        case 8
            T = QAM256_LLR;
    end
    % Truncate to an fp16 representation
    colfp16 = fp16nv(T, SimCtrl.fp16AlgoSel);
    % Convert back to fp16 for interpolation below
    T = double(colfp16);
end

pam = qam / 2; %number of bits in PAM constellations

%compute LLRs:

nQam = length(Qams_eq);
LLRseq = [];
for idxQam = 1:nQam    
    x_est = Qams_eq(idxQam);        
    % N0 = Ree(portIdx(s),freqIdx(f));
    N0 = Ree;    
    if SimCtrl.useCuphySoftDemapper == 0
        %estimate real/imag pam llrs:
        llr_real = max_pam_LLR(real(x_est),N0,d,pam);
        llr_imag = max_pam_LLR(imag(x_est),N0,d,pam);
        %combine into qam llrs:
        llr_matlab = zeros(qam,1);
        for j = 1 : pam
            llr_matlab(2*(j-1) + 1) = llr_real(j);
            llr_matlab(2*(j-1) + 2) = llr_imag(j);
        end
        llr = llr_matlab;
    else
        % Model for cuPHY soft demapper
        llr_cuphy = cuPhySoftDemapper(x_est, T, qam, N0);
        llr = llr_cuphy;
    end    
    % force NaN to 1
    for j = 1:2*pam
        if isnan(llr(j))
            llr(j) = 1;
        end
    end    
    LLRseq((idxQam-1)*qam+1:idxQam*qam) = real(llr);    
end

LLRseq = LLRseq(:);

return