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

function [r_dmrs, lowPaprGroupNumber, lowPaprSequenceNumber] = gen_dmrs_dftsofdm(Nf, startPrb, nPrb, N_slot_frame, N_symb_slot, idxSym_dmrs, ...
    idxSlotInFrame, puschIdentity, groupOrSequenceHopping)

n_ID_RS = puschIdentity;

if groupOrSequenceHopping == 1
    c = build_Gold_sequence(floor(n_ID_RS/30), 10 * N_slot_frame * N_symb_slot);
elseif groupOrSequenceHopping == 2
    c = build_Gold_sequence(n_ID_RS, 10 * N_slot_frame * N_symb_slot);    
end

M_zc = nPrb * 6;
freqIdx = (startPrb-1)*6+1:(startPrb-1+nPrb)*6;
N_sc_RB = 12;

r_dmrs = zeros(Nf/2, N_symb_slot);
lowPaprGroupNumber = zeros(length(idxSym_dmrs), 1);
lowPaprSequenceNumber = zeros(length(idxSym_dmrs), 1);
countSym = 1; 
for idxSym = idxSym_dmrs - 1
    if groupOrSequenceHopping == 0
        f_gh = 0;
        v = 0;
    elseif groupOrSequenceHopping == 1
        f_gh = 0;
        for m = 0:7
            idxSeq = 8 * (idxSlotInFrame * N_symb_slot + idxSym) + m;
            f_gh = f_gh + c(idxSeq + 1) * 2^m;
        end
        f_gh = mod(f_gh, 30);
        v = 0;
    elseif groupOrSequenceHopping == 2
        f_gh = 0;
        if M_zc >= 6 * N_sc_RB
            idxSeq = idxSlotInFrame * N_symb_slot + idxSym;
            v = c(idxSeq + 1);
        else
            v = 0;
        end
    else
        error('groupOrSequenceHopping is not supported ...\n');
    end
    u = mod(f_gh + n_ID_RS, 30);
    lowPaprGroupNumber(countSym) = u;
    lowPaprSequenceNumber(countSym) = v;
    r_dmrs(freqIdx, idxSym + 1) = LowPaprSeqGen(M_zc, u, v);
    countSym = countSym + 1;
end

return