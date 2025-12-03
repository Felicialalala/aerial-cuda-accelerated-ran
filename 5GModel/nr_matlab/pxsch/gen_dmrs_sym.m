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

function [r_dmrs, scrSeq] = gen_dmrs_sym(slotNumber, Nf, Nt, N_dmrs_id)

% slotNumber = carrier.idxSlot + carrier.idxSubframe * carrier.N_slot_subframe_mu;
% Nf = carrier.N_sc;
% Nt = carrier.N_symb_slot;
% N_id = dmrs.dlDmrsScramblingId; 

r_dmrs = zeros(Nf/2,Nt,2);
scrSeq = zeros(Nf,Nt,2);
for n_scid = 0 : 1
    for t = 1 : Nt        
        %compute seed to Gold sequence:
        c_init = 2^17*(slotNumber*Nt + t)*(2*N_dmrs_id + 1) + 2*N_dmrs_id + n_scid;
        c_init = mod(c_init,2^31);        
        %build the Gold sequence
        c = build_Gold_sequence(c_init,Nf);   
        scrSeq(:, t, n_scid+1) = c;
        
        %build the scrambling sequence:
        r_dmrs(:,t,n_scid + 1) = build_freq_scrambling_sequence(c,Nf/2);        
    end
end

return
