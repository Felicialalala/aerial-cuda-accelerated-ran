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

function [rxSamp, Chan_UL_interference] = addInterference(rxSamp, SIR, Chan_UL_interference, carrier, SimCtrl)
dim_rxSamp = size(rxSamp);
%generate and add interference
num_interference_UE = length(Chan_UL_interference);
for ii = 1:num_interference_UE
    if SimCtrl.timeDomainSim
        dim_tx_interf_TD = size(rxSamp);%
        dim_tx_interf_TD(end) = Chan_UL_interference{ii}.Nin;
        white_signal = sqrt(10^(-SIR/10/num_interference_UE))*sqrt(0.5)*(randn(dim_tx_interf_TD)+1j*randn(dim_tx_interf_TD));
    else % Freq domain
        dim_tx_interf_FD = dim_rxSamp;
        dim_tx_interf_FD(end) = Chan_UL_interference{ii}.Nin; % dim: [num_REs_in_Grid, num_sym_per_slot, num_tx_ant]
        white_signal = sqrt(10^(-SIR/10/num_interference_UE))*sqrt(0.5)*(randn(dim_tx_interf_FD)+1i*randn(dim_tx_interf_FD));
    end
    [interference_this_UE,Chan_UL_interference{ii}] = Channel(white_signal, Chan_UL_interference{ii}, SimCtrl, carrier);
    rxSamp = rxSamp + interference_this_UE; 
    if isfield(Chan_UL_interference{ii}, 'chanMatrix_FD_oneSlot')
        Chan_UL_interference{ii}.interfChanMatrix_FD_oneSlot = Chan_UL_interference{ii}.chanMatrix_FD_oneSlot;
    end
end

return