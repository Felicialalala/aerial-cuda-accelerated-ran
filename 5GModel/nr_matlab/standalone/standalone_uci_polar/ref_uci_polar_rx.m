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

function cwLLRs_cell = ref_uci_polar_rx(nPolUciSegs, A_seg, E_seg, polUciSegLLRs_cell)

% Function runs uci polar reciever pipeline using reference scripts (for example 5g toolbox).
% Outputs intermediate buffers.

%inputs:
% nPolUciSegs        --> number of polar uci segments to process
% A_seg              --> number of info bits per segment. Dim: nPolUciSegs x 1
% E_seg              --> number of tx bits per segment.   Dim: nPolUciSegs x 1
% polUciSegLLRs_cell --> cell containing LLRs of polar uci segments. Dim: nPolUciSegs x 1

%outputs
% cwLLRs_cell --> cell containing codeword LLRs. Dim: nPolUciSegs x 1

%%
% DERIVE UCI PARAMATERS

polarUciSegPrms_cell = cell(nPolUciSegs,1);

for segIdx = 0 : (nPolUciSegs - 1)
    polarUciSegPrms_cell{segIdx + 1} = derive_polarUciSegPrms(A_seg(segIdx + 1), E_seg(segIdx + 1));
end

[totNumPolCbs, polarCbPrms_cell, polarUciSegPrms_cell] = derive_polarCbPrms(nPolUciSegs, polarUciSegPrms_cell);
 

%%
% RUN


% segment LLRs to codewords, deInterleave, deRateMatch
cwLLRs_cell = cell(totNumPolCbs,1);

for segIdx = 0 : (nPolUciSegs - 1)
    
    polarUciSegPrms = polarUciSegPrms_cell{segIdx + 1};
    [~, cwLLRs]     = PUCCH_decoder(polUciSegLLRs_cell{segIdx + 1}', A_seg(segIdx + 1), 1, 1);
    
    for i = 0 : (polarUciSegPrms.nCbs - 1)
        cbIdx                  = polarUciSegPrms.childCbIdxs(i + 1);
        cwLLRs_cell{cbIdx + 1} = cwLLRs(i + 1,:)';
    end
end

end


