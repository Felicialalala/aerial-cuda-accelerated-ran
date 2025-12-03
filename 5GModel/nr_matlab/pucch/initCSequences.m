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

function [pucch] = initCSequences(pucch)     
    n_ID = pucch.hoppingId;
    groupHopping = pucch.groupHopFlag;
    sequenceHopping = pucch.sequenceHopFlag;
    cSequenceGH = [];
    
    if groupHopping && ~sequenceHopping
        c_init = floor(n_ID/30);
        %Maximum number of slots per frame: 80 under numerology 3
        cSequenceGH = build_Gold_sequence(c_init, 8*(2*80+1)+7);
    elseif ~groupHopping && sequenceHopping
        c_init = 2^5*floor(n_ID/30)+mod(n_ID, 30);
        cSequenceGH = build_Gold_sequence(c_init, 2*80+1);
    elseif groupHopping && sequenceHopping
        error('Error. \nGroup hopping and sequence hopping cannot be both enabled.')
    end
    
    pucch.cSequenceGH = cSequenceGH;
end
