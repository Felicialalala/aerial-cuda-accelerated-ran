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

function uciSegEst = combine_uciCbEsts(cbEst0, cbEst1, polarUciSegPrms)

%%
%PARAMATERS

A_seg          = polarUciSegPrms.A_seg; % number of bits in uci segment
zeroInsertFlag = polarUciSegPrms.zeroInsertFlag; 

%%
%START

A_seg_half = floor(A_seg / 2);

uciSegEst = zeros(A_seg,1);
uciSegEst(1 : A_seg_half)       = cbEst0(zeroInsertFlag + 1 : end);
uciSegEst(A_seg_half + 1 : end) = cbEst1;

end
