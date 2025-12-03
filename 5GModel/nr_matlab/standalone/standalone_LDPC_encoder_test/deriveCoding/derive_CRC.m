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

function PuschCfg = derive_CRC(PuschCfg)

%Function derives CRC used to encode transport block.
%Follows 38.212 section 6.2.1

%outputs:
%PuschCfg.coding.CRC      --> '16' or '24A'. Indicates which CRC polynomial used to encode TB
%PuschCfg.coding.sizes.B  --> size of CRC encoded transport block

%%
%PARAMATERS

TBS = PuschCfg.coding.TBS; %size of transport block

%%
%START

if TBS > 3824
    CRC = '24A';
    B = TBS + 24;
else
    CRC = '16';
    B = TBS + 16;
end

%%
%WRAP

PuschCfg.coding.CRC = CRC;
PuschCfg.coding.B = B;


end


