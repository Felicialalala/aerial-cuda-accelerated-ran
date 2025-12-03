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

function PuschCfg = derive_BGN(PuschCfg)

%For user iue, determine which base graph to use.
%Follows 38.212 section 6.2.2

%outputs:
%PuschCfg.coding.BGN --> 1 or 2. Indicates which base graph used


%%
%PARAMATERS

TBS = PuschCfg.coding.TBS;     %size of transport blockcodeRate = sp.gnb.codeRate(iue); %users target code rate
codeRate = PuschCfg.coding.codeRate; %target code rate

%%
%START

if (TBS <= 292) || ((TBS <= 3824) && (codeRate <= 0.67)) || (codeRate <= 0.25)
    BGN = 2;
else
    BGN = 1;
end

%%
%WRAP

PuschCfg.coding.BGN = BGN;


end


        
