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

function PuschCfg = generate_rnd_cfg

%generate a random Pusch configuration for a user

%coding configuration:
%PuschCfg.coding.mcs        --> mcs assgined to user
%PuschCfg.coding.mcsTable   --> mcs table assigned to user

%allocation configuration:
%PuschCfg.alloc.nprb_alloc  --> number of prbs in allocation
%PuschCfg.alloc.N_data      --> number of data TF resource in allocation
%PuschCfg.alloc.nl          --> number of user layers

%%
%CODING

%choose between table 1 and 2:
c = rand;
if c <= 0.5
    mcsTable = 1;
else
    mcsTable = 2;
end

%choose a rnd mcs for the user:
if mcsTable == 1
    mcs = floor(rand*29);
else
    mcs = floor(rand*28);
end

%%
%ALLOC

%choose rnd number of PRBs
nprb_alloc = ceil(rand*273);

%choose rnd number of layers:
nl = ceil(rand*8);

%compute number of TF resources, assuming 2 control symbols
if nl <= 4
    N_data = 11*12*nprb_alloc; %1 dmrs symbol
else
    N_data = 11*12*nprb_alloc; %2 dmrs symbols
end

%%
%WRAP

PuschCfg.coding.mcs = mcs;
PuschCfg.coding.mcsTable = mcsTable; 
PuschCfg.alloc.nprb_alloc = nprb_alloc;
PuschCfg.alloc.N_data = N_data;
PuschCfg.mimo.nl = nl;         

end




