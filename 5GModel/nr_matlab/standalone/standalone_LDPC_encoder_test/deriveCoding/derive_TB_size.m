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

function PuschCfg = derive_TB_size(PuschCfg)

%For user iue, determines transport block size.
%Follows TS38.214 section 5.1.3.2

%outputs:
%PuschCfg.coding.sizes.TBS --> number of information bits in users transport block


%%
%PARAMATERS

%allocation paramaters:
nprb_alloc = PuschCfg.alloc.nprb_alloc;   % number of prbs in allocation
N_data = PuschCfg.alloc.N_data;           % number of data TF resource in allocation
nl = PuschCfg.mimo.nl;                    % number of user layers

%coding paramaters:
qam = PuschCfg.coding.qam;                % number of bits in user's QAM
codeRate = PuschCfg.coding.codeRate;      % users target code rate

%%
%SETUP
load(fullfile('./deriveCoding/TBS_table.mat'));
% load('.\deriveCoding\TBS_table.mat');

%%
%START

%compute number of avaliable TF resources avaliable to the UE
Nre = min(156, N_data / nprb_alloc) * nprb_alloc;
 
%approximate number information bits (given code rate, qam, layers)
Ninfo = Nre * codeRate * qam * nl;

if Ninfo <= 3824       
    %for "small" sizes, look up TBS in a table. First round the
    %number of information bits.
    n = max(3,(floor(log2(Ninfo)) - 6));
    Ninfo_prime = max(24, 2^n*floor(Ninfo / 2^n));

    %next lookup in table closest TBS (without going over).
    compare = Ninfo_prime - TBS_table;
    compare(compare > 0) = -100000;
    [~,max_index] = max(compare);
    TBS = TBS_table(max_index);
    C = 1;
else
    %for "large" sizes, compute TBS. First round the number of
    %information bits to a power of two.
     n = floor(log2(Ninfo-24)) - 5;
     Ninfo_prime = max(3840, 2^n*round((Ninfo-24)/2^n));

    %Next, compute the number of code words. For large code rates,
    %use base-graph 1. For small code rate use base-graph 2.
    if codeRate < 1/4
        C = ceil( (Ninfo + 24) / 3816);
        TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
    else
        if Ninfo_prime > 8424
            C = ceil( (Ninfo_prime + 24) / 8424);
            TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
        else
            C = 1;
            TBS = 8*C*ceil( (Ninfo_prime + 24) / (8*C) ) - 24;
        end
    end
end
 
%%
%WRAP

PuschCfg.coding.TBS = TBS;

     
