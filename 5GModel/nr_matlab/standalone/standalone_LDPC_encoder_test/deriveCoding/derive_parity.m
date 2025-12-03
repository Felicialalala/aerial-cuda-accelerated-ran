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

function PuschCfg = derive_parity(PuschCfg)

%function derives the number of parity nodes added to each code block.

%outputs:
%PuschCfg.coding.nV_parity --> number of parity nodes

%%
%PARAMATERS

%coding paramaters:
C  = PuschCfg.coding.C;            % number of codeblocks
qam = PuschCfg.coding.qam;         % bits per qam
BGN = PuschCfg.coding.BGN;         % 1 or 2. Indicates which base graph used
Zc = PuschCfg.coding.Zc;           % lifting size
F = PuschCfg.coding.F;             % number of filler bits per codeblock

%allocation paramaters:
nl = PuschCfg.mimo.nl;             %number of layers transmited by user
N_data = PuschCfg.alloc.N_data;   %number of TF data resources in allocation

%%
%START

%number of bits to be transmitted:
G = N_data * qam * nl; 

%derive number of rate matched bits per codeblock:
E = zeros(C,1);

for r = 0 : (C - 1) 
    if r <= (C - mod( G / (nl * qam) , C) - 1)
        E(r + 1) = nl * qam * floor( G / (C * nl * qam) );
    else
        E(r + 1) = nl * qam * ceil( G / (C * nl * qam) );
    end
end

%add filler and punctured bits to the size:
E = E + F + 2*Zc;

%compute number of variable nodes:
nV = ceil(max(E) / Zc);

%compute number of parity nodes:
if BGN == 1
    nV_parity = nV - 22;
    
    if nV_parity >= 46
        nV_parity = 46;
    end
else
    nV_parity = nV - 10;
    
    if nV_parity >= 42
        nV_parity = 42;
    end
end

%%
%WRAP
PuschCfg.coding.nV_parity = nV_parity;



