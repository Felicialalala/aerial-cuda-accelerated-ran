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

function Xtf = generate_pucch_tf_signal(x,pucch,carrier)

%function generates pucch tf signal. 

%inputs:
% x    --> bpsk or qpsk symbol to be transmited

%outputs:
% Xtf  --> time-frequency signal. Dim: Nf x Nt

%%
%PARAMATERS

%gnb:
Nf = carrier.N_sc;      % total number of subcarriers
Nt = carrier.N_symb_subframe_mu;      % total number of OFDM symbols

%pucch:
tOCCidx = pucch.tOCCidx;     % index of time covering code
startSym = pucch.startSym;   % staring pucch symbol (1-10)
prbIdx = pucch.prbIdx;       % index of pucch prb
nSym_data = pucch.nSym_data; % number of data symbols
nSym_dmrs = pucch.nSym_dmrs; % number of dmrs symbols
u = pucch.u;                 % group id
cs = pucch.cs;               % cyclic shift. Dim: nSym x 1
cs_freq = derive_cs_freq;


%%
%SETUP

%load base sequence:
load('r_pucch.mat');
r = r(:,u+1);

%load time codes:
load('tOCC_pucch.mat');
tOCC_data = tOCC{nSym_data,tOCCidx};
tOCC_dmrs = tOCC{nSym_dmrs,tOCCidx};

%pucch frequency idx:
freqIdx = 12*(prbIdx - 1) + 1 : 12*prbIdx;

%%
%START

Xtf = zeros(Nf,Nt);

%dmrs:
for i = 1 : nSym_dmrs    
    symIdx = 2*(i-1);       
    Xtf(freqIdx, symIdx + startSym + 1) = ...
        tOCC_dmrs(i) * r .* cs_freq(:,cs(symIdx+1)+1);    
end

%data:
for i = 1 : nSym_data    
    symIdx = 2*(i-1) + 1; 
    Xtf(freqIdx, symIdx + startSym + 1) = ...
        x * tOCC_data(i) * r .* cs_freq(:,cs(symIdx+1)+1);    
end

end



    
    
    
    



























