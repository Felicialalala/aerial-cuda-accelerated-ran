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

function [Y_dmrs_iue, Y_data_iue] = remove_pucch_OCC(Y_dmrs,Y_data,PucchCfg,reciever,sp)

%function removes OCCs corresponding to the iue user

%inputs:
% Y_dmrs --> recieved pucch dmrs signal. Dim: 12 x nSym_dmrs x L_BS
% Y_data --> recieved pucch data siganl. Dim: 12 x nSym_data x L_BS

%outputs:
% Y_dmrs_iue --> pucch dmrs signal w/h iue codes removed. Dim: 12 x nSym_dmrs x L_BS
% Y_data_iue --> pucch dmrs signal w/h iue codes removed. Dim: 12 x nSym_dmrs x L_BS

%%
%PARAMATERS

%gnb:
L_BS = sp.gnb.numerology.L_BS;           % total number of bs antennas

%pucch:
cs = PucchCfg.cs;                        % cyclic shifts. Dim: nSym x 1
nSym_dmrs = PucchCfg.nSym_dmrs;          % number of dmrs symbols
nSym_data = PucchCfg.nSym_data;          % number of data symbols
tOCCidx = PucchCfg.tOCCidx;              % time OCC index (1-nSym_data)

%pucch reciever:
cs_freq = reciever.cs_freq;              % frequency reprentation of cyclic shifts. Dim: 12 x 12

%%
%SETUP

%load data/dmrs tOCC:
load('tOCC_pucch.mat');

tOCC_dmrs = tOCC{nSym_dmrs,tOCCidx};
tOCC_data = tOCC{nSym_data,tOCCidx};

%%
%START

%remove dmrs codes:
Y_dmrs_iue = Y_dmrs;

for i = 1 : nSym_dmrs
    symIdx = 2*(i-1) + 1;
    
    for j = 1 : L_BS
        Y_dmrs_iue(:,i,j) = conj(tOCC_dmrs(i)) * conj(cs_freq(:,cs(symIdx) + 1)) .* ...
            Y_dmrs_iue(:,i,j);
    end
    
end


%remove data codes:
Y_data_iue = Y_data;

for i = 1 : nSym_data
    symIdx = 2*(i-1) + 2;
    
    for j = 1 : L_BS
        Y_data_iue(:,i,j) = conj(tOCC_data(i)) * conj(cs_freq(:,cs(symIdx) + 1)) .* ...
            Y_data_iue(:,i,j);
    end
    
end


end












