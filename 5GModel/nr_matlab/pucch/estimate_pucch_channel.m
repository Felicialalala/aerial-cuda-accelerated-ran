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

function H_est = estimate_pucch_channel(Y_dmrs_iue,PucchCfg,reciever,sp)

%function estimates the pucch channel for the iue user

%inputs:
% Y_dmrs_iue --> pucch dmrs signal. Dim: 12 x nSym_dmrs x L_BS

%outputs:
% H_est      --> estimate of pucch channel. Dim: 12 x nSym_data x L_BS

%%
%PARAMATERS

%gnb:
L_BS = sp.gnb.numerology.L_BS;  % total number of bs antennas

%pucch:
nSym_data = PucchCfg.nSym_data; % number of data symbols

%reciever:
Wf = reciever.ChEst.Wf;               % frequency ChEst filter. Dim: 12 x 12
Wt = reciever.ChEst.Wt;               % time ChEst filter. Dim: nSym_dmrs x nSym_data
s = reciever.ChEst.s;                 % delay centering sequence. Dim: 12 x 1

%%
%START

%apply filters:
H_est = zeros(12,nSym_data,L_BS);

for i = 1 : L_BS
    H_est(:,:,i) = (Wf * Y_dmrs_iue(:,:,i)) * Wt;
end

%undo delay centering:
for i = 1 : nSym_data
    for j = 1 : L_BS
        H_est(:,i,j) = conj(s) .* H_est(:,i,j);
    end
end

