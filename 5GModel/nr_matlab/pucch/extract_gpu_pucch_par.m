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

function [nUe_pucch,Pucch_common,Pucch_ue_cell,Pucch_receiver] = extract_gpu_pucch_par(sp)

%function extracts pucch paramaters needed by gpu

%outputs:
% nUe_pucch     --> number of pucch users
% Pucch_common  --> pucch paramaters shared by all users
% Pucch_ue_cell --> cell containing user specific pucch paramaters. Dim: nUe_pucch x 1
% Pucch_reciver --> structure containing filters and sequences needed by pucch reciever


%Pucch_common
% startSym  --> index of starting pucch symbol (c 0 indexing!)
% nSym      --> number of pucch symbols
% nSym_data --> number of pucch data symbols
% nSym_dmrs --> number of pucch dmrs symbols
% prbIdx    --> index of pucch prb (c 0 indexing!)
% u         --> index of low-papr sequence (c 0 indexing!)
% L_BS      --> number of base-station antennas

%Pucch_ue
% cs       --> index of cyclic shifts. Dim: nSym x 1. (c 0 indexing!)
% tOCCidx  --> index of time cover-code. (matlab 0 index!)
% nbits    --> 1 or 2. Number of transmitted bits.

%Pucch_reciver
% cs_freq   --> frequency representation of cyclic shifts. Dim: 12 x 12
% r         --> low papr pucch sequences. Dim: 12 x 30
% tOCC_cell --> cell containing time orthognal covering codes. Dim: 7 x 1
% Wf        --> frequency ChEst filter. Dim: 12 x 12
% s         --> delay shift sequence. Dim: 12 x 1
% Wt_cell   --> Cell of time ChEst filters. Dim: 11 x 1


%%
%PARAMATERS

PucchCfg_cell = sp.gnb.pucch.PucchCfg_cell;
reciever = sp.gnb.pucch.reciever;
nUe_pucch = length(PucchCfg_cell);

%%
%START

%common pucch paramaters:
Pucch_common = [];

%extract paramaters:
PucchCfg = PucchCfg_cell{1};
Pucch_common.startSym = PucchCfg.startSym;
Pucch_common.nSym = PucchCfg.nSym;
Pucch_common.nSym_data = PucchCfg.nSym_data;
Pucch_common.nSym_dmrs = PucchCfg.nSym_dmrs;
Pucch_common.prbIdx = PucchCfg.prbIdx - 1;
Pucch_common.u = PucchCfg.u;
Pucch_common.L_BS = sp.gnb.numerology.L_BS;
Pucch_common.mu = sp.gnb.mu; % numerology
Pucch_common.slotNumber = sp.gnb.pusch.slotNumber;
Pucch_common.hoppingId =  sp.gnb.hop_id;

%user specfic pucch paramaters:
Pucch_ue_cell = cell(nUe_pucch,1);

for iue = 1 : nUe_pucch
    Pucch_ue = [];
    PucchCfg = PucchCfg_cell{iue};
    
    %extract paramaters:
    Pucch_ue.cs = PucchCfg.cs;
    Pucch_ue.tOCCidx = PucchCfg.tOCCidx - 1;
    Pucch_ue.nBits = PucchCfg.nBits;
    Pucch_ue.cs0 = PucchCfg.cs0;
    
    %wrap:
    Pucch_ue_cell{iue} = Pucch_ue;
end
    
%pucch reciever:
Pucch_receiver = [];

%extract paramaters:
load('tOCC_pucch2.mat');
load('r_pucch.mat');
Pucch_receiver.cs_freq = reciever.cs_freq;
Pucch_receiver.r = r;
Pucch_receiver.tOCC_cell = tOCC_cell;
Pucch_receiver.Wf = reciever.ChEst.Wf;
Pucch_receiver.s = reciever.ChEst.s;
Pucch_receiver.Wt_cell = reciever.ChEst.Wt_cell;




