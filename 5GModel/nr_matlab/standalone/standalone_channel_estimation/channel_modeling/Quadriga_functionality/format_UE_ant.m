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

function H = format_UE_ant(H,num_UE,PAR)

%function reformuates TF channel. Collapses UE polarization and UE index to
%a single dimension --> UE antenna.

%inputs:
%H --> dim: Nf x Nt x L_BS x 2 x num_UE

%outputs:
%H --> dim: Nf x Nt x L_BS x num_UE_ant. Where num_UE_ant = 2*num_UE

%%
%PARAMETERS


%modulation parameters:
Nf = PAR.mod.Nf; %number of subcarriers
Nt = PAR.mod.Nt; %number of OFDM symbols

%antenna parameters:
L_BS = PAR.ant.L_BS; %number of BS antennas

%%
%START

H = reshape(H,Nf,Nt,L_BS,2*num_UE);

end
