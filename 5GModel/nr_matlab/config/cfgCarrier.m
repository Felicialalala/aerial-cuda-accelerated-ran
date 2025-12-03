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

function carrier = cfgCarrier

carrier.carrierFreq = 3.5;          % carrier freq (GHz)
carrier.duplex = 1;                 % 0: FDD, 1: TDD
carrier.CpType = 0;                 % 0: normal, 1: extended
carrier.mu = 1;                     % 0, 1, 2, 3, 4
carrier.N_grid_start_mu = 0;        % grid start in PRB
carrier.N_grid_size_mu = 273;       % grid size in PRB
carrier.N_ID_CELL = 41;             % Cell ID
carrier.Nant_gNB = 4;               % number of antennas at gNB
carrier.Nant_gNB_srs = 4;           % number of gNB antenna ports sent to the O-DU for the SRS channel
carrier.N_FhPort_DL = 4;            % number of FH ports for DL
carrier.N_FhPort_UL = 4;            % number of FH ports for UL
carrier.Nant_UE = 4;                % number of antennas at UE
carrier.freqShift7p5KHz = 0;        % UL 7.5KHz freq shift  
carrier.SFN_start = 0;              % starting SFN number
carrier.dmrsTypeAPos = 2;           % DMRS type A position

% TDD config
carrier.TddPeriod = 2.5;         % ms
ulSlot = ones(1, 14);                   % all ul symbols
dlSlot = zeros(1, 14);                  % all dl symbols
specialSlot = [0 0 0 0 0 0 -1 -1 -1 -1 1 1 1 1]; 
carrier.SlotConfig = [dlSlot dlSlot specialSlot ulSlot ulSlot];

% PDSCH config


return
