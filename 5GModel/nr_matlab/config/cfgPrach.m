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

function prach = cfgPrach

% prach related config
prach.configurationIndex = 158;       % 0 - 255 
prach.restrictedSet = 0; % 0: unrestricted, 1: typeA, 2: typeB
prach.rootSequenceIndex = 0;        % 0, 1, ..., L_RA-2
prach.zeroCorrelationZone = 10;     % 0, 1, 2, ..., 15 (select N_CS)
prach.n_RA_start = 0;               % starting freq alloc for PRACH 
prach.msg1_FDM = 1;                 % number of freq alloc for PRACH
prach.N_nc = 1; % number of Non-coherent combining for repetition preamble
prach.prmbIdx = 10;      % preamble index
prach.ssbIdx = 0;
prach.digBFInterfaces = 4;
prach.beamIdx = [1 2 3 4];
prach.betaPrach = 1;    % PRACH power scaling factor
prach.allSubframes = 0;
prach.force_thr0 = 0;

prach.idxUE = 0;

return
