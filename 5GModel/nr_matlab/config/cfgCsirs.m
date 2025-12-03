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

function csirs = cfgCsirs

% CSI-RS related config
csirs.BWPSize = 273;
csirs.BWPStart = 0;
csirs.StartRB = 100; % Per FAPI spec, related to common resource block #0 (CRB#0), not BWP 
csirs.NrOfRBs = 52;
csirs.CSIType = 1; % 0:TRS, 1:CSI-RS NZP, 2:CSI-RS ZP
csirs.Row = 3;
csirs.FreqDomain = ones(1, 12);
csirs.SymbL0 = 0;
csirs.SymbL1 = 10;
csirs.CDMType = 1; % 0: noCDM, 1: fd-CDM2, 2: cdm4-FD2-TD2, 3: cdm8-FD2-TD4
csirs.FreqDensity = 0; % 0: dot5 (even RB), 1: dot5 (odd RB), 2: one, 3: three
csirs.ScrambId = 0;
csirs.prcdBf = 0;
csirs.digBFInterfaces = 2;
csirs.beamIdx = [1 2];
csirs.powerControlOffset = 0 + 8; % SCF-FAPIv2 PDSCH/CSIRS, [0:23] = [-8:15] dB
csirs.powerControlOffsetSS = 1; % SCF-FAPIv2 CSIRS/SSB, [0, 1, 2, 3] = [-3, 0, 3, 6] dB

csirs.idxUE = 0;

return
