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

function srs = cfgSrs

% srs related config
srs.RNTI = 10;
srs.BWPSize = 273;
srs.BWPStart = 0;
srs.numAntPorts = 1;
srs.numSymbols = 1;
srs.numRepetitions = 1;
srs.timeStartPosition = 12;
srs.configIndex = 63; % configIndex > 0 
srs.sequenceId = 0;
srs.bandwidthIndex = 0; % fixed to 0
srs.combSize = 4;
srs.combOffset = 1;
srs.cyclicShift = 1;
srs.frequencyPosition = 1;
srs.frequencyShift = 1;
srs.frequencyHopping = 0;
srs.groupOrSequenceHopping = 0;
srs.resourceType = 0;
srs.Tsrs = 1;
srs.Toffset = 0;
srs.Beamforming = 0;
srs.digBFInterfaces = 4;
srs.beamIdx = [1 2 3 4];
srs.numTotalUeAntennas = 1;             % added in FAPIv4
srs.ueAntennasInThisSrsResourceSet = 1; % added in FAPIv4
srs.sampledUeAntennas = 1;              % added in FAPIv4
srs.usage = 0;

srs.idxUE = 0;
srs.prgSize = 2; % For SRS chEst granularity

return
