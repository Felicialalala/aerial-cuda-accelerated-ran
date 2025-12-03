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

function pdsch = cfgPdsch

% pdsch related config
pdsch.pduBitmap = 0;
pdsch.BWPSize = 273;
pdsch.BWPStart = 0;
pdsch.RNTI = 46;
pdsch.NrOfCodewords = 1; % not used
pdsch.targetCodeRate = 0; % valid only when mcsIndex >= 28(29), FAPI format: raw coderate x 1024 x 10
pdsch.qamModOrder = 0; % valid only when mcsIndex >= 28(29)
pdsch.TBSize = 0; % valide only when mcsIndex >= 28(29)
pdsch.mcsIndex = 27;
pdsch.mcsTable = 1;
pdsch.rvIndex = 0;
pdsch.dataScramblingId = 40;
pdsch.nrOfLayers = 4;
pdsch.portIdx = [0:pdsch.nrOfLayers-1];
pdsch.transmissionScheme = 0; % not used
pdsch.refPoint = 0;
pdsch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 0 0 0];
pdsch.DmrsMappingType = 0; % 0:typeA, 1:typeB (not defined in FAPI PDU, for compliance test only)
pdsch.dmrsConfigType = 0; % 1: not supported
pdsch.DmrsScramblingId = 40;
pdsch.SCID = 0;
pdsch.numDmrsCdmGrpsNoData = 2;
pdsch.resourceAlloc = 1; 
% define rbBitmap according to FAPI format. Each array element corresponding 
% to one byte (uint8), and totally 36 bytes. Bit from left to right corresponding 
% to [VRB7,VRB6,...,VRB0, VRB15,VRB14,...,VRB8,...]
pdsch.rbBitmap = [255*ones(1, 34),1,0]; 
pdsch.rbStart = 120; %
pdsch.rbSize = 100; %
pdsch.VRBtoPRBMapping = 0;  % 1: not supported
pdsch.StartSymbolIndex = 2;
pdsch.NrOfSymbols = 10;
pdsch.prcdBf = 0;
pdsch.digBFInterfaces = 4;
pdsch.beamIdx = [1 2 3 4];
pdsch.powerControlOffset = 8; % SCF-FAPIv2 PDSCH/CSIRS, [0:23] = [-8:15] dB
pdsch.powerControlOffsetSS = 1; % SCF-FAPIv2 CSIRS/SSS, [0, 1, 2, 3] = [-3, 0, 3, 6] dB
pdsch.IsLastCbPresent = 0;
pdsch.isInlineTbCrc = 0;
pdsch.dlTbCrc = 0;
pdsch.testModel = 0; % support DL test models specified in 38.141-1 4.9.2.3
% for LBRM
pdsch.I_LBRM = 1;
pdsch.maxLayers = 4;
pdsch.maxQm = 8;
pdsch.n_PRB_LBRM = 273;
pdsch.Xtf_remap = [];
pdsch.seed = 0;
pdsch.payload = [];
pdsch.idxUE = 0;
pdsch.idxUeg = 0;
pdsch.nlAbove16 = 0;

return
