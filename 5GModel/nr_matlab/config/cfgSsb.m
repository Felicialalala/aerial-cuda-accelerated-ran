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

function ssb = cfgSsb

% ssb related config
ssb.ssbSubcarrierOffset = 12;   % subcarrier offset
ssb.SsbOffsetPointA = 6;        % RB offset to Point A
ssb.periodFrame = 1;            % SSB set period (frame)
ssb.caseType = 'case_C';        % SSB set case type
ssb.n_hf = 0;                   % half frame index
% support up to 4 SSBs per slot per cell
ssb.prcdBf_vec = [0 0 0 0];     
% ssb 0 
ssb.digBFInterfaces_0 = 1;
ssb.beamIdx_0 = [1];
% ssb 1
ssb.digBFInterfaces_1 = 1;
ssb.beamIdx_1 = [2];
% ssb 2
ssb.digBFInterfaces_2 = 1;
ssb.beamIdx_2 = [3];
% ssb 3
ssb.digBFInterfaces_3 = 1;
ssb.beamIdx_3 = [4];

% The SSB power control parameter in SCF-FAPIv2 is used here.
% 0 = 0 dB PSS to SSS power ratio
% 1 = 3 dB PSS to SSS power ratio
ssb.betaPss = 0;
ssb.seed = 0;
% rng(ssb.seed);
ssb.mib = round(rand(1, 24));   % hex2dec('012345');    % payload for mib

% When ssb.cfgTx = 1, set the following ssb params.
% Otherwise, they will be derived based on [fc, duplex, caseType]
ssb.cfgTx = 1;
ssb.L_max = 8; 
ssb.symIdxInFrame = [2, 8, 16, 22, 30, 36, 44, 50];
ssb.ssbBitMap = [0 1 0 0 0 0 0 0];

ssb.idxUE = 0;

return
