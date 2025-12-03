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

function pdcch = cfgPdcch

% pdcch related config
pdcch.BWPSize = 273;
pdcch.BWPStart = 0;
pdcch.StartSymbolIndex = 0;
pdcch.DurationSymbols = 2;
pdcch.coresetIdx = 1; % set to 0 for coreset0
pdcch.coresetMap = [1 1 1 1 1 1 1 1];
pdcch.CceRegMappingType = 0; 
pdcch.RegBundleSize = 6;
pdcch.InterleaverSize = 2;
pdcch.ShiftIndex = 1;
pdcch.precoderGranularity = 0;
pdcch.numDlDci = 1;
pdcch.testModel = 0; % support DL test models specified in 38.141-1 4.9.2.3
pdcch.isCSS = 0;
pdcch.forceCceIndex = 0;
for idxDCI = 1:pdcch.numDlDci
    DCI.RNTI = idxDCI*3;
    DCI.ScramblingId = idxDCI*7;
    DCI.ScramblingRNTI = DCI.RNTI+1;    

    DCI.beta_PDCCH_1_0 = 8; % SCF-FAPIv2 not used [0:16] = [-8:8] dB
    DCI.powerControlOffsetSS = 1; % SCF-FAPIv2 PDCCH/SSB, [0, 1, 2, 3] = [-3, 0, 3, 6] dB
    DCI.powerControlOffsetSSProfileNR = 0; % SCF-FAPIv4 PDCCH/SSB, [-8:8] = [-8:8] dB
    DCI.AggregationLevel = 1;
    DCI.cceIndex = idxDCI - 1;
    DCI.PayloadSizeBits = 45;
    DCI.prcdBf = 0;
    DCI.digBFInterfaces = 1;
    DCI.beamIdx = [1];
    DCI.seed = 0;
%     rng(DCI.seed);
    DCI.Payload = round(rand(1, DCI.PayloadSizeBits));
    pdcch.DCI{idxDCI} = DCI;
end
pdcch.dciUL = 0;
pdcch.idxUE = 0;

return
