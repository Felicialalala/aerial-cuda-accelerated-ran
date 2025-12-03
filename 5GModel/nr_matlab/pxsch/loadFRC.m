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

function pusch = loadFRC(pusch_in, FRCcfg)

pusch = pusch_in;

switch FRCcfg
    case 'L1T1M00B024'
        pusch.mcsIndex = 0;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M01B024'
        pusch.mcsIndex = 1;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M02B024'
        pusch.mcsIndex = 2;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M03B024'
        pusch.mcsIndex = 3;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M04B024'
        pusch.mcsIndex = 4;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M05B024'
        pusch.mcsIndex = 5;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M06B024'
        pusch.mcsIndex = 6;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M07B024'
        pusch.mcsIndex = 7;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M08B024'
        pusch.mcsIndex = 8;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M09B024'
        pusch.mcsIndex = 9;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M10B024'
        pusch.mcsIndex = 10;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M11B024'
        pusch.mcsIndex = 11;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M12B024'
        pusch.mcsIndex = 12;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M13B024'
        pusch.mcsIndex = 13;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M14B024'
        pusch.mcsIndex = 14;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M15B024'
        pusch.mcsIndex = 15;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M16B024'
        pusch.mcsIndex = 16;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M17B024'
        pusch.mcsIndex = 17;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M18B024'
        pusch.mcsIndex = 18;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M19B024'
        pusch.mcsIndex = 19;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M20B024'
        pusch.mcsIndex = 20;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M21B024'
        pusch.mcsIndex = 21;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M22B024'
        pusch.mcsIndex = 22;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M23B024'
        pusch.mcsIndex = 23;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M24B024'
        pusch.mcsIndex = 24;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M25B024'
        pusch.mcsIndex = 25;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M26B024'
        pusch.mcsIndex = 26;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M27B024'
        pusch.mcsIndex = 27;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L2T1M00B024'
        pusch.mcsIndex = 0;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L2T1M09B024'
        pusch.mcsIndex = 9;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L2T1M18B024'
        pusch.mcsIndex = 18;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L2T1M27B024'
        pusch.mcsIndex = 27;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T4M00B024'
        pusch.mcsIndex = 0;
        pusch.mcsTable = 4;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T4M01B024'
        pusch.mcsIndex = 1;
        pusch.mcsTable = 4;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T4M02B024'
        pusch.mcsIndex = 2;
        pusch.mcsTable = 4;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T4M03B024'
        pusch.mcsIndex = 3;
        pusch.mcsTable = 4;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T4M04B024'
        pusch.mcsIndex = 4;
        pusch.mcsTable = 4;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T4M05B024'
        pusch.mcsIndex = 5;
        pusch.mcsTable = 4;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M00B273'
        pusch.mcsIndex = 0;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M01B273'
        pusch.mcsIndex = 1;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M02B273'
        pusch.mcsIndex = 2;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M03B273'
        pusch.mcsIndex = 3;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M04B273'
        pusch.mcsIndex = 4;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M05B273'
        pusch.mcsIndex = 5;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M06B273'
        pusch.mcsIndex = 6;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M07B273'
        pusch.mcsIndex = 7;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M08B273'
        pusch.mcsIndex = 8;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M09B273'
        pusch.mcsIndex = 9;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M10B273'
        pusch.mcsIndex = 10;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M11B273'
        pusch.mcsIndex = 11;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M12B273'
        pusch.mcsIndex = 12;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M13B273'
        pusch.mcsIndex = 13;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M14B273'
        pusch.mcsIndex = 14;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M15B273'
        pusch.mcsIndex = 15;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M16B273'
        pusch.mcsIndex = 16;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M17B273'
        pusch.mcsIndex = 17;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M18B273'
        pusch.mcsIndex = 18;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M19B273'
        pusch.mcsIndex = 19;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M20B273'
        pusch.mcsIndex = 20;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M21B273'
        pusch.mcsIndex = 21;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M22B273'
        pusch.mcsIndex = 22;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M23B273'
        pusch.mcsIndex = 23;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M24B273'
        pusch.mcsIndex = 24;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M25B273'
        pusch.mcsIndex = 25;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M26B273'
        pusch.mcsIndex = 26;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M27B273'
        pusch.mcsIndex = 27;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L2T1M00B273'
        pusch.mcsIndex = 0;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L2T1M09B273'
        pusch.mcsIndex = 9;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L2T1M18B273'
        pusch.mcsIndex = 18;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L2T1M27B273'
        pusch.mcsIndex = 27;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T4M00B273'
        pusch.mcsIndex = 0;
        pusch.mcsTable = 4;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T4M01B273'
        pusch.mcsIndex = 1;
        pusch.mcsTable = 4;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T4M02B273'
        pusch.mcsIndex = 2;
        pusch.mcsTable = 4;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T4M03B273'
        pusch.mcsIndex = 3;
        pusch.mcsTable = 4;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T4M04B273'
        pusch.mcsIndex = 4;
        pusch.mcsTable = 4;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T4M05B273'
        pusch.mcsIndex = 5;
        pusch.mcsTable = 4;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'L1T1M00B001'
        pusch.mcsIndex = 0; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 1; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14;         
    case 'L1T1M00B008'
        pusch.mcsIndex = 0; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 8; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14;                 
    case 'L1T1M19B008'
        pusch.mcsIndex = 19; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 8; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14;             
    case 'L2T1M19B273'
        pusch.mcsIndex = 19; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 273; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14;                
    case 'L2T1M27B008'
        pusch.mcsIndex = 27; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 8; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14;          
    case 'L3T1M27B273'
        pusch.mcsIndex = 27; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 3;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 273; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14;          
    case 'L4T1M27B273'
        pusch.mcsIndex = 27; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 4;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'G-FR1-A1-5'
        pusch.mcsIndex = 4;
        pusch.mcsTable = 0;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 51;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'G-FR1-A2-5'
        pusch.mcsIndex = 16;
        pusch.mcsTable = 0;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 51;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'G-FR1-A3-10'
        pusch.mcsIndex = 1;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 106;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'G-FR1-A3-14'
        pusch.mcsIndex = 1; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 273; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14;     
    case 'G-FR1-A3-24'
        pusch.mcsIndex = 1; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 106; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14;        
    case 'G-FR1-A3-28'
        pusch.mcsIndex = 1; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 273; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14;     
    case 'G-FR1-A3-31'
        pusch.mcsIndex = 1;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 25;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'G-FR1-A3-32'
        pusch.mcsIndex = 1;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'G-FR1-A3A-3'
        pusch.mcsIndex = 5;
        pusch.mcsTable = 2;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'G-FR1-A4-10'
        pusch.mcsIndex = 10; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 106; 
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'G-FR1-A4-11'
        pusch.mcsIndex = 10;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 24;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'G-FR1-A4-14'
        pusch.mcsIndex = 10;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 273; 
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'G-FR1-B4-14'
        pusch.mcsIndex = 10;
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 1 0 0 0 0 0 0 1 1 0 0];
        pusch.rbStart = 0;
        pusch.rbSize = 273;
        pusch.StartSymbolIndex = 0;
        pusch.NrOfSymbols = 14;
    case 'G-FR1-A4-24'
        pusch.mcsIndex = 10; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 106; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14;        
    case 'G-FR1-A4-28'
        pusch.mcsIndex = 10; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 2;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 273; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14; 
    case 'G-FR1-A5-10'
        pusch.mcsIndex = 13; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 106; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14;          
    case 'G-FR1-A5-14'
        pusch.mcsIndex = 13; 
        pusch.mcsTable = 1;
        pusch.nrOfLayers = 1;
        pusch.DmrsSymbPos = [0 0 1 0 0 0 0 0 0 0 0 1 0 0];
        pusch.rbStart = 0; 
        pusch.rbSize = 273; 
        pusch.StartSymbolIndex = 0; 
        pusch.NrOfSymbols = 14;        
    otherwise
        error('FRCcfg is not supported ... \n');
end

return
