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

function testInitialAccess(SysPar)

SysPar = initSysPar;

% ssb related config
SysPar.ssb.ssbSubcarrierOffset = 12;   % subcarrier offset
SysPar.ssb.SsbOffsetPointA = (136-10)*2;        % RB offset to Point A

% pdsch related config
SysPar.pdsch{1}.mcsIndex = 1;
SysPar.pdsch{1}.rbStart = 137+10;
SysPar.pdsch{1}.rbSize = 100;

SysPar.SimCtrl.timeDomainSim = 1;
SysPar.SimCtrl.enableUeRx = 1;
SysPar.SimCtrl.N_slot_run = 4;
SysPar.SimCtrl.plotFigure.tfGrid = 0; 
SysPar.SimCtrl.enableCS = 1;

SysPar.Chan{1}.SNR = 0;

[SysPar, UE, gNB] = nrSimulator(SysPar);

return




    