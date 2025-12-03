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

function bfw = cfgBfw

% updated PDU:
bfw.nUes  = 1;
bfw.bfwUL = 0;

% Per user paramaters:
bfw.RNTI           = 1;
bfw.pduIndex       = 0;
bfw.numOfUeAnt     = 1;
bfw.gnbAntIdxStart = 0;
bfw.gnbAntIdxEnd   = 31;
bfw.ueAntIdx0      = 0;
bfw.ueAntIdx1      = 0;
bfw.ueAntIdx3      = 0;
bfw.ueAntIdx4      = 0;

% ue grp common paramaters:
bfw.rbStart    = 0;
bfw.rbSize     = 272;
bfw.numPRGs    = 136;
bfw.prgSize    = 2;


return
