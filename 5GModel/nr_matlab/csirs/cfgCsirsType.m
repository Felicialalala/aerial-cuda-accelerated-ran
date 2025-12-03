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

function csirsType = cfgCsirsType()
% cfgCsirsType - Centralized CSI-RS type constants
%
% This function returns a structure containing CSI-RS type constants
% to avoid hard-coding magic numbers across scripts.
%
% Returns:
%   csirsType - Structure containing CSI-RS type constants
%
% CSI-RS Types:
%   TRS_CSI_RS   = 0  % Time/frequency tracking reference signal
%   NZP_CSI_RS   = 1  % Non-zero power CSI-RS
%   ZP_CSI_RS    = 2  % Zero power CSI-RS

csirsType.TRS = 0;  % Time/frequency tracking reference signal
csirsType.NZP = 1;  % Non-zero power CSI-RS
csirsType.ZP  = 2;   % Zero power CSI-RS

return 