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

    RIV = 273*(273 - 273 + 1) + (273 - 1 - (1-1));
    RIV_length = ceil(log2(273*(273+1)/2));
    DCI_bitFields = gen_DCIFormat0_0_payload(RIV_length);
    DCI_bitFields.Identifier = 0; % always set to 0, indicating an UL DCI format, refer to Sec. 7.3.1.1.1 in TS 38.212
    DCI_bitFields.FrequencyDomainResources = RIV;
    DCI_bitFields.TimeDomainResources = 0; % hard coding for now
    DCI_bitFields.FrequencyHoppingFlag = 0;
    DCI_bitFields.ModulationCoding = 1;
    DCI_bitFields.NewDataIndicator = 0;
    DCI_bitFields.RedundancyVersion = 0;
    DCI_bitFields.HARQprocessNumber = 0;
    DCI_bitFields.TPCcommand = 0; % hard coding for now
    DCI_bitFields.AlignedWidth = 40;
    payload_bits = toBits(DCI_bitFields);