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

function r = oranDecompress_fromFile(filename,Nsym,Nprb,Nant,beta_scale)

iqWidth = 14;

if (nargin == 4)
    % If beta_scale isn't provided, calculate it from default ORAN CUS parameters
    MAX_FP16 = 65504;
    m = 9;
    e = 4;
    FSoffset = 0;
    FS0 = (2^(m-1)*2^(2^e-1))^2;
    FS = FS0 * 2^(-FSoffset);
    beta_scale = MAX_FP16 / sqrt(FS)
end

%filename = '~/git/5GModel/scripts/20210219_120654_frame_118_subframe_2_slot_1_iq.txt';
%filename = '~/git/5GModel/scripts/20210219_144558_frame_111_subframe_2_slot_1_iq.txt';
FID = fopen(filename);
c = textscan(FID, '%s');

cSamples_uint8 = hex2dec(c{1});

r = oranDecompress(cSamples_uint8,iqWidth,Nsym,Nprb,Nant,beta_scale);
%r = oranDecompress(cSamples_uint8,14,1,273,1);
