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

function beta_scale = oranCalcBeta(sim_is_uplink, iqWidth, FSoffset, ref_c, total_num_REs, max_amp_ul)
% beta_scale = oranCalcBeta(sim_is_uplink, iqWidth, FSoffset, ref_c, total_num_REs)
% Calculate the beta scale factor from gNB perspective.
%    - for DL it's used for compression
%    - for UL it's used for decompression
%    - Calling function should invert beta if the intended use is opposite this.
if nargin < 3
    FSoffset = 0;
end
if nargin < 4
    ref_c = 0;
end
if nargin < 5
    total_num_REs = 273*12;
end   
if nargin < 6
    max_amp_ul = 65504;
end

m = iqWidth;
e = 4;
FS0 = (2^(m-1)*2^(2^e-1))^2;
FS = FS0 * 2^(-FSoffset);

if (sim_is_uplink)
   beta_scale = max_amp_ul / sqrt(FS);
else
    beta_scale = sqrt(FS*10^(ref_c/10)/total_num_REs);
end

beta_scale = double(single(beta_scale));
