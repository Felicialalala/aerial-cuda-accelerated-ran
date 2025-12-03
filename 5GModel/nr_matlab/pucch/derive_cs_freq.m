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

function cs_freq = derive_cs_freq

%function derives frequency domain representations of cyclic shifts

%outputs:
%cs_freq --> cyclic shift in freq domain. Dim: 12 x 12

%%
%START

cs_freq = eye(12);
cs_freq = fft(cs_freq,12,1);

% correction 12/17/19, conjugate cs_freq:
cs_freq = conj(cs_freq);

end

