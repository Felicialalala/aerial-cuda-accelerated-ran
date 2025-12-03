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

function data_est = estimate_pucch_data(H_est,Y_data_iue)

%function applies a matched filter to estimate pucch data

%inputs:
%H_est      --> estimate of pucch channel. Dim: 12 x nSym_data x L_BS
%Y_data_iue --> pucch data signal. Dim: 12 x nSym_data x L_BS

%outputs:
%data_est      --> hard estimate of pucch data (scaler).

%%
%START

%apply match filter:
m = conj(H_est) .* Y_data_iue;
m = sum(m(:));

%hard slice:
if real(m) <= 0
    data_est_real = -1 / sqrt(2);
else
    data_est_real = 1 / sqrt(2);
end

if imag(m) <= 0
    data_est_imag = -1 / sqrt(2);
else
    data_est_imag = 1 / sqrt(2);
end

data_est = data_est_real + 1i*data_est_imag;

end
