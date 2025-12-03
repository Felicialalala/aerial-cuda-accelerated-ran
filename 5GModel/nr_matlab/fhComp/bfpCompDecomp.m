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

function cSamples = bfpCompDecomp(X_tf,iqWidth, Ref_c, FSOffset, Nre_max, max_amp_ul, sim_is_uplink)

ucSamples = X_tf(:);

beta = oranCalcBeta(sim_is_uplink, iqWidth, FSOffset, Ref_c, Nre_max, max_amp_ul);
if (sim_is_uplink)
    beta = 1/beta; % invert since we're generating the UL signal
end

ucSamples = ucSamples * beta;

nSamp = length(ucSamples);
nPRB = nSamp/12;

iqSamples = floor([real(ucSamples(:))'; imag(ucSamples(:))']);

fPRB = reshape(iqSamples, 12*2, nPRB);
maxValue = max([max(fPRB); abs(min(fPRB))-1]);
maxValue(maxValue < 1) = 1; % Clamp minimum to 1

raw_exp = floor(log2(maxValue)+1);
exponent = max(raw_exp-iqWidth+1, zeros(1, nPRB));

scaler = 2.^-exponent;
sPRB = fPRB.*repmat(scaler, 24, 1);

cSamples = floor(sPRB(1:2:end, :)) + 1j * floor(sPRB(2:2:end, :));
cSamples = cSamples./repmat(scaler, 12, 1);

cSamples = cSamples/beta;

size_Xtf = size(X_tf);
cSamples = reshape(cSamples, size_Xtf);

return;

