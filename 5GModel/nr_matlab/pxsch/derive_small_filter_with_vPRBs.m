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

function W = derive_small_filter_with_vPRBs(nprb,nprb_valid, nprb_offset, N0,df)

% function derives mmse interpolation filter for a small cluster of prbs

% inputs:
% nprb --> number of prbs in cluster
% N0 --> noise variance
% df --> subcarrier spacing
delaySpread   = 2.00*10^(-6); % assume CP length

global SimCtrl
if SimCtrl.delaySpread > 0
    delaySpread = SimCtrl.delaySpread;
end

channelEnergy = 1;            % assume average channel energy is 1
dmrsEnergy    = 2;            % type 1 dmrs, no data interleaving

% outputs:
% W         --> interpolation filter. Dim: (Nfc + 1) x Nfc_dmrs

%%
%SETUP

% sizes:
Nfc = nprb * 12;
Nfc_dmrs = nprb * 6;
Nfc_valid = nprb_valid * 12;

% data grid:
f = (0 : Nfc_valid);
f = df * f';

% dmrs grid:
f_dmrs = (0 : 2 : (Nfc - 2)) + nprb_offset*12;
f_dmrs = f_dmrs + 1;
f_dmrs = df * f_dmrs';

% fOCC
fOCC = ones(Nfc_dmrs,1);
fOCC(mod(1:Nfc_dmrs,2) == 0) = -1;

%%
%COVARIANCE

% channel covariance:
[F1,F2] = meshgrid(f_dmrs,f_dmrs);
RYY = sinc_tbf(delaySpread*(F1 - F2));

% add fOCC:
RYY = (fOCC * fOCC' + ones(Nfc_dmrs)) .* RYY;

% add energy:
RYY = dmrsEnergy * channelEnergy * RYY;

% add noise:
RYY = RYY + N0 * eye(Nfc_dmrs);

%%
%CORRELATION

% channel correlation:
[F1,F2] = meshgrid(f_dmrs,f);
RXY = sinc_tbf(delaySpread*(F1 - F2));

% add energy:
RXY = sqrt(dmrsEnergy) * channelEnergy * RXY;

%%
%FILTER

W = RXY * pinv(RYY);

a = 2;











