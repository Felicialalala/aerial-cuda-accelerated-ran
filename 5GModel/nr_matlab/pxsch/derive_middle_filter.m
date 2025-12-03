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

function W_middle = derive_middle_filter(N0,df,win_size)

% function derives mmse interpolation filter for a cluster of two prbs. Extracts
% additional dmrs signal from adjacent lower and upper Prbs.
%inputs:
% N0 --> noise variance
% df --> subcarrier spacing
global SimCtrl;
if nargin < 3
    delaySpread = 2e-6; %  sp.gnb.pusch.reciever.ChEst.delaySpread;     % delay spread assumed by channel esitmation block (s)    
    if SimCtrl.delaySpread > 0
        delaySpread = SimCtrl.delaySpread;
    end
else
    delaySpread = win_size;
end

channelEnergy = 1;            % assume average channel energy is 1
dmrsEnergy    = 2;            % type 1 dmrs, no data interleaving

% outputs:
% W_lower --> lower interpolation filter. Dim: 25 x 24


%%
%SETUP

% sizes:
Nfc = 24;              % number of subcarriers to estimation channel on
Nfc_dmrs = 24;         % number of subcarriers to extract dmrs from

% data grid:
f = (12) : (12 + 24);
f = df * f';

% dmrs grid:
f_dmrs = 0 : (Nfc_dmrs - 1);
f_dmrs = 2 * f_dmrs + 1;
f_dmrs = df * f_dmrs';

% fOCC:
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

W_middle = RXY * pinv(RYY);

%%
%EXPECTED ERROR

% REE = W_middle*RYY*W_middle' - W_middle*RXY' - RXY*W_middle' + eye(25);
% Ree = diag(REE);
% Ree = Ree(2 : end);
% snr = -10*log10(mean(diag(REE)))

% a = 2;











