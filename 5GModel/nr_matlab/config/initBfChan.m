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

function chan_BF = initBfChan(srsChEstBuffPrms)

nUeAnt     = srsChEstBuffPrms.nUeAnt;
nGnbAnt    = srsChEstBuffPrms.nGnbAnt;
prbGrpSize = srsChEstBuffPrms.prbGrpSize;
gainDB     = srsChEstBuffPrms.gainDB;

chan_BF = sqrt(10^(gainDB/10)) * sqrt(1/2) * (randn(ceil(273/prbGrpSize),nGnbAnt, nUeAnt) + 1i*randn(ceil(273/prbGrpSize),nGnbAnt, nUeAnt));

% Make partial PRG duplicate of prior as means of interpolating for partial bundle
chan_BF(ceil(273/prbGrpSize),:,:) = chan_BF(floor(273/prbGrpSize),:,:);


end
