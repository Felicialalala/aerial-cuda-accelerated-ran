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

function nrSlot = wrap_pucch_estimtate(b_est,Pucch_ue_cell,nrSlot)

%function save the pucch data estimates

%inputs:
%b_est --> estimates of pucch data. Dim: nUe_pucch x 2

%%
%START

nUe_pucch = size(b_est,1);

for iue = 1 : nUe_pucch
    nBits = Pucch_ue_cell{iue}.nBits;
    nrSlot.pucch.rxData_cell{iue}.b_est = b_est(iue,1:nBits);
end

end
