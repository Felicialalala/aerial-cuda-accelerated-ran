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

function bitMap = rbgIdx2rbMap(idxRBG, P, BWPStart, BWPSize)

RBGsize_0 = P-mod(BWPStart, P);
if mod(BWPStart + BWPSize, P) > 0
    RBGsize_last = mod(BWPStart + BWPSize, P);
else
    RBGsize_last = P;
end
nRBG = (BWPSize - (RBGsize_0 + RBGsize_last))/P + 2;

bitMap = zeros(1, 36*8);
for idx = idxRBG
    if idx == 0
        bitMap(1:RBGsize_0) = 1;
    elseif idx < nRBG-1
        bitMap(((idx-1)*P+1:idx*P)+RBGsize_0) = 1;
    elseif idx == nRBG-1
        bitMap(((idx-1)*P+1:(idx-1)*P+RBGsize_last)+RBGsize_0) = 1;
    else
        error('idxRBG is out of bound ...');
    end
end

return