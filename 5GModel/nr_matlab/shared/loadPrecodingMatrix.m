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

function [enablePrcdBf, PM_W] = loadPrecodingMatrix(prcdBf, table)

global SimCtrl

if ~isempty(prcdBf)
    if prcdBf > 0
        enablePrcdBf = 1;
        PM_W = table.PM_W{prcdBf};        
        PM_W = fp16nv(real(PM_W), SimCtrl.fp16AlgoSel) + 1j*fp16nv(imag(PM_W), SimCtrl.fp16AlgoSel);
    else
        enablePrcdBf = 0;
        PM_W = [];
    end
else
    enablePrcdBf = 0;
    PM_W = [];
end

return