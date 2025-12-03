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

function idxAlloc = findAlloc(alloc, type)
% function idxAlloc = findAlloc(alloc, type)
%
% This function finds the index to the alloc with a specifed type
%

findFlag = 0;
for idxAlloc = 1:length(alloc)
    if strcmp(alloc{idxAlloc}.type, type)
        findFlag = 1;
        break;
    end
end

if findFlag == 0
    idxAlloc = 0;
end
