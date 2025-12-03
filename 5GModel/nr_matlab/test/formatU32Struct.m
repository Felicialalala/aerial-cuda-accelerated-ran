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

function struct = formatU32Struct(struct, singleField)

if nargin < 2
    singleField = {};
end

fieldName = fieldnames(struct);
nField = length(fieldName);

for k = 1:nField
    val = getfield(struct, fieldName{k});
    isSingle = ismember(fieldName{k}, singleField);
    if isSingle
        if ~isstruct(val)
            struct = setfield(struct, fieldName{k}, single(val));
        else
            struct = rmfield(struct, fieldName{k});
        end
    else
        if ~isstruct(val)
            struct = setfield(struct, fieldName{k}, uint32(val));
        else
            struct = rmfield(struct, fieldName{k});
        end
    end
end

return
