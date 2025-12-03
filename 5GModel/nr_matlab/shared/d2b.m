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

function y = d2b(x)
% Convert a decimanl number into a binary array
% 
% Similar to dec2bin but yields a numerical array instead of a string and is found to
% be rather faster
if x == 0
    y = 0;
else
    c = ceil(log(x)/log(2)); % Number of divisions necessary ( rounding up the log2(x) )
    y(c) = 0; % Initialize output array
    for i = 1:c
        r = floor(x / 2);
        y(c+1-i) = x - 2*r;
        x = r;
    end
end
