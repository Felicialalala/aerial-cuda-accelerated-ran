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

function [X,error] = CRC_decode(Y,crcstr, table)

%function performs crc check and removes crc bits

%input:
%Y      --> crc encoded bits
%crcstr --> string indicating which crc is used

%outputs:
%X     --> Y with crc bits removed
%error --> 0 or 1. Indicates if crc passes or fails


%%
%SETUP

%number of crc bits
switch crcstr
    case('24A')
        ncrc = 24;
    case('24B')
        ncrc = 24;
    case('16')
        ncrc = 16;
end


%%
%START

%extract crc bits from data:
crc1 = Y(end - ncrc + 1 : end);

%remove crc bits from bit:
X = Y(1 : end - ncrc);

%compute crc:
[~,crc2] = add_CRC_LUT(X,crcstr, table);

%check if they are the same:
if sum(abs(crc1 - crc2)) > 0
    error = 1;
else
    error = 0;
end

%remove crc bits:
X = Y(1 : end - ncrc);

end

