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

function crcTable = genCrcTable(crc_str)

switch crc_str
    case '24A'
        g = [1 1 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 1 1 1 1 1 0 1 1];
    case '24B'
        g = [1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1];
    case  '24C'
        g = [1 1 0 1 1 0 0 1 0 1 0 1 1 0 0 0 1 0 0 0 1 0 1 1 1];
    case '16'
        g = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1];  
    case '11'
        g = [1 1 1 0 0 0 1 0 0 0 0 1];
end
g1 = bin2dec(num2str(g(2:end)));
len = length(g)-1;
for k = 1:256
    byte = (k-1)*2^(len-8);
    for b = 1:8
        if byte >= 2^(len-1)
            byte = bitxor((byte-2^(len-1))*2, g1);
        else
            byte = byte*2;
        end
    end
    crcTable(k) = byte;
end

return
