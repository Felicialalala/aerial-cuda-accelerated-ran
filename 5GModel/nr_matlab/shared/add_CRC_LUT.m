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

function [ y, crc ] = add_CRC_LUT(x,crc_str, crc_table)

%function computes an appends CRC bits

%inputs:
%x       --> input bit sequence
%crc_str --> string indicating which crc polynomial to use

%outputs:
%y      --> input bits + crc
%crc    --> crc bits

%%
%SETUP

if nargin < 3
    useCrcTable = 0;
else
    useCrcTable = 1;
end

switch crc_str
    case '24A'
        g = [1 1 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 1 1 1 1 1 0 1 1];
        if useCrcTable
            crcTable = crc_table.crcTable_24A;
        end
    case '24B'
        g = [1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1];
        if useCrcTable
            crcTable = crc_table.crcTable_24B;
        end
    case  '24C'
        g = [1 1 0 1 1 0 0 1 0 1 0 1 1 0 0 0 1 0 0 0 1 0 1 1 1];
        if useCrcTable
            crcTable = crc_table.crcTable_24C;
        end
    case '16'
        g = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1];
        if useCrcTable
            crcTable = crc_table.crcTable_16;
        end
    case '11'
        g = [1 1 1 0 0 0 1 0 0 0 0 1];
        if useCrcTable
            crcTable = crc_table.crcTable_11;
        end
end

g = logical(g);

%compute constants:
n = length(x);     % number of input bits
r = length(g) - 1; % number of crc bits

%%
%START

%copy sequence:
z = x;

%append zero bits:
z = [z ; zeros(r,1)];

%long division:

if useCrcTable
    z1 = reshape(z, 8, length(z)/8)';
    v = 2.^[7:-1:0];
    z2 = z1 * v';    
    crc1 = uint32(0);               
    for i = 1:n/8
        byt = uint32(z2(i));
        pos = bitshift(bitxor(crc1, bitshift(byt, r-8)), -(r-8));
        crc1 = bitxor(mod(bitshift(crc1, 8), 2^r), crcTable(pos+1));
    end
    crc1 = dec2bin(crc1, r)-'0';
    crc = crc1';
else
    z = logical(z);
    g1 =logical(g(1+1:1+r)');
    j = [1:r];
    for i = 1 : n
        if z(i)
            z(i+j) = xor(z(i+j), g1);
        end
    end
    crc = z(end - r + 1 : end);
end

%append crc:
y = [x ; crc];

%logical case:
y = logical(y);

end
