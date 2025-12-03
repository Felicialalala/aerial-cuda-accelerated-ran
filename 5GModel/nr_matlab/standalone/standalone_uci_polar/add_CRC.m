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

function [y,crc] = add_CRC(x,crc_str)

%function computes an appends CRC bits

%inputs:
%x       --> input bit sequence
%crc_str --> string indicating which crc polynomial to use

%outputs:
%y      --> input bits + crc
%crc    --> crc bits

%%
%SETUP

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
    case '6'
        g = [1 1 0 0 0 0 1];
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
z = logical(z);

%long division:
for i = 1 : n
    if z(i)
        for j = 1 : r
            z(i+j) = xor(z(i+j),g(j+1)); 
        end
    end
end

%extract crc:
crc = z(end - r + 1 : end);

%append crc:
y = [x ; crc];

%logical case:
y = logical(y);

end




            
            
            


            

    
