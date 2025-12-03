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

function data_bytes = uint8_convert(data_bits, b7msb)

% function converts a bit array into a uint8 byte array.

%inputs:
% data_bits  --> array of bits
% b7msb -> 0: first input bit goes to MSB of first output byte 
%          1: first input bit goes to LSB of first output byte

%outputs:
% data_bytes --> array of uint8 bytes

%%
%SETUP

if nargin == 1
    b7msb = 1;
end    

data_bits = data_bits(:);

nBits = length(data_bits);
nBytes = ceil(nBits / 8);
padding = zeros(nBytes*8-nBits, 1);
data_bits = [data_bits; padding]; 

%%
%START

data_bits = reshape(data_bits,8,nBytes);

% b7msb -> 0: first input bit goes to MSB of first output byte 
%          1: first input bit goes to LSB of first output byte
if b7msb
    data_bits = data_bits .* ( 2.^([0:7])');
else
    data_bits = data_bits .* ( 2.^([7:-1:0])');
end
data_bytes = sum(data_bits,1);
data_bytes = uint8(data_bytes)';

end


