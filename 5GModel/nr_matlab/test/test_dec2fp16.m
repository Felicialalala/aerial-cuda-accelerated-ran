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

clear all;
close all;

x = [
   0,
   1,
   hex2dec('3ff'),
   hex2dec('400'),
   hex2dec('3555'),
   hex2dec('3bff'),
   hex2dec('3c00'),
   hex2dec('3c01'),
   hex2dec('7bff'),
   hex2dec('7c00'),
   hex2dec('8000'),
   hex2dec('c000'),
   hex2dec('fc00')
];

y_gold = [
   0,
   2^(-14) * (1/1024),
   2^(-14) * (1023/1024),
   2^(-14),
   2^(-2) * (1 + 341/1024),
   2^(-1) * (1 + 1023/1024),
   1,
   1 + 1/1024,
   2^15 * (1+1023/1024),
   inf,
   -0,
   -2,
   -inf
];

y = dec2fp16(x);
e = y ~= y_gold;
sum(e)

