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

function prachFormat = convertPrachFormatToSCF(preambleFormat)

switch preambleFormat
    case '0'
        prachFormat = 0;
    case '1'
        prachFormat = 1;
    case '2'
        prachFormat = 2;
    case '3'
        prachFormat = 3;
    case 'A1'
        prachFormat = 4;
    case 'A2'
        prachFormat = 5;
    case 'A3'
        prachFormat = 6;
    case 'B1'
        prachFormat = 7;
    case 'B4'
        prachFormat = 8;
    case 'C0'
        prachFormat = 9;
    case 'C2'
        prachFormat = 10;
    case 'A1/B1'
        prachFormat = 11;
    case 'A2/B2'
        prachFormat = 12;
    case 'A3/B3'
        prachFormat = 13;
    otherwise
        error('preambleFormat is not supported ...\n');
end


return
        
