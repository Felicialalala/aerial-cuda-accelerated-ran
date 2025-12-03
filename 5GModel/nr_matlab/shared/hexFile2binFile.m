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

function hexFile2binFile(hexFileName, binFileName, hexFormat, lenByte)

% hexFormat = 0
% 3E2803...
%
% hexFormat = 1
% 3E
% 28
% 03
% ...

if nargin < 3
    hexFormat = 0;
end

hexFileId = fopen(hexFileName);
binFileId = fopen(binFileName, 'w');

if hexFormat == 0
    hexData = textscan(hexFileId, '%s');
    nByte = length(hexData{1}{1}) / 2;

    for n = 1:nByte
        byt = hexData{1}{1}(2*n-1:2*n);
        bin = dec2bin(hex2dec(byt), 8)-'0';
        for m = 1:8
            fprintf(binFileId, '%1d,', bin(m));
        end
        fprintf(binFileId, '\n');
    end
elseif hexFormat == 1
    hexData = textscan(hexFileId, '%x');
    nByte = size(hexData{1});

    for n = 1:nByte
        byt = hexData{1}(n);
        bin = dec2bin(byt, 8)-'0';
        for m = 1:8
            fprintf(binFileId, '%1d,', bin(m));
        end
        fprintf(binFileId, '\n');
    end
end

fclose(hexFileId);

if (nargin >= 4)
    PAD_BYTES = lenByte - nByte;
    display(['Adding ',num2str(PAD_BYTES),' pad bytes']);
    for k = 1:PAD_BYTES
       for m = 1:8
           fprintf(binFileId, '%1d,', 0);
       end
       fprintf(binFileId, '\n');
    end
end

fclose(binFileId);
