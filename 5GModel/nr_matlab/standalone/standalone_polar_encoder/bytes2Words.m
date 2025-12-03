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

 %%
function [words] = bytes2Words(bytes)
    j = 1;
    nBytes = length(bytes);
    nWords = ceil(nBytes/8);
    words = uint32(zeros(1,nWords));
    for i = 1:4:nBytes
        startBytePos = i;
        endBytePos = i+3;
        if endBytePos > nBytes
            endBytePos = nBytes;            
        end
        % flip - put first byte in LSB pos
        % string - stringyfy 8 byte and join in prep to bin2dec
        % bin2dec - convert word to decimal
        %words(j) = uint32(bin2dec(strjoin(string(flip(bytes(startBytePos:endBytePos))))));
        words(j) = typecast((bytes(startBytePos:endBytePos)), 'uint32');
        j = j+1;
    end    
end
