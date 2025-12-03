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
function [bytes] = bits2bytes(bits)
    j = 1;
    nBits = length(bits);
    nBytes = ceil(nBits/8);
    bytes = uint8(zeros(1,nBytes));
    for i = 1:8:nBits
        startBitPos = i;
        endBitPos = i+7;
        if endBitPos > nBits
            endBitPos = nBits;            
        end
        % flip - put first bit in LSB pos
        % string - stringyfy 8 bits and join in prep to bin2dec
        % bin2dec - convert byte to decimal
        bytes(j) = uint8(bin2dec(strjoin(string(flip(bits(startBitPos:endBitPos))))));
        j = j+1;
    end    
end
