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

function [X_tf] = oranDecompress(cSamples_uint8,iqWidth,Nprb,Nsym,Nant,beta_scale, doPermute)

if nargin < 7
    doPermute = 1;
end

RE_PER_PRB = 12;

PRB_SIZE_BYTES = (iqWidth*2*RE_PER_PRB)/8 + 1;
nPrb = length(cSamples_uint8)/PRB_SIZE_BYTES;

X_tf = zeros(nPrb*RE_PER_PRB,1);
for k=1:nPrb
    cStart = (k-1)*PRB_SIZE_BYTES + 1;
    cEnd = cStart + PRB_SIZE_BYTES-1;
    cBytes = cSamples_uint8(cStart:cEnd);
    cExp = cBytes(1);
    bitStream = dec2bin(cBytes(2:PRB_SIZE_BYTES),8);
    reshapeBitStream = transpose(reshape(transpose(bitStream),iqWidth,RE_PER_PRB*2));

    nStart = (k-1)*RE_PER_PRB + 1;
    nEnd = nStart + RE_PER_PRB - 1;
    x = bin2dec(reshapeBitStream);
    x(x >= 2^(iqWidth-1)) = x(x >= 2^(iqWidth-1)) - 2^iqWidth;
    x = x * 2^cExp;
    X_tf(nStart:nEnd) = x(1:2:end) + j*x(2:2:end);
    %display(['cStart: ',num2str(cStart),' cEnd:',num2str(cEnd),' cExp: ',num2str(cExp)]);
end

if doPermute
    X_tf = reshape(X_tf,Nprb*RE_PER_PRB,Nant,Nsym);
    X_tf = permute(X_tf, [1 3 2]);
else
    X_tf = reshape(X_tf,Nprb*RE_PER_PRB,Nsym,Nant);
end
X_tf = beta_scale * X_tf;
