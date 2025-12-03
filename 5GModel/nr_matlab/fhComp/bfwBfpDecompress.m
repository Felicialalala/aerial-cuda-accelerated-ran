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

function bfw = bfwBfpDecompress(cSamples_uint8, iqWidth, beta, nRxAnt, nPrbGrpBfw, nLayers)

cSamples_uint8 = cSamples_uint8(:)';

bfw = bfw_bfp_decompress(cSamples_uint8, iqWidth, nRxAnt, nPrbGrpBfw*nLayers);

bfw = reshape(bfw,nRxAnt, nPrbGrpBfw, nLayers);
bfw = permute(bfw, [1 3 2]);

bfw = bfw/beta;

return

function bfw = bfw_bfp_decompress(cSamples_uint8, iqWidth, nSampPerBatch, nBatch)

BatchSizeBytes = (iqWidth*2*nSampPerBatch)/8 + 1;

bfw = zeros(nBatch*nSampPerBatch,1);
for k=1:nBatch
    cStart = (k-1)*BatchSizeBytes + 1;
    cEnd = cStart + BatchSizeBytes-1;
    cBytes = cSamples_uint8(cStart:cEnd);
    cExp = double(cBytes(1));
    bitStream = dec2bin(cBytes(2:BatchSizeBytes),8);
    reshapeBitStream = transpose(reshape(transpose(bitStream),iqWidth,nSampPerBatch*2));

    nStart = (k-1)*nSampPerBatch + 1;
    nEnd = nStart + nSampPerBatch - 1;
    x = bin2dec(reshapeBitStream);
    x(x >= 2^(iqWidth-1)) = x(x >= 2^(iqWidth-1)) - 2^iqWidth;
    x = x * 2^cExp;
    bfw(nStart:nEnd) = x(1:2:end) + j*x(2:2:end);
end

return
