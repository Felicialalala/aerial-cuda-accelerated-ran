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

function cSamples_uint8 = bfwBfpCompress(bfw, compressBitWidth, beta, nRxAnt, nLayers, nPrbGrpBfw)

% nRxAnt x nLayers x nPrbGrpBfw => nRxAnt x nPrbGrpBfw x nLayers
bfw = permute(bfw, [1, 3, 2]);

ucSamples = reshape(beta * bfw, nRxAnt * nPrbGrpBfw * nLayers, 1);

cSamples_uint8 = bfw_bfp_compress(ucSamples, compressBitWidth, nRxAnt, nPrbGrpBfw * nLayers);

BatchSizeBytes = (compressBitWidth*2*nRxAnt)/8 + 1;
cSamples_uint8 = reshape(cSamples_uint8, BatchSizeBytes, nPrbGrpBfw, nLayers);

return


function cSamples_uint8 = bfw_bfp_compress(ucSamples, iqWidth, nSampPerBatch, nBatch)

iqSamples = floor([real(ucSamples(:))'; imag(ucSamples(:))']);

fBatch = reshape(iqSamples, nSampPerBatch*2, nBatch);
maxValue = max([max(fBatch); abs(min(fBatch))-1]);
maxValue(maxValue < 1) = 1; % Clamp minimum to 1

raw_exp = floor(log2(maxValue)+1);
exponent = max(raw_exp-iqWidth+1, zeros(1, nBatch));

exponent_b = dec2bin(exponent, 8)';

scaler = 2.^-exponent;
sBatch = fBatch.*repmat(scaler, 2*nSampPerBatch, 1);

negComp = 2^iqWidth;
cBatch = sBatch + (sBatch < 0).*negComp;

bBatch0 = dec2bin(cBatch, iqWidth);
bBatch1 = bBatch0';
bBatch2 = reshape(bBatch1, 2*nSampPerBatch*iqWidth, nBatch);

bBatch3 = [exponent_b; bBatch2];

bBatch4 = bBatch3(:);

bBatch5 = reshape(bBatch4, 8, length(bBatch4)/8);

cSamples_uint8 = bin2dec(bBatch5');

cSamples_uint8 = cSamples_uint8';

return

