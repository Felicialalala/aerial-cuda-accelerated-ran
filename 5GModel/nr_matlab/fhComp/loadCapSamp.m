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

function Xtf = loadCapSamp(capFileName, sim_is_uplink, iqWidth, FSOffset, Ref_c, total_num_REs, max_amp_ul, Nprb, Nsym, Nant)

PURE_CSV=1;
fprintf('Load samples from %s\n', capFileName);
fid = fopen(capFileName);
if (iqWidth >= 9) && (iqWidth <= 14)
    c = textscan(fid, '%s');
    cSamples_uint8 = hex2dec(c{1});
    RE_PER_PRB = 12;
    PRB_SIZE_BYTES = (iqWidth*2*RE_PER_PRB)/8 + 1;
    len = Nprb*PRB_SIZE_BYTES*Nsym*Nant;
    if length(cSamples_uint8) < len
        cSamples_uint8(end+1:len) = 0;
    end

    beta_scale = oranCalcBeta(sim_is_uplink, iqWidth, FSOffset, Ref_c, total_num_REs, max_amp_ul);
    if (sim_is_uplink == 0)
        beta_scale = 1/beta_scale; % invert since we're reading from a file that already had this compression applied
    end

    Xtf = oranDecompress(cSamples_uint8,iqWidth,Nprb,Nsym,Nant,beta_scale);
elseif (PURE_CSV)
    r = csvread(capFileName);
    rc = r(:,1) + j*r(:,2);
    rcr = reshape(rc,Nprb*12,Nsym,Nant);
    Xtf = rcr;
else
    c = textscan(fid, '%s');
    cSamples_uint8 = hex2dec(c{1});
    RE_PER_PRB = 12;
    PRB_SIZE_BYTES = (iqWidth*2*RE_PER_PRB)/8;
    nPrb = length(cSamples_uint8)/PRB_SIZE_BYTES;
    for k=1:nPrb
        cStart = (k-1)*PRB_SIZE_BYTES+1;
        cEnd = cStart + PRB_SIZE_BYTES-1;
        cBytes = cSamples_uint8(cStart:cEnd);
        bitStream = dec2bin(cBytes(1:PRB_SIZE_BYTES),8);
        reshapeBitStream = transpose(reshape(transpose(bitStream),iqWidth,RE_PER_PRB*2));
        % Little endian conversion
        reshapeBitStream = [reshapeBitStream(:,9:16) reshapeBitStream(:,1:8)];

        nStart = (k-1)*RE_PER_PRB + 1;
        nEnd = nStart + RE_PER_PRB - 1;

        x = bin2dec(reshapeBitStream);
        x_fp16 = dec2fp16(x);
        X_tf(nStart:nEnd) = x_fp16(1:2:end) + j*x_fp16(2:2:end);
        %display(['cStart: ',num2str(cStart),' cEnd:',num2st
    end
    Xtf = reshape(X_tf,Nprb*RE_PER_PRB,Nant,Nsym);
    Xtf = permute(Xtf, [1 3 2]);
end

return
