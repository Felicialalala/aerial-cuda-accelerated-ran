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

% function cSamples = fh_bfp_compress(ucSamples, iqWidth, bypass)

% Compresses data compressed using block floating point algorithm (O-RAN
% WG4.CUS.0-v02.00, Section A.1.1
% inputs:
%   ucSamples, uncompressed I+Q data arranged in PRBs (uni-dimensional
%   array)
%   iqWidth, bit width of compressed data (9 bit and 14 bit currently
%   supported)
%   bypass, flag to bypass compression (when set to 1)
% outputs:
%   cSamples, compressed I+Q data arranged in PRBs (uni-dimensional
%   array)

% Load data from test vector
% tvData = hdf5_load_nv('GPU_test_input/TV_cuphy_perf-pusch-TC310_snrdb40.00_MIMO1x4_PRB272_DataSyms11_qam64.h5');
% 
% %% extract entry DataRx
% data = tvData.DataRx(:);
% dataRxSize = size(tvData.DataRx);
% 
% %% Output TV bin file to write into
% endianess = 'ieee-le';
% fTV = 'TV_cuphy_perf-pusch-TC310_snrdb40.00_MIMO1x4_PRB272_DataSyms11_qam64_iq.bin';
% 
% if exist(fTV, 'file')==2
%   delete(fTV);
% end
% 
% fid = fopen(fTV, 'w');
% 
% %% Interleave and write entry into bin file
% if ~isreal(data)
%     dataInterleaved = zeros(2*prod(dataRxSize),1);
%     dataInterleaved(1:2:end) = real(data.');
%     dataInterleaved(2:2:end) = imag(data.');
% else
%     dataInterleaved = zeros(prod(dataRxSize),1);
%     dataInterleaved(:) = data.';
% end
% fwrite(fid, dataInterleaved.', 'single', endianess);
% fclose(fid);
% 
% fid = fopen(fTV);
% dataRxReadStrm = fread(fid, 'single', endianess);
% fclose(fid);
% 
% %% Verify
% dataRxRead = dataRxReadStrm(1:2:end) + 1i*dataRxReadStrm(2:2:end);
% dataRxRead = reshape(dataRxRead, dataRxSize);
% %compare(dataRxRead, tvData.DataRx, 'Input samples', 1);

%ucSamples = data(1:12*20).';
ucSamples = [ 0.707106781186548 + 1i*0.707106781186547
-0.707106781186547 + 1i*0.707106781186548
-0.707106781186548 - 1i*0.707106781186547
  0.707106781186547 - 1i*0.707106781186548
  0.707106781186548 + 1i*0.707106781186547
-0.707106781186547 + 1i*0.707106781186548
-0.707106781186548 - 1i*0.707106781186547
  0.707106781186547 - 1i*0.707106781186548
  0.707106781186548 + 1i*0.707106781186547
-0.707106781186547 + 1i*0.707106781186548
-0.707106781186548 - 1i*0.707106781186547
  0.707106781186547 - 1i*0.707106781186548];

ucSamples = [0.383544922 + 1i*0.383544922
-0.383544922 + 1i*0.383544922
0.383544922 - 1i*0.383544922
-0.383544922 - 1i*0.383544922
0.84375 + 1i*0.383544922
-0.84375 + 1i*0.383544922
0.84375	- 1i*0.383544922
-0.84375 - 1i*0.383544922
0.383544922	+ 1i*0.84375
-0.383544922 + 1i*0.84375
0.383544922	- 1i*0.84375
-0.383544922 - 1i*0.84375];

ucSamples = zeros(12,1) + 1i*zeros(12,1);

%beta = 3316293.30546018;
beta = 3316293.25000; %rounded 
iqWidth = 14;
bypass = 0;
ucSamples = (double(fp16nv(real(ucSamples), 2)) + 1i*double(fp16nv(imag(ucSamples), 2)))*beta;

cSamples_uint8_ReadStrm = fh_bfp_compress(ucSamples, iqWidth, bypass);

% Verify

% Read file
% fid = fopen(fTV);
% cSamples_uint8_ReadStrm = fread(fid, 'uint8');
% fclose(fid);

fSamples = [];
% Decompress samples (14-bit compression)
prblen = (24*iqWidth+8)/8;
numprb = length(cSamples_uint8_ReadStrm)/prblen;
%numprb = length(cSamples_uint8)/prblen;

for ii=1:numprb
    cPRB = cSamples_uint8_ReadStrm((ii-1)*prblen+1:ii*prblen);
    %exponent = cPRB(end);
    exponent = cPRB(1);
    cPRB(1) = [];
    scaler = 2^exponent;
    % expand to bits, re-arrange in blocks of 14 samples
    bitPRB = '';
    for jj=1:prblen-1
        bitPRB = strcat(bitPRB, dec2bin(cPRB(jj),8));
    end
    for jj=1:12
        if (bitPRB((jj-1)*2*iqWidth+1)) == '0' && (bitPRB((jj-1)*2*iqWidth+15)) == '0'
             fPRB(jj) = scaler * bin2dec(bitPRB((jj-1)*2*iqWidth+2:jj*2*iqWidth-iqWidth)) + ...
                1i *  (scaler * bin2dec(bitPRB((jj-1)*2*iqWidth+16:jj*2*iqWidth)));
        elseif (bitPRB((jj-1)*2*iqWidth+1)) == '0' && (bitPRB((jj-1)*2*iqWidth+15)) == '1'
            fPRB(jj) = scaler * bin2dec(bitPRB((jj-1)*2*iqWidth+2:jj*2*iqWidth-iqWidth)) + ...
                1i *  -(scaler * bin2dec(bitPRB((jj-1)*2*iqWidth+16:jj*2*iqWidth)));
        elseif (bitPRB((jj-1)*2*iqWidth+1)) == '1' && (bitPRB((jj-1)*2*iqWidth+15)) == '0'
             fPRB(jj) = -scaler * bin2dec(bitPRB((jj-1)*2*iqWidth+2:jj*2*iqWidth-iqWidth)) + ...
                1i *  (scaler * bin2dec(bitPRB((jj-1)*2*iqWidth+16:jj*2*iqWidth)));
        elseif (bitPRB((jj-1)*2*iqWidth+1)) == '1' && (bitPRB((jj-1)*2*iqWidth+15)) == '1'
             fPRB(jj) = -scaler * bin2dec(bitPRB((jj-1)*2*iqWidth+2:jj*2*iqWidth-iqWidth)) + ...
                1i *  -(scaler * bin2dec(bitPRB((jj-1)*2*iqWidth+16:jj*2*iqWidth)));
        end
    end
    fSamples = [fSamples fPRB];
end
% Compare IQ samples
%plot(abs(ucSamples - fSamples))
abs(ucSamples - fSamples.')/var(ucSamples)
%compare(dataRxRead, tvData.DataRx, 'Input samples', 1);
%dec2hex(cPRB)







