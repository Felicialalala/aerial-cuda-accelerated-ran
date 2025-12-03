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

function cSamples_uint8 = fh_bfp_compress(ucSamples, iqWidth, bypass)

alg_sel = 0;

if alg_sel == 0

    nSamp = length(ucSamples);
    nPRB = nSamp/12;

    iqSamples = floor([real(ucSamples(:))'; imag(ucSamples(:))']);

    fPRB = reshape(iqSamples, 12*2, nPRB);
    maxValue = max([max(fPRB); abs(min(fPRB))-1]);
    maxValue(maxValue < 1) = 1; % Clamp minimum to 1

    raw_exp = floor(log2(maxValue)+1);
    exponent = max(raw_exp-iqWidth+1, zeros(1, nPRB));

    exponent_b = dec2bin(exponent, 8)';

    scaler = 2.^-exponent;
    sPRB = fPRB.*repmat(scaler, 24, 1);

    negComp = 2^iqWidth;
    cPRB = sPRB + (sPRB < 0).*negComp;

    bPRB0 = dec2bin(cPRB, iqWidth);
    bPRB1 = bPRB0';
    bPRB2 = reshape(bPRB1, 2*12*iqWidth, nPRB);

    bPRB3 = [exponent_b; bPRB2];

    bPRB4 = bPRB3(:);

    bPRB5 = reshape(bPRB4, 8, length(bPRB4)/8);

    % Optimized version: Replace bin2dec with vectorized bit manipulation
    % bPRB5 is 8 x N matrix of '0'/'1' chars, convert to uint8 directly
    bPRB5_logical = (bPRB5 == '1');  % Convert to logical matrix
    powers_of_2 = 2.^(7:-1:0)';  % [128, 64, 32, 16, 8, 4, 2, 1] as double
    cSamples_uint8 = uint8(bPRB5_logical' * powers_of_2);

    cSamples_uint8 = cSamples_uint8';

elseif alg_sel == 1

    if (iqWidth ~= 14)
        error('Unsupported iqWidth');
    end

    cSamples_uint8 = [];
    for ii=1:length(ucSamples)/12;

        fPRB = ucSamples((ii-1)*12+1:ii*12);
        %Compress n PRB of data
        if bypass
            outSamples = inSamples;
            break;
        end

        %maxV = max(max(real(ucSamples)), max(imag(ucSamples)));
        %minV = min(min(real(ucSamples)), min(imag(ucSamples)));
        maxV = max(max(real(fPRB)), max(imag(fPRB)));
        minV = min(min(real(fPRB)), min(imag(fPRB)));
        maxValue = max(maxV, abs(minV)-1);

        % Calculate exponent
        raw_exp = floor(log2(maxValue)+1);
        exponent = max(raw_exp -iqWidth +1, 0);
        exponentB = dec2bin(exponent, 8);
        % Calculate scaler
        scaler = 2^-exponent;
        % Scale and quantize
        outB = '';
        outB = strcat(outB, exponentB);
        for iRe = 1:length(fPRB)
            cPRB(iRe) = scaler*(real(fPRB(iRe)) + 1i*(imag(fPRB(iRe))));
            if real(cPRB(iRe)) >= 0
                outB = strcat(outB, dec2bin(real(cPRB(iRe)), iqWidth));
            else
                outB = strcat(outB, dec2bin(real(cPRB(iRe)+2^iqWidth), iqWidth));
            end
            if imag(cPRB(iRe)) >= 0
                outB = strcat(outB, dec2bin(imag(cPRB(iRe)), iqWidth));
            else
                outB = strcat(outB, dec2bin(imag(cPRB(iRe))+2^iqWidth, iqWidth));
            end
        end
        cSamples = outB;%strcat(outB, exponentB);


        % convert tb_data to uint8_t format
        cSamples_PRB_uint8 = zeros(1, length(cSamples)/8);
        for ii=1:length(cSamples)/8
            tmp = num2str(cSamples((ii-1)*8+1:ii*8)');
            cSamples_PRB_uint8(ii) = bin2dec(tmp');
        end
        cSamples_uint8 = [cSamples_uint8 cSamples_PRB_uint8];
        %outSamples = cPRB;
        %exponentB
        %outB
        %dec2hex(cSamples_PRB_uint8)
    end
end

return



