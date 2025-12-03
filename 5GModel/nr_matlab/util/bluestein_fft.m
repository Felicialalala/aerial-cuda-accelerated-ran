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

function result = bluestein_fft(x, N2)
    N = length(x);
    if nargin < 2
        N2 = 2^nextpow2(2*N - 1);
    else
        if N2 < 2*N-1
            error('Error: Bluestein FFT length should be at least 2N-1!');
        end
    end
    % Step 1: Zero-pad the input to the next power of 2
    x_padded = zeros(N2, 1);
    x_padded(1:N) = x;
    
    % Step 2: Create two twiddle factors
    n1 = 0:N-1;
    twf1 = zeros(N2,1);
    twf1(1:N) = exp(-1i*pi*n1.^2/N);
    twf2 = zeros(N2,1);
    twf2(1:N) = exp(1i*pi*n1.^2/N);
    twf2(N2-N+2:N2) = exp(1i*pi*[-N+1:-1].^2/N);    
    
    % Step 3: Convolution through FFT and IFFT, and output results
    Xq = fft(x_padded.*twf1, N2);
    Wq = fft(twf2, N2);
    tmp = twf1.*ifft(Xq.*Wq, N2);
    result = tmp(1:N);   
end