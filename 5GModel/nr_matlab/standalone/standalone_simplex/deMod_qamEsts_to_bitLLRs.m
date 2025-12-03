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

function bitLLRs = deMod_qamEsts_to_bitLLRs(y,N0,nQams,nBitsPerQam, pi2Bpsk)

%inputs:
% y           --> estimates of tx qams. Dim: nQams x 1
% N0          --> noise variance of estimates
% nQams       --> number of transmitted qams
% nBitsPerQam --> number of bits per Qam. 1,2,4,16,64,256

%outputs:
% bitLLRs --> LLRs of qam bits. Dim: nBitsPerQam*nQams x 1

%%
% SETUP

load('qam_dist.mat');

switch nBitsPerQam
 case 2
     d = d_qpsk;
 case 4
     d = d_qam16;
 case 6
     d = d_qam64;
 case 8
     d = d_qam256;
end

nBitsPerPam = nBitsPerQam / 2;
 
%%
%START

bitLLRs = zeros(nBitsPerQam,nQams);

if (nBitsPerQam == 1)
    if pi2Bpsk
        bitLLRs = real(2/sqrt(2) * (1-1i) * y / N0);
    else
        bitLLRs = real(2 * y / N0);
    end
else
    for qamIdx = 1 : nQams
        % Estimate real/imag Pam LLRs:
        real_pam_llrs = max_pam_LLR(real(y(qamIdx)),N0,d,nBitsPerPam);
        imag_pam_llrs = max_pam_LLR(imag(y(qamIdx)),N0,d,nBitsPerPam);

        % combine into Qam LLRs:
        for pamIdx = 1 : nBitsPerPam
            bitLLRs(2*(pamIdx - 1) + 1, qamIdx) = real_pam_llrs(pamIdx);
            bitLLRs(2*(pamIdx - 1) + 2, qamIdx) = imag_pam_llrs(pamIdx);
        end
    end
end

bitLLRs = bitLLRs(:);
 
end

    
        
     
     
 
 
 
 
 
 
