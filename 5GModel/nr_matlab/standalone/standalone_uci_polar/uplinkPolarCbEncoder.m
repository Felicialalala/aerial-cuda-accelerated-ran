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

function polCw = uplinkPolarCbEncoder(polarCbPrms, cwBitTypes, polCbCrcEncoded)

% function perform uplink polar encoding on a single codeblock

%inputs:
% polarCbPrms     --> polar codeblock paramaters
% cwBitTypes      --> types of each codeword bit. 0: frozen, 1: payload, 2: input parity. Dim: N_cb x 1
% polCbCrcEncoded --> crc encoded codeblock. Dim: K_cb x 1

%outputs:
% polCw --> polar codeword. Dim: N_cb x 1

%%
%PARAMATERS

N_cw = polarCbPrms.N_cw; % length of polar codeword
n_cw = polarCbPrms.n_cw; % log2(N_cb)

%%
% BUTTERLY INPUT
% 38.212 section 5.3.1.2 

polCw = zeros(N_cw,1);
inputBitIdx  = 0;
y0 = 0;
y1 = 0;
y2 = 0;
y3 = 0;
y4 = 0;

for cwIdx = 0 : (N_cw - 1)
    yt = y0;
    y0 = y1;
    y1 = y2;
    y2 = y3;
    y3 = y4;
    y4 = yt;
    
    switch cwBitTypes(cwIdx + 1)
        case 1
            polCw(cwIdx + 1) = polCbCrcEncoded(inputBitIdx + 1);
            y0               = mod(y0 + polCbCrcEncoded(inputBitIdx + 1), 2);
            inputBitIdx      = inputBitIdx + 1;
            
        case 2
            polCw(cwIdx + 1) = y0;
    end
end

%%
% BUTTERFLY

for i = 0 : (n_cw - 1) %loop over n_cb stages 
     
    s = 2^i;
    m = N_cw / (2*s);
    
    %parallel start (N_cb/2 parallel XORS)
    for j = 1 : m
        start_idx = 2*s*(j-1);
        
        for k = 1 : s
            polCw(start_idx + k) = xor(polCw(start_idx + k), polCw(start_idx + k + s) );
        end
    end
    %parallel end
end





            
            
            


    
    
