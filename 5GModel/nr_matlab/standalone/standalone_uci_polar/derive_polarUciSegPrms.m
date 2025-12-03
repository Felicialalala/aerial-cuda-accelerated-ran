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

function polarUciSegPrms = derive_polarUciSegPrms(A_seg, E_seg)

% inputs:
% A_seg --> number of bits in uci segment
% E_seg --> number of tx bits

% outputs:
% polarUciSegPrms 

%%
%START

polarUciSegPrms = [];
polarUciSegPrms.A_seg = A_seg;
polarUciSegPrms.E_seg = E_seg;

% crc size (38.212 6.3.1.2.1)
if(A_seg <= 19)
    polarUciSegPrms.nCrcBits = 6;
else
    polarUciSegPrms.nCrcBits = 11;
end
   
% code block segmentation (38.212 6.3.1.3.1)
% code block size         (38.212 5.2.1)
if(((A_seg >= 360) && (E_seg >= 1088)) || (A_seg >= 1013))
    polarUciSegPrms.nCbs           = (2);
    polarUciSegPrms.K_cw           = ceil(A_seg / 2) + polarUciSegPrms.nCrcBits;
    polarUciSegPrms.E_cw           = floor(E_seg / 2);
    polarUciSegPrms.zeroInsertFlag = mod(A_seg, 2);
else
    polarUciSegPrms.nCbs           = (1);
    polarUciSegPrms.K_cw           = A_seg + polarUciSegPrms.nCrcBits;
    polarUciSegPrms.E_cw           = E_seg;
    polarUciSegPrms.zeroInsertFlag = 0;
end

% encoded cb(s) size (38.212 5.3.1)
n_temp = ceil(log2(polarUciSegPrms.E_cw)) - 1;
if ((polarUciSegPrms.E_cw <= 9/8*2^(n_temp)) && (polarUciSegPrms.K_cw/polarUciSegPrms.E_cw <= 9/16))
    n_1 = n_temp;
else
    n_1 = n_temp + 1;
end

n_2   = ceil(log2(polarUciSegPrms.K_cw * 8));
n_min = 5;
n_max = 10;

polarUciSegPrms.n_cw = max(min([n_1 n_2 n_max]), n_min);
polarUciSegPrms.N_cw = 2^polarUciSegPrms.n_cw;




end
