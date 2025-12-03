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

function PuschCfg = derive_lifting(PuschCfg)

%Function derives number of code blocks and users lifting set
%Follows 138.212 section 5.2.2

%outputs:
%PuschCfg.coding.C             --> number of codeblocks
%PuschCfg.coding.Zc            --> lifting size
%PuschCfg.coding.i_LS          --> lifting set index
%PuschCfg.coding.sizes.K       --> number of systematic bits per codeblock
%PuschCfg.coding.sizes.F       --> number of filler bits per codeblock
%PuschCfg.coding.sizes.K_prime --> number of CRC encoded bits per codeblock

%%
%PARAMATERS

B = PuschCfg.coding.B;   %size of CRC encoded transport block
BGN = PuschCfg.coding.BGN;     %1 or 2. Indicates which base graph used

%%
%DERIVE C

%derive max number of bits per codeblock:
if BGN == 1
    K_cb = 8448;
else
    K_cb = 3840;
end


%derive number of codeblocks
if B <= K_cb
    C = 1;
    B_prime = B;
else
    L = 24; %CRC bits per code block
    C = ceil(B / (K_cb - L)); %number of code blocks
    B_prime = B + C*L; %total number of bits
end

%bits per code block:
K_prime = B_prime / C;


%number of systematic information bits:
if BGN == 1
    K_b = 22;
else
    if B > 640
        K_b = 10;
    else
        if B > 560
            K_b = 9;
        else
            if B > 192
                K_b = 8;
            else
                K_b = 6;
            end
        end
    end
end

%%
%DERIVE Zc

Z = [2, 4, 8, 16, 32, 64, 128, 256,...
    3, 6, 12, 24, 48, 96, 192, 384,...
    5, 10, 20, 40, 80, 160, 320,...
    7, 14, 28, 56, 112, 224, ...
    9, 18, 36, 72, 144, 288,...
    11, 22, 44, 88, 176, 352,...
    13, 26, 52, 104, 208,...
    15, 30, 60, 120, 240];

%find smallest Z such that Z*K_b >= K_prime:
Diff = Z*K_b - K_prime;
Diff(Diff < 0) = Inf;

[~,index] = min(Diff);
Zc = Z(index);

%%
%LIFTING SET

if (1 <= index) && (index <= 8)
    i_LS = 1;
end

if (9 <= index) && (index <= 16)
    i_LS = 2;
end

if (17 <= index) && (index <= 23)
    i_LS = 3;
end

if (24 <= index) && (index <= 29)
    i_LS = 4;
end

if (30 <= index) && (index <= 35)
    i_LS = 5;
end

if (36 <= index) && (index <= 41)
    i_LS = 6;
end

if (42 <= index) && (index <= 46)
    i_LS = 7;
end

if (47 <= index) && (index <= 51)
    i_LS = 8;
end

%%
%FILLER BITS

%number of systematic bits:
if BGN == 1
    K = Zc*22;
else
    K = Zc*10;
end

%number of filler bits:
F = K - K_prime;

%%
%WRAP

PuschCfg.coding.C = C;
PuschCfg.coding.Zc = Zc;
PuschCfg.coding.i_LS = i_LS;
PuschCfg.coding.K = K;
PuschCfg.coding.K_b = K_b;
PuschCfg.coding.F = F;
PuschCfg.coding.K_prime = K_prime;


