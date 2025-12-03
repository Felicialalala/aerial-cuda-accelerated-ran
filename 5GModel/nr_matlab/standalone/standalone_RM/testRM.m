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

function [berImpl, ber5GToolbox] = testRM(K, E, snr, N, rndSeed, useToolbox)
%
% This function performs compliance test for the Simplex code
% implementation under AWGN
%
% Input:    K:       the number of information bits
%           E:       the length of bit sequence after rate matching
%           snr:     the SNR for AWGN generation
%           N:       the number of transmissions for computing BER
%           rndSeed: randomness seed
%           useToolbox: whether to use 5G Toolbox or not
%           
%
% Output:   berImpl: the BER achieved by the Simplex code implementation, 
%           ber5G:   the BER achieved by the 5G Toolbox functions.
%
%

%% validate inputs

if K < 3 && K > 11
    error('Invalid input: 3 <= K <= 11 for RM code');
end

if N<1000
    error('Invalid input: N is too small: should be >= 1000 to ensure BER accuracy');
end

%% compute BER achieved by Simplex code implementation

rng(rndSeed);
errBits        = 0;
ToolboxErrBits = 0;

for expIdx = 1:N
    
    payload = randi([0,1], [K, 1]);
    
    %% encode
    fecEnc = FecRmObj(1, E, K);
    outEncode = fecEnc(payload);
    
    rmBitLLRs = apply_RM_channel(snr, outEncode); % channel
    
    %% decode
    fecEnc = FecRmObj(0, E, K);
    decodedBits = fecEnc(rmBitLLRs);
    errBits = errBits + sum(abs(payload - str2num(decodedBits)));
    
    if(useToolbox)
        ToolboxOutEnc = nrUCIEncode(int8(payload), E, 'pi/2-BPSK');
        
        ToolboxLLRs = apply_RM_channel(snr, double(ToolboxOutEnc)); % channel
        ToolboxDecBits = nrUCIDecode(ToolboxLLRs, K, 'pi/2-BPSK');
        ToolboxErrBits = ToolboxErrBits + sum(abs(int8(payload) - ToolboxDecBits));
    end
end

berImpl = errBits/N/K;
ber5GToolbox = ToolboxErrBits/N/K;

end
