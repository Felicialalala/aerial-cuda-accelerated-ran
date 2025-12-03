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

function [berImpl, ber5GToolbox] = testSimplex(Qm, K, E, snr, N, rndSeed, nRNTI, nID, useToolbox)
%
% This function performs compliance test for the Simplex code
% implementation under AWGN
%
% Input:    Qm:      modulation order, should be in {1, 2, 4, 6, 8}
%           K:       the number of information bits
%           E:       the length of bit sequence after rate matching
%           snr:     the SNR for AWGN generation
%           N:       the number of transmissions for computing BER
%           rndSeed: randomness seed
%           nRNTI:   RNTI as described in Sec. 6.3.1.1, TS38.211
%           nID:     dataScramblingId or N_ID^cell as described in Sec. 6.3.1.1, TS38.211
%           useToolbox: whether to use 5G Toolbox or not
%           
%
% Output:   berImpl: the BER achieved by the Simplex code implementation, 
%           ber5G:   the BER achieved by the 5G Toolbox functions.
%
%

%% validate inputs
if Qm ~= 1 && Qm ~= 2 && Qm ~= 4 && Qm ~= 6 && Qm ~= 8
    error('Invalid input: Modulation order Qm must be in {1, 2, 4, 6, 8}');
end

if K ~= 1 && K ~= 2
    error('Invalid input: number of information bits K must be either 1 or 2');
end

if K == 1
    if E < Qm
        error('Invalid input: For 1-bit information, the length of rate-matched sequence E must be no less than Qm');
    end
end

if N<1000
    error('Invalid input: N is too small: should be >= 1000 to ensure BER accuracy');
end

%% setup

QmSeq = [];
switch Qm
    case 1
        QmSeq = 'pi/2-BPSK';
    case 2
        QmSeq = 'QPSK';
    case 4
        QmSeq = '16QAM';
    case 6
        QmSeq = '64QAM';
    case 8
        QmSeq = '256QAM';
end


%% compute BER achieved by Simplex code implementation

rng(rndSeed);
errBits        = 0;
ToolboxErrBits = 0;


for expIdx = 1:N
    
    payload = randi([0,1], [1, K]);
    outEncode = simplexEncode(payload, K, E, Qm);
    outScr = scramble(outEncode, E, nRNTI, nID, 0, 0);
    outScrLLR = apply_simplex_channel(snr, Qm, outScr);
    outDescrLLR = SimplexDescramble(outScrLLR, K, E, Qm, nRNTI, nID, 0, 0);
    
    decodedBits = simplexDecode(outDescrLLR, K, E, Qm);
    errBits = errBits + sum(abs(payload - decodedBits));
    
    if(useToolbox)
        ToolboxDecBits = nrUCIDecode(outDescrLLR, K, QmSeq);
        ToolboxErrBits = ToolboxErrBits + sum(abs(int8(transpose(payload)) - ToolboxDecBits));
    end
end

berImpl = errBits/N/K;
ber5GToolbox = ToolboxErrBits/N/K;

end
