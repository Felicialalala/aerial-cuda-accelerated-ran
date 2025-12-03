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

tic;

A = 16;
E = 288;
nUci_vec = [1000 2000 3000 4000 5000 6000];
SNR_vec = [-8:1:-3];
nSNR = length(SNR_vec);
listLength = 8;

BLER1 = zeros(1, nSNR);
BLER2 = zeros(1, nSNR);
parfor idxSnr = 1:length(SNR_vec)
    SNR = SNR_vec(idxSnr)
    err1 = 0;
    err2 = 0;
    nUci = nUci_vec(idxSnr);
    for idxUci = 1:nUci
        txucibits = round(rand(1, A))';
        codeduci = nrUCIEncode(txucibits, E);
        txucisym = 1-2*codeduci;
        rxucisym = txucisym + randn(length(txucisym), 1)*10^(-SNR/20);
        rxucibits1 = double(nrUCIDecode(rxucisym, A, 'listLength', listLength));
        if sum(abs(rxucibits1 - txucibits))
            err1 = err1 + 1;
        end
        rxucibits2 = uciSegPolarDecode(A, E, listLength, rxucisym);
        if sum(abs(rxucibits2 - txucibits))
            err2 = err2 + 1;
        end
    end
    BLER1(idxSnr) = err1/nUci;
    BLER2(idxSnr) = err2/nUci;
end

figure; 
semilogy(SNR_vec, BLER1); 
hold on; grid on;
semilogy(SNR_vec, BLER2); 
legend('toolbox', '5gmodel');
xlabel('SNR (dB)');
ylabel('BLER');
title('Polar decoder performance (A=16, E=288, AWGN)');

toc;