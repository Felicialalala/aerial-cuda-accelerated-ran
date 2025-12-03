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

load ToCfoEst
SNRv = [-5:5:30];
cfoTable = CFOERR(end-12+1:end-6+1, :);
toTable = TOERR(end-12+1:end-6+1, :);
colorVec = 'rbgk';
markVec = 'xo+';

figure(1);
for idx = 1:6
    chanIdx = floor((idx-1)/3);
    prbIdx = mod(idx-1, 3);
    line = ['-', colorVec(chanIdx+1), markVec(prbIdx+1)];
    semilogy(SNRv, cfoTable(idx, :), line, 'LineWidth', 2);
    hold on; grid on;
end
xlabel('SNR (dB)');
ylabel('CFO estimation MMSE (Hz)');
title('CFO estimation error');
legend('AWGN, nPRB = 1', 'AWGN, nPRB = 8','AWGN, nPRB = 273','fading, nPRB = 1','fading, nPRB = 8','fading, nPRB = 273')

figure(2);
for idx = 1:6
    chanIdx = floor((idx-1)/3);
    prbIdx = mod(idx-1, 3);
    line = ['-', colorVec(chanIdx+1), markVec(prbIdx+1)];
    semilogy(SNRv, toTable(idx, :)*1e6, line, 'LineWidth', 2);
    hold on; grid on;
end
xlabel('SNR (dB)');
ylabel('TO estimation MMSE (us)');
title('TO estimation error');
legend('AWGN, nPRB = 1', 'AWGN, nPRB = 8','AWGN, nPRB = 273','fading, nPRB = 1','fading, nPRB = 8','fading, nPRB = 273')
