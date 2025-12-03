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

%%
%PARAMATERS 


Qm         = 6;         % modulation order, should be in {1, 2, 4, 6, 8}
K          = 2;         % the number of information bits
E          = Qm*3*1;    % the length of bit sequence after rate matching. Should be mult of Qm.
N          = 1000;      % the number of transmissions for computing BER
rndSeed    = 1;         % randomness seed
nRNTI      = 1;         % RNTI as described in Sec. 6.3.1.1, TS38.211
nID        = 1;         % dataScramblingId or N_ID^cell as described in Sec. 6.3.1.1, TS38.211
useToolbox = 1;         % whether to use 5G Toolbox or not
 
%%
%START

snr  = -5 : 0.5 : 5;
nSnr = length(snr);

berImpl      = zeros(nSnr,1);
ber5GToolbox = zeros(nSnr,1);

for i = 1 : nSnr
    [berImpl(i), ber5GToolbox(i)] = testSimplex(Qm, K, E, snr(i), N, rndSeed, nRNTI, nID, useToolbox);
    percent_done = i / nSnr
end

%%
%PLOT

figure
semilogy(snr,ber5GToolbox);
hold on
semilogy(snr,berImpl,'*');
grid on
legend('5g toolbox','cuphy');
xlabel('snr (dB)');
ylabel('ber');
title_str = strcat('K = ',num2str(K),',E = ', num2str(E), ',Qm = ', num2str(Qm));
title(title_str); 

