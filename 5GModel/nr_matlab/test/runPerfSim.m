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

function runPerfSim(sim_mode)

TSTART = tic;

if nargin == 0
    sim_mode = 0;
end

run_pusch = 1;
run_prach = 1;
run_pucch = 1;

if sim_mode == 0
    nFrame_pusch_data = 1; % 50;
    nFrame_pusch_uci = 10; % 500;
    nFrame_pusch_prcd = 1; % 50;
    nFrame_pusch_bler = 10; % 500
    nFrame_prach_false = 10; % 500;
    nFrame_prach_missed = 1; % 50;    
    nFrame_pucch_0 = 1; % 50;
    nFrame_pucch_1 = 1; % 50;
    nFrame_pucch_2 = 1; % 50;
    nFrame_pucch_3 = 1; % 50;
else
    nFrame_pusch_data = 50;
    nFrame_pusch_uci = 500;
    nFrame_pusch_prcd = 50;
    nFrame_pusch_bler = 500;
    nFrame_prach_false = 500;
    nFrame_prach_missed = 50;    
    nFrame_pucch_0 = 50;
    nFrame_pucch_1 = 50;
    nFrame_pucch_2 = 50;
    nFrame_pucch_3 = 50;
end
pusch_data_caseSet = [7004, 7005, 7006, 7012, 7013];
pusch_prcd_caseSet = [7056];
pusch_uci_caseSet = [7051, 7052, 7053, 7054];
pusch_bler_caseSet = [7101];
prach_caseSet = [5003, 5004];
pucch0_missed_caseSet = [6003, 6004];
pucch1_nack2ack_caseSet = [6102];
pucch1_missed_caseSet = [6108];
pucch2_missed_caseSet = [6202];
pucch2_uci_caseSet = [6212];
pucch3_uci_caseSet = [6303, 6304, 6308];

if run_pusch
    % PUSCH data, SNRoffset = 0
    [nPerf, nFail, CBer, TBer] = testPerformance_pusch(pusch_data_caseSet, 0, nFrame_pusch_data, 0);
    % PUSCH UCI, SNRoffset = 0
    [nPerf, nFail, CBer, TBer] = testPerformance_pusch(pusch_uci_caseSet, 0, nFrame_pusch_uci, 1);
    % PUSCH transform precoding, SNRoffset = 0
    [nPerf, nFail, CBer, TBer] = testPerformance_pusch(pusch_prcd_caseSet, 0, nFrame_pusch_prcd, 2);
    % PUSCH 0.001% BLER, SNRoffset = 0
    [nPerf, nFail, CBer, TBer] = testPerformance_pusch(pusch_bler_caseSet, 0, nFrame_pusch_bler, 3);    
end
if run_prach
    % SNRoffset = 0, falseAlarmTest = 1 (false detection)
    [nPerf, nFail, Pfd, Pmd] = testPerformance_prach(prach_caseSet, 0, nFrame_prach_false, 1);
    % SNRoffset = 0, falseAlarmTest = 0 (missed detection)
    [nPerf, nFail, Pfd, Pmd] = testPerformance_prach(prach_caseSet, 0, nFrame_prach_missed, 0);
end
if run_pucch
    % format-0
    % SNRoffset = 0, format-0, pucchTestMode = 2 (false detection)
    [nPerf, nFail, P_err] = testPerformance_pucch(pucch0_missed_caseSet, 0, nFrame_pucch_0, 0, 1);
    % SNRoffset = 0, format-0, pucchTestMode = 3 (ACK missed detection)
    [nPerf, nFail, P_err] = testPerformance_pucch(pucch0_missed_caseSet, 0, nFrame_pucch_0, 0, 3);
    % format-1
    % SNRoffset = 0, format-1, pucchTestMode = 1 (false detection)
    [nPerf, nFail, P_err] = testPerformance_pucch(pucch1_missed_caseSet, 0, nFrame_pucch_1, 1, 1);
    % SNRoffset = 0, format-1, pucchTestMode = 2 (NACK to ACK detection)
    [nPerf, nFail, P_err] = testPerformance_pucch(pucch1_nack2ack_caseSet, 0, nFrame_pucch_1, 1, 2);
    % SNRoffset = 0, format-1, pucchTestMode = 3 (ACK missed detection)
    [nPerf, nFail, P_err] = testPerformance_pucch(pucch1_missed_caseSet, 0, nFrame_pucch_1, 1, 3);
    % format-2
    % SNRoffset = 0, format-2, pucchTestMode = 1 (false detection)
    [nPerf, nFail, P_err] = testPerformance_pucch(pucch2_missed_caseSet, 0, nFrame_pucch_2, 2, 1);
    % SNRoffset = 0, format-2, pucchTestMode = 3 (ACK missed detection)
    [nPerf, nFail, P_err] = testPerformance_pucch(pucch2_missed_caseSet, 0, nFrame_pucch_2, 2, 3);
    % SNRoffset = 0, format-2, pucchTestMode = 4 (UCI BLER)
    [nPerf, nFail, P_err] = testPerformance_pucch(pucch2_uci_caseSet, 0, nFrame_pucch_2, 2, 4);
    % format-3
    % SNRoffset = 0, format-3, pucchTestMode = 4 (UCI BLER)
    [nPerf, nFail, P_err] = testPerformance_pucch(pucch3_uci_caseSet, 0, nFrame_pucch_3, 3, 4);
end

fprintf('\nTotal time for runPerfSim is %d seconds\n', round(toc(TSTART)));

return