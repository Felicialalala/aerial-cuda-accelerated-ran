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

function errCnt = genCfgTV_demo(caseSet)

TC_list = ones(1, 19);
errCnt = 0;

[status,msg] = mkdir('GPU_test_input');

parfor idxTc = 1:length(TC_list)
    TcList = zeros(1, length(TC_list));
    TcList(idxTc) = TC_list(idxTc);
    errCnt = genDemoTV(TcList) + errCnt;
end   

return

function err = genDemoTV(TcList)

err = 0;

if TcList(1)
    err = runSim('demo_ssb.yaml', 'demo_ssb');
end

if TcList(2)
    err = runSim('demo_coreset0.yaml', 'demo_coreset0');
end

if TcList(3)
    err = runSim('demo_msg1.yaml', 'demo_msg1');
end

if TcList(4)
    err = runSim('demo_msg2.yaml', 'demo_msg2');
end

if TcList(5)
    err = runSim('demo_msg2_4_ant_pdcch_dl.yaml', 'demo_msg2_4_ant_pdcch_dl');
end

if TcList(6)
    err = runSim('demo_msg3.yaml', 'demo_msg3');
end

if TcList(7)
    err = runSim('demo_msg4.yaml', 'demo_msg4');
end

if TcList(8)
    err = runSim('demo_msg4_4_ant_pdcch_ul.yaml', 'demo_msg4_4_ant_pdcch_ul');
end

if TcList(9)
    err = runSim('demo_msg5_pusch.yaml', 'demo_msg5_pusch');
end

if TcList(10)
    err = runSim('demo_traffic_dl.yaml', 'demo_traffic_dl');
end

if TcList(11)
    err = runSim('demo_traffic_ul_pdcch.yaml', 'demo_traffic_ul_pdcch');
end

if TcList(12)
    err = runSim('demo_traffic_ul_pusch.yaml', 'demo_traffic_ul_pusch');
end

if TcList(13)
    err = runSim('demo_ssb_fxn.yaml', 'demo_ssb_fxn');
end

if TcList(14)
    err = runSim('demo_sib1_fxn.yaml', 'demo_sib1_fxn');
end

if TcList(15)
    err = runSim('demo_msg1_fxn.yaml', 'demo_msg1_fxn');
end

if TcList(16)
    err = runSim('demo_msg2_fxn.yaml', 'demo_msg2_fxn');
end

if TcList(17)
    err = runSim('tv_s_slot_dl.yaml', 'tv_s_slot_dl');
end

if TcList(18)
    err = runSim('tv_s_slot_ul.yaml', 'tv_s_slot_ul');
end

if TcList(19)
    err = runSim('demo_msg3_fxn.yaml', 'demo_msg3_fxn');
end

return
