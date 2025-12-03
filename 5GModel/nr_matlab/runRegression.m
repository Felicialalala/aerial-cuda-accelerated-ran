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

function [nTC_total, err_total, nCuphyTV_total, nFapiTV_total] = runRegression(testSet, channelSet, caseSet, subSetMod, relNum)

TSTART = tic;

startup;

switch nargin
    case 0
        testSet = {'allTests'};
        channelSet = {'allChannels'};
        caseSet = 'full';
        subSetMod = [0, 1];
        relNum = 10000;
        printUsage;
    case 3
        subSetMod = [0, 1];
        relNum = 10000;
    case 4
        relNum = 10000;
    case 5
        % 
    otherwise
        printUsage;
        nTC_total = 0;
        err_total = 1;
        nCuhpyTV_total = 0;
        nFapiTV_total = 0;
        return;
end

run_testCompliance = 0;
run_genTestVector = 0;
run_testPerformance = 0;
run_perfPattern = 0;

run_ssb = 0;
run_pdcch = 0;
run_pdsch = 0;
run_csirs = 0;
run_dlmix = 0;
run_prach = 0;
run_pucch = 0;
run_pusch = 0;
run_srs = 0;
run_ulmix = 0;
run_bfw = 0;
run_simplex = 0;

run_perf_tvs = 0;
run_cuphycp_tvs = 0;
run_cfgtemplate = 0;
run_genLaunchPatternFile = 0;
run_genCfgTV_nvbug = 0;

for idxTestSet = 1:length(testSet)
    switch testSet{idxTestSet}
        case 'Compliance'
            run_testCompliance = 1;
        case 'TestVector'
            run_genTestVector = 1;
        case 'Performance'
            run_testPerformance = 1;
        case 'PerfPattern'
            run_perfPattern = 1;
        case 'allTests'
            run_testCompliance = 1;
            run_genTestVector = 1;
            run_testPerformance = 1;
            % run_perfPattern will not be run since it's already part of run_genTestVector
        otherwise
            printUsage();
            nTC_total = 0;
            err_total = 1;
            nCuphyTV_total = 0;
            nFapiTV_total = 0;
            return;
    end
end

if run_testCompliance && run_genTestVector
    compTvMode = 'both';
elseif run_testCompliance
    compTvMode = 'testCompliance';
elseif run_genTestVector
    compTvMode = 'genTV';
elseif run_perfPattern
    compTvMode = 'perfPattern';
else
    compTvMode = 'none';
end

if ~run_perfPattern
    for idxChannelSet = 1:length(channelSet)
        switch channelSet{idxChannelSet}
            case 'ssb'
                run_ssb = 1;
            case 'pdcch'
                run_pdcch = 1;
            case 'pdsch'
                run_pdsch = 1;
            case 'csirs'
                run_csirs = 1;
            case 'dlmix'
                run_dlmix = 1;
            case 'allDL'
                run_ssb = 1;
                run_pdcch = 1;
                run_pdsch = 1;
                run_csirs = 1;
                run_dlmix = 1;
            case 'prach'
                run_prach = 1;
            case 'pucch'
                run_pucch = 1;
            case 'pusch'
                run_pusch = 1;
            case 'srs'
                run_srs = 1;
            case 'simplex'
                run_simplex = 1;
            case 'ulmix'
                run_ulmix = 1;
            case 'allUL'
                run_prach = 1;
                run_pucch = 1;
                run_pusch = 1;
                run_srs = 1;
                run_ulmix = 1;
                run_simplex = 1;
            case 'bfw'
                run_bfw = 1;
            case 'cuPHY-CP_TVs'
                run_cuphycp_tvs = 1;
            case 'perf_TVs'
                run_perf_tvs = 1;
            case 'cfgTemplate'
                run_cfgtemplate = 1;
            case 'launchPatternFile'
                run_genLaunchPatternFile = 1;
            case 'CfgTV_nvbug'
                run_genCfgTV_nvbug = 1;
            case 'allChannels'
                run_ssb = 1;
                run_pdcch = 1;
                run_pdsch = 1;
                run_csirs = 1;
                run_dlmix = 1;
                run_prach = 1;
                run_pucch = 1;
                run_pusch = 1;
                run_srs = 1;
                run_simplex = 1;
                run_ulmix = 1;
                run_bfw = 1;
                run_cuphycp_tvs = 1;
                run_perf_tvs = 1;
                run_cfgtemplate = 1;
                run_genLaunchPatternFile = 1;
                run_genCfgTV_nvbug = 1;
            otherwise
                printUsage();
                nTC_total = 0;
                err_total = 1;
                nCuphyTV_total = 0;
                nFapiTV_total = 0;
                return;
        end
    end
end

caseSetList = {'compact', 'full', 'selected'};
validInputArgFlag = (run_perfPattern && isnumeric(caseSet)) || ismember(caseSet, caseSetList);
if ~validInputArgFlag
    printUsage();
    nTC_total = 0;
    err_total = 1;
    nCuphyTV_total = 0;
    nFapiTV_total = 0;
    return;
end

nTC_ssb = 0;
nTC_pdcch = 0;
nTC_pdsch = 0;
nTC_csirs = 0;
nTC_dlmix = 0;
nTC_prach = 0;
nTC_pucch = 0;
nTC_pusch = 0;
nTC_srs = 0;
nTC_simplex = 0;
nTC_ulmix = 0;
nTC_bfw = 0;

err_ssb = 0;
err_pdcch = 0;
err_pdsch = 0;
err_csirs = 0;
err_dlmix = 0;
err_prach = 0;
err_pucch = 0;
err_pusch = 0;
err_srs = 0;
encErr_simplex = 0;
scrErr_simplex = 0;
err_ulmix = 0;
err_bfw = 0;
errFlagExist = 0;

detErr_ssb = 0;
detErr_pdcch = 0;
detErr_pdsch = 0;
detErr_csirs = 0;
detErr_dlmix = 0;
detErr_prach = 0;
detErr_pucch = 0;
detErr_pusch = 0;
detErr_srs = 0;
detErr_ulmix = 0;
detErr_bfw = 0;

nTV_ssb = 0;
nTV_pdcch = 0;
nTV_pdsch = 0;
nTV_csirs = 0;
nCuphyTV_dlmix = 0;
nFapiTV_dlmix = 0;
nTV_prach = 0;
nTV_pucch = 0;
nTV_pusch = 0;
nTV_srs = 0;
nTV_simplex = 0;
nCuphyTV_ulmix = 0;
nFapiTV_ulmix = 0;
nTV_bfw = 0;

if (run_cfgtemplate)
    fprintf('\nRun genCfgTemplate ...');
    genCfgTemplate;
    fprintf('\nTest runSim for DL ...');
    genCfgTemplate('cfg_template_DL.yaml', 'DL');
    errFlag = runSim('cfg_template_DL.yaml', 'template_DL');
    delete cfg_template_DL.yaml;
    delete GPU_test_input/template_DL*;
    if errFlag, fprintf('==> FAIL\n'); errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    
    fprintf('Test runSim for UL ...');
    genCfgTemplate('cfg_template_UL.yaml', 'UL');
    errFlag = runSim('cfg_template_UL.yaml', 'template_UL');
    delete cfg_template_UL.yaml;
    delete GPU_test_input/template_UL*;
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    
    fprintf('Test runSim_multiSlot ...');
    errFlag = runSim_multiSlot('cfg_list.yaml', 'cfg_list');
    delete GPU_test_input/cfg_list*;
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
end

if run_testCompliance
    if (run_pusch)
        fprintf('Test runSim for PUSCH pcap analysis ...');
        errFlag = runSim('cfg_pusch_pcap_example.yaml', 'pusch_pcap_example', 'pusch_pcap_example.txt');
        delete GPU_test_input/pusch_pcap_example*;
        if errFlag, fprintf('==> FAIL\n'); errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    end

    if (run_pusch)
        fprintf('Test runSim for UCI on PUSCH pcap analysis ...');
        errFlag = runSim('cfg_UciOnPusch_pcap_example.yaml', 'UciOnPusch_pcap_example', 'UciOnPusch_pcap_example.txt');
        delete GPU_test_input/UciOnPusch_pcap_example*;
        if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    end
    
    if (run_prach)
        fprintf('Test runSim for PRACH pcap analysis ...');
        errFlag = runSim('cfg_prach_pcap_example.yaml', 'prach_pcap_example', 'prach_pcap_example.txt');
        delete GPU_test_input/prach_pcap_example*;
        if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    end
    
    if (run_ssb)
        fprintf('Test runSim for SSB pcap analysis ...');
        errFlag = runSim('cfg_ssb_pcap_example.yaml', 'ssb_pcap_example', 'ssb_pcap_example.txt');
        delete GPU_test_input/ssb_pcap_example*;
        if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    end

    if (run_pdcch)
        fprintf('Test runSim for PDCCH pcap analysis ...');
        errFlag = runSim('cfg_pdcch_pcap_example.yaml', 'pdcch_pcap_example', 'pdcch_pcap_example.txt');
        delete GPU_test_input/pcsch_pcap_example*;
        if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    end
    
    if (run_pdsch)
        fprintf('Test runSim for PDSCH pcap analysis ...');
        errFlag = runSim('cfg_pdsch_pcap_example.yaml', 'pdsch_pcap_example', 'pdsch_pcap_example.txt');
        delete GPU_test_input/pdsch_pcap_example*;
        if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    end
end

if run_genTestVector && run_perf_tvs
    fprintf('\nTest genCfgTV_perf_ss ...');

    % Can access these files because of paths in nr_matlab/startup.m
    errFlag = genCfgTV_perf_ss_json('performance.xlsm', 'testcases.json', 0); 
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    % errFlag = genCfgTV_perf_ss_json('performance-avg.xlsm', 'testcases.json', 0); 
    % if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    genCfgTV_perf_ss_bwc('performance-avg.xlsm');
    errFlag = genCfgTV_perf_pdcch(["F08", "F09", "F14"]); 
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    errFlag = genCfgTV_perf_pucch(["F08", "F09", "F14"]); 
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    errFlag = genCfgTV_perf_prach(["F08", "F09", "F14"]); 
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    errFlag = genCfgTV_perf_srs(["F09", "F14"]); % no SRS TV for F08
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    errFlag = genCfgTV_perf_csirs(["F08", "F09", "F14"]); 
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    errFlag = genCfgTV_perf_ssb(["F08", "F09", "F14"]);
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    errFlag = genCfgTV_perf_ss_vf_ueg('performance-avg-fdm.xlsm',['U08 - PUSCH', 'V08 - PDSCH', 'U09 - PUSCH', 'V09 - PDSCH', 'U14 - PUSCH', 'V14 - PDSCH'], 0); % disable FAPI TV
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end

    if strcmp(caseSet, 'selected')
        errFlag = genCfgTV_perf_ss('performance_example.xlsm', [], 1);
    else
        errFlag = genCfgTV_perf_ss('performance_example.xlsm', [], 1);
    end
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end

    fprintf('\nTest genCfgTV_perf_ss_vf ...');
    if strcmp(caseSet, 'selected')
        errFlag = genCfgTV_perf_ss_vf('performance-vf.clean.xlsx', [], 1);
    else
        errFlag = genCfgTV_perf_ss_vf('performance-vf.clean.xlsx', [], 1);
    end
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
end
if run_genTestVector && run_cuphycp_tvs
    fprintf('\nTest genCuPhyChEstCoeffs ...');
    genCuPhyChEstCoeffs;
    fprintf('==> PASS\n\n')
    
    fprintf('\nTest genCfgTV_cuphycp ...');
    errFlag = genCfgTV_cuphycp(caseSet);
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    
    fprintf('\nTest genCfgTV_demo ...');
    errFlag = genCfgTV_demo(caseSet);
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
end
if run_genTestVector && run_genLaunchPatternFile
    fprintf('\nTest genLP_cuphycp ...');
    errFlag = genLP_cuphycp(caseSet);
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    
    fprintf('\nTest genLP_POC2 ...');
    errFlag = genLP_POC2(caseSet);
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
    
    fprintf('\nTest genLaunchPatternFile ...');
    errFlag = genLaunchPatternFile(caseSet);
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
end
if run_genTestVector && run_genCfgTV_nvbug
    fprintf('\nTest genCfgTV_nvbug ...');
    errFlag = genCfgTV_nvbug(caseSet);
    if errFlag, fprintf('==> FAIL\n');  errFlagExist = 1; else fprintf('==> PASS\n\n'); end
end

fprintf('\nerrFlagExist = %d. \nStart running testCompGenTV/testPerformance ...\n\n', errFlagExist);

if ~strcmp(compTvMode, 'none')
    if run_ssb
        [nTC_ssb, err_ssb, nTV_ssb, detErr_ssb] = testCompGenTV_ssb(caseSet, compTvMode, subSetMod, relNum);
    end
    if run_pdcch
        [nTC_pdcch, err_pdcch, nTV_pdcch, detErr_pdcch] = testCompGenTV_pdcch(caseSet, compTvMode, subSetMod, relNum);
    end
    if run_pdsch
        [nTC_pdsch, err_pdsch, nTV_pdsch, detErr_pdsch] = testCompGenTV_pdsch(caseSet, compTvMode, subSetMod, relNum);
    end
    if run_csirs
        [nTC_csirs, err_csirs, nTV_csirs, detErr_csirs] = testCompGenTV_csirs(caseSet, compTvMode, subSetMod, relNum);
    end
    if run_dlmix && run_genTestVector
        [nTC_dlmix, err_dlmix, nCuphyTV_dlmix, nFapiTV_dlmix, detErr_dlmix] = testCompGenTV_dlmix(caseSet, compTvMode, subSetMod, relNum);
    end
    if run_prach
        [nTC_prach, err_prach, nTV_prach, detErr_prach] = testCompGenTV_prach(caseSet, compTvMode, subSetMod, relNum);
    end
    if run_pucch
        [nTC_pucch, err_pucch, nTV_pucch, detErr_pucch] = testCompGenTV_pucch(caseSet, compTvMode, subSetMod, relNum);
    end
    if run_pusch
        [nTC_pusch, err_pusch, nTV_pusch, detErr_pusch] = testCompGenTV_pusch(caseSet, compTvMode, subSetMod, relNum);
    end
    if run_srs
        [nTC_srs, err_srs, nTV_srs, detErr_srs] = testCompGenTV_srs(caseSet, compTvMode, subSetMod, relNum);
    end
    if run_simplex
        [nTC_simplex, encErr_simplex, scrErr_simplex, nTV_simplex] = testCompGenTV_simplex(caseSet, compTvMode);
    end
    if run_ulmix && run_genTestVector
        [nTC_ulmix, err_ulmix, nCuphyTV_ulmix, nFapiTV_ulmix, detErr_ulmix] = testCompGenTV_ulmix(caseSet, compTvMode, subSetMod, relNum);
    end
    if run_bfw && run_genTestVector
        [nTC_bfw, err_bfw, nTV_bfw, detErr_bfw] = testCompGenTV_bfw(caseSet, compTvMode, subSetMod, relNum);
    end
    if run_perfPattern
        % Call genPerfPattern and unpack results
        [perfResults, ~, ~, ~] = genPerfPattern(caseSet, channelSet);

        % Add perfPattern results to global counters for summary
        nTC_dlmix      = nTC_dlmix   + perfResults.dlmix.nTC;
        err_dlmix      = err_dlmix   + perfResults.dlmix.err;
        nCuphyTV_dlmix = nCuphyTV_dlmix + perfResults.dlmix.nCuphyTV;
        nFapiTV_dlmix  = nFapiTV_dlmix  + perfResults.dlmix.nFapiTV;
        detErr_dlmix   = detErr_dlmix  + perfResults.dlmix.detErr;

        nTC_ulmix      = nTC_ulmix   + perfResults.ulmix.nTC;
        err_ulmix      = err_ulmix   + perfResults.ulmix.err;
        nCuphyTV_ulmix = nCuphyTV_ulmix + perfResults.ulmix.nCuphyTV;
        nFapiTV_ulmix  = nFapiTV_ulmix  + perfResults.ulmix.nFapiTV;
        detErr_ulmix   = detErr_ulmix  + perfResults.ulmix.detErr;

        nTC_bfw        = nTC_bfw     + perfResults.bfw.nTC;
        err_bfw        = err_bfw     + perfResults.bfw.err;
        nTV_bfw        = nTV_bfw     + perfResults.bfw.nTV;
        detErr_bfw     = detErr_bfw  + perfResults.bfw.detErr;
    end
end

nPerf_pusch = 0;
nFail_pusch = 0;
nPerf_prach = 0;
nFail_prach = 0;
nPerf_pucch = 0;
nFail_pucch = 0;
nPerf_srs = 0;
nFail_srs = 0;
nPerf_pdsch = 0;
nFail_pdsch = 0;

if run_testPerformance
    if strcmp(compTvMode, 'none')
        nFrame_pusch_data = 10;
        nFrame_pusch_uci = 100;
        nFrame_pusch_prcd = 100;
        nFrame_pusch_bler = 1000;
        nFrame_prach = 100;
        nFrame_pucch_0 = 10;
        nFrame_pucch_1 = 10;
        nFrame_pucch_2 = 10;
        nFrame_pucch_3 = 10;
        nFrame_pdsch_data = 10;
    else
        nFrame_pusch_data = 1;
        nFrame_pusch_uci = 1;
        nFrame_pusch_prcd = 1;
        nFrame_pusch_bler = 1;
        nFrame_prach = 1;
        nFrame_pucch_0 = 1;
        nFrame_pucch_1 = 1;
        nFrame_pucch_2 = 1;
        nFrame_pucch_3 = 1;
        nFrame_pdsch_data = 1;
    end

    if run_pusch
        % PUSCH data, SNRoffset = 0
        [nPerf, nFail, CBer, TBer] = testPerformance_pusch(caseSet, 0, nFrame_pusch_data, 0);
        nPerf_pusch = nPerf_pusch + nPerf;
        nFail_pusch = nFail_pusch + nFail;
        % PUSCH UCI, SNRoffset = 0
        [nPerf, nFail, CBer, TBer] = testPerformance_pusch(caseSet, 0, nFrame_pusch_uci, 1);
        nPerf_pusch = nPerf_pusch + nPerf;
        nFail_pusch = nFail_pusch + nFail;
        % PUSCH transform precoding, SNRoffset = 0
        [nPerf, nFail, CBer, TBer] = testPerformance_pusch(caseSet, 0, nFrame_pusch_prcd, 2);
        nPerf_pusch = nPerf_pusch + nPerf;
        nFail_pusch = nFail_pusch + nFail;
        % PUSCH data 0.001% BLER
        [nPerf, nFail, CBer, TBer] = testPerformance_pusch(caseSet, 0, nFrame_pusch_bler, 3);
        nPerf_pusch = nPerf_pusch + nPerf;
        nFail_pusch = nFail_pusch + nFail;
        % PUSCH with perfect channel estimation
        [nPerf, nFail, CBer, TBer] = testPerformance_pusch(7992, 0, nFrame_pusch_bler);
        nPerf_pusch = nPerf_pusch + nPerf;
        nFail_pusch = nFail_pusch + nFail;
        [nPerf, nFail, CBer, TBer] = testPerformance_pusch([7701, 7716, 7717, 7732, 7733, 7748], 0, nFrame_pusch_data, 7);
        nPerf_pusch = nPerf_pusch + nPerf;
        nFail_pusch = nFail_pusch + nFail;
    end
    if run_prach
        % SNRoffset = 0, falseAlarmTest = 1 (false detection)
        [nPerf, nFail, Pfd, Pmd] = testPerformance_prach(caseSet, 0, nFrame_prach, 1);
        nPerf_prach = nPerf_prach + nPerf;
        nFail_prach = nFail_prach + nFail;
        % SNRoffset = 0, falseAlarmTest = 0 (missed detection)
        [nPerf, nFail, Pfd, Pmd] = testPerformance_prach(caseSet, 0, nFrame_prach, 0);
        nPerf_prach = nPerf_prach + nPerf;
        nFail_prach = nFail_prach + nFail;
    end
    if run_pucch
        % format-0
        % SNRoffset = 0, format-0, pucchTestMode = 1 (false detection)
        [nPerf, nFail, P_err] = testPerformance_pucch(caseSet, 0, nFrame_pucch_0, 0, 1);
        nPerf_pucch = nPerf_pucch + nPerf;
        nFail_pucch = nFail_pucch + nFail;
        % SNRoffset = 0, format-0, pucchTestMode = 2 (NACK to ACK detection)
        [nPerf, nFail, P_err] = testPerformance_pucch(caseSet, 0, nFrame_pucch_0, 0, 2);
        nPerf_pucch = nPerf_pucch + nPerf;
        nFail_pucch = nFail_pucch + nFail;
        % SNRoffset = 0, format-0, pucchTestMode = 3 (ACK missed detection)
        [nPerf, nFail, P_err] = testPerformance_pucch(caseSet, 0, nFrame_pucch_0, 0, 3);
        nPerf_pucch = nPerf_pucch + nPerf;
        nFail_pucch = nFail_pucch + nFail;
        % format-1
        % SNRoffset = 0, format-1, pucchTestMode = 1 (false detection)
        [nPerf, nFail, P_err] = testPerformance_pucch(caseSet, 0, nFrame_pucch_1, 1, 1);
        nPerf_pucch = nPerf_pucch + nPerf;
        nFail_pucch = nFail_pucch + nFail;
        % SNRoffset = 0, format-1, pucchTestMode = 2 (NACK to ACK detection)
        [nPerf, nFail, P_err] = testPerformance_pucch(caseSet, 0, nFrame_pucch_1, 1, 2);
        nPerf_pucch = nPerf_pucch + nPerf;
        nFail_pucch = nFail_pucch + nFail;
        % SNRoffset = 0, format-1, pucchTestMode = 3 (ACK missed detection)
        [nPerf, nFail, P_err] = testPerformance_pucch(caseSet, 0, nFrame_pucch_1, 1, 3);
        nPerf_pucch = nPerf_pucch + nPerf;
        nFail_pucch = nFail_pucch + nFail;
        % format-2
        % SNRoffset = 0, format-2, pucchTestMode = 1 (false detection)
        [nPerf, nFail, P_err] = testPerformance_pucch(caseSet, 0, nFrame_pucch_2, 2, 1);
        nPerf_pucch = nPerf_pucch + nPerf;
        nFail_pucch = nFail_pucch + nFail;
        % SNRoffset = 0, format-2, pucchTestMode = 3 (ACK missed detection)
        [nPerf, nFail, P_err] = testPerformance_pucch(caseSet, 0, nFrame_pucch_2, 2, 3);
        nPerf_pucch = nPerf_pucch + nPerf;
        nFail_pucch = nFail_pucch + nFail;
        % SNRoffset = 0, format-2, pucchTestMode = 4 (UCI BLER)
        [nPerf, nFail, P_err] = testPerformance_pucch(caseSet, 0, nFrame_pucch_2, 2, 4);
        nPerf_pucch = nPerf_pucch + nPerf;
        nFail_pucch = nFail_pucch + nFail;
        % format-3
        % SNRoffset = 0, format-3, pucchTestMode = 4 (UCI BLER)
        [nPerf, nFail, P_err] = testPerformance_pucch(caseSet, 0, nFrame_pucch_3, 3, 4);
        nPerf_pucch = nPerf_pucch + nPerf;
        nFail_pucch = nFail_pucch + nFail;
    end
    if run_srs && 0 % disable SRS perf test
        [nPerf, nFail] = testPerformance_srs(caseSet);
        nPerf_srs = nPerf_srs + nPerf;
        nFail_srs = nFail_srs + nFail;
    end
    if run_pdsch
        [nPerf, nFail, CBer, TBer] = testPerformance_pdsch([3704, 3720, 3736, 3750], 0, nFrame_pdsch_data, 7);
        nPerf_pdsch = nPerf_pdsch + nPerf;
        nFail_pdsch = nFail_pdsch + nFail;
    end
end

nTC_total = nTC_ssb + nTC_pdcch + nTC_pdsch + nTC_csirs + nTC_dlmix ...
    + nTC_prach + nTC_pucch + nTC_pusch + nTC_srs + nTC_simplex + nTC_ulmix + nTC_bfw;

compErr_total = err_ssb + err_pdcch + err_pdsch + err_csirs + err_dlmix ...
    + err_prach + err_pucch + err_pusch + err_srs + (encErr_simplex+scrErr_simplex) + err_ulmix + err_bfw;

nCuphyTV_total = nTV_ssb + nTV_pdcch + nTV_pdsch + nTV_csirs + nCuphyTV_dlmix ...
    + nTV_prach + nTV_pucch + nTV_pusch + nTV_srs + nTV_simplex + nCuphyTV_ulmix + nTV_bfw;

nFapiTV_total = nTV_ssb + nTV_pdcch + nTV_pdsch + nTV_csirs + nFapiTV_dlmix ...
        + nTV_prach + nTV_pucch + nTV_pusch + nTV_srs + nTV_simplex + nFapiTV_ulmix + nTV_bfw;

detErr_total = detErr_ssb + detErr_pdcch + detErr_pdsch + detErr_csirs + detErr_dlmix ...
    + detErr_prach + detErr_pucch + detErr_pusch + detErr_srs + detErr_ulmix + detErr_bfw;

nPerf_total = nPerf_pusch + nPerf_prach + nPerf_pucch + nPerf_srs + nPerf_pdsch;
nFail_total = nFail_pusch + nFail_prach + nFail_pucch + nFail_srs + nFail_pdsch;


fprintf('Channel  Compliance_Test  Error   CUPHY_Test_Vector   FAPI_Test_Vector  Error  Performance_Test   Fail\n')
fprintf('------------------------------------------------------------------------------\n')
fprintf('SSB          %3d          %3d           %3d                 %3d          %3d         %3d          %3d\n', nTC_ssb, err_ssb, nTV_ssb, nTV_ssb, detErr_ssb, 0, 0);
fprintf('PDCCH        %3d          %3d           %3d                 %3d          %3d         %3d          %3d\n', nTC_pdcch, err_pdcch, nTV_pdcch, nTV_pdcch, detErr_pdcch, 0, 0);
fprintf('PDSCH        %3d          %3d           %3d                 %3d          %3d         %3d          %3d\n', nTC_pdsch, err_pdsch, nTV_pdsch, nTV_pdsch, detErr_pdsch, nPerf_pdsch, nFail_pdsch);
fprintf('CSIRS        %3d          %3d           %3d                 %3d          %3d         %3d          %3d\n', nTC_csirs, err_csirs, nTV_csirs, nTV_csirs, detErr_csirs, 0, 0);
fprintf('DLMIX        %3d          %3d           %3d                 %3d          %3d         %3d          %3d\n', nTC_dlmix, err_dlmix, nCuphyTV_dlmix, nFapiTV_dlmix, detErr_dlmix, 0, 0);
fprintf('PRACH        %3d          %3d           %3d                 %3d          %3d         %3d          %3d\n', nTC_prach, err_prach, nTV_prach, nTV_prach, detErr_prach, nPerf_prach, nFail_prach);
fprintf('PUCCH        %3d          %3d           %3d                 %3d          %3d         %3d          %3d\n', nTC_pucch, err_pucch, nTV_pucch, nTV_pucch, detErr_pucch, nPerf_pucch, nFail_pucch);
fprintf('PUSCH        %3d          %3d           %3d                 %3d          %3d         %3d          %3d\n', nTC_pusch, err_pusch, nTV_pusch, nTV_pusch, detErr_pusch, nPerf_pusch, nFail_pusch);
fprintf('SRS          %3d          %3d           %3d                 %3d          %3d         %3d          %3d\n', nTC_srs, err_srs, nTV_srs, nTV_srs, detErr_srs, nPerf_srs, nFail_srs);
fprintf('SIMPLEX      %3d          %3d           %3d                 %3d          %3d         %3d          %3d\n', nTC_simplex, (encErr_simplex+scrErr_simplex), nTV_simplex, nTV_simplex, 0, 0, 0)
fprintf('ULMIX        %3d          %3d           %3d                 %3d          %3d         %3d          %3d\n', nTC_ulmix, err_ulmix, nCuphyTV_ulmix, nFapiTV_ulmix, detErr_ulmix, 0, 0);
fprintf('BFW          %3d          %3d           %3d                 %3d          %3d         %3d          %3d\n', nTC_bfw, err_bfw, nTV_bfw, nTV_bfw, detErr_bfw, 0, 0);
fprintf('------------------------------------------------------------------------------\n')
fprintf('Total       %4d          %3d          %4d                %4d          %3d         %3d          %3d\n', nTC_total, compErr_total, nCuphyTV_total, nFapiTV_total, detErr_total, nPerf_total, nFail_total);

err_total = errFlagExist + compErr_total + detErr_total + nFail_total;

fprintf('\nTotal time for runRegression is %d seconds\n', round(toc(TSTART)));

return;

function printUsage

fprintf('Usage: runRegression(testSet, channelSet, caseSet) \n');
fprintf('testSet: Compliance, TestVector, Performance, PerfPattern, allTests\n');
fprintf('channelSet: ssb, pdcch, pdsch, csirs, dlmix, allDL, prach, pucch, pusch, srs, ulmix, allUL, bfw, cuPHY-CP_TVs, perf_TVs, cfgTemplate, launchPatternFile, CfgTV_nvbug, allChannels \n')
fprintf('caseSet: compact, full, selected (choose one set only) \n')
fprintf('Example: runRegression({''Compliance'', ''TestVector''}, {''allDL'', ''prach'', ''srs''}, ''compact'')\n');
fprintf('runRegression() = runRegression({''allTests''}, {''allChannels''}, ''full'')\n');
fprintf('runRegression() = runRegression({''PerfPattern''}, {''allChannels''}, 59.3)\n');

return
