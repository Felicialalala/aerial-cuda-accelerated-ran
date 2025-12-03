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

%
% This is the main program to run nrSimulator for algorithm design and
% new feature development. 
%
%

function [errFlag, SysPar, UE, gNB] = runSim(cfgFileName, tvFileName, rxFileName)

% rng(0); move downwards so that we can read seed from cfgFie

if nargin == 0
    if isdeployed
        error('In deployed mode with batchsim launch, it should not reach here! Expect cfgFileName in deployed mode.')
    end
    printUsage;
    testAlloc.dl = 1;
    testAlloc.ul = 0;    
    if testAlloc.dl
        testAlloc.ssb = 1;
        testAlloc.pdcch = 1;
        testAlloc.pdsch = 1;
        testAlloc.csirs = 1;
    end    
    if testAlloc.ul
        testAlloc.prach = 1;
        testAlloc.pucch = 0;
        testAlloc.pusch = 0;
        testAlloc.srs = 0;
    end    
    SysPar = initSysPar(testAlloc);  
else
    testAlloc.dl = 0;
    testAlloc.ul = 0;
    testAlloc.ssb = 0;
    testAlloc.pdcch = 0;
    testAlloc.pdsch = 0;
    testAlloc.csirs = 0;
    testAlloc.prach = 0;
    testAlloc.pucch = 0;
    testAlloc.pusch = 0;
    testAlloc.srs = 0;
    SysPar = initSysPar(testAlloc); 
    SysPar_yaml = ReadYaml(cfgFileName, 0, 0);
    SysPar = updateStruct(SysPar, SysPar_yaml);
    if isdeployed 
        if isempty(fieldnames(SysPar_yaml)) % in deployed mode with batchsim launch, SysPar_yaml should not be empty!
            error('Parsing cfg yaml file failed in deployed mode!')      % WriteYaml(fullfile(fileparts(cfgFileName),'cfg_template_sysparyaml.yaml'), SysPar_yaml);
        end
        if isfile(fullfile(ctfroot,'/runSim/cfg_template.yaml'))
            error(sprintf('%s should not be here! Please exclude it from your compilation!\n', fullfile(ctfroot,'/runSim/cfg_template.yaml')))
        end
    end
    fprintf('\nRead config from %s\n', cfgFileName);      
    if nargin > 1
        SysPar.SimCtrl.genTV.enable = 1;
        SysPar.SimCtrl.genTV.TVname = tvFileName;
        SysPar.SimCtrl.genTV.forceSlotIdxFlag = 1;
    end
    if nargin == 3
        SysPar.SimCtrl.capSamp.enable = 1;
        SysPar.SimCtrl.capSamp.fileName = rxFileName;
    end    
end

if SysPar.SimCtrl.capSamp.enable
    SysPar.SimCtrl.enableUeRx = 1;
end

SysPar = updateAlgFlag(SysPar);

if SysPar.SimCtrl.ml.dataset.enable_save_dataset
    SysPar.SimCtrl = init_table_ML_datasets(SysPar.SimCtrl);
    SysPar.SimCtrl.ml.dataset.current_time_sec = 0;
end

% run nrSimulator
rng(SysPar.SimCtrl.seed,'threefry') % by default the MATLAB client uses generator 'twister' while the workers (under parfor loops in testPerformance_xxxxx.m) use 'threefry'. In order to keep them the same, force the the client generator here to 'threefry' as well. Refer to https://www.mathworks.com/help/matlab/ref/rng.html ; refer to https://www.mathworks.com/help/parallel-computing/what-is-parallel-computing.html for what are client and worker.
[SysPar, UE, gNB] = nrSimulator(SysPar);

SimCtrl = SysPar.SimCtrl;

testAlloc = SysPar.testAlloc;
fprintf('\nChannel: ssb pdcch pdsch csirs prach pucch pusch srs\n');
fprintf('----------------------------------------------------\n');
fprintf('Alloc: %4d %4d  %4d  %4d  %4d  %4d  %4d %4d\n\n', testAlloc.ssb, testAlloc.pdcch,...
    testAlloc.pdsch, testAlloc.csirs, testAlloc.prach, testAlloc.pucch,...
    testAlloc.pusch, testAlloc.srs);

errFlag = 0;
printFlag = 1;
if (checkDetError(testAlloc, SimCtrl, printFlag))
    errFlag = 1;
end

if SimCtrl.batchsim.save_results
    cfg_file_folder = fileparts(cfgFileName);
    results_folder = fullfile(cfg_file_folder,'/results/');
    if ~exist(results_folder,'dir')
        mkdir(results_folder)
    end        
    if SysPar.SimCtrl.ml.dataset.enable_save_dataset
        dataset_filename = fullfile(results_folder,'/ML_dataset.parquet');
        SimCtrl.ml.dataset.table_ML_datasets.to_parquet(pyargs('path',dataset_filename, 'engine','pyarrow','compression','snappy'))
%         parquetwrite(dataset_filename,SimCtrl.ml.dataset.table_ML_datasets);
    end    
    enable_save_Hgenie_in_separate_file = 0;
    if enable_save_Hgenie_in_separate_file
        
        Hgenie = SysPar.SimCtrl.results.pusch{1}.Hgenie;
        [Nrx,Ntx,Nsc,Nsym,Nslots] = size(Hgenie);
        num_zeros_lower = (SysPar.carrier.Nfft-Nsc)/2;
        num_zeros_upper = num_zeros_lower;
        zp_lower = zeros(Nrx,Ntx,num_zeros_lower,Nsym, Nslots);
        zp_upper = zeros(Nrx,Ntx,num_zeros_upper,Nsym, Nslots);
        Hgenie = cat(3,zp_lower, Hgenie,zp_upper);
        if SysPar.SimCtrl.N_Interfering_UE_UL>0
            interf_Hgenie = SysPar.SimCtrl.results.pusch{1}.interf_Hgenie;
            interf_Hgenie = cat(3,zp_lower, interf_Hgenie,zp_upper);
            genie_nCov = SysPar.SimCtrl.results.pusch{1}.genie_nCov;
        end
        numSlotsPerFrame = 10*2^(SysPar.carrier.mu);
        N_frames = Nslots/numSlotsPerFrame;
        for idx_frame = 1:N_frames
            idxStart = (idx_frame-1)*numSlotsPerFrame+1;
            idxEnd = idx_frame*numSlotsPerFrame;
            Hgenie_thisFrame = Hgenie(:,:,:,:,idxStart:idxEnd);
            results_filename = fullfile(results_folder,sprintf('/genieCFR_frame%d.mat',idx_frame-1));
            save(results_filename,'Hgenie_thisFrame');
            if SysPar.SimCtrl.N_Interfering_UE_UL>0
                interf_Hgenie_thisFrame = interf_Hgenie(:,:,:,:,idxStart:idxEnd);
                results_filename = fullfile(results_folder,sprintf('/interf_genieCFR_frame%d.mat',idx_frame-1));
                save(results_filename,'interf_Hgenie_thisFrame');
                genie_nCov_thisFrame = genie_nCov(:,:,:,idxStart:idxEnd);
                results_filename = fullfile(results_folder,sprintf('/genie_nCov_frame%d.mat',idx_frame-1));
                save(results_filename,'genie_nCov_thisFrame');
            end
        end
        SysPar.SimCtrl.results.pusch{1}.Hgenie = [];
        if SysPar.SimCtrl.N_Interfering_UE_UL>0
            SysPar.SimCtrl.results.pusch{1}.interf_Hgenie = [];
        end
    end
    results_filename = fullfile(results_folder,'/results.mat');
    if SimCtrl.batchsim.save_results_short
        SysParShort = struct();
        SysParShort.SimCtrl.results = SysPar.SimCtrl.results;
        SysParShort.Chan = SysPar.Chan;
        save(results_filename,'SysParShort')
    else
        save(results_filename,'SysPar')
    end
end

if SimCtrl.plotFigure.tfGrid
    testAlloc = SysPar.testAlloc;
    % plot TX signal
    if testAlloc.dl
        Xtf_frame = gNB.Phy.tx.Xtf_frame(:, :, 1);
        figure; mesh(abs(Xtf_frame)); view(2);
        xlabel('symbol index');
        ylabel('subcarrier index');
        title('gNB TX OFDM symbols');
        if SimCtrl.plotFigure.constellation
            figure;
            for idxSym = 1:14
                subplot(4, 4, idxSym);
                plot(complex(Xtf_frame(:, idxSym)), '.');
                hold on; grid on;
                title(['I/Q on sym ', num2str(idxSym-1)]);
            end
        end
    end
    
    N_UE = SysPar.SimCtrl.N_UE;
    if testAlloc.ul
        if SysPar.SimCtrl.capSamp.enable
            Xtf_frame = gNB.Phy.rx.Xtf(:, :, 1);
            figure; mesh(abs(Xtf_frame)); view(2);
            xlabel('symbol index');
            ylabel('subcarrier index');
            title('gNB RX OFDM symbols from capture');
        else
            for idxUE = 1:N_UE
                Xtf_frame = UE{idxUE}.Phy.tx.Xtf_frame(:, :, 1);
                figure;
                mesh(abs(Xtf_frame));
                view(2);
                xlabel('symbol index');
                ylabel('subcarrier index');
                title('UE TX OFDM symbols');
            end
        end
    end    
end

pause(1);

return

function printUsage

fprintf('Usage: Run a single test case with user defined test configuration through input yaml file.\n');  
fprintf('1. Run genCfgTemplate to generate a configuration template file ''cfg_template.yaml''\n'); 
fprintf('2. Edit the template file with specific configuration and save it to another yaml file name\n');
fprintf('3. Run runSim(cfgFileName, tvFileName). For example, runSim(''pdsch_f14_cfg.yaml'', ''pdsch_f14'')\n');
fprintf('runSim() will run with the default configuration.\n');

return
