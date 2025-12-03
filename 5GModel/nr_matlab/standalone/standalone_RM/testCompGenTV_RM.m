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

function  testCompGenTV_RM(caseSet, compTvMode)

% evaluate BER achieved by 5G Toolbox functions
evalBERtoolbox = 1;

if nargin == 0
    caseSet = 'full';
    compTvMode = 'both';
elseif nargin == 1
    compTvMode = 'both';
end

switch compTvMode
        case 'both'
        genTvFlag = 1;
        testCompliance = 1;
        evalBER = 1;
    case 'genTV'
        genTvFlag = 1;
        testCompliance = 0;
        evalBER = 0;
    case 'testCompliance'
        genTvFlag = 0;
        testCompliance = 1;
        evalBER = 0;
    case 'evalBER'
        genTvFlag = 0;
        testCompliance = 0;
        evalBER = 1;
    otherwise
        error('compTvMode is not supported...\n');
end

compact_TC  = [62000:62004];
full_TC     = [62000:62004];
selected_TC = [62000:62004];

if isnumeric(caseSet)
    TcToTest = caseSet;
else    
    switch caseSet
        case 'compact'
            TcToTest = compact_TC;
        case 'full'
            TcToTest = full_TC;
        case 'selected'
            TcToTest = selected_TC;
        otherwise
            error('caseSet is not supported...\n');
    end
end

CFG = {...
% TC#    payload   E    SNR
62000  [1 1 1 1 1 0 1 0 1 1 0]   32   10 % default
62001  [1 1 1 1 1 0 1 0 1 1 0]   64   10 % vary E
62002  [1 1 1 1 1 0 1 0 1 1 0]   16   10 % vary E
62003  [1 1 1 1 1 0 1 0 1 1 0]   16   0 % vary E, SNR
62004  [1 1 1 1 1 0 1 0 1 1 0]   16   -5 % vary E, SNR
};

[NallTest, ~] = size(CFG);
nCompEncErrs    = 0;
nCompChecks     = 0;
nTvGen          = 0;

tvDirName = 'GPU_test_input';
fp16AlgoSel = 0;

for i = 1:NallTest
    caseNum = CFG{i, 1};
    if ismember(caseNum, TcToTest)
        rng(caseNum);
        nTvGen = nTvGen + genTvFlag;
        
        payload = CFG{i, 2};
        K       = length(payload);
        E       = CFG{i, 3};
        snr = CFG{i, 4};
        
        fecEnc = FecRmObj(1, E, K);
        outEncode = fecEnc(payload');
        rmBitLLRs = apply_RM_channel(snr, outEncode); % channel
        
        if genTvFlag
            TVname = [sprintf('TV_%05d', caseNum), '_RM'];
            saveTV_Simplex_cuphy(tvDirName, TVname, fp16AlgoSel, payload, K, E, rmBitLLRs);
        end
        
        if evalBER
            [berImpl, ber5GToolbox] = testRM(K, E, snr, 1000, 0, evalBERtoolbox);
            
            fprintf('\n BER evaluation for test case %d: K = %d, E = %d, SNR = %d dB, BER = %f%%, BER(toolbox) = %f%% \n', ...
                caseNum, K, E, snr, berImpl*100, ber5GToolbox*100);
        end
        
        if(testCompliance)
            nCompChecks = nCompChecks + 1;
            
            ToolboxOutEnc = nrUCIEncode(int8(transpose(payload)), E, 'pi/2-BPSK');
            
            err_bit = sum(abs(int8(outEncode)-ToolboxOutEnc));
            if err_bit > 0
                nCompEncErrs = nCompEncErrs + 1;
                fprintf('\n complience error with simplex encoder in test case: %d \n', caseNum);
            end
        end 
    end
end
fprintf('\n Generated %d test vectors', nTvGen);
fprintf('\n Performed %d complience tests, finding %d encoder errors \n\n', nCompChecks, nCompEncErrs);
end

function saveTV_Simplex_cuphy(tvDirName, TVname, fp16AlgoSel, payload, K, E, rmBitLLRs)
%%create h5 file
[status,msg]  = mkdir(tvDirName);
h5File        = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

RMPar.K       = uint8(K);
RMPar.E       = uint32(E);
payload       = reshape(payload, [], 1);

rmBitLLRs = fp16nv(rmBitLLRs, fp16AlgoSel);
payload   = fp16nv(payload, fp16AlgoSel);

hdf5_write_nv_exp(h5File, 'RMPar', RMPar);
hdf5_write_nv(h5File, 'payload', payload);
hdf5_write_nv(h5File, 'payload_uint8', uint8(payload));
hdf5_write_nv(h5File, 'rmBitLLRs', rmBitLLRs);


end
