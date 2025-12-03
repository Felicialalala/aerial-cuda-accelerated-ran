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

function oranCompDecomp_unitest(X_tf_orig)
% ORANCOMPDECOMP_UNITTEST Test function for ORAN compression and decompression
%
% This function tests the ORAN compression and decompression functionality by:
%   1. Compressing input data using both BFP9 and BFP14 formats
%   2. Decompressing the data
%   3. Calculating and displaying error statistics
%   4. Generating error distribution plots
%
% Inputs:
%   X_tf_orig - (Optional) Complex input data array of size [Nprb*12 x Nsym x Nant]
%               If not provided, generates random test data with default dimensions
%               where:
%               - Nprb: Number of physical resource blocks
%               - Nsym: Number of OFDM symbols
%               - Nant: Number of antennas
%               - 12: Number of resource elements per PRB
%
% Outputs:
%   None - Results are displayed and plots are saved to files:
%          - Error statistics (RMSE, SNR, Max Error)
%          - Error distribution histograms
%          - Sample values at maximum error location
%          - Compression test plots (compression_test_bfp*.png)
%
% Example:
%   oranCompDecomp_unittest()  % Run with default random data
%   oranCompDecomp_unittest(my_data)  % Run with custom input data
    
    RE_PER_PRB = 12;
    sim_is_uplink = true;  % Set based on your needs
    
    % If X_tf_orig is provided, use its dimensions
    % Otherwise use default values to generate new data
    if nargin < 1 || isempty(X_tf_orig)
        % Default test parameters
        Nprb = 273;
        Nsym = 14;
        Nant = 8;
        
        % Generate random complex data
        rng(0);  % Set random seed for reproducibility
        X_tf_orig = (randn(Nprb*RE_PER_PRB, Nsym, Nant) + ...
                    1j * randn(Nprb*RE_PER_PRB, Nsym, Nant)) / sqrt(2);
    else
        % Extract dimensions from provided X_tf_orig
        [NprbRE, Nsym, Nant] = size(X_tf_orig);
        Nprb = NprbRE / RE_PER_PRB;
    end

    % Test different compression widths
    iqWidths = [9, 14];  % Test both BFP9 and BFP14
    
    % Initialize SimCtrl global variable needed by oranCompress
    global SimCtrl;
    SimCtrl.oranComp.iqWidth = iqWidths;
    SimCtrl.oranComp.Ref_c = [0, 0];
    SimCtrl.oranComp.FSOffset = [0, 0];
    SimCtrl.oranComp.Nre_max = Nprb * RE_PER_PRB;
    SimCtrl.oranComp.max_amp_ul = 65504;
    SimCtrl.fp16AlgoSel = 2;  % FP16 conversion algorithm selection
    SimCtrl.oranCompressBetaForce = 0;  % Force beta values
    
    % Quantize X_tf_orig to FP16
    X_tf_orig = fp16nv(real(X_tf_orig), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(X_tf_orig), SimCtrl.fp16AlgoSel);
    
    % Test each compression width
    for idx = 1:length(iqWidths)
        iqWidth = iqWidths(idx);
        fprintf('\nTesting compression with IQ width %d:\n', iqWidth);
    
        iqWidth = SimCtrl.oranComp.iqWidth(idx);
        Ref_c = SimCtrl.oranComp.Ref_c(idx);
        FSOffset = SimCtrl.oranComp.FSOffset(idx);
        Nre_max = SimCtrl.oranComp.Nre_max;
        max_amp_ul = SimCtrl.oranComp.max_amp_ul;
        % model compress and decompress loss
        X_tf_compDecomp = bfpCompDecomp(X_tf_orig, iqWidth, Ref_c, FSOffset, Nre_max, max_amp_ul, 1);
        X_tf_compDecomp = fp16nv(real(X_tf_compDecomp), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(X_tf_compDecomp), SimCtrl.fp16AlgoSel);  % data for 5GModel processing
        % Compress
        [compressed_data] = oranCompress(X_tf_orig, sim_is_uplink);
        
        % Get beta scale based on iqWidth
        FSOffset = SimCtrl.oranComp.FSOffset(idx);
        Ref_c = SimCtrl.oranComp.Ref_c(idx);
        Nre_max = SimCtrl.oranComp.Nre_max;
        max_amp_ul = SimCtrl.oranComp.max_amp_ul;
        beta = oranCalcBeta(sim_is_uplink, iqWidth, FSOffset, Ref_c, Nre_max, max_amp_ul);
        % hack to force beta
        if SimCtrl.oranCompressBetaForce == 1
            if iqWidth == 9
                beta = 65536;
            elseif iqWidth == 14
                beta = 2097152;
            end           
        end
        
        % Decompress
        if ~sim_is_uplink
            beta = 1 / beta;
        end
        decompressed_data = oranDecompress(compressed_data{idx}, iqWidth, ...
                                         Nprb, Nsym, Nant, beta, 0);  % not using permute
        decompressed_data = fp16nv(real(decompressed_data), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(decompressed_data), SimCtrl.fp16AlgoSel);  % data for cuPHY processing
        % Calculate error statistics
        error = abs(decompressed_data - X_tf_compDecomp);
        max_error = max(error(:));
        [max_error_idx_linear] = find(error == max_error, 1);
        [i, j, k] = ind2sub(size(error), max_error_idx_linear);
        
        rmse = sqrt(mean(error(:).^2));
        
        % Calculate SNR
        signal_power = mean(abs(X_tf_compDecomp(:)).^2);
        noise_power = mean(abs(error(:)).^2);
        snr_db = 10 * log10(signal_power/noise_power);
        
        % Print statistics
        fprintf('RMSE: %.2e\n', rmse);
        fprintf('SNR: %.2f dB\n', snr_db);
        fprintf('Max Error: %.2e\n', max_error);
        fprintf('Max Error Location: (%d, %d, %d)\n', i, j, k);
        fprintf('Ref value at max error: (%.6f%+.6fj)\n', ...
            real(X_tf_compDecomp(i,j,k)), imag(X_tf_compDecomp(i,j,k)));
        fprintf('Decompressed value at max error: (%.6f%+.6fj)\n', ...
            real(decompressed_data(i,j,k)), imag(decompressed_data(i,j,k)));
        fprintf('Max Error: %.2e\n', max_error);
        
        % % Plot error histogram
        % figure;
        % histogram(error(:), 100);
        % title(sprintf('Error Distribution - BFP%d', iqWidth));
        % xlabel('Error Magnitude');
        % ylabel('Count');
        % grid on;
        
        % % Add text box with statistics
        % stats_str = sprintf('Max Error: %.2e\nMean Error: %.2e\nRMSE: %.2e\nSNR: %.2f dB', ...
        %     max_error, mean_error, rmse, snr_db);
        % annotation('textbox', [0.65 0.7 0.3 0.2], 'String', stats_str, ...
        %     'FitBoxToText', 'on', 'BackgroundColor', 'white');
        
        % % Save the figure
        % saveas(gcf, sprintf('compression_test_bfp%d.png', iqWidth));
        
        % % Verify expected error bounds
        % if iqWidth == 9
        %     expected_max_error = 1/256;  % Adjust based on your requirements
        % else  % iqWidth == 14
        %     expected_max_error = 1/65536;  % Adjust based on your requirements
        % end
        % 
        % % Assert tests
        % assert(max_error < expected_max_error, ...
        %     'Max error %.2e exceeds expected bound %.2e for BFP%d', ...
        %     max_error, expected_max_error, iqWidth);
        % 
        % assert(snr_db > 30, ...  % Adjust threshold based on your requirements
        %     'SNR %.2f dB is below minimum threshold for BFP%d', ...
        %     snr_db, iqWidth);
    end
end