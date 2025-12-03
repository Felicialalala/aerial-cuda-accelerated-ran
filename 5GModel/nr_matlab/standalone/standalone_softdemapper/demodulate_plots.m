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

%PAM_range = 15 / sqrt(170);
PAM_range = 16 / sqrt(170);
PAM = linspace(-PAM_range, PAM_range)';
symbols = PAM + (i * PAM);
demod_matlab           = llr_vec_to_bits(     nrSymbolDemodulate(symbols, '256QAM', 2),                      8);
demod_table_model      = llr_vec_to_bits(table_symbol_demodulate_fp16(symbols, '256QAM', 2),                      8);
demod_table_model_fp16 = llr_vec_to_bits(table_symbol_demodulate_fp16(symbols, '256QAM', 2, 'Precision', 'fp16'), 8);

cuPHYOutput = hdf5_load_nv('sym_demod_fp16_QAM256.h5');
sym_cuphy              = real(cuPHYOutput.sym);
demod_cuphy            = llr_vec_to_bits(cuPHYOutput.llr, 8);

f = figure;
for s = 1:4
    subplot(4, 1, s);
    plot(PAM, demod_matlab(:,s),      'bo-', ...
         PAM, demod_table_model(:,s),       'gs-', ...
         sym_cuphy, demod_cuphy(:,s), 'rx-');
end

% Input LLR values are grouped by symbol, with log2QAM values
% per symbol. For this test, we will only look at the even
% bits, since (for 5G and the test structure chosen here) the
% odd bits will have the same values.
function demod_bits = llr_vec_to_bits(v, log2QAM)
    demod_bits = reshape(v, log2QAM, [])';
    demod_bits = demod_bits(:,1:2:end);
end
