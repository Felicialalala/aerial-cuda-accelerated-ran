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

function out = table_symbol_demodulate(in, mod, QAM_noise_var)
  % Implements symbol demodulation (otherwise known as soft demapping)
  % using generated tables, with results intended to be identical to
  % the MATLAB nrSymbolDemodulate() function.
  %---------------------------------------------------------------------
  % PAM noise is 1/2 QAM noise, assuming noise power is equally
  % distributed between the in-phase and quadrature components
  PAM_noise_var = QAM_noise_var / 2;
  %---------------------------------------------------------------------
  % Load a table that matches the input modulation
  switch(mod)
    case 'BPSK'
      % Reuse the QPSK table here!
      T = readtable('QAM4_LLR.txt');
      A = 1 / sqrt(2);
      QAM_bits = uint32(1);
    case 'QPSK'
      T = readtable('QAM4_LLR.txt');
      A = 1 / sqrt(2);
      QAM_bits = uint32(2);
    case '16QAM'
      T = readtable('QAM16_LLR.txt');
      A = 1 / sqrt(10);
      QAM_bits = uint32(4);
    case '64QAM'
      T = readtable('QAM64_LLR.txt');
      A = 1 / sqrt(42);
      QAM_bits = uint32(6);
    case '256QAM'
      T = readtable('QAM256_LLR.txt');
      A = 1 / sqrt(170);
      QAM_bits = uint32(8);
    otherwise
      error('Invalid modulation: %s', mod)
  end
  num_symbols = size(in, 1);
  out = zeros(num_symbols * QAM_bits, 1);
  %---------------------------------------------------------------------
  PAM_bits = max(QAM_bits / 2, 1);
  %---------------------------------------------------------------------
  in_phase   = reshape(real(in), 1, length(in));
  quadrature = reshape(imag(in), 1, length(in));
  LLR_mat    = zeros(QAM_bits, length(in));
  for ii = 1:PAM_bits
    LLR_mat(ii * 2 - 1,:) = interp1(T.Zr, T.(sprintf('bit%d', ii-1)), in_phase, 'linear');
    if QAM_bits > 1
      LLR_mat(ii * 2 - 0,:) = interp1(T.Zr, T.(sprintf('bit%d', ii-1)), quadrature, 'linear');
    end
  end
  out = PAM_noise_var * LLR_mat(:);
end
