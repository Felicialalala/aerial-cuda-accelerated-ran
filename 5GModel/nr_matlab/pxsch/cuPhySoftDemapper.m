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

function out = cuPhySoftDemapper(in, T, QAM_bits, QAM_noise_var)
% Implements symbol demodulation (otherwise known as soft demapping)
% using generated tables, with results intended to provide the same
% outputs as the MATLAB nrSymbolDemodulate() function, but using the
% cuPHY library implementation.
%---------------------------------------------------------------------

% PAM noise is 1/2 QAM noise, assuming noise power is equally
% distributed between the in-phase and quadrature components
PAM_noise_var = QAM_noise_var / 2;
num_symbols = size(in, 1);
out = zeros(num_symbols * QAM_bits, 1);

%---------------------------------------------------------------------
PAM_bits = max(QAM_bits / 2, 1);
%---------------------------------------------------------------------

in_phase   = reshape(real(in), 1, length(in));
quadrature = reshape(imag(in), 1, length(in));
LLR_mat    = zeros(QAM_bits, length(in));

if QAM_bits == 1
    LLR_mat = in_phase;
else
    for ii = 1:PAM_bits
        %     LLR_mat(ii * 2 - 1,:) = interp1(T(:,1), T(:,ii+1), in_phase, 'linear');
        LLR_mat(ii * 2 - 1,:) = linearInterp(T(:,1), T(:,ii+1), in_phase);
        if QAM_bits > 1
            %         LLR_mat(ii * 2 - 0,:) = interp1(T(:,1), T(:,ii+1), quadrature, 'linear');
            LLR_mat(ii * 2 - 0,:) = linearInterp(T(:,1), T(:,ii+1), quadrature);
        end
    end
end

% Comment out to save runtime
% if SimCtrl.fp16AlgoSel == 0
%     LLR_mat_fp16 = half(LLR_mat);
% else
%     LLR_mat_fp16 = vfp16(LLR_mat);
% end
% LLR_mat = double(LLR_mat_fp16);

out = 1/PAM_noise_var * LLR_mat(:);

return

function out = linearInterp(x, y, in)

Nx = length(x);
idx = find(x  > in, 1);

if isempty(idx)
    idx1 = Nx-1;
    idx2 = Nx;
elseif idx == 1
    idx1 = 1;
    idx2 = 2;
else
    idx1 = idx - 1;
    idx2 = idx;
end

out = (y(idx2)-y(idx1))/(x(idx2)-x(idx1))*(in-x(idx1)) + y(idx1);

return

