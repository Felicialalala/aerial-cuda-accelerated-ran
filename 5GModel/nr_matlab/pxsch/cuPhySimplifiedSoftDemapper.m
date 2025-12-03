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

function out = cuPhySimplifiedSoftDemapper(in, T, QAM_bits, QAM_noise_var)
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

if QAM_bits == 8     % 256QAM
    A = 1/sqrt(170);
    LLR_mat(1,:) = in_phase;
    LLR_mat(3,:) = -abs(in_phase) + 8*A;
    LLR_mat(5,:) = -abs(abs(in_phase)-8*A) + 4*A;
    LLR_mat(7,:) = -abs(abs(abs(in_phase)-8*A)-4*A) + 2*A;
    LLR_mat(2,:) = quadrature;
    LLR_mat(4,:) = -abs(quadrature) + 8*A;
    LLR_mat(6,:) = -abs(abs(quadrature)-8*A) + 4*A;
    LLR_mat(8,:) = -abs(abs(abs(quadrature)-8*A)-4*A) + 2*A; 
    LLR_mat = LLR_mat*2*A; 
elseif QAM_bits == 6 % 64QAM
    A = 1/sqrt(42);
    LLR_mat(1,:) = in_phase;
    LLR_mat(3,:) = -abs(in_phase) + 4*A;
    LLR_mat(5,:) = -abs(abs(in_phase)-4*A) + 2*A;
    LLR_mat(2,:) = quadrature;
    LLR_mat(4,:) = -abs(quadrature) + 4*A;
    LLR_mat(6,:) = -abs(abs(quadrature)-4*A) + 2*A;
    LLR_mat = LLR_mat*2*A; 
elseif QAM_bits == 4 % 16QAM  
    A = 1/sqrt(10);
    LLR_mat(1,:) = in_phase;
    LLR_mat(3,:) = -abs(in_phase) + 2*A;
    LLR_mat(2,:) = quadrature;
    LLR_mat(4,:) = -abs(quadrature) + 2*A;
    LLR_mat = LLR_mat*2*A; 
elseif QAM_bits == 2 % 4QAM  
    A = 1/sqrt(2);
    LLR_mat(1,:) = in_phase;
    LLR_mat(2,:) = quadrature;
    LLR_mat = LLR_mat*2*A; 
elseif QAM_bits == 1 % BPSK  
    A = 1;
    LLR_mat(1,:) = in_phase;
    LLR_mat = LLR_mat*2*A; 
else
    error('Unknown QAM type!')
end
% % plot soft-demapper for 256QAM
% fig=figure; 
% t = tiledlayout('flow');%,TileSpacing='compact'
% x = T(:,1);
% nexttile;plot(T(:,1), T(:,2),'bo-', DisplayName='original', LineWidth=2, MarkerSize=12);hold on;plot(x, 2*A*(x), 'rx--', DisplayName='simplified', LineWidth=2, MarkerSize=12);title('The first bit'); grid minor;legend();xlim([-16*A,16*A]);
% nexttile;plot(T(:,1), T(:,3),'bo-', DisplayName='original', LineWidth=2, MarkerSize=12);hold on;plot(x, 2*A*(-abs(x) + 8*A), 'rx--', DisplayName='simplified', LineWidth=2, MarkerSize=12);title('The second bit'); grid minor;xlim([-16*A,16*A]);
% nexttile;plot(T(:,1), T(:,4),'bo-', DisplayName='original', LineWidth=2, MarkerSize=12);hold on;plot(x, 2*A*(-abs(abs(x)-8*A) + 4*A), 'rx--', DisplayName='simplified', LineWidth=2, MarkerSize=12);title('The third bit'); grid minor;xlim([-16*A,16*A]);
% nexttile;plot(T(:,1), T(:,5),'bo-', DisplayName='original', LineWidth=2, MarkerSize=12);hold on;plot(x, 2*A*(-abs(abs(abs(x)-8*A)-4*A) + 2*A), 'rx--', DisplayName='simplified', LineWidth=2, MarkerSize=12);title('The fourth bit'); grid minor;xlim([-16*A,16*A]);
% set(findall(gcf,'-property','FontSize'),'FontSize',16)  
% xlabel(t,'Real or Imag part of postEq symbol','FontSize',20);
% ylabel(t,'LLR','FontSize',20);



% Comment out to save runtime
% if SimCtrl.fp16AlgoSel == 0
%     LLR_mat_fp16 = half(LLR_mat);
% else
%     LLR_mat_fp16 = vfp16(LLR_mat);
% end
% LLR_mat = double(LLR_mat_fp16);

out = 1/PAM_noise_var * LLR_mat(:);

return


