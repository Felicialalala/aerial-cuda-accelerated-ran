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

function [H_est, ChEst_par] = chan_estimation(Y, table, slotNumber, Nf, Nt, N_dmrs_id, ...
        nl, portIdx, n_scid, symIdx_dmrs, Nt_dmrs, nPrb, startPrb, delta_f, ...
        numDmrsCdmGrpsNoData, enableTfPrcd, N_slot_frame, N_symb_slot, ...
        puschIdentity, groupOrSequenceHopping)

% function applies legacy ChEst

%inputs:
% Y --> signal recieved by base station. Dim: Nf x Nt x L_BS

%outputs:
% H_est --> estimated channel. Dim: L_BS x L_UE x Nf

%%

if enableTfPrcd == 0
    r_dmrs  = conj(gen_dmrs_sym(slotNumber, Nf, Nt, N_dmrs_id));
else
    [r_dmrs, ~, ~] = gen_dmrs_dftsofdm(Nf, startPrb, nPrb, N_slot_frame, N_symb_slot, symIdx_dmrs, ...
        slotNumber, puschIdentity, groupOrSequenceHopping);
    r_dmrs = conj(r_dmrs);
end

%PARAMATERS
%shift:
df = delta_f;
tau = (2.0*10^(-6) - 2*10^(-6)/10) / 2; % delay shift

global SimCtrl
if SimCtrl.delaySpread > 0
    tau = (SimCtrl.delaySpread-SimCtrl.delaySpread/8)/2;
end

f = 0 : Nf;
f = f.'*df;

f_dmrs = 0 : 2 : (Nf - 1);
f_dmrs = f_dmrs + 1;
f_dmrs = f_dmrs.'*df;
sd = exp(2*pi*1i*tau*f_dmrs);
% sd = repmat(sd,[1 Nt_dmrs]) .* conj(r_dmrs(:,symIdx_dmrs));
sd = repmat(sd,[1 Nt_dmrs]) .* (r_dmrs(:,symIdx_dmrs));

s = exp(-2*pi*1i*tau*f);
d = conj(r_dmrs(:,symIdx_dmrs));  % dmrs descrambling sequence. Dim: Nf / 2 x Nt_dmrs
s_grid = exp(2*pi*1i*tau*f_dmrs); % 1st shift sequence. Applied on dmrs grid, centers delay to zero.

global SimCtrl;
s = fp16nv(real(s), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(s), SimCtrl.fp16AlgoSel);
s_grid = fp16nv(real(s_grid), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(s_grid), SimCtrl.fp16AlgoSel);

% simulation paramaters:

[~, ~, L_BS] = size(Y);  % total number of bs antennas
L_UE = nl;       % max number of spatially multiplexed layers

W_lower = table.W_lower; %  derive_legacy_lower(10^(-3),df);
W_middle = table.W_middle;
W_upper = table.W_upper;

% inititialize channel estimate:
H_est = zeros(Nf,L_UE,L_BS);

ChEst_par.W = W_lower;
ChEst_par.nPrb_dmrs = 8;
ChEst_par.startPrb_dmrs = startPrb;
ChEst_par.nPrb_est = 4;
ChEst_par.startPrb_est = startPrb;

% launch a "thread block" for each BS antenna:
for bs = 1 : L_BS    
    H_est(:,:,bs) = legacy_ChEst_kernel(H_est(:,:,bs),Y(:,:,bs), table, r_dmrs, ChEst_par, ...
        nl, portIdx, n_scid, symIdx_dmrs, Nt_dmrs, nPrb, startPrb, s, s_grid);
end

% estimate middle prbs:
num_middle_cluster = floor( (nPrb - 8) / 4);

for i = 1 : num_middle_cluster
    
    % configure thread block:
    ChEst_par = [];
    ChEst_par.W = W_middle;
    ChEst_par.nPrb_dmrs = 8;
    ChEst_par.startPrb_dmrs = startPrb + i*4 - 2;
    ChEst_par.nPrb_est = 4;
    ChEst_par.startPrb_est = startPrb + i*4;
    
    % launch a "thread block" for each BS antenna:
    for bs = 1 : L_BS
        H_est(:,:,bs) = legacy_ChEst_kernel(H_est(:,:,bs),Y(:,:,bs), table, r_dmrs, ChEst_par, ...
            nl, portIdx, n_scid, symIdx_dmrs, Nt_dmrs, nPrb, startPrb, s, s_grid);
    end
    
end

% configure upper ChEst "thread block":
ChEst_par = [];
ChEst_par.W = W_upper;
ChEst_par.nPrb_dmrs = 8;
ChEst_par.startPrb_dmrs = startPrb + nPrb - 8;
ChEst_par.nPrb_est = 4;
ChEst_par.startPrb_est = startPrb + nPrb - 4;

% launch a "thread block" for each BS antenna:
for bs = 1 : L_BS
    H_est(:,:,bs) = legacy_ChEst_kernel(H_est(:,:,bs),Y(:,:,bs), table, r_dmrs, ChEst_par, ...
        nl, portIdx, n_scid, symIdx_dmrs, Nt_dmrs, nPrb, startPrb, s, s_grid);
end

% permute:
H_est = permute(H_est,[3 2 1]);

if numDmrsCdmGrpsNoData == 1
    H_est = H_est * sqrt(2);
end

ChEst_par.filterArray = 0; 
ChEst_par.W_upper = W_upper;
ChEst_par.W_lower = W_lower;
ChEst_par.W_middle = W_middle;
ChEst_par.s = s;
ChEst_par.sd = sd;
ChEst_par.s_grid = s_grid;

return
