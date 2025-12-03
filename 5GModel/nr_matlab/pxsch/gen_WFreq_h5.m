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

function [WFreq, W4Freq, WSmallFreq, shiftSeq, shiftSeq4, unShiftSeq, unShiftSeq4] = gen_WFreq_h5(h5filename, mu, N0, delaySpread, fp16AlgoSel)

if nargin == 0
    h5filename = 'WFreq.h5';
    mu = 1;
    N0 = 1e-3;
    delaySpread = 2e-6;
    fp16AlgoSel = 2;
    SysPar = initSysPar();
    global SimCtrl;
    SimCtrl = SysPar.SimCtrl;
    SimCtrl.delaySpread = delaySpread;
end

% Generate WFreq

delta_f = 15e3 * 2^mu;
h5File  = H5F.create(h5filename, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

W_lower  = derive_legacy_lower(N0, delta_f);
W_middle = derive_legacy_middle(N0, delta_f);
W_upper  = derive_legacy_upper(N0, delta_f);

W4_upper  = derive_upper_filter(N0, delta_f);
W4_middle = derive_middle_filter(N0, delta_f);
W4_lower  = derive_lower_filter(N0, delta_f);

W3 = derive_small_filter(3,N0, delta_f);
W2 = derive_small_filter(2,N0, delta_f);
W1 = derive_small_filter(1,N0, delta_f);


WFreq      = single(reshape([W_middle W_lower W_upper], [size(W_middle,1) size(W_middle, 2) 3]));
hdf5_write_nv_exp(h5File, 'WFreq', WFreq);

W4Freq      = single(reshape([W4_middle W4_lower W4_upper], [size(W4_middle,1) size(W4_middle, 2) 3]));
hdf5_write_nv_exp(h5File, 'WFreq4', W4Freq);

W3_padded = W3;

W2_padded = zeros(37,18);
W2_padded(1 : 25, 1:12) = W2;

W1_padded = zeros(37,18);
W1_padded(1 : 13, 1 : 6) = W1;

WSmallFreq = single(reshape([W1_padded W2_padded W3_padded], [size(W1_padded,1) size(W1_padded, 2) 3]));
hdf5_write_nv_exp(h5File, 'WFreqSmall', WSmallFreq);


% Geneerate ShiftSeq
tau = (delaySpread-delaySpread/8)/2;

f_dmrs = 0 : 2 : (8*12 - 1);
f_dmrs = delta_f * f_dmrs';

f_data = -1 : (8*12 - 1);
f_data = delta_f * f_data';

shiftSeq  = exp(2*pi*1i*tau*f_dmrs);
shiftSeq  = fp16nv(real(shiftSeq), fp16AlgoSel) + 1i*fp16nv(imag(shiftSeq), fp16AlgoSel);
shiftSeq4 = shiftSeq(1 : 4*6);

unShiftSeq  = exp(-2*pi*1i*tau*f_data);
unShiftSeq  = fp16nv(real(unShiftSeq), fp16AlgoSel) + 1i*fp16nv(imag(unShiftSeq), fp16AlgoSel);
unShiftSeq4 = unShiftSeq(1 : (12*4 + 1));

hdf5_write_nv(h5File, 'ShiftSeq', single(shiftSeq(1:8*6)),'fp16');
hdf5_write_nv(h5File, 'UnShiftSeq', single(unShiftSeq(1:97)),'fp16');

hdf5_write_nv(h5File, 'ShiftSeq4', single(shiftSeq4),'fp16');
hdf5_write_nv(h5File, 'UnShiftSeq4', single(unShiftSeq4),'fp16');


H5F.close(h5File);

return
