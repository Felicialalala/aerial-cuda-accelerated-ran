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

function cuphySRSpar = set_cuphy_srs_paramaters(gnb,srs,srsChEst)

% function sets cuphy srs paramaters. Several hardcoded for now.

%%
%ZC PARAMATERS

% number of prb fixed to 272
nPrb = 272;

% length of srs sequence:
n_srs = 3*nPrb;

% zc prime:
load('primes.mat');
idx = find(p < n_srs,1,'last');
N_zc = p(idx);

% zc seqeunce number (no group/sequence hopping):
u = mod(srs.pdu_cell{1}.sequenceId,30);
v = 0; 
q_bar = N_zc*(u+1)/31;
q = floor(q_bar+0.5) + v*(-1)^floor(2*q_bar);

%%
%WRAP

cuphySRSpar.Lgnb               = gnb.L;
cuphySRSpar.nUe                = 8;
cuphySRSpar.nPrb               = gnb.Nf / 12;
cuphySRSpar.scs                = gnb.df;
cuphySRSpar.nSym               = 2;
cuphySRSpar.symIdx             = srs.symIdx;
cuphySRSpar.N_zc               = N_zc;
cuphySRSpar.q                  = q;
cuphySRSpar.delaySpread        = srsChEst.delaySpread;  
cuphySRSpar.nPrbPerThreadBlock = srsChEst.nPrbPerThreadBlock;

end














