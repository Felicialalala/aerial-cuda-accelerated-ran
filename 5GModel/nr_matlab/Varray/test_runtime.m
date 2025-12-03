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

% test runtime
clear all
Nitr = 273*12;
A = randn(4,4,Nitr)+1i*randn(4,4,Nitr);
C = pagemtimes(A,'ctranspose',A,'none');

% % test chol
% tic;
% for i = 1:Nitr
% %     tmp = Varray_util_chol(Varray(C(:,:,i)),'lower');
% %     tmp = tmp_util_chol(C(:,:,i),'lower');
% %     tmp = util_quant_chol(C(:,:,i),'lower');  
% end
% tmp = Varray_util_chol_3d(Varray(C),'lower',0);
% toc;

% % test LDL
% tic;
% [L,D,U] = Varray_util_LDL_3d(Varray(C),0);
% toc;

% % test forward sub
% tic;
% [Linv] = Varray_util_forward_sub_3d(Varray(C),Varray(C),0);
% toc;

% % test backward sub
% tic;
% [Linv] = Varray_util_backward_sub_3d(Varray(C),Varray(C),0);
% toc;