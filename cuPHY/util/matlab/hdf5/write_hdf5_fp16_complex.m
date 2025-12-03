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

% ----------------------------------------------------------------------
h5File  = H5F.create('fp16_cmplx_example_single.h5', 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
A = single(rand(3) + (i*rand(3)));
hdf5_write_nv3(h5File, 'A', A, 'fp16');
H5F.close(h5File);
