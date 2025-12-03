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

h5File  = H5F.create('vlen_example.h5', 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

%
% Initialize variable-length data.  wdata{1} is a countdown of
% length LEN0, wdata{2} is a Fibonacci sequence of length LEN1.
%
wdata{1} = int32(10:-1:1);
wdata{2} = int32([ 1 4 9 16 25 36 49 64 81 100 121]);

hdf5_write_nv_vlen(h5File, 'wdata', wdata);

H5F.close(h5File);
