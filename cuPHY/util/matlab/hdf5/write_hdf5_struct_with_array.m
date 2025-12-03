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

h5File  = H5F.create('struct_with_array_example.h5', 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

% Create an array of structures, using cell arrays to provide a value
% for each element of the array.
A = struct('serial_no', {1153; 1184; 1027; 1313}, ...
           'temperature', {53.23; 55.12; 130.55; 1252.89}, ...
           'pressure', {24.57; 22.95; 31.23; 84.11}, ...
           'array', {[int32(0), int32(0), int32(1)]; [int32(0), int32(1), int32(0)]; [int32(0), int32(1), int32(1)]; [int32(1), int32(2), int32(3)]});

hdf5_write_nv(h5File, 'A', A);

H5F.close(h5File);
