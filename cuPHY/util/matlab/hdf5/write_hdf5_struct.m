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

h5File  = H5F.create('struct_example.h5', 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
A = struct('one_as_double',  1,         ...
           'two_as_single',  single(2), ...
           'three_as_uint8', uint8(3),  ...
           'four_as_uint16', uint16(4), ...
           'five_as_uint32', uint32(5), ...
           'six_as_int8',    int8(6),   ...
           'seven_as_int16', int16(7),  ...
           'eight_as_int32', int32(8),  ...
           'nine_as_int64',  int64(9));

hdf5_write_nv(h5File, 'A', A);

H5F.close(h5File);
