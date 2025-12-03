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

%test_types = {'double', 'single', 'uint32', 'int32', 'uint16', 'int16', 'uint8', 'int8'};
test_types = {'double', 'single'};

for idx = 1:numel(test_types)
    t = test_types{idx};
    fname = strcat('magic_', t, '_complex', '.h5');
    A = cast(magic(16) - i*magic(16), t);
    h5File  = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
    hdf5_write_nv(h5File, 'A', A);
    H5F.close(h5File);

    check = hdf5_load_nv(fname);
    success_count = 0;
    if size(A) ~= size(check.A)
        fprintf('Size error for %s\n', fname);
        size(A)
        size(check.A)
    else
        success_count = success_count + 1;
    end
    if any(A(:) ~= check.A(:))
        fprintf('Data mismatch for %s\n', fname);
        A
        check.A
    else
        success_count = success_count + 1;
    end
    if 2 == success_count
        fprintf('%s: SUCCESS\n', fname);
        delete(fname);
    end
end
