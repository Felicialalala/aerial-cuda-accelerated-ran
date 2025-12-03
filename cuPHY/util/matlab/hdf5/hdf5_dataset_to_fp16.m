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

function hdf5_dataset_to_fp16(infile, outfile, varargin)
  % HDF5_DATSET_TO_FP16 Convert one or more datasets to fp16
  h5File = H5F.create(outfile, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
  A = hdf5_load_nv(infile);
  names = fieldnames(A);
  for iName = 1:length(names)
      name = names{iName};
      for iConvertName = 1:numel(varargin)
          if strcmp(varargin{iConvertName}, name)
              fprintf('Converting %s to fp16\n', name);
              hdf5_write_nv(h5File, name, A.(name), 'fp16');
          else
              fprintf('Writing %s (unmodified)\n', name);
              hdf5_write_nv(h5File, name, A.(name));
          end   
      end
  end
  H5F.close(h5File);
