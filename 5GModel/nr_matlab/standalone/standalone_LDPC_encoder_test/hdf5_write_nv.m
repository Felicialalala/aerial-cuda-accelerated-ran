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

function hdf5_write_nv(hdf5loc, name, A)
  % HDF5_WRITE_NV Write a variable to an HDF5 file
  % HDF5_WRITE_NV(loc, name, A) writes variable A to the location loc
  % in an HDF5 file.
  % Location can be an HDF5 file handle or a HDF5 group handle.
  %
  % Files written are standard HDF5 files, and contents can be viewed
  % with standard HDF5 utilities (e.g. h5dump). However, certain
  % conventions are used for compatibility with the NVIDIA cuPHY library:
  %
  % - Complex arrays are written as HDF5 arrays of a COMPOUND type, where
  %   the compound type has fields 're' and 'im'. (HDF5 does not have a
  %   native concept for complex values.)
  % - Array dimension ordering is reversed, as MATLAB uses column-major
  %   ordering, whereas HDF5 uses row-major
  %
  % Example usage:
  %   h5File  = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
  %   A = rand(12, 4);
  %   b = single(randn(3, 9));
  %   hdf5_write_nv(h5File, 'A', A);
  %   hdf5_write_nv(h5File, 'b', b);
  %   H5F.close(h5File);

  % ------------------------------------------------------------------
  % Determine the element type
  H5TypeString = get_hdf5_type(A);
  if isempty(H5TypeString)
      error('Unexpected element class (%s)', class(A));
  end
  % ------------------------------------------------------------------
  if strcmp('H5T_COMPOUND', H5TypeString)
      if length(A) > 1
          error('Struct exports are limited to length 1');
      end
      % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      % Determine the size and type of each MATLAB struct field
      compoundSizes = [];
      elemTypes = {};
      names = fieldnames(A);
      for iName = 1:length(names)
          elemTypes{end + 1} = get_hdf5_type(A.(names{iName}));
          if strcmp(elemTypes{end},'H5T_COMPOUND')
              error('Exported structures cannot have nested structure fields.');
          end
          if isempty(elemTypes{end})
              error('Exported structures must only contain supported primitive types.');
          end
          compoundSizes(end + 1) = H5T.get_size(elemTypes{end});
      end
      % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      % Create a new HDF5 type to represent the MATLAB struct
      compoundOffsets= [0 cumsum(compoundSizes(1:end-1))];
      compoundType = H5T.create('H5T_COMPOUND', sum(compoundSizes));
      for iName = 1:length(names)
          H5T.insert(compoundType, names{iName}, compoundOffsets(iName), elemTypes{iName});
      end

      % Create a dataspace. MATLAB is column major, whereas HDF5 is
      % row major, so we do a fliplr on the dimensions
      % Setting maximum size to [] indicates that the maximum size is
      % current size.
      arrayDataspace = H5S.create_simple(1, [1], []);

      % Create a dataset and write it to the file
      arrayDataset = H5D.create(hdf5loc, name, compoundType, arrayDataspace, 'H5P_DEFAULT');
      H5D.write(arrayDataset, compoundType, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', A);

      % Cleanup
      H5D.close(arrayDataset);
      H5S.close(arrayDataspace);
      H5T.close(compoundType);
  else
      elemType       = H5T.copy(H5TypeString);
      if isreal(A)
          Awrite = A;
          fileType = elemType;
      else
          % Create an HDF5 complex data type
          compoundSizes   = [H5T.get_size(elemType) H5T.get_size(elemType)];
          compoundOffsets = [0 cumsum(compoundSizes(1:end-1))];
          complexType     = H5T.create('H5T_COMPOUND', sum(compoundSizes));
          H5T.insert(complexType, 're', compoundOffsets(1), elemType);
          H5T.insert(complexType, 'im', compoundOffsets(2), elemType);
          % Copy real and imaginary data to a local struct with two fields
          Awrite.re = real(A);
          Awrite.im = imag(A);
          fileType = complexType;
      end
  
      % Create a dataspace. MATLAB is column major, whereas HDF5 is
      % row major, so we do a fliplr on the dimensions
      % Setting maximum size to [] indicates that the maximum size is
      % current size.
      arrayDataspace = H5S.create_simple(ndims(A), fliplr(size(A)), []);

      % Create a dataset and write it to the file
      arrayDataset = H5D.create(hdf5loc, name, fileType, arrayDataspace, 'H5P_DEFAULT');
      H5D.write(arrayDataset, fileType, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', Awrite);

      % Cleanup
      H5D.close(arrayDataset);
      H5S.close(arrayDataspace);
      H5T.close(fileType);
  end
end

function H5TypeString = get_hdf5_type(varA)
  % Determine the element type
  switch(class(varA))
    case 'single'
      H5TypeString = 'H5T_NATIVE_FLOAT';
    case 'double'
      H5TypeString = 'H5T_NATIVE_DOUBLE';
    case 'uint8'
      H5TypeString = 'H5T_NATIVE_UINT8';
    case 'uint16'
      H5TypeString = 'H5T_NATIVE_UINT16';
    case 'uint32'
      H5TypeString = 'H5T_NATIVE_UINT32';
    case 'uint64'
      H5TypeString = 'H5T_NATIVE_UINT64';
    case 'int8'
      H5TypeString = 'H5T_NATIVE_INT8';
    case 'int16'
      H5TypeString = 'H5T_NATIVE_INT16';
    case 'int32'
      H5TypeString = 'H5T_NATIVE_INT32';
    case 'int64'
      H5TypeString = 'H5T_NATIVE_INT64';
    case 'struct'
      H5TypeString = 'H5T_COMPOUND';
    otherwise
      H5TypeString = '';
  end
end
