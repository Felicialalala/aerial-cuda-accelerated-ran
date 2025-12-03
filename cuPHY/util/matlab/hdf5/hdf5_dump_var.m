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

function hdf5_dump_var(infile, varname, outfile)
  % HDF5_DUMP_VAR Write a single variable from an HDF5 file to
  % a binary file.
  %
  % Example usage:
  %   hdf5_dump_var('TV_cuphy_pusch-TC1_snrdb40.00_iter1_MIMO4x4_PRB272_DataSyms9_qam256.h5', 'DataRx', 'out.bin');

  %---------------------------------------------------------------------
  % Load the input HDF5 file
  A = hdf5_load_nv(infile);
  outarray = A.(varname);
  if nargin < 3
      outfile = get_filename(infile, varname, outarray);
  end
  %---------------------------------------------------------------------
  % Open the output file
  fprintf('Output file: %s\n', outfile);
  fileID = fopen(outfile, 'w');
  %---------------------------------------------------------------------
  if isstruct(outarray)
      error('struct dump not implemented');
  elseif isreal(outarray)
      write_real(fileID, outarray);
  else
      write_complex(fileID, outarray);
  end
  %---------------------------------------------------------------------
  fclose(fileID);
end

function write_real(fileID, a)
    fprintf('Writing %d values of type %s\n', numel(a), class(a));
    fwrite(fileID, a(:), class(a));
end

function write_complex(fileID, a)
    % Create a single array of interleaved values
    a_real = [real(a(:))         zeros(numel(a), 1)]';
    a_imag = [zeros(numel(a), 1) imag(a(:))        ]';
    a_interleaved = a_real(:) + a_imag(:);
    fprintf('Writing %d values of type %s (2 values per complex element)\n', numel(a_interleaved), class(a_interleaved));
    fwrite(fileID, a_interleaved, class(a));
end

function fname = get_filename(infile, varname, var)
    [filepath, name, ext] = fileparts(infile);
    nd = ndims(var);
    sz = size(var);
    %-------------------------------------------------------------------
    % Try to get rid of singleton dimensions
    if sz(1) == 1
        sz(1)=[];
    end
    if sz(end) == 1
        sz(end) = [];
    end
    %-------------------------------------------------------------------
    % Create a string with the dimensions
    szstr = sprintf('%d', sz(1));
    sz(1) = [];
    for s  = sz
        szstr = strcat(szstr, '_', sprintf('%d', s));
    end
    %-------------------------------------------------------------------
    % Create a string with the type
    typestr = class(var);
    if ~isreal(var)
        typestr = strcat(typestr, '_complex');
    end
    fname = strcat(filepath, name, '_', varname, '_', szstr, '_', typestr, '.bin');
end
