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

function differences = struct_compare(s1, s2)
    differences = do_struct_compare(0, s1, s2);
end

function differences = do_struct_compare(diff_previous, s1, s2)
    differences = diff_previous;
    fnames = unique([ fieldnames(s1); fieldnames(s2) ]);
    for idx = 1:length(fnames)
        fname = fnames{idx};
        %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Check that the field exists in structure 1
        if ~isfield(s1, fname)
            fprintf('Field %s is not present in structure 1\n', fname);
            differences = differences + 1;
            continue
        end
        %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Check that the field exists in structure 2            
        if ~isfield(s2, fname)
            fprintf('Field %s is not present in structure 2\n', fname);
            differences = differences + 1;
            continue
        end
        %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Compare field types
        field1 = getfield(s1, fname);
        field2 = getfield(s2, fname);
        if ~strcmp(class(field1), class(field2))
            fprintf('Field %s is of class %s in struct 1 and %s in struct 2', ...
                    fname,                                                    ...
                    class(field1),                                            ...
                    class(field2));
            differences = differences + 1;
            continue
        end
        %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Compare field dimensions
        if any(size(field1) ~= size(field2))
              fprintf('Size of field %s differs in struct 1 and struct 2: \n', fname);
              fprintf('\tstruct 1: ');
              fprintf('%d ', size(field1));
              fprintf('\n\tstruct 2: ');
              fprintf('%d ', size(field2));
              fprintf('\n');
              differences = differences + 1;
              continue
        end
        %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Compare individual array values
        if strcmp('struct', class(field1))
            for elem_idx = 1:length(field1)
                differences = do_struct_compare(differences, field1, field2);
            end
        else
            if any(field1 ~= field2, 'all')
                fprintf('Values in field %s differ\n', fname);
                differences = differences + 1;
            end
        end
    end % for idx = 1:length(fnames)
end
