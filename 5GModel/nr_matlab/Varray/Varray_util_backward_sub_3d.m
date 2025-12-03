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

% given UX=B, where U is a upper triangular matrix, use backward substitution to get X
function X = Varray_util_backward_sub_3d(U, B, fp_flag)
    if nargin == 2
        fp_flag = 0;        % 0: double format
    end
    [numRows,numEqus,M] = size(B.value); % M: num of matrices
    X = Varray(zeros(numRows,numEqus,M), fp_flag);
    for idx_row = numRows:-1:1
        tmp_sum = pagemtimes(U(idx_row,idx_row+1:numRows,:), X(idx_row+1:numRows,:,:));
        X(idx_row,:,:) = pagemtimes((B(idx_row,:,:) - tmp_sum), (Varray(1.0, fp_flag)./U(idx_row,idx_row,:)));
    end 
end

% nested for-loop version. Slow
% function X = Varray_util_backward_sub(U, B)
%     [numRows,numEqus] = size(B.value);
%     X = Varray(zeros(numRows,numEqus));
%     for idx_equ = 1:numEqus
%         for idx_row = numRows:-1:1
%             tmp_sum = Varray(0);
%             for k = idx_row+1:numRows
%                 tmp_sum = tmp_sum + U(idx_row,k)*X(k,idx_equ);
%             end
%             X(idx_row,idx_equ) = (B(idx_row,idx_equ) - tmp_sum) * (Varray(1)/U(idx_row,idx_row));
%         end
%     end 
% end