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

% function to get inverse of a triangular matrix
function Ainv = Varray_util_inv_tri(A,tri_flag,fp_flag)
    [N,~,M] = size(A.value); % matrix dimension. M: num of 2d matrices
    Imat = Varray(repmat(eye(N),[1,1,M]), fp_flag);
    if (nargin == 1) || strcmp(tri_flag, 'lower')
        Ainv = Varray_util_forward_sub(A, Imat, fp_flag);
    elseif strcmp(tri_flag, 'upper')
        Ainv = Varray_util_backward_sub(A, Imat, fp_flag);
    else
        error('Undefined tri_flag!')
    end
end

% refer to G. W. Stewart: Matrix Algorithms Volume I:Basic Decompositions,
% page 94
% function Ainv = Varray_util_inv_tri(A,tri_flag)
%     Linv = Varray(zeros(N,N)); % initialize the output matrix
%     for k = 1:N
%         Linv(k,k) = Varray(1)/L(k,k);
%         for i = k+1:N
%             tmp_sum = Varray(0);
%             for m = k:i-1
%                 tmp_sum = tmp_sum + L(i,m)*Linv(m,k);
%             end
%             Linv(i,k) = tmp_sum*(Varray(-1)/L(i,i));
%         end        
%     end
% end