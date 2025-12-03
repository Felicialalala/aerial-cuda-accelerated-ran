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

function L = Varray_util_chol_3d(A, tri_flag, fp_flag)
    if nargin==1
        tri_flag = 'lower'; % return lower triangular
        fp_flag = 0;        % 0: double format
    elseif nargin==2
        fp_flag = 0;        % double
    end
    [N, ~, M] = size(A.value);          % matrix dimension. M: num of square matrices
    L = Varray(zeros(N,N,M),fp_flag);   % initialize the output matrix
    for j = 1:N
        Lj = L(j,1:j-1,:); 
        tmp_sum1 = pagemtimes(Lj,'none',Lj,'ctranspose');
        Ljj = sqrt(A(j,j,:)-tmp_sum1);
        L(j,j,:) = Ljj;
        Dj_inv = Varray(1.0,fp_flag)./Ljj;
    
        tmp_sum2 = pagemtimes(L(j+1:N,1:j-1,:),'none',L(j,1:j-1,:),'ctranspose');
        L(j+1:N,j,:) = Dj_inv.*(A(j+1:N,j,:)-tmp_sum2);
    end
    if strcmp(tri_flag,'lower')
        L = L;
    elseif strcmp(tri_flag,'upper')
        L = pagectranspose(L);
    else
        error('[Varray_util_chol_3d.m] Undefined flag!')
    end
end

% nested for-loop version. Slow
% function L = Varray_util_chol(A, flag)
%     N = size(A.value,1); % matrix dimension
%     L = Varray(zeros(N,N)); % initialize the output matrix
%     for j = 1:N
%         tmp_sum = Varray(0);
%         for k = 1:j-1
%             tmp_sum = tmp_sum + conj(L(j,k))*L(j,k);
%         end
%         L(j,j) = sqrt(A(j,j)-tmp_sum);
% 
%         for i = j+1:N
%             tmp_sum = Varray(0);
%             for k = 1:j-1
%                 tmp_sum = tmp_sum + conj(L(j,k))*L(i,k);
%             end
%             L(i,j) = (Varray(1.0)/L(j,j))*(A(i,j)-tmp_sum);
%         end
%     end
%     if strcmp(flag, 'lower')
%         L = L;
%     elseif strcmp(flag, 'upper')
%         L = L';
%     else
%         error('[Varray_util_chol.m] Undefined flag!')
%     end
% end