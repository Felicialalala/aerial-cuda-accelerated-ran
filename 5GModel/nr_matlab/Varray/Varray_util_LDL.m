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

function [L, D, U] = Varray_util_LDL(G, fp_flag)
    if nargin == 1
        fp_flag = 0;        % 0: double format
    end
    N = size(G.value,1);    % matrix dimension
    U = G;                  % initialize the output matrix with matrix G
    for i = 1:N
        Ui = U(1:i-1,i);
        sum1 = sum(real(diag(U(1:i-1,1:i-1))).*real(conj(Ui).*Ui));
        Uii = real(G(i,i)) - sum1;
        U(i,i) = Uii;
        Dinv_i = Varray(1.0, fp_flag)/Uii;

        sum2 = ( diag(real(diag(U(1:i-1,1:i-1))))*U(1:i-1,i))'*U(1:i-1,i+1:N);
        U(i,i+1:N) = (G(i,i+1:N)-sum2) * Dinv_i;        
    end
    D = diag(diag(U));
    L = tril(U',-1) + Varray(eye(N), fp_flag);
    U = L';    
end

% nested for-loop version. Slow
% function [L, D, U] = Varray_util_LDL(G)
%     N = size(G.value,1); % matrix dimension
%     U = G; % initialize the output matrix with matrix G
%     D = Varray(zeros(N,N));
%     Dinv = Varray(zeros(N,1));
%     for i = 1:N
%         sum1 = Varray(0);
%         for j = 1:i-1
%             sum1 = sum1 + real(U(j,j))*real(conj(U(j,i))*U(j,i));
%         end 
%         U(i,i) = real(G(i,i)) - sum1;
%         Dinv(i) = Varray(1)/U(i,i);
% 
%         for j = i+1:N
%             sum2 = Varray(0);
%             for k = 1:i-1
%                 sum2 = sum2 + (conj(U(k,i))*U(k,j)) * real(U(k,k));
%             end
%             U(i,j) = (G(i,j)-sum2) * Dinv(i);
%         end
%     end
%     D = diag(diag(U));
%     L = tril(U',-1) + Varray(eye(N));
%     U = L';    
% end