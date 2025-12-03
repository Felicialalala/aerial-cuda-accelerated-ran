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

function [U,D,Linv,A0,A1] = lu_solve2(G,H)

%%

% Joint method:
% Append [G | I | H] and LU factorize [G | I | H] to yield [U | Linv | M],
% where M = Linv*H
% Compute W = (Linv*H)' * Linv = M' * Linv
I = eye(size(G));
A0 = [G I H];
[rowsA, colsA] = size(A0);

A1 = A0;
% # of threads required in the thread block for LU: (rowsA-1)*(colsA-1)
% # of threads in thread block = floor(((rowsA-1)*(colsA-1) + WARP_SIZE - 1)/WARP_SIZE)*WARP_SIZE
% e.g. # of threads in thread block = floor(((4-1)*(4+14+4-1) + 32 - 1)/32)*32

% LU factorization (without pivoting) using the outer product method (or rank-1 update)
for k = 1:rowsA-1
   % A(k+1:rowsA,k) = -A(k+1:rowsA,k)/A(k,k);
   for i = k+1:rowsA
       % Compute the multipliers which are the non-zero elements of the Gauss vector
       % The Gauss vectors are the columns of L 
       % For storage compactness the multipliers (non-zero elements of Gauss vector/column of L) 
       % are stored in the annihilated zero locations in the columns of U
       % This way the Gauss vector is formed inplace in A
       A1(i,k) = -A1(i,k)/A1(k,k);
       
       % Now apply Gaussian elimination to the ith row of A. This is accomplished in the following 
       % form of a rank-1 update to A (note: A(:,k)*A(k,:) is the outerproduct)
       for j = k+1:colsA
           A1(i,j) = A1(i,j) + A1(i,k)*A1(k,j);
       end
   end
end

[nLayers, nBSAnts] = size(H);
U = triu(A1(:,1:nLayers));
D = diag(diag(U));
Linv = A1(:,nLayers+1:2*nLayers);
M = A1(:,2*nLayers+1:end); % Linv*H


end
