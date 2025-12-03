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

rng(111);
% debug Varray LDL
fp_flag = 3;
Nant = 4;
N = 100;
a = randn(Nant,Nant,N)+1i*randn(Nant,Nant,N);

g = pagemtimes(a,'ctranspose',a,'none');
% change condition number
C = 1e6; % desired condition number
for ii = 1:N
    [u,s,v]= svd(g(:,:,ii));
    
    s = diag(s);           % s is vector
    % ===== linear stretch of existing s
    s = s(1)*( 1-((C-1)/C)*(s(1)-s)/(s(1)-s(end))) ;
    % =====
    s = diag(s);           % back to matrix
    g(:,:,ii) = u*s*v';
end
G = Varray(g, fp_flag);

% 
[L1,D1,~] = ldl(G, fp_flag);
D1_min_vec = zeros(N,1);
for ii = 1:N
    D1_min_vec(ii) = min(diag(getValue(D1(:,:,ii))));
end
figure;
plot(D1_min_vec)
%
L2 = Varray(zeros(Nant,Nant,N), fp_flag);
D2 = Varray(zeros(Nant,Nant,N), fp_flag);
for ii = 1:N
    [L2(:,:,ii), D2(:,:,ii), ~] = ldl(G(:,:,ii),fp_flag);
    obtained = getValue(L2(:,:,ii))*getValue(D2(:,:,ii))*getValue(L2(:,:,ii))';
%     assert(all(all(abs(getValue(G(:,:,ii)) - obtained) < eps(200))))
end
%
L3 = zeros(Nant,Nant,N);%Varray(zeros(Nant,Nant,N), fp_flag);
D3 = zeros(Nant,Nant,N);%Varray(zeros(Nant,Nant,N), fp_flag);
P3 = zeros(Nant,Nant,N);
for ii = 1:N
    [L3(:,:,ii), D3(:,:,ii)] = ldl_golub(getValue(G(:,:,ii)));
%     L3(:,:,ii) = P3(:,:,ii)*L3(:,:,ii);
    obtained = L3(:,:,ii)*D3(:,:,ii)*L3(:,:,ii)';
    assert(all(all(abs(getValue(G(:,:,ii)) - obtained) < eps(200))))
end

figure;
diff_L12 = getValue(L1(:)-L2(:));
diff_D12 = getValue(D1(:)-D2(:));
plot(abs(diff_D12));
title('Diff 1 and 2')
figure
diff_L23 = getValue(L2(:))-L3(:);
diff_D23 = getValue(D2(:))-D3(:);
plot(abs(diff_D23));
title('Diff 2 and 3')
figure
diff_L13 = getValue(L1(:))-L3(:);
diff_D13 = getValue(D1(:))-D3(:);
plot(abs(diff_L13));
title('Diff 1 and 3')



