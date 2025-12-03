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

rng(0);

X_tf = sqrt(0.5) * (randn(273*12, 14, 4) + j*randn(273*12, 14, 4));

iqWidth = 16;
beta = 2^11;

X_tf_fx= oranFL2FX(X_tf, iqWidth, beta);

X_tf_size = size(X_tf);
X_tf_fl = oranFX2FL(X_tf_fx, iqWidth, beta, X_tf_size);

err = X_tf - X_tf_fl;

err_pw = 10*log10(mean(abs(err(:).^2)))