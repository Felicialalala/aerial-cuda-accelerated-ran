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

function [errCnt, nTV] = genCfgTV_perf_ss_json(spreadsheet, json_file, is_fapi)
    buffer = fileread(json_file);
    data = jsondecode(buffer);
    key = replace(spreadsheet, '.xlsm','_xlsm');
    key = replace(key, '-','_');
    
    cases = string(data.(key));
    
    [errCnt, nTV] = genCfgTV_perf_ss(spreadsheet, cases, is_fapi);
end
