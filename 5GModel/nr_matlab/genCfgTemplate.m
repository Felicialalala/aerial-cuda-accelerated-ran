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

function genCfgTemplate(fileName, dlul)

rng(0);

SysPar = initSysPar;

if nargin == 0
    fileName = 'cfg_template.yaml';
    dlul = 'DL';
elseif nargin == 1
    dlul = 'DL';
end

if strcmp(dlul, 'DL')
    SysPar.testAlloc.dl = 1;
    SysPar.testAlloc.ul = 0;
    SysPar.testAlloc.ssb = 1;
    SysPar.testAlloc.pdcch = 1;
    SysPar.testAlloc.pdsch = 1;
    SysPar.testAlloc.csirs = 1;
    SysPar.testAlloc.prach = 0;
    SysPar.testAlloc.pucch = 0;
    SysPar.testAlloc.pusch = 0;    
    SysPar.testAlloc.srs = 0;   
else
    SysPar.testAlloc.dl = 0;
    SysPar.testAlloc.ul = 1;
    SysPar.testAlloc.ssb = 0;
    SysPar.testAlloc.pdcch = 0;
    SysPar.testAlloc.pdsch = 0;
    SysPar.testAlloc.csirs = 0;
    SysPar.testAlloc.prach = 1;
    SysPar.testAlloc.pucch = 1;
    SysPar.testAlloc.pusch = 1;    
    SysPar.testAlloc.srs = 1;   
end

tempFileName = 'temp_cfg_template.yaml';
WriteYaml(tempFileName, SysPar);

st1 = fileread('header4yaml.txt');
st2 = fileread(tempFileName);
[fid,msg] = fopen(fileName,'wt');
assert(fid>=3,msg)
fprintf(fid,'%s\n\n%s',st1,st2);
fclose(fid);

delete temp_cfg_template.yaml;

return
