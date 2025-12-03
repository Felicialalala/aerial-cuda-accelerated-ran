# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import aerial_mcore as NRSimulator
import matlab
import time
import sys

ARGC = len(sys.argv)
if ARGC < 3:
   print(f"Usage: {sys.argv[0]} cfgFileName tvFileName rxFileName(optional)")
   sys.exit(1)

cfgFileName = sys.argv[1]
tvFileName = sys.argv[2]
rxFileName = None
if ARGC > 3:
    rxFileName = sys.argv[3]

eng = NRSimulator.initialize()
#eng.cfg_parfor(0,nargout=0)

tic = time.time()
if rxFileName is None:
    errFlag, SysPar, UE, gNB = eng.runSim(cfgFileName, tvFileName, nargout=4)
else:
    errFlag, SysPar, UE, gNB = eng.runSim(cfgFileName, tvFileName, rxFileName, nargout=4)
toc = time.time()

print(f"errFlag: {errFlag}")
print(f"Elapsed time: {toc-tic} seconds")
if errFlag != 0:
    sys.exit(1)

