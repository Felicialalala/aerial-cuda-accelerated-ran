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

import os
import aerial_mcore as NRSimulator
import matlab
import sys


def run5GModelCicd(cicdMode: int, channel: str, relNum: int) -> int:
    '''
        Runs 5GModel regression, returns number of tests with errors

        cicdMode: 0 for MR mode, 1 for nightly regression mode, 2 for compact TV generation

        relNum: 10000 for latest, 2240 for Rel-22.4
    '''

    assert cicdMode>= 0
    assert cicdMode<3

    eng = NRSimulator.initialize()

    # Generate TVs based on cicdMode
    mrSubSetMod = matlab.double([0, 1])
    nightlySubSetMod = matlab.double([0, 1])

    if cicdMode == 0:
        # MR mode

        # generate subset of TVs supported by cuPHY
        nTC, errCnt = eng.runRegression(['allTests'], [channel], 'compact', mrSubSetMod, relNum, nargout=2)

        # The above regression test may generate TVs that are not supported by
        # cuPHY. So remove the disabled TVs.
        os.system('rm GPU_test_input/disabled*')

    elif cicdMode == 1:
        # nightly mode

        # Full regression test for 5GModel
        nTC,errCnt = eng.runRegression(['allTests'], [channel], 'full', nightlySubSetMod, relNum, nargout=2);

        # The above regression test may generate TVs that are not supported by
        # cuPHY. So remove the disabled TVs.
        os.system('rm GPU_test_input/disabled*')

    elif cicdMode == 2:
        nTC,errCnt = eng.runRegression(['TestVector'], [channel], 'compact', mrSubSetMod, relNum, nargout=2);

        # The above regression test may generate TVs that are not supported by
        # cuPHY. So remove the disabled TVs.
        os.system('rm GPU_test_input/disabled*')
    return errCnt

def printUsage():
    print("""
Usage:
    export REGRESSION_MODE=<value>
    export RELEASE_NUMBER=<value>
    python3 example_5GModel_regression.py <channel>

    where REGRESSION_MODE is
       - 0 for MR mode (subset of tests), or
       - 1 for nightly regression mode (all tests)
       - 2 for nightly regression mode (test vectors)

    where RELEASE_NUMBER is
       - 10000 for latest, or
       - 2240 for Rel-22-4
""")
    sys.exit(1)

if __name__ == "__main__":
    cicdMode = os.environ.get('REGRESSION_MODE')
    if cicdMode is None:
        printUsage()
    cicdMode = int(cicdMode)
    if (cicdMode < 0) or (cicdMode >= 3):
        printUsage()

    relNum = os.environ.get('RELEASE_NUMBER')
    if relNum is None:
        relNum = 10000
    relNum = int(relNum)

    channel = sys.argv[1]
    print("run5GModelCicd with relNum " + str(relNum));
    errCnt = run5GModelCicd(cicdMode,channel,relNum)
    if errCnt > 0:
        sys.exit(1)
