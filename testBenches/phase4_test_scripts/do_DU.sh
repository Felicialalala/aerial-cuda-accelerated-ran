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

# Apply CONFIG_DIR logic after all parsing is complete
CONFIG_DIR=$cuBB_SDK
echo "CONFIG_DIR: $CONFIG_DIR"
#!/bin/bash
TEST_CONFIG_FILE=$CONFIG_DIR/testBenches/phase4_test_scripts/test_config_summary.sh
rm -f $TEST_CONFIG_FILE
testCase=$1
./testBenches/phase4_test_scripts/setup1_DU.sh -y nrSim_SCF_CG1_$testCase

source $TEST_CONFIG_FILE
while [[ ! -v RU_SETUP_COMPLETE ]]; do
	echo "RU Setup not complete. Waiting"
	sleep 5
	source $TEST_CONFIG_FILE
done
./testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh --memtrace;

