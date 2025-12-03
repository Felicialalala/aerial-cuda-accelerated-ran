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

# rnti_mode: 0 - default, normal mode; 1 - negative test

python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_config_mac_rnti.py --rnti_mode 1

# sleep long enough to make sure HARQ buffer exhaustion occurs
sleep 60

python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_config_mac_rnti.py --rnti_mode 0
