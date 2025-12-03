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

#Start Cell 0
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_ctrl_cmd.py --server_ip localhost --cell_id 0 --cmd 2 --target_cell_id 3
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_ctrl_cmd.py --server_ip localhost --cell_id 0 --cmd 2 --target_cell_id 1
sleep 20
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_ctrl_cmd.py --server_ip localhost --cell_id 0 --cmd 0

#Switch to cell 1
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_ctrl_cmd.py --server_ip localhost --cell_id 0 --cmd 2 --target_cell_id 1
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_multi_attrs_update.py --cell_id 1 --dst_mac_addr 26:04:9D:9E:29:B3 --vlan_id 2 --pcp 7
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_ctrl_cmd.py --server_ip localhost --cell_id 0 --cmd 1
sleep 20
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_ctrl_cmd.py --server_ip localhost --cell_id 0 --cmd 0

#Switch to cell 2
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_ctrl_cmd.py --server_ip localhost --cell_id 0 --cmd 2 --target_cell_id 2
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_multi_attrs_update.py --cell_id 1 --dst_mac_addr 20:34:9A:9E:29:B3 --vlan_id 2 --pcp 7
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_ctrl_cmd.py --server_ip localhost --cell_id 0 --cmd 1
sleep 20
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_ctrl_cmd.py --server_ip localhost --cell_id 0 --cmd 0

#Switch back to cell 0
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_ctrl_cmd.py --server_ip localhost --cell_id 0 --cmd 2 --target_cell_id 0
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_multi_attrs_update.py --cell_id 1 --dst_mac_addr 20:04:9B:9E:27:A3 --vlan_id 2 --pcp 7
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_ctrl_cmd.py --server_ip localhost --cell_id 0 --cmd 1
sleep 20
python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_cell_ctrl_cmd.py --server_ip localhost --cell_id 0 --cmd 0
