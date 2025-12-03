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

import yaml
import os
import socket
import uuid
import argparse
import socket
import glob
#Use last two digits of mac address as the variable.

def getmac(interface):

  try:
    mac = open('/sys/class/net/'+interface+'/address').readline()
  except:
    mac = "00:00:00:00:00:00"

  return mac[0:17]

base = argparse.ArgumentParser()
base.add_argument('--id', type=str, dest='id', help='Manual identifier', required=False)
args = base.parse_args()
interface_name = 'ens6f0'

print ("The MAC address in formatted way for inteface " + interface_name + " is : " + getmac(interface_name))

cuBB_SDK = os.getenv('cuBB_SDK')
if cuBB_SDK is None:
    print("Please set cuBB_SDK!")
    exit(1)

ru_config_file_name = cuBB_SDK + '/cuPHY-CP/ru-emulator/config/config.yaml'
mac_addresses = cuBB_SDK + '/cuPHY-CP/ru-emulator/scripts/mac_addresses.yaml'

configs = glob.glob(cuBB_SDK + '/cuPHY-CP/cuphycontroller/config/cuphycontroller_*.yaml')

with open(mac_addresses, 'r') as base:
    try:
        mac_addresses = yaml.safe_load(base)
        mac_addresses = mac_addresses['mac_addresses']
    except yaml.YAMLError as exc:
        print(exc)

# Static mac mapping follows the format of 20:20:20:DEVKIT:INTERFACE:CELL
if interface_name == 'ens6f0':
    interface_mac = '00'
elif interface_name == 'ens6f1':
    interface_mac = '01'
else:
    print(f'WARNING: Unknown interface_name: {interface_name}. Defaulting to interface_mac = 00')
    interface_mac = '00'

for i in range(len(mac_addresses)):
    mac_addresses[i] = mac_addresses[i].replace('22', interface_mac)
    if args.id is not None:
        mac_addresses[i] = mac_addresses[i].replace('11', str(hex(max(int(args.id[-2:], 16), int('ff', 16)))[2:]))
    else:
        mac_addresses[i] = mac_addresses[i].replace('11', socket.gethostname()[-2:])

print('Using below mac address for each cell')    
print(mac_addresses)

with open(ru_config_file_name, 'r') as base:
    try:
        ru_config = yaml.safe_load(base)
        for cell, mac in zip(ru_config['ru_emulator']['cell_configs'], mac_addresses):
            cell['eth'] = mac
        
        ru_config['ru_emulator']['peers'][0]['peerethaddr'] = getmac(interface_name)
        print('Set RU config peerethaddr to ' + getmac(interface_name))

    except yaml.YAMLError as exc:
        print(exc)

with open(ru_config_file_name, 'w') as out:
    print("Editting file " + ru_config_file_name)
    data = yaml.dump(ru_config, out, default_flow_style=False, sort_keys=False)

for c in configs:
    with open(c, 'r') as base:
        try:
            cuphycontroller_config = yaml.safe_load(base)
            for cell, mac in zip(cuphycontroller_config['cuphydriver_config']['cells'], mac_addresses):
                cell['dst_mac_addr'] = mac
        except yaml.YAMLError as exc:
            print(exc)

    with open(c, 'w') as out:
        print("Editting file " + c)
        data = yaml.dump(cuphycontroller_config, out, default_flow_style=False, sort_keys=False)
