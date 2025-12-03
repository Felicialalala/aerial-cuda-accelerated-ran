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

"""Aerial Build Development image hpccm recipe using Ubuntu base OS
Usage:
$ hpccm --recipe aerial_build_devel_recipe.py --format docker
"""

# Check if AERIAL_REPO user argument exists
AERIAL_REPO = USERARG.get('AERIAL_REPO')
if AERIAL_REPO is None:
    raise RuntimeError("User argument AERIAL_REPO must be set")

AERIAL_VERSION_TAG = USERARG.get('AERIAL_VERSION_TAG')
if AERIAL_VERSION_TAG is None:
    raise RuntimeError("User argument AERIAL_VERSION_TAG must be set")


if cpu_target == 'x86_64':
    TARGETARCH='amd64'
elif cpu_target == 'aarch64':
    TARGETARCH='arm64'
else:
    raise RuntimeError("Unsupported platform")

# Use Aerial base image
Stage0 += baseimage(image=f'{AERIAL_REPO}aerial_base:{AERIAL_VERSION_TAG}', _arch=cpu_target, _distro='ubuntu22')

ospackages=[
        'autoconf',
        'automake',
        'autotools-dev',
        'bc',
        'bison',
        'debhelper',
        'check',
        'chrpath',
        'dpatch',
        'ethtool',
        'flex',
        'gdb',
        'git-lfs',
        'help2man',
        'htop',
        'iproute2',
        'jq',
        'libbsd-dev',
        'libcairo2',
        'libcurl4-openssl-dev',
        'libglib2.0-dev',
        'libjson-c-dev',
        'libltdl-dev',
        'libmnl-dev',
        'libnghttp2-dev',
        'libnl-route-3-dev',
        'libnl-3-dev',
        'libnuma-dev',
        'libpcap-dev',
        'libsubunit0',
        'libsubunit-dev',
        'liburiparser-dev',
        'lsof',
        'libssl-dev',
        'm4',
        'mlocate',
        'net-tools',
        'ninja-build',
        'pciutils',
        'pkg-config',
        'pybind11-dev',
        'python3-cairo',
        'python3-pyelftools',
        'python3-testresources',
        'python3.10-venv',
        'psmisc',
        'quilt',
        'rt-tests',
        'screen',
        'software-properties-common',
        'swig',
        'tcpdump',
        'tmux',
        'numactl',
        'zip',
        'binutils-dev',     # Needed for backward-cpp to pretty-print and elaborated stacktrace
        'libdwarf-dev',     # Needed for backward-cpp to pretty-print and elaborated stacktrace
        ]

Stage0 += user(user='root')
Stage0 += packages(ospackages=ospackages)

if cpu_target == 'x86_64':
    tensorrt=[
        "libnvinfer10",
        "libnvinfer-vc-plugin10",
        "libnvinfer-lean10",
        "libnvinfer-dispatch10",
        "libnvonnxparsers10",
        "libnvinfer-bin",
        "libnvinfer-samples",
        "python3-libnvinfer",
        "python3-libnvinfer-lean",
        "python3-libnvinfer-dispatch",
        "libnvinfer-win-builder-resource10",
        "libnvinfer-win-builder-resource10",
        "tensorrt",
        "libnvinfer-headers-python-plugin-dev",
        "libnvinfer-plugin10",
        "libnvinfer-headers-dev",
        "libnvinfer-headers-plugin-dev",
        "libnvinfer-dev",
        "libnvinfer-lean-dev",
        "libnvinfer-plugin-dev",
        "libnvinfer-vc-plugin-dev",
        "libnvinfer-dispatch-dev",
        "libnvonnxparsers-dev",
        "libnvinfer-headers-python-plugin-dev",
        "libnvinfer-dev",
        "libnvinfer-lean-dev",
        "libnvinfer-dispatch-dev",
        "libnvinfer-plugin-dev",
        "libnvinfer-vc-plugin-dev",
        "libnvonnxparsers-dev",
        "python3-libnvinfer-dev",
        "tensorrt-dev",
        ]

    Stage0 += packages(ospackages=[f"{pack}=10.12.0.36-1+cuda12.9" for pack in tensorrt])

if cpu_target == 'aarch64':
    Stage0 += packages(ospackages=[
        "tensorrt",
        "tensorrt-dev",
        ])

Stage0 += environment(variables={"PATH":"$PATH:/usr/src/tensorrt/bin"})

# Screen setup
Stage0 += shell(commands=[
    'echo "logfile screenlog_%t.log" >> /etc/screenrc',
    'echo "logfile flush 1" >> /etc/screenrc',
    'echo "defshell -bash" >> /etc/screenrc',
    ])

Stage0 += pip(pip="pip3", requirements=f'requirements.txt')

# Handle platform specific version differences
# Explicitly installing these is not necessary and causes some confusion for now. Ignore - they come from apt installs.
#if cpu_target == 'x86_64':
#    Stage0 += pip(pip="pip3", requirements=f'requirements_x86_64.txt')
#
#if cpu_target == 'aarch64':
#    Stage0 += pip(pip="pip3", requirements=f'requirements_aarch64.txt')

# Install nsight-systems
if cpu_target == 'x86_64':
    cli_package_url = 'https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb'

if cpu_target == 'aarch64':
    cli_package_url = 'https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/nsight-systems-cli-2025.3.1_2025.3.1.90-1_arm64.deb'

Stage0 += shell(commands=[
    f'wget {cli_package_url}',
    f'dpkg -i {os.path.basename(cli_package_url)}',
    f'rm {os.path.basename(cli_package_url)}',
])

# Workaround - Needed so host-launched graphs appear under the right green context
if cpu_target == 'x86_64':
    Stage0 += shell(commands=["cp /usr/local/cuda/lib64/libcupti.so /opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/libcupti.so.12.9"])

if cpu_target == 'aarch64':
    Stage0 += shell(commands=["cp /usr/local/cuda/lib64/libcupti.so /opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-sbsa-armv8/libcupti-sbsa.so.12.9"])

if cpu_target == 'aarch64':
    yq_binary='wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_arm64 -O /usr/bin/yq'
else:
    yq_binary='wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq'
Stage0 += shell(commands=[
    yq_binary,
    'chmod +x /usr/bin/yq',
    ])

Stage0 += user(user='aerial')

Stage0 += workdir(directory='$cuBB_SDK')

