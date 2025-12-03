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

set(SOURCES cuphy_ex_polar_encoder.cpp dummy.cu)
add_executable(cuphy_ex_polar_encoder ${SOURCES})
target_link_libraries(cuphy_ex_polar_encoder cuphy_hdf5 cuphy CLI11::CLI11)

install(FILES CMakeLists.release.txt DESTINATION examples/polar_encoder  RENAME CMakeLists.txt)
