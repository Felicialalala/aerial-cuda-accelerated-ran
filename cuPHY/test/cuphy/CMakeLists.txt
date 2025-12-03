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

########################################################################
# cuphy_hpp_tests
add_executable(cuphy_hpp_tests)
target_sources(cuphy_hpp_tests PRIVATE cuphy_hpp_tests.cpp)
target_link_libraries(cuphy_hpp_tests cuphy gtest gtest_main)
add_test(NAME cuphy_hpp_tests COMMAND cuphy_hpp_tests)
