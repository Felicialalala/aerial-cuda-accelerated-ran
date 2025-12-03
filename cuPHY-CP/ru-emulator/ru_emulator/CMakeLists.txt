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

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/helper.cmake)

set(SOURCES main.cpp
            ${SOURCES}
            )

set(TARGET ru_emulator)
add_executable(${TARGET} ${SOURCES})
link_target(${TARGET})

# ----------------------------------------------------------------------
# Functional test
# ----------------------------------------------------------------------
set(STANDALONE ru_standalone)
add_executable(${STANDALONE} ${SOURCES})
link_target(${STANDALONE})
target_compile_definitions(${STANDALONE} PRIVATE STANDALONE=1)


set(DEBUG ru_debug)
add_executable(${DEBUG} ${SOURCES})
link_target(${DEBUG})
target_compile_definitions(${DEBUG} PRIVATE RU_EM_UL_TIMERS_ENABLE=1)
