/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_ENGINE_PARAMS_HPP
#define TRT_ENGINE_PARAMS_HPP

#include <string>

#include "cuphy.h"

namespace trt_engine {

/**
 * @brief Holds cuphyTrtTensorPrms_t
 * @details cache the string since it's lifetime
 *  will be shorter than this modules' trtEngine types.
 */
struct TrtParams final {
 TrtParams() = default;
 explicit TrtParams(const cuphyTrtTensorPrms_t& p) :
         params(p), name(p.name) {}
 cuphyTrtTensorPrms_t params{};
 std::string name;
};

} // namespace trt_engine

#endif //TRT_ENGINE_PARAMS_HPP
