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

#ifndef CUPHY_CH_EST_UTILS_HPP
#define CUPHY_CH_EST_UTILS_HPP

#include <cstdint>

// Currently serving channel estimate purpose.
// When this is extended to the rest of the kernels node manipulations, it can
// be moved to cuphy or any generic namespace.
namespace ch_est {
    /**
     * @struct ChestCudaUtils is a simple utility wrapper needed by CUDA Chest Graph Manager.
     * It takes bool and convert it to @enum DisableAllNodes
     */
    struct ChestCudaUtils final {
        enum class DisableAllNodes : std::uint8_t {
            TRUE,
            FALSE,
        };
        /**
         * Given a boolean, translate it to one of the above strong enums values
         * @param disableAllNodes boolean to translate
         * @return Enum type see @enum DisableAllNodes
         */
        [[nodiscard]] static DisableAllNodes toDisableAllNodes(const bool disableAllNodes){
            return disableAllNodes ? DisableAllNodes::TRUE : DisableAllNodes::FALSE;
        }
    };

} // namespace ch_est

#endif //CUPHY_CH_EST_UTILS_HPP
