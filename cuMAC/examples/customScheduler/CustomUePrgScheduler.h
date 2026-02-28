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

#pragma once

#include <cstdint>
#include "api.h"

namespace cumac {

class CustomUePrgScheduler {
public:
    struct Config {
        float sinrWeight = 1.0f;
        float pfWeight = 0.25f;
        float bufferWeight = 0.05f;
    };

    CustomUePrgScheduler();
    explicit CustomUePrgScheduler(const Config& cfg);

    void run(cumacCellGrpUeStatus* cellGrpUeStatusCpu,
             cumacSchdSol* schdSolCpu,
             cumacCellGrpPrms* cellGrpPrmsCpu,
             cumacSchdSol* schdSolGpu,
             cudaStream_t stream) const;

private:
    struct SelectedUe {
        uint16_t schedSlot;
        uint16_t activeUeId;
        float score;
    };

    Config m_cfg;

    float ueScore(uint16_t activeUeId,
                  const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                  const cumacCellGrpPrms* cellGrpPrmsCpu) const;

    uint32_t activeTrafficBytes(uint16_t activeUeId,
                                const cumacCellGrpUeStatus* cellGrpUeStatusCpu) const;
};

} // namespace cumac
