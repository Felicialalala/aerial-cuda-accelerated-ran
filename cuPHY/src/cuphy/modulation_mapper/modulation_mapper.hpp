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

#include "tensor_desc.hpp"

struct modulationDescr
{
    PdschDmrsParams* d_params;
    const uint32_t* modulation_input;
    const PdschPerTbParams* workspace;
    __half2* modulation_output;
    int max_bits_per_layer;
};
typedef struct modulationDescr modulationDescr_t;

namespace cuphy_i
{

////////////////////////////////////////////////////////////////////////
// symbol_modulate()
cuphyStatus_t symbol_modulate(const tensor_desc& tSym,
                              void*              pSym,
                              const tensor_desc& tBits,
                              const void*        pBits,
                              int                log2_QAM,
                              cudaStream_t       strm);
  
} // namespace cuphy_i
