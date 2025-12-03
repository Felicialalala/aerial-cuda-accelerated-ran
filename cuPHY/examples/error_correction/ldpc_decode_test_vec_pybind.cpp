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

#include <iostream>
#include <stdio.h>
#include "ldpc_decode_test_vec_pybind.hpp"

///////////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_pybind::ldpc_decode_test_vec_pybind()
ldpc_decode_test_vec_pybind::ldpc_decode_test_vec_pybind(const test_vec_pybind_params& pyparams) :
    ldpc_decode_test_vec(pyparams.LLRtype),
    inputLLR_(pyparams.inputLLR),
    num_cw_limit_(pyparams.num_cw_limit)
{
    //----------------------------------------------------------------------------
    // Get the LLR Data
    tLLR_ = inputLLR_;
    //----------------------------------------------------------------------------
    // Populate LDPC "configuration" data
    populate_config(pyparams); 
    //----------------------------------------------------------------------------
    if(num_cw_limit_ > 0)
    {
        // Can be added as a feature to process limited number of codewords
        ;
    }
    else
    {
        set_LLR_desc(tLLR_.desc());
    }

}

///////////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_pybind::desc()
const char* ldpc_decode_test_vec_pybind::desc() const
{
    return "Pybind data";
}

///////////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_pybind::populate_config()
void ldpc_decode_test_vec_pybind::populate_config(const test_vec_pybind_params& pyparams)
{
    config_.BG          = pyparams.BG;
    config_.num_cw      = (pyparams.num_cw_limit > 0) ?
                            pyparams.num_cw_limit : tLLR_.dimensions()[1];

    config_.B           = pyparams.B;
    config_.Z           = find_lifting_size(config_.BG, config_.B);
    config_.Kb          = get_num_info_nodes(config_.BG, config_.B);
    config_.F           = (config_.Z * ((1 == config_.BG) ? 22 : 10)) - config_.B;
 
    config_.N           = tLLR_.dimensions()[0];
    config_.R           = static_cast<float>(config_.B) / static_cast<float>(config_.N);

    config_.P  = 0;

    config_.K           = config_.B + config_.F; 
    config_.mb = static_cast<int>(std::ceil((config_.N - config_.K) / static_cast<float>(config_.Z)));

    config_.QAM         = "Unknown";
    config_.punc        = LLR_PUNCTURE_STATUS_UNKNOWN;

}

///////////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_pybind::generate()
void ldpc_decode_test_vec_pybind::generate()
{
    // Nothing to do for pybind based test vectors
}
