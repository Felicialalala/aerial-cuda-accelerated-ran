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

#if !defined(LDPC_DECODE_TEST_VEC_PYBIND_HPP_INCLUDED_)
#define LDPC_DECODE_TEST_VEC_PYBIND_HPP_INCLUDED_

#include <string>
#include "ldpc_decode_test_vec.hpp"

struct test_vec_pybind_params
{
    test_vec_pybind_params(cuphy::tensor_device ip_llr,
                            cuphyDataType_t     llr_type,
                            uint16_t            b,
                            uint16_t            bg,
                            uint16_t            num_cw_lim) :
        inputLLR(ip_llr),
        LLRtype(llr_type),
        B(b),
        BG(bg),
        num_cw_limit(num_cw_lim)
        {

        }

        cuphy::tensor_device    inputLLR;
        cuphyDataType_t         LLRtype;
        uint16_t                B;
        uint16_t                BG;
        uint16_t                num_cw_limit;  
};

///////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_pybind
class ldpc_decode_test_vec_pybind : public ldpc_decode_test_vec
{
    public:
        //------------------------------------------------------------------
        // ldpc_decode_test_vec_pybind()
        ldpc_decode_test_vec_pybind(const test_vec_pybind_params& pyparams);
        //------------------------------------------------------------------
        // ~ldpc_decode_test_vec_pybind()
        virtual ~ldpc_decode_test_vec_pybind() {}
        //------------------------------------------------------------------
        // desc()
        // Descriptive string
        virtual const char* desc() const override;
        //------------------------------------------------------------------
        // Prepare a test vector
        virtual void generate() override;
        
    private:
        //------------------------------------------------------------------
        // populate_config()
        void populate_config(const test_vec_pybind_params& pyparams);
        //------------------------------------------------------------------
        // Data
        cuphy::tensor_device     inputLLR_; // Input LLR Tensor 
        uint16_t                 num_cw_limit_; // Restrict processing to partiular size
};

#endif //!defined(LDPC_DECODE_TEST_VEC_PYBIND_HPP_INCLUDED_)