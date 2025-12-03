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

#if !defined(LDPC_2_HPP_INCLUDED_)
#define LDPC_2_HPP_INCLUDED_

#include "ldpc.hpp"

////////////////////////////////////////////////////////////////////////
// Functions specific to the "LDPC2" family of implementations
namespace ldpc2
{

union word_t
{
    float       f32;
    uint32_t    u32;
    int32_t     i32;
    __half_raw  f16;
    __half2_raw f16x2;
    ushort2     u16x2;
    short2      i16x2;
};

////////////////////////////////////////////////////////////////////////
// LDPC_kernel_params
struct LDPC_kernel_params
{
    const char* input_llr;
    char*       out;
    int         input_llr_stride_elements;
    int         output_stride_words;
    int         max_iterations;
    int         outputs_per_codeword;       // The number of outputs/ints per codeword.
    word_t      norm;
    void*       workspace;
    int         Z;
    int         z2;
    int         z4;
    int         z8;
    int         z16;
    int         mbz8;
    int         mbz16;
    int         num_parity_nodes;
    int         num_var_nodes;             // (1 == BG) ? (22 + mb) : (10 + mb)
    int         K;                         // number of bits: (1 == BG) ? (22 * Z) : (10 * Z)
    int         Kb;                        // num info nodes (22 for BG 1, {6, 8, 9, 10} for BG2)
    int         KbZ;
    int         Z_var;                     // Z * num_var_nodes
    int         Z_var_szelem;              // Z * num_var_nodes * sizeof(app_t)
    int         num_codewords;
    int         soft_out_stride_elements;
    float       clamp_value;
    void*       soft_out;
    LDPC_kernel_params(const cuphyLDPCDecodeConfigDesc_t& cfg,
                       const_tensor_pair&                 tLLR,
                       LDPC_output_t&                     tDst,
                       const cuphy_optional<tensor_pair>& optSoftOutputs) :
        input_llr((const char*)tLLR.second),
        out((char*)tDst.addr()),
        input_llr_stride_elements(tLLR.first.get().layout().strides[1]),
        output_stride_words(tDst.layout().strides[0]),
        max_iterations(cfg.max_iterations),
        outputs_per_codeword(((cfg.Kb * cfg.Z) + 31) / 32),
        workspace(cfg.workspace),
        Z(cfg.Z),
        z2(cfg.Z * 2),
        z4(cfg.Z * 4),
        z8(cfg.Z * 8),
        z16(cfg.Z * 16),
        mbz8(cfg.num_parity_nodes * cfg.Z * 8),
        mbz16(cfg.num_parity_nodes * cfg.Z * 16),
        num_parity_nodes(cfg.num_parity_nodes),
        num_var_nodes(cfg.num_parity_nodes + ((1 == cfg.BG) ? 22 : 10)),
        K((1 == cfg.BG) ? (22 * cfg.Z) : (10 * cfg.Z)),
        Kb(cfg.Kb),
        KbZ(cfg.Kb*cfg.Z),
        Z_var(cfg.Z * num_var_nodes),
        Z_var_szelem(cfg.Z * num_var_nodes * ((CUPHY_R_16F == cfg.llr_type) ?  2 : 4)),
        num_codewords(tLLR.first.get().layout().dimensions[1]),
        soft_out_stride_elements(static_cast<bool>(optSoftOutputs) ? optSoftOutputs.value().first.get().layout().strides[1] : 0),
        clamp_value(cfg.clamp_value),
        soft_out(static_cast<bool>(optSoftOutputs) ? optSoftOutputs.value().second : nullptr)
    {
        // When the LLR type is F32, we convert LLR values to FP16 when
        // loading.
        // The cuPHY API dictates that the config normalization should match
        // the LLR type, so we convert to FP16 here if necessary.
        norm.f16x2 = (CUPHY_R_32F == cfg.llr_type)                            ?
                     static_cast<__half2_raw>(__float2half2_rn(cfg.norm.f32)) :
                     cfg.norm.f16x2;
    }
    LDPC_kernel_params(const cuphyLDPCDecodeConfigDesc_t& cfg,
                       int                                input_stride_elem,
                       const void*                        input_addr,
                       int                                out_stride_words,
                       void*                              out_addr,
                       int                                soft_out_stride_elem,
                       void*                              soft_out_addr,
                       int                                num_cw) :
        input_llr((const char*)input_addr),
        out((char*)out_addr),
        input_llr_stride_elements(input_stride_elem),
        output_stride_words(out_stride_words),
        max_iterations(cfg.max_iterations),
        outputs_per_codeword(((cfg.Kb * cfg.Z) + 31) / 32),
        workspace(cfg.workspace),
        Z(cfg.Z),
        z2(cfg.Z * 2),
        z4(cfg.Z * 4),
        z8(cfg.Z * 8),
        z16(cfg.Z * 16),
        mbz8(cfg.num_parity_nodes * cfg.Z * 8),
        mbz16(cfg.num_parity_nodes * cfg.Z * 16),
        num_parity_nodes(cfg.num_parity_nodes),
        num_var_nodes(cfg.num_parity_nodes + ((1 == cfg.BG) ? 22 : 10)),
        K((1 == cfg.BG) ? (22 * cfg.Z) : (10 * cfg.Z)),
        Kb(cfg.Kb),
        KbZ(cfg.Kb*cfg.Z),
        Z_var(cfg.Z * num_var_nodes),
        Z_var_szelem(cfg.Z * num_var_nodes * ((CUPHY_R_16F == cfg.llr_type) ?  2 : 4)),
        num_codewords(num_cw),
        soft_out_stride_elements(soft_out_stride_elem),
        clamp_value(cfg.clamp_value),
        soft_out(soft_out_addr)
    {
        // When the LLR type is F32, we convert LLR values to FP16 when
        // loading.
        // The cuPHY API dictates that the config normalization should match
        // the LLR type, so we convert to FP16 here if necessary.
        norm.f16x2 = (CUPHY_R_32F == cfg.llr_type)                            ?
                     static_cast<__half2_raw>(__float2half2_rn(cfg.norm.f32)) :
                     cfg.norm.f16x2;
    }
};

////////////////////////////////////////////////////////////////////////
// shmem_llr_buffer_size()
// Returns the size required to store LLR/APP values in shared memory
CUDA_BOTH_INLINE
uint32_t shmem_llr_buffer_size(uint32_t vnodes, uint32_t Z, uint32_t elem_size)
{
    return (vnodes * Z * elem_size);
}

////////////////////////////////////////////////////////////////////////
// shmem_llr_buffer_size()
// Returns the size required to store LLR/APP values in shared memory,
// including end padding that may be required by the shared memory
// storage type used by the loader. (For example, if the loader loads
// a uint4 from global memory and stores that to shared memory, we need
// to round up the shared memory so that the end uint4 store targets a
// valid shared memory address.) Note, however, that shared memory is
// allocated in increments of the shared memory unit allocation size,
// which according to the CUDA Occupancy Calculator, is 256. Therefore,
// rounding up as this function does may not be necessary.
template <typename T, typename TStore>
inline
uint32_t shmem_llr_buffer_size_padded(uint32_t vnodes, uint32_t Z)
{
    return round_up_to_next(vnodes * Z * sizeof(T), sizeof(TStore));
}

////////////////////////////////////////////////////////////////////////
// get_device_max_shmem_per_block_option()
// Returns the maximum shared memory per block (optin) values as would
// be returned via a query of the CUDA device properties. Returns -1
// on error.
int32_t get_device_max_shmem_per_block_optin();

} // namespace ldpc2

#endif // !defined(LDPC_2_HPP_INCLUDED_)
