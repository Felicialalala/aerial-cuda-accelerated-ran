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

#if !defined(LDPC2_DEC_OUTPUT_CUH_INCLUDED_)
#define LDPC2_DEC_OUTPUT_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// output_codeword_addr
// Provides the output address for a codeword when using the tensor-
// based LDPC decoder interface.
struct output_codeword_addr
{
    static __device__ uint32_t* get(const LDPC_kernel_params& params, int idx)
    {
        return reinterpret_cast<uint32_t*>(params.out + (idx * sizeof(uint32_t) * params.output_stride_words));
    }
};

////////////////////////////////////////////////////////////////////////
// output_LLR_addr
// Provides the output address for codeword LLV values when using the
// tensor-based LDPC decoder interface.
// Each thread will write 32 bits. Since only fp16 is supported for
// soft outputs, that means that each thread will write 2 fp16 values.
// The stride must be a multiple of 2 (elements) for uint32_t storage.
struct output_LLR_addr
{
    static __device__ uint32_t* get(const LDPC_kernel_params& params, int idx)
    {
        __half* hOut = static_cast<__half*>(params.soft_out);
        return reinterpret_cast<uint32_t*>(hOut + (idx * params.soft_out_stride_elements));
    }
};

////////////////////////////////////////////////////////////////////////
// decode_desc_output_addr()
// Provides the output address for a codeword when using the transport
// block-based LDPC decoder interface.
template <typename T> struct decode_desc_output_addr
{
    __device__
    static uint32_t* get(const cuphyLDPCDecodeDesc_t& decodeDesc, int cwIndex)
    {
        uint32_t* addr = nullptr;
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                if(cwIndex < decodeDesc.tb_output[i].num_codewords)
                {
                    addr = decodeDesc.tb_output[i].addr + (cwIndex * decodeDesc.tb_output[i].stride_words);
                    break;
                }
                cwIndex -= decodeDesc.tb_output[i].num_codewords;
            }
        }
        return addr;
    }
};

////////////////////////////////////////////////////////////////////////
// decode_desc_soft_output_addr()
// Provides the output address for a codeword when using the transport
// block-based LDPC decoder interface.
struct decode_desc_soft_output_addr
{
    __device__
    static uint32_t* get(const cuphyLDPCDecodeDesc_t& decodeDesc, int cwIndex)
    {
        uint32_t* addr = nullptr;
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                if(cwIndex < decodeDesc.llr_output[i].num_codewords)
                {
                    __half* ph = static_cast<__half*>(decodeDesc.llr_output[i].addr);
                    addr = reinterpret_cast<uint32_t*>(ph + (cwIndex * decodeDesc.llr_output[i].stride_elements));
                    break;
                }
                cwIndex -= decodeDesc.llr_output[i].num_codewords;
            }
        }
        return addr;
    }
};

////////////////////////////////////////////////////////////////////////
// decode_desc_output_addr
// Specialization of decode_desc_output_addr for __half2
//template <> struct decode_desc_output_addr<__half2>
//{
//    __device__
//    static uint32_t* get()
//    {
//        return nullptr;
//    }
//};

////////////////////////////////////////////////////////////////////////
// num_cta_output_codewords()
// Returns the number of output codewords for a CTA.
// For 1x codeword at a time kernels, the return value is always 1.
// For 2x codeword at a time kernels, the return value will be 2 in all
// cases, except for the last CTA when the number of output codewords is
// odd. (In that case the return value will be 1.)
// This function is useful as written when the number of CTAs is equal
// to the number of codewords, but should not be used for looping.
template <typename T> __device__
int num_cta_output_codewords(int total_num_cw)
{
    if(1 == codewords_per_CTA<T>::value)
    {
        return 1;
    }
    else
    {
        if((blockIdx.x * codewords_per_CTA<T>::value + 1) < total_num_cw)
        {
            return 2;
        }
        else
        {
            return 1;
        }
    }
}


////////////////////////////////////////////////////////////////////////
// output_token
// Class wrapper to disambiguate an additional argument passed to
// output_params constructors.
struct output_token
{
    tb_token token;
    __device__ explicit
    output_token(tb_token tok) : token(tok) {}
};

template <typename T> struct ldpc_dec_output_params;

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_params<>
template <typename T> struct ldpc_dec_output_params
{
    uint32_t* dst_gmem;         // output address for this CTA
    int       num_cw_bits;      // needed for variable outputs
    int       out_words_per_cw; // NOTE: can be derived from above...
    __device__
    ldpc_dec_output_params(const LDPC_kernel_params& params, int cwIndex) :
        dst_gmem(output_codeword_addr::get(params, cwIndex)),
        num_cw_bits(get_num_output_bits(params)),
        out_words_per_cw((num_cw_bits + 31) / 32)
    {
    }
    __device__
    ldpc_dec_output_params(uint32_t* dst,
                           int       nbits,
                           int       nwords) :
        dst_gmem(dst),
        num_cw_bits(nbits),
        out_words_per_cw(nwords)
    {
    }
    __device__
    ldpc_dec_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc, int cwIndex) :
        dst_gmem(decode_desc_output_addr<T>::get(decodeDesc, cwIndex)),
        num_cw_bits(get_num_output_bits(decodeDesc)),
        out_words_per_cw((num_cw_bits + 31) / 32)
    {
    }
    __device__
    ldpc_dec_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc,
                           const output_token&          out_tok) :
        dst_gmem(nullptr),
        num_cw_bits(get_num_output_bits(decodeDesc)),
        out_words_per_cw((num_cw_bits + 31) / 32)
    {
        int  tb     = tb_from_token(out_tok.token);
        int  offset = offset_from_token(out_tok.token);
        int  stride = decodeDesc.tb_output[tb].stride_words;
        dst_gmem    = decodeDesc.tb_output[tb].addr + (offset * stride);
    }
};

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_params<>
// specialization for __half2 (2 codewords at a time)
template <>
struct ldpc_dec_output_params<__half2>
{
    uint32_t* dst_gmem;
    int       num_cw_bits;      // needed for variable outputs
    int       out_words_per_cw; // NOTE: can be derived from above...
    int       out_stride_words; // needed for 2x codewords, where output stores two consecutive outputs
    int       num_out_cw;       // number of output codewords (by this CTA!), needed for 2x codeword kernels only
    __device__
    ldpc_dec_output_params(const LDPC_kernel_params& params) :
        dst_gmem(output_codeword_addr::get(params, blockIdx.x * codewords_per_CTA<__half2>::value)),
        num_cw_bits(get_num_output_bits(params)),
        out_words_per_cw((num_cw_bits + 31) / 32),
        out_stride_words(params.output_stride_words),
        num_out_cw(num_cta_output_codewords<__half2>(params.num_codewords))
    {
    }
    __device__
    ldpc_dec_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc) :
        dst_gmem(nullptr),
        num_cw_bits(get_num_output_bits(decodeDesc)),
        out_words_per_cw((num_cw_bits + 31) / 32),
        out_stride_words(0),
        num_out_cw(0)
    {
        int blkIndex = blockIdx.x;
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                int iBlocksClaimed = (decodeDesc.tb_output[i].num_codewords + 1) / 2;
                if(blkIndex < iBlocksClaimed)
                {
                    out_stride_words = decodeDesc.tb_output[i].stride_words;
                    dst_gmem         = decodeDesc.tb_output[i].addr + (blkIndex * 2 * out_stride_words);
                    // Last block may have only 1 codeword...
                    num_out_cw       = ((blkIndex*2 + 1) == decodeDesc.tb_output[i].num_codewords) ? 1 : 2;
                    break;
                }
                blkIndex -= iBlocksClaimed;
            }
        }
    }
    __device__
    ldpc_dec_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc,
                           const output_token&          out_tok) :
        dst_gmem(nullptr),
        num_cw_bits(get_num_output_bits(decodeDesc)),
        out_words_per_cw((num_cw_bits + 31) / 32),
        out_stride_words(0),
        num_out_cw(0)
    {
        int  tb          = tb_from_token(out_tok.token);
        int  offset      = offset_from_token(out_tok.token);
        bool is_partial  = is_partial_from_token(out_tok.token);
        out_stride_words = decodeDesc.tb_output[tb].stride_words;
        dst_gmem         = decodeDesc.tb_output[tb].addr + (offset * out_stride_words);
        num_out_cw       = is_partial ? 1 : 2;
    }
};

template <typename T> struct ldpc_dec_soft_output_params;

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output_params
// This struct converts from the two different parameter structures
// (LDPC_kernel_params for the legacy tensor interface and cuphyLDPCDecodeDesc_t
// for the transport block interface) to a common structure, allowing
// a single function to implement the looping logic.
template <>
struct ldpc_dec_soft_output_params<__half>
{
    uint32_t* dst_gmem;      // output address for this CTA
    int       num_cw_values;
    __device__
    ldpc_dec_soft_output_params(const LDPC_kernel_params& params, int cwIndex) :
        dst_gmem(output_LLR_addr::get(params, cwIndex)),
        num_cw_values(get_num_output_bits(params))
    {
    }
    __device__
    ldpc_dec_soft_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc, int cwIndex) :
        dst_gmem(decode_desc_soft_output_addr::get(decodeDesc, cwIndex)),
        num_cw_values(get_num_output_bits(decodeDesc))
    {
    }
    // Constructor to set up the soft output parameters using the transport
    // block token, used by some kernels to store information on the specific
    // codeword targeted by a CTA.
    __device__
    ldpc_dec_soft_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                const output_token&          out_tok) :
        dst_gmem(nullptr),
        num_cw_values(get_num_output_bits(decodeDesc))
    {
        int  tb     = tb_from_token(out_tok.token);
        int  offset = offset_from_token(out_tok.token);
        int  stride = decodeDesc.llr_output[tb].stride_elements;
        __half* h   = static_cast<__half*>(decodeDesc.llr_output[tb].addr);
        dst_gmem    = reinterpret_cast<uint32_t*>(h + (offset * stride));
    }
};

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output_params
// Specialization of ldpc_dec_soft_output_params for kernels that decode
// two codewords in a single CTA.
template <>
struct ldpc_dec_soft_output_params<__half2>
{
    uint32_t* dst_gmem;         // output address for this CTA
    int       num_cw_values;
    int       out_stride_elems; // needed for 2x codewords, where output stores two consecutive outputs
    int       num_out_cw;       // number of output codewords (by this CTA!), needed for 2x codeword kernels only

    __device__
    ldpc_dec_soft_output_params(const LDPC_kernel_params& params) :
        dst_gmem(output_LLR_addr::get(params, blockIdx.x * codewords_per_CTA<__half2>::value)),
        num_cw_values(get_num_output_bits(params)),
        out_stride_elems(params.soft_out_stride_elements),
        num_out_cw(num_cta_output_codewords<__half2>(params.num_codewords))
    {
    }
    __device__
    ldpc_dec_soft_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc) :
        dst_gmem(nullptr),
        num_cw_values(get_num_output_bits(decodeDesc)),
        out_stride_elems(0),
        num_out_cw(0)
    {
        int blkIndex = blockIdx.x;
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                int iBlocksClaimed = (decodeDesc.tb_output[i].num_codewords + 1) / 2;
                if(blkIndex < iBlocksClaimed)
                {
                    out_stride_elems = decodeDesc.llr_output[i].stride_elements;
                    __half* h = static_cast<__half*>(decodeDesc.llr_output[i].addr);
                    h += (blkIndex * 2 * out_stride_elems);
                    dst_gmem         = reinterpret_cast<uint32_t*>(h);
                    // Last block may have only 1 codeword...
                    num_out_cw       = ((blkIndex*2 + 1) == decodeDesc.llr_output[i].num_codewords) ? 1 : 2;
                    break;
                }
                blkIndex -= iBlocksClaimed;
            }
        }
    }
    // Constructor to set up the soft output parameters using the transport
    // block token, used by some kernels to store information on the specific
    // codeword targeted by a CTA.
    __device__
    ldpc_dec_soft_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                const output_token&          out_tok) :
        dst_gmem(nullptr),
        num_cw_values(get_num_output_bits(decodeDesc)),
        out_stride_elems(0),
        num_out_cw(0)
    {
        int  tb          = tb_from_token(out_tok.token);
        int  offset      = offset_from_token(out_tok.token);
        bool is_partial  = is_partial_from_token(out_tok.token);
        out_stride_elems = decodeDesc.llr_output[tb].stride_elements;
        __half* h = static_cast<__half*>(decodeDesc.llr_output[tb].addr);
        dst_gmem         = reinterpret_cast<uint32_t*>(h + (offset * out_stride_elems));
        num_out_cw       = is_partial ? 1 : 2;
    }
};


////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_fixed()
//template <typename T, int NODES, int Z>
//static inline __device__ void ldpc_dec_output_fixed(const ldpc_dec_output_params<T>& params,
//                                                    const float*                     app_smem)
//{
//    // The number of threads per warp.
//    enum
//    {
//        THREADS_PER_WARP = 32
//    };
//
//    // Decompose the thread indices into warp/lane.
//    int warp = threadIdx.x / THREADS_PER_WARP;
//    int lane = threadIdx.x % THREADS_PER_WARP;
//
//    // The output per thread.
//    uint32_t output = 0;
//
//    // Each warp reads 32*THREADS_PER_WARP elements.
//    int idx = warp * 32 * THREADS_PER_WARP + lane;
//    for(int ii = 0; ii < 32; ++ii)
//    {
//        float app = 0.f;
//        if(idx + ii * THREADS_PER_WARP < NODES * Z)
//        {
//            app = app_smem[idx + ii * THREADS_PER_WARP];
//        }
//
//        unsigned int vote = __ballot_sync(0xffffffff, signbit(app));
//        if(lane == ii)
//        {
//            output = vote;
//        }
//    }
//
//    // Output the result.
//    if(threadIdx.x < params.out_words_per_cw)
//    {
//        params.dst_gmem[threadIdx.x] = output;
//    }
//}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_fixed()
//template <typename T, int NODES, int Z>
//static inline __device__ void ldpc_dec_output_fixed(const ldpc_dec_output_params<T>& params,
//                                                    const __half*                    app_smem)
//{
//    // The number of threads per warp.
//    enum
//    {
//        THREADS_PER_WARP = 32
//    };
//
//    // Decompose the thread indices into warp/lane.
//    int warp = threadIdx.x / THREADS_PER_WARP;
//    int lane = threadIdx.x % THREADS_PER_WARP;
//
//    // The output per thread.
//    uint32_t output = 0;
//
//    // Each warp reads 32*THREADS_PER_WARP elements.
//    int idx = warp * 32 * THREADS_PER_WARP + lane;
//    for(int ii = 0; ii < 32; ++ii)
//    {
//        __half app = __float2half(0.0f);
//        if((idx + ii * THREADS_PER_WARP) < (NODES * Z))
//        {
//            app = app_smem[idx + ii * THREADS_PER_WARP];
//        }
//
//        unsigned int vote = __ballot_sync(0xffffffff, signbit(app));
//        if(lane == ii)
//        {
//            output = vote;
//        }
//    }
//
//    // Output the result.
//    if(threadIdx.x < params.out_words_per_cw)
//    {
//        params.dst_gmem[threadIdx.x] = output;
//    }
//}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
//template <typename T>
//static inline __device__ void ldpc_dec_output_variable(const ldpc_dec_output_params<T>& params,
//                                                       const float*                     app_smem)
//{
//    // The number of threads per warp.
//    enum
//    {
//        THREADS_PER_WARP = 32
//    };
//
//    // Decompose the thread indices into warp/lane.
//    int warp = threadIdx.x / THREADS_PER_WARP;
//    int lane = threadIdx.x % THREADS_PER_WARP;
//
//    // The output per thread.
//    uint32_t output = 0;
//
//    // Each warp reads 32*THREADS_PER_WARP elements.
//    int idx = warp * 32 * THREADS_PER_WARP + lane;
//    for(int ii = 0; ii < 32; ++ii)
//    {
//        float app = 0.f;
//        if((idx + ii * THREADS_PER_WARP) < params.num_cw_bits)
//        {
//            app = app_smem[idx + ii * THREADS_PER_WARP];
//        }
//
//        unsigned int vote = __ballot_sync(0xffffffff, signbit(app));
//        if(lane == ii)
//        {
//            output = vote;
//        }
//    }
//
//    // Output the result.
//    if(threadIdx.x < params.out_words_per_cw)
//    {
//        params.dst_gmem[threadIdx.x] = output;
//    }
//}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
template <typename T>
static inline __device__ void ldpc_dec_output_variable(const ldpc_dec_output_params<T>& params,
                                                       const __half*                   app_smem)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };
    //---------------------------------------------------------------
    // Each warp reads 32*THREADS_PER_WARP=1024 APP values and writes
    // 1024 bits in the form of 32 uint32_t values.
    const int WARP_IDX      = threadIdx.x / THREADS_PER_WARP;
    const int LANE          = threadIdx.x % THREADS_PER_WARP;
    const int BITS_PER_WARP = THREADS_PER_WARP * sizeof(uint32_t) * CHAR_BIT;
    //---------------------------------------------------------------
    // Check for early exit
    const int NUM_WARPS_REQ = (params.num_cw_bits + BITS_PER_WARP - 1) / BITS_PER_WARP;
    if(WARP_IDX >= NUM_WARPS_REQ)
    {
        return;
    }

    int output_idx = threadIdx.x;
    int start_idx  = WARP_IDX * BITS_PER_WARP + LANE;
    {
        uint32_t  output_value = 0;
        for(int ii = 0; ii < 32; ++ii)
        {
            const int    APP_IDX = start_idx + (ii * THREADS_PER_WARP);

            // Load soft decision from shared memory.
            // If index out of range, load value that is 0b as hard decision.
            const __half APP     = (APP_IDX < params.num_cw_bits) ?
                                   app_smem[APP_IDX]              :
                                   __float2half(1.0f);
            const uint32_t VOTE  = __ballot_sync(0xffffffff, llr_hard_decision(APP));
            if(LANE == ii)
            {
                output_value = VOTE;
            }
        }
        // Output the result.
        if(output_idx < params.out_words_per_cw)
        {
            params.dst_gmem[output_idx] = output_value;
        }
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
template <typename T>
static inline __device__ void ldpc_dec_output_variable_loop(const ldpc_dec_output_params<T>& params,
                                                            const __half*                   app_smem)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };
    //---------------------------------------------------------------
    // Each warp reads 32*THREADS_PER_WARP=1024 APP values and writes
    // 1024 bits in the form of 32 uint32_t values.
    const int WARP_IDX      = threadIdx.x / THREADS_PER_WARP;
    const int LANE          = threadIdx.x % THREADS_PER_WARP;
    const int BITS_PER_WARP = THREADS_PER_WARP * sizeof(uint32_t) * CHAR_BIT;
    //---------------------------------------------------------------
    // Check for early exit
    //const int NUM_WARPS          = (blockDim.x + 31) / 32;
    const int NUM_FULL_WARPS     = (blockDim.x / 32);
    const int NUM_FULL_WARPS_REQ = (params.num_cw_bits + BITS_PER_WARP - 1) / BITS_PER_WARP;
    const int NUM_ACTIVE_WARPS   = min(NUM_FULL_WARPS, NUM_FULL_WARPS_REQ);
    if(WARP_IDX >= NUM_ACTIVE_WARPS)
    {
        return;
    }

    int output_idx = threadIdx.x;
    int start_idx  = WARP_IDX * BITS_PER_WARP + LANE;
    do
    {
        uint32_t  output_value = 0;
        for(int ii = 0; ii < 32; ++ii)
        {
            const int    APP_IDX = start_idx + (ii * THREADS_PER_WARP);

            // Load soft decision from shared memory.
            // If index out of range, load value that is 0b as hard decision.
            const __half APP     = (APP_IDX < params.num_cw_bits) ?
                                   app_smem[APP_IDX]              :
                                   __float2half(1.0f);
            const uint32_t VOTE  = __ballot_sync(0xffffffff, llr_hard_decision(APP));
            if(LANE == ii)
            {
                output_value = VOTE;
            }
        }
        // Output the result.
        if(output_idx < params.out_words_per_cw)
        {
            params.dst_gmem[output_idx] = output_value;
        }
        // Advance
        output_idx += (NUM_ACTIVE_WARPS * THREADS_PER_WARP);
        start_idx  += (NUM_ACTIVE_WARPS * BITS_PER_WARP);
    } while(start_idx < params.num_cw_bits);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
static inline __device__ void ldpc_dec_output_variable(const LDPC_kernel_params& kernelParams,
                                                       const __half*             app_smem)
{
    ldpc_dec_output_params<__half> params(kernelParams, blockIdx.x);
    ldpc_dec_output_variable(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_multi()
static inline __device__ void ldpc_dec_output_variable_multi(const LDPC_kernel_params&    kernelParams,
                                                             const __half*                app_smem,
                                                             const multi_codeword_config& mconfig)
{
    const int LLR_STRIDE_VALUES   = round_up_to_next(get_num_LLRs(kernelParams),
                                                     static_cast<int>(sizeof(ldpc_traits<__half>::llr_sts_t) / sizeof(__half)));
    const int OUTPUT_STRIDE_BYTES = sizeof(uint32_t) * kernelParams.output_stride_words;
    for(int i = 0; i < mconfig.cta_codeword_count; ++i)
    {
        char* dst = kernelParams.out + ((mconfig.cta_start_index + i) * OUTPUT_STRIDE_BYTES);
        ldpc_dec_output_params<__half> params(reinterpret_cast<uint32_t*>(dst),               // output address
                                              get_num_output_bits(kernelParams),              // output bits
                                              (get_num_output_bits(kernelParams) + 31) / 32); // num output words
        ldpc_dec_output_variable(params, app_smem + (i * LLR_STRIDE_VALUES));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
static inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                       const __half*                app_smem)
{
    ldpc_dec_output_params<__half> params(decodeDesc, blockIdx.x);
    ldpc_dec_output_variable(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_multi()
static inline __device__ void ldpc_dec_output_variable_multi(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                             const __half*                app_smem,
                                                             const multi_codeword_config& mconfig)
{
    const int LLR_STRIDE_VALUES = round_up_to_next(get_num_LLRs(decodeDesc),
                                                   static_cast<int>(sizeof(ldpc_traits<__half>::llr_sts_t) / sizeof(__half)));
    for(int i = 0; i < mconfig.cta_codeword_count; ++i)
    {
        ldpc_dec_output_params<__half> params(decodeDesc,
                                              mconfig.cta_start_index + i);
        ldpc_dec_output_variable(params, app_smem + (i * LLR_STRIDE_VALUES));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
static inline __device__ void ldpc_dec_output_variable_loop(const LDPC_kernel_params& kernelParams,
                                                            const __half*             app_smem)
{
    ldpc_dec_output_params<__half> params(kernelParams, blockIdx.x);
    ldpc_dec_output_variable_loop(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
static inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                            const __half*                app_smem)
{
    ldpc_dec_output_params<__half> params(decodeDesc, blockIdx.x);
    ldpc_dec_output_variable_loop(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
static inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                       tb_token                     token,
                                                       const __half*                app_smem)
{
    ldpc_dec_output_params<__half> params(decodeDesc, output_token(token));
    ldpc_dec_output_variable(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
static inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                            tb_token                     token,
                                                            const __half*                app_smem)
{
    ldpc_dec_output_params<__half> params(decodeDesc, output_token(token));
    ldpc_dec_output_variable_loop(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
// Overload of hard decision output function for kernels that process
// two codewords at a time (using fp16x2 APP values in shared memory).
// "Variable" output functions use parameters that are not known at
// compile time.
template <typename T>
static inline __device__ void ldpc_dec_output_variable(const ldpc_dec_output_params<T>& params,
                                                       const __half2*                   app_smem)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };
    //---------------------------------------------------------------
    // Each warp reads 32*THREADS_PER_WARP=1024 APP values and writes
    // 1024 bits in the form of 32 uint32_t values.
    const int WARP_IDX      = threadIdx.x / THREADS_PER_WARP;
    const int LANE          = threadIdx.x % THREADS_PER_WARP;
    const int BITS_PER_WARP = THREADS_PER_WARP * sizeof(uint32_t) * CHAR_BIT;
    //---------------------------------------------------------------
    // Check for early exit
    const int NUM_WARPS_REQ = (params.num_cw_bits + BITS_PER_WARP - 1) / BITS_PER_WARP;
    if(WARP_IDX >= NUM_WARPS_REQ)
    {
        return;
    }

    int output_idx = threadIdx.x;
    int start_idx  = WARP_IDX * BITS_PER_WARP + LANE;
    {
        // The output per thread.
        uint32_t output[2] = {0, 0};

        for(int ii = 0; ii < 32; ++ii)
        {
            word_t       app;
            const int    APP_IDX = start_idx + (ii * THREADS_PER_WARP);
            // Load soft decision from shared memory.
            // If index out of range, load value that is 0b as hard decision.
            app.f16x2 = __half2_raw(__float2half2_rn(1.0));
            if(APP_IDX < params.num_cw_bits)
            {
                app.f16x2 = app_smem[APP_IDX];
            }
            //word_t app_sign_mask = fp16x2_sign_mask(app);
            //unsigned int vote0 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x00008000));
            //unsigned int vote1 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x80000000));
            __half2 fp16x2(app.f16x2);
            unsigned int vote0 = __ballot_sync(0xffffffff, (llr_hard_decision(fp16x2.x)));
            unsigned int vote1 = __ballot_sync(0xffffffff, (llr_hard_decision(fp16x2.y)));
            if(LANE == ii)
            {
                output[0] = vote0;
                output[1] = vote1;
            }
        }
        // Output the result.
        if(output_idx < params.out_words_per_cw)
        {
            params.dst_gmem[output_idx] = output[0];
            // Avoid writes past the end of the output when the number of
            // codewords is odd.
            if(2 == params.num_out_cw)
            {
                params.dst_gmem[output_idx + params.out_stride_words] = output[1];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
// Overload of hard decision output function for kernels that process
// two codewords at a time (using fp16x2 APP values in shared memory).
// "Variable" output functions use parameters that are not known at
// compile time.
template <typename T>
static inline __device__ void ldpc_dec_output_variable_loop(const ldpc_dec_output_params<T>& params,
                                                            const __half2*                   app_smem)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };
    //---------------------------------------------------------------
    // Each warp reads 32*THREADS_PER_WARP=1024 APP values and writes
    // 1024 bits in the form of 32 uint32_t values.
    const int WARP_IDX      = threadIdx.x / THREADS_PER_WARP;
    const int LANE          = threadIdx.x % THREADS_PER_WARP;
    const int BITS_PER_WARP = THREADS_PER_WARP * sizeof(uint32_t) * CHAR_BIT;
    //---------------------------------------------------------------
    // Check for early exit
    //const int NUM_WARPS          = (blockDim.x + 31) / 32;
    const int NUM_FULL_WARPS     = (blockDim.x / 32);
    const int NUM_FULL_WARPS_REQ = (params.num_cw_bits + BITS_PER_WARP - 1) / BITS_PER_WARP;
    const int NUM_ACTIVE_WARPS   = min(NUM_FULL_WARPS, NUM_FULL_WARPS_REQ);
    if(WARP_IDX >= NUM_ACTIVE_WARPS)
    {
        return;
    }

    int output_idx = threadIdx.x;
    int start_idx  = WARP_IDX * BITS_PER_WARP + LANE;
    do
    {
        // The output per thread.
        uint32_t output[2] = {0, 0};

        for(int ii = 0; ii < 32; ++ii)
        {
            word_t       app;
            const int    APP_IDX = start_idx + (ii * THREADS_PER_WARP);
            // Load soft decision from shared memory.
            // If index out of range, load value that is 0b as hard decision.
            app.f16x2 = __half2_raw(__float2half2_rn(1.0));
            if(APP_IDX < params.num_cw_bits)
            {
                app.f16x2 = app_smem[APP_IDX];
            }
            //word_t app_sign_mask = fp16x2_sign_mask(app);
            //unsigned int vote0 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x00008000));
            //unsigned int vote1 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x80000000));
            __half2 fp16x2(app.f16x2);
            unsigned int vote0 = __ballot_sync(0xffffffff, (llr_hard_decision(fp16x2.x)));
            unsigned int vote1 = __ballot_sync(0xffffffff, (llr_hard_decision(fp16x2.y)));
            if(LANE == ii)
            {
                output[0] = vote0;
                output[1] = vote1;
            }
        }
        // Output the result.
        if(output_idx < params.out_words_per_cw)
        {
            params.dst_gmem[output_idx] = output[0];
            // Avoid writes past the end of the output when the number of
            // codewords is odd.
            if(2 == params.num_out_cw)
            {
                params.dst_gmem[output_idx + params.out_stride_words] = output[1];
            }
        }
        // Advance
        output_idx += (NUM_ACTIVE_WARPS * THREADS_PER_WARP);
        start_idx  += (NUM_ACTIVE_WARPS * BITS_PER_WARP);
    } while(start_idx < params.num_cw_bits);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
static inline __device__ void ldpc_dec_output_variable(const LDPC_kernel_params& kernelParams,
                                                       const __half2*            app_smem)
{
    ldpc_dec_output_params<__half2> params(kernelParams);
    ldpc_dec_output_variable(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
static inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                       const __half2*               app_smem)
{
    ldpc_dec_output_params<__half2> params(decodeDesc);
    ldpc_dec_output_variable(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
static inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                       tb_token                     token,
                                                       const __half2*               app_smem)
{
    ldpc_dec_output_params<__half2> params(decodeDesc, output_token(token));
    ldpc_dec_output_variable(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
static inline __device__ void ldpc_dec_output_variable_loop(const LDPC_kernel_params& kernelParams,
                                                            const __half2*            app_smem)
{
    ldpc_dec_output_params<__half2> params(kernelParams);
    ldpc_dec_output_variable_loop(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
static inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                            const __half2*               app_smem)
{
    ldpc_dec_output_params<__half2> params(decodeDesc);
    ldpc_dec_output_variable_loop(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
static inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                            tb_token                     token,
                                                            const __half2*               app_smem)
{
    ldpc_dec_output_params<__half2> params(decodeDesc, output_token(token));
    ldpc_dec_output_variable_loop(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_fixed()
// Overload of hard decision output function for kernels that process
// two codewords at a time (using fp16x2 APP values in shared memory).
// "Fixed" output functions use parameters (NODES, Z) that are known at
// compile time.
//template <typename T, int NODES, int Z>
//static inline __device__ void ldpc_dec_output_fixed(const ldpc_dec_output_params<T>& params,
//                                                    const __half2*                   app_smem)
//{
//    // The number of threads per warp.
//    enum
//    {
//        THREADS_PER_WARP = 32
//    };
//
//    // Decompose the thread indices into warp/lane.
//    const int WARP_IDX = threadIdx.x / THREADS_PER_WARP;
//    const int LANE     = threadIdx.x % THREADS_PER_WARP;
//
//    // The output per thread.
//    uint32_t output[2] = {0, 0};
//
//    // Each warp reads 32*THREADS_PER_WARP elements.
//    int idx = (WARP_IDX * 32 * THREADS_PER_WARP) + LANE;
//    for(int ii = 0; ii < 32; ++ii)
//    {
//        word_t app;
//        app.u32 = 0;
//        if(idx + (ii * THREADS_PER_WARP) < (NODES * Z))
//        {
//            app.f16x2 = app_smem[idx + (ii * THREADS_PER_WARP)];
//        }
//        word_t app_sign_mask = fp16x2_sign_mask(app);
//
//        unsigned int vote0 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x00008000));
//        unsigned int vote1 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x80000000));
//        if(LANE == ii)
//        {
//            output[0] = vote0;
//            output[1] = vote1;
//        }
//    }
//
//    // Output the result.
//    if(threadIdx.x < params.out_words_per_cw)
//    {
//        params.dst_gmem[threadIdx.x]                             = output[0];
//        // Avoid writes past the end of the output when the number of
//        // codewords is odd.
//        if(2 == params.num_out_cw)
//        {
//            params.dst_gmem[threadIdx.x + params.out_stride_words] = output[1];
//        }
//    }
//}


////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output()
static inline __device__ void ldpc_dec_soft_output(const ldpc_dec_soft_output_params<__half>& params,
                                                   const __half*                             app_smem)
{
    // Each thread will write a pair of fp16 values as a uint32_t to
    // the global memory address.
    // We are currently writing an LLR value for information bits, and it does
    // not appear that this can be an odd number. (There are 22 info nodes for
    // BG1, and the only odd number of info nodes for BG2 is 9, but this only
    // occurs for Z values that are even (60 and 64).) However, we will keep
    // the logic here in case we change the number of values written in the
    // future to allow odd numbers of values.
    const int NUM_VALUES = params.num_cw_values;
    for(int app_idx = (threadIdx.x * 2);
        app_idx < NUM_VALUES;
        app_idx += (blockDim.x * 2))
    {
        word_t w;
        w.u32 = 0;
        if((app_idx + 1) == NUM_VALUES)
        {
            // Odd number of values: load a single value
            w.f16x2.x = app_smem[app_idx];
        }
        else
        {
            // Load a pair of values
            w.u32 = *reinterpret_cast<const uint32_t*>(app_smem + app_idx);
        }
        // Adjust the index to account for 32-bit writes and store to global memory
        params.dst_gmem[app_idx / 2] = w.u32;
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output()
// Adaptor function for writing soft outputs with the legacy tensor
// interface to the LDPC decoder. Constructs an instance of the
// ldpc_dec_soft_output_params structure that is common to both the
// legacy and transport block interfaces, and forwards to the function
// with the actual write logic.
static inline __device__ void ldpc_dec_soft_output(const LDPC_kernel_params& kernelParams,
                                                   const __half*             app_smem)
{
    ldpc_dec_soft_output_params<__half> params(kernelParams, blockIdx.x);
    ldpc_dec_soft_output(params, app_smem);
}


////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output()
// Adaptor function for writing soft outputs with the transport block
// interface to the LDPC decoder. Constructs an instance of the
// ldpc_dec_soft_output_params structure that is common to both the
// legacy and transport block interfaces, and forwards to the function
// with the actual write logic.
static inline __device__ void ldpc_dec_soft_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                   tb_token                     token,
                                                   const __half*                app_smem)
{
    ldpc_dec_soft_output_params<__half> params(decodeDesc, output_token(token));
    ldpc_dec_soft_output(params, app_smem);
}
static inline __device__ void ldpc_dec_soft_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                   const __half*                app_smem)
{
    ldpc_dec_soft_output_params<__half> params(decodeDesc, blockIdx.x);
    ldpc_dec_soft_output(params, app_smem);
}

static inline __device__ void ldpc_dec_soft_output(const ldpc_dec_soft_output_params<__half2>& params,
                                                   const __half2*                             app_smem)
{
    // Each thread will write a pair of fp16 values as a uint32_t to
    // the global memory address for EACH of TWO CODEWORDS.
    // We are currently writing an LLR value for information bits, and it does
    // not appear that this can be an odd number. (There are 22 info nodes for
    // BG1, and the only odd number of info nodes for BG2 is 9, but this only
    // occurs for Z values that are even (60 and 64).) However, we will keep
    // the logic here in case we change the number of values written in the
    // future to allow odd numbers of values.
    const int NUM_VALUES = params.num_cw_values;
    for(int app_idx = (threadIdx.x * 2);
        app_idx < NUM_VALUES;
        app_idx += (blockDim.x * 2))
    {
        word_t src0, src1, dst0, dst1;
        src0.u32 = src1.u32 = 0;
        src0.f16x2 = app_smem[app_idx];
        if((app_idx + 1) < NUM_VALUES)
        {
            // Load a second pair of values
            src1.f16x2 = app_smem[app_idx + 1];
        }
        dst0.f16x2 = __lows2half2(src0.f16x2, src1.f16x2);
        dst1.f16x2 = __highs2half2(src0.f16x2, src1.f16x2);
        // Shuffle from interleaved values to per-codeword values
        // Adjust the index to account for 32-bit stores to global memory (instead of
        // 16-bit half precision values).
        int u32_idx = app_idx / 2;
        params.dst_gmem[u32_idx] = dst0.u32;
        // For 2 codeword per CTA kernels, we may not have a second codeword
        if(2 == params.num_out_cw)
        {
            u32_idx = (app_idx + params.out_stride_elems) / 2;
            params.dst_gmem[u32_idx] = dst1.u32;
        }
    }
}

static inline __device__ void ldpc_dec_soft_output(const LDPC_kernel_params& kernelParams,
                                                   const __half2*            app_smem)
{
    ldpc_dec_soft_output_params<__half2> params(kernelParams);
    ldpc_dec_soft_output(params, app_smem);
}

static inline __device__ void ldpc_dec_soft_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                   tb_token                     token,
                                                   const __half2*               app_smem)
{
    ldpc_dec_soft_output_params<__half2> params(decodeDesc, output_token(token));
    ldpc_dec_soft_output(params, app_smem);
}
static inline __device__ void ldpc_dec_soft_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                   const __half2*               app_smem)
{
    ldpc_dec_soft_output_params<__half2> params(decodeDesc);
    ldpc_dec_soft_output(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output_multi()
static inline __device__ void ldpc_dec_soft_output_multi(const LDPC_kernel_params&    kernelParams,
                                                         const __half*                app_smem,
                                                         const multi_codeword_config& mconfig)
{
    const int LLR_STRIDE_VALUES   = round_up_to_next(get_num_LLRs(kernelParams),
                                                     static_cast<int>(sizeof(ldpc_traits<__half>::llr_sts_t) / sizeof(__half)));
    for(int i = 0; i < mconfig.cta_codeword_count; ++i)
    {
        ldpc_dec_soft_output_params<__half> params(kernelParams, mconfig.cta_start_index + i);
        ldpc_dec_soft_output(params, app_smem + (i * LLR_STRIDE_VALUES));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output_multi()
static inline __device__ void ldpc_dec_soft_output_multi(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                         const __half*                app_smem,
                                                         const multi_codeword_config& mconfig)
{
    const int LLR_STRIDE_VALUES = round_up_to_next(get_num_LLRs(decodeDesc),
                                                   static_cast<int>(sizeof(ldpc_traits<__half>::llr_sts_t) / sizeof(__half)));
    for(int i = 0; i < mconfig.cta_codeword_count; ++i)
    {
        ldpc_dec_output_params<__half> params(decodeDesc,
                                              mconfig.cta_start_index + i);
        ldpc_dec_output_variable(params, app_smem + (i * LLR_STRIDE_VALUES));
    }
}

} // namespace ldpc2

#endif // !defined(LDPC2_DEC_OUTPUT_CUH_INCLUDED_)