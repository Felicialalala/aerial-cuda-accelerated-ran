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

#include <math.h>
#include <stdio.h>
#include "cuphy.h"
#include "cuphy_internal.h"
#include "rate_matching.hpp"
#include "descrambling.cuh"
#include "crc.hpp"
#include "descrambling.hpp"
#include "derate_matching_modulo.hpp"

// max possible value for NUM_LLRS_PROCESSED_PER_THRD is 32, as this should not exceed warp size
// using NUM_LLRS_PROCESSED_PER_THRD = 32 will decrease the number of CTAs, but it increases the number of sequential iterations
// using smaller NUM_LLRS_PROCESSED_PER_THRD would increase number of CTAs, as a result increases number of instructions that are per CTA,
// but it reduces the number of iterations in the main for loop in de_rate_matching_global2
#define NUM_LLRS_PROCESSED_PER_THRD 32


using namespace cuphy_i;
using namespace descrambling;
using namespace crc;

__device__ inline int isnan_(float f) { return isnan(f); }
__device__ inline int isnan_(__half h) { return isnan(__half2float(h)); }

__device__ inline float rate_match_xor_sign(uint32_t seq, int bit_index, float llr_input)
{
    union u
    {
        float    f32;
        uint32_t u32;
    };
    u input, output;
    input.f32 = llr_input;
    // Extract the desired bit from the sequence and XOR with the input
    // float to (possibly) modify the sign bit of the input.
    output.u32 = (((seq << (31 - bit_index)) & 0x80000000) ^ input.u32);
    return output.f32;
}

__device__ inline __half rate_match_xor_sign(uint32_t seq, int bit_index, __half llr_input)
{
    // Shift the desired bit from the sequence to the sign position for
    // a half precision value (bit 15).
    uint32_t   half_sign_mask = (seq >> bit_index) << 15;
    __half_raw hraw           = llr_input;
    uint32_t   hraw32         = hraw.x;
    // XOR the sign mask with the original value to (possibly) modify
    // the sign bit of the input
    uint32_t out32 = (half_sign_mask & 0x00008000) ^ hraw32;
    hraw.x         = (unsigned short)out32;
    return __half(hraw);
}

// Flip sign bits of two FP16 LLRs packed in a __half2 using scrambling bits.
//   seqWord0/1  : 32-bit Gold seq words for LLR0 and LLR1.
//   bit_index0/1: bit positions inside each word.
//   llr_pair    : {low, high} halves are the two LLRs.
// Builds a mask with bits in FP16 sign positions (15, 31) and XORs in place
__device__ inline __half2 rate_match_xor_sign_pair(uint32_t seqWord0, uint32_t seqWord1, int bit_index0, int bit_index1, __half2 llr_pair)
{
    const uint32_t s0   = (seqWord0 >> bit_index0) & 1u;
    const uint32_t s1   = (seqWord1 >> bit_index1) & 1u;
    const uint32_t mask = (s0 << 15) | (s1 << 31); // low-half sign @bit15, high-half @bit31

    uint32_t bits;
    memcpy(&bits, &llr_pair, sizeof(bits));
    bits ^= mask;
    memcpy(&llr_pair, &bits, sizeof(bits));

    return llr_pair;
}

__device__ __forceinline__
float2 rate_match_xor_sign_pair(uint32_t seqWord0, uint32_t seqWord1, int bit_index0, int bit_index1, float2 llr_pair)
{
    // Extract the two sign bits to apply
    const uint32_t s0 = (seqWord0 >> bit_index0) & 1u;  // for llr_pair.x
    const uint32_t s1 = (seqWord1 >> bit_index1) & 1u;  // for llr_pair.y

    // Build per-lane sign masks (float sign is bit 31)
    const uint32_t mask0 = s0 << 31;
    const uint32_t mask1 = s1 << 31;

    // Flip sign bits by XORing the bit patterns
    uint32_t bx = __float_as_uint(llr_pair.x) ^ mask0;
    uint32_t by = __float_as_uint(llr_pair.y) ^ mask1;

    llr_pair.x = __uint_as_float(bx);
    llr_pair.y = __uint_as_float(by);
    return llr_pair;
}

__device__ inline uint32_t compute_llr_index(int tid, uint32_t j, uint32_t k,
                                             uint32_t codeBlockQAMStartIndex, uint32_t adjustedCodeBlockQAMStartIndex,
                                             uint32_t Nl, uint32_t nBBULayers, uint32_t* layer_map_array, uint8_t uciOnPuschFlag)
{
    uint32_t llr_idx;
    if(uciOnPuschFlag)
    {
        llr_idx = codeBlockQAMStartIndex + tid;
    }
    else
    {
        if (Nl < nBBULayers)
        {
            // jl can be interpreted as reIdx as there are Nl qams for this user in a singe resource element
            uint32_t jl = j / Nl;
            // LLR buffer has dimension QAM_STRIDE x nBBULayers x nRe
            // First dimension: bitIdxInQam = k
            // Second dimension: layerIdxWithinUeGrp = layer_map_array[(j - jl * Nl)]
            // Third dimension:  reIdx = jl
            // The location of the llr would be: llrIdx = bitIdxInQam + (QAM_STRIDE * layerIdxWithinUeGrp + reIdx * nBBULayers * QAM_STRIDE)
            llr_idx = (adjustedCodeBlockQAMStartIndex * nBBULayers / Nl) + (k + (jl * nBBULayers + layer_map_array[(j - jl * Nl)]) * QAM_STRIDE);
        }
        else
        {
            // when nBBULayers==Nl, the general index logic above to retrieve llr can be simplified as below
            llr_idx = adjustedCodeBlockQAMStartIndex + k + j * QAM_STRIDE;
        }
    }
    return llr_idx;
}


template <typename T>
__device__ inline T atomicMaxCustom(T* address, T val)
{
    // Default implementation using CUDA intrinsic atomicMax
    return atomicMax(address, val);
}

// specialization for __half using custom implementation with atomicCAS
template <>
__device__ inline __half atomicMaxCustom<__half>(__half* address, __half val){
    unsigned short* address_as_ushort = reinterpret_cast<unsigned short*>(address);
    unsigned short old = *address_as_ushort, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ushort, assumed, __half_as_ushort(__hmax(val, __ushort_as_half(assumed))));
    } while (assumed != old);
    return __ushort_as_half(old);
}

// specialization for float using custom implementation with atomicCAS
template <>
__device__ inline float atomicMaxCustom<float>(float* address, float val){
    unsigned int* address_as_uint = reinterpret_cast<unsigned int*>(address);
    unsigned int old = *address_as_uint, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(fmaxf(val, __uint_as_float(assumed))));
    } while (assumed != old);
    return __uint_as_float(old);
}

template <typename T>
__device__ inline T atomicMinCustom(T* address, T val)
{
    // Default implementation using CUDA intrinsic atomicMin
    return atomicMin(address, val);
}

// specialization for __half using custom implementation with atomicCAS
template <>
__device__ inline __half atomicMinCustom<__half>(__half* address, __half val){
    unsigned short* address_as_ushort = reinterpret_cast<unsigned short*>(address);
    unsigned short old = *address_as_ushort, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ushort, assumed, __half_as_ushort(__hmin(val, __ushort_as_half(assumed))));
    } while (assumed != old);
    return __ushort_as_half(old);
}

// specialization for float using custom implementation with atomicCAS
template <>
__device__ inline float atomicMinCustom<float>(float* address, float val){
    unsigned int* address_as_uint = reinterpret_cast<unsigned int*>(address);
    unsigned int old = *address_as_uint, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(fminf(val, __uint_as_float(assumed))));
    } while (assumed != old);
    return __uint_as_float(old);
}

template <typename T_OUT>
__device__ __forceinline__
void processOneLLR(  uint32_t           jIdx,          // = tid /  Qm
                     uint32_t           kIdx,          // = tid %  Qm
                     T_OUT              llr,           // preloaded
                     int                EoverQm,
                     /* --- rate-matching / combining ------------------------- */
                     uint32_t           Kd,
                     uint32_t           F,
                     uint32_t           k0,
                     uint32_t           Ncb,
                     int                potentialRaceIfPositive,
                     bool               ndi,
                     /* --- misc ---------------------------------------------- */
                     bool               descramblingOn,
                     T_OUT              LLR_CLAMP_MIN,
                     T_OUT              LLR_CLAMP_MAX,
                     /* --- destination --------------------------------------- */
                     T_OUT*    __restrict__ out)
{
    if (jIdx >= static_cast<uint32_t>(EoverQm)) return;   // out of range

    // de-rate-match index -------------------------------------------
    const uint32_t inIdx  = kIdx * EoverQm + jIdx;
    const uint32_t outIdx = derate_match_fast_calc_modulo(inIdx, Kd, F, k0, Ncb);
    //outIdx = inIdx % (Ncb - F); // this will not generate the correct out index for all scenarios

    const bool useAtomics = (potentialRaceIfPositive > 0) && (outIdx < potentialRaceIfPositive);

    // Write / combine -----------------------------------------------
    // ndi 1: no LLR combining, just write to memory;
    // ndi 0: LLR combining, use atomicAdd
    if(ndi)
    {
        if(!useAtomics)
        {
            out[outIdx] = llr;
        }
        else
        {
            T_OUT prev = atomicAdd(out + outIdx, llr);
            llr += prev;
            // clamp the llr
            if(llr > LLR_CLAMP_MAX) atomicMinCustom(out + outIdx, LLR_CLAMP_MAX);
            else if(llr < LLR_CLAMP_MIN) atomicMaxCustom(out + outIdx, LLR_CLAMP_MIN);
        }
    }
    else
    {
        if(!useAtomics)
        {
            llr += out[outIdx];
            // clamp the llr
            if constexpr (std::is_same<T_OUT, __half>::value)
            {
               llr = __hmax(__hmin(llr, LLR_CLAMP_MAX), LLR_CLAMP_MIN);
            } else
            {
                llr = max(min(LLR_CLAMP_MAX, llr), LLR_CLAMP_MIN);
            }
            // write the updated LLR.  No need for atomic, different threads work on different outIdx.
            out[outIdx] = llr;
        }
        else
        {
            T_OUT prev_llr = atomicAdd(out + outIdx, llr);
            llr += prev_llr;
            // clamp the llr
            if(llr > LLR_CLAMP_MAX) atomicMinCustom(out + outIdx, LLR_CLAMP_MAX);
            else if(llr < LLR_CLAMP_MIN) atomicMaxCustom(out + outIdx, LLR_CLAMP_MIN);
        }
    }
}

// Vectorized zero for a linear T_OUT-range [S, S+L) in out[].
// each CTA writes its own contiguous slice; uses 8-byte stores.
// ToDo: to be further optimized for devices with cc 10 and beyond
template <typename T_OUT>
__device__ __forceinline__
void zeroRangeVec(T_OUT* __restrict__ out,
                  uint32_t            start,
                  uint32_t            end,
                  uint32_t            tid,
                  uint32_t            stride)
{
    if(start >= end) return;

    // ----- Head: scalar store until we hit 16B alignment -----
    T_OUT*   base  = out + start;
    uint32_t total = end - start;

    // elements to reach 16B alignment from 'base'
    const uintptr_t addr           = reinterpret_cast<uintptr_t>(base);
    const uint32_t  bytes_to_align = (16u - (addr & 15u)) & 15u;
    const uint32_t  head           = min(total, bytes_to_align / static_cast<uint32_t>(sizeof(T_OUT)));

    // handle unaligned indices [0, head)
    for(uint32_t i = tid; i < head; i += stride) { base[i] = static_cast<T_OUT>(0.0f); }

    // Advance to aligned boundary
    base += head;
    total -= head;
    if(!total) return;

    // ----- 16B vector stores (8 halves or 4 floats per chunk) -----
    //static_assert(16 % sizeof(T_OUT) == 0, "T_OUT size must divide 16");
    constexpr uint32_t ELEMS_PER_VEC = 16u / sizeof(T_OUT);

    uint4*   ptrVec   = reinterpret_cast<uint4*>(base);
    uint32_t vecCount = total / ELEMS_PER_VEC;  // number of 16-byte stores

    // Each global lane writes vector chunks: j = tid; j < vecCount; j += stride
    for(uint32_t j = tid; j < vecCount; j += stride)
    {
        //p16[i] = {0u, 0u, 0u, 0u}; // 128-bit zero
        // slightly lower L1 pollution using .cs (evict-first) policy
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(ptrVec + j), "r"(0), "r"(0), "r"(0), "r"(0));
    }

    // ----- Tail: scalar store of leftovers (remaining elements < ELEMS_PER_VEC) -----
    T_OUT*   tailPtr = reinterpret_cast<T_OUT*>(ptrVec + vecCount);
    uint32_t tail    = total % ELEMS_PER_VEC;

    // handle remaining unaligned indices [0, tail)
    for(uint32_t i = tid; i < tail; i += stride) { tailPtr[i] = static_cast<T_OUT>(0.0f); }
}

// k0 for LDPC BG1/BG2 (38.212), rv in [0..3]
__device__ __forceinline__
uint32_t k0_from_bg_rv(uint8_t bg, uint8_t rv, uint32_t Zc, uint32_t Ncb)
{
    // assuming BG1 ⇒ Ncb == 66 * Zc and BG2 ⇒ Ncb == 50 * Zc
    // k0 computation can be simplified to avoid integer division
    uint32_t k0 = 0;
    if(bg == 1)
    {
        if(rv == 0) { k0 = 0; }
        else if(rv == 1) { k0 = (17 * Ncb / (66 * Zc)) * Zc; }
        else if(rv == 2) { k0 = (33 * Ncb / (66 * Zc)) * Zc; }
        else if(rv == 3) { k0 = (56 * Ncb / (66 * Zc)) * Zc; }
    }
    else if(bg == 2)
    {
        if(rv == 0) { k0 = 0; }
        else if(rv == 1) { k0 = (13 * Ncb / (50 * Zc)) * Zc; }
        else if(rv == 2) { k0 = (25 * Ncb / (50 * Zc)) * Zc; }
        else if(rv == 3) { k0 = (43 * Ncb / (50 * Zc)) * Zc; }
    }
    return k0;
}

template <typename T_IN, typename T_OUT, int Qm>
__device__ __forceinline__ void deRateMatchingKernelInner(puschRxRateMatchDescr_t* pRmDesc)
{
    using T_OUT_PAIR = typename std::conditional<std::is_same<T_OUT, __half>::value, __half2, float2>::type;

    constexpr int WARP_SIZE = 32;
    const T_OUT LLR_CLAMP_MAX = static_cast<T_OUT>(10000.0f);
    const T_OUT LLR_CLAMP_MIN = static_cast<T_OUT>(-10000.0f);
    // PUSCH kernel descriptor
    puschRxRateMatchDescr_t& rmDesc = *pRmDesc;
    const uint32_t nFracCbs = gridDim.x;
    const uint32_t fracCbIdx = blockIdx.x;
    const uint32_t cbIdx = blockIdx.y;
    const uint32_t tbIdx = blockIdx.z;
    const uint16_t ueIdx = rmDesc.schUserIdxs[tbIdx];

    // Output tensor
    // @todo: rmDesc.out which holds an array of pointers to HARQ buffers (in GPU memory) lives in host pinned memory,
    // check performance impact of accessing this memory
    T_OUT* out = static_cast<T_OUT*>(rmDesc.out[ueIdx]);

    // Array of transport block parameters structs
    const PerTbParams& tbPrms = rmDesc.tbPrmsArray[ueIdx];
    // code block index
    uint32_t r = cbIdx + tbPrms.firstCodeBlockIndex;

    // Output code block stride
    uint32_t Ncb_padded = tbPrms.Ncb_padded;
    uint32_t cbStartOffset = r * Ncb_padded;

    // Adjust for codeblock offset
    out += cbStartOffset;


    // Enable/Disable descrambling
    int descramblingOn = rmDesc.descramblingOn;
    // Input LLR tensor
    const T_IN* llr_vec_in = static_cast<const T_IN*>(rmDesc.llr_vec_in[tbIdx]);

    //******** The following parameters are invariant for all CTAs working on the same transport block*******/
    // They only vary along the y-dimension of the grid, namely across transport blocks

    // Output de-rate matched code block size excluding punctured bits
    uint32_t Ncb = tbPrms.Ncb;
    // number of code blocks in transport block
    uint32_t C = tbPrms.num_CBs;
    // base graph index
    uint32_t bg = tbPrms.bg;
    // redundancy version
    uint32_t rv = tbPrms.rv;
    // new data indicator
    uint32_t ndi = tbPrms.ndi;

    // lifting factor
    uint32_t Zc = tbPrms.Zc;
    // Number of UE layers (Number of layers occupied by transport block tbIdx)
    uint32_t Nl = tbPrms.Nl;

    // Total number of layers from all UEs (Number of BBU antenna layers, KERNEL LEVEL parameter)
    uint32_t nBBULayers = tbPrms.nBBULayers;

    // layer mapping
    __shared__ uint32_t layer_map_array[MAX_N_LAYERS_PUSCH];

    /************/

    if(r < C)
    {   // Only executes code if thread is allocated a valid  codeblock (some threads will be idle)

        if (nBBULayers > Nl)
        {
            for (int i = threadIdx.x; i < Nl; i += blockDim.x)
            {
                layer_map_array[i] = tbPrms.layer_map_array[i];
            }
            __syncthreads();
        }

        // Determine input rate matched block size E and start index codeBlockQAMStartIndex

        //******** The following parameters are invariant for all CTAs working on the same transport block*******/

        // index at which the first LLR of code block r starts within transport block tbIdx
        uint32_t codeBlockQAMStartIndex;
        // Size (number of LLRs) of input rate-matched code block r
        uint32_t E;
        // Number of layers times modulation index: determines how many LLRs are read from each block of NBBULayers
        uint32_t TBLLRsPerNBBULayers = Nl * Qm;
        // total number of LLRs to be read for current transport block
        uint32_t totalNLLRsForTB = TBLLRsPerNBBULayers * C;

        // encodedSize is size (number of LLRs) of current transport block; q1 is number of NBBULayers blocks the transport block is spread over
        uint32_t q1 = tbPrms.uciOnPuschFlag ? tbPrms.G / TBLLRsPerNBBULayers : tbPrms.encodedSize / TBLLRsPerNBBULayers; // exact division
        // number of NBBULayers blocks each code block is spread over
        uint32_t q = q1 / C;

        // This is straight from the spec: compute size E of each code block of current transport block
        uint32_t rr = C - (q1 - q * C) - 1;
        // smaller code blocks size
        uint32_t El = Nl * Qm * q;
        // larger code block size
        //uint32_t Eh = Nl * Qm * ((Ncb + totalNLLRsForTB - 1) / totalNLLRsForTB);
        uint32_t Eh = El + TBLLRsPerNBBULayers * (q * totalNLLRsForTB < tbPrms.encodedSize);

        if(r <= rr)
        {
            E                      = El;
            codeBlockQAMStartIndex = r * El;
        }
        else
        {
            E                      = Eh;
            codeBlockQAMStartIndex = (rr + 1) * El + (r - rr - 1) * Eh;
        }

        uint32_t EoverQm = E / Qm;

        // For incremental redundancy transmission: determine k0 based on rv and bg(base graph)
        // Uninitialized scalar variable this changes will treat rv 0 (detected if k0=0): no LLR combining, just write to memory; write
        uint32_t k0 = k0_from_bg_rv(bg, rv, Zc, Ncb);
        /************/

        // First code block LLR index for current CTA within transport block, used for descrambling sequence generation
        const uint32_t ctaLLROffset = fracCbIdx * blockDim.x * NUM_LLRS_PROCESSED_PER_THRD;
        const uint32_t cbLLRStartIndex = codeBlockQAMStartIndex + ctaLLROffset;

        //====================================================================================================================================================
        // each thread block will process (blockDim.x * NUM_LLRS_PROCESSED_PER_THRD) LLRs, i.e. we'll have a for loop that each thread in the CTA will iterate
        // maximum NUM_LLRS_PROCESSED_PER_THRD times.
        // at iteration 0, each warp reads "mySeq" from thread 0 of that warp. "mySeq" has 32 bits (WORD_SIZE), each bit associates with one thread in the warp (WARP_SIZE).
        // hence this implementation relies on the fact that WORD_SIZE == WARP_SIZE
        // at iteration i, each warp reads the corresponding word from the golden sequence, and that is word index NUM_WARPS_PER_TB * i + WARP_IDX  where
        // NUM_WARPS_PER_TB = blockDim.x / WARP_SIZE, and WARP_IDX = threadIdx.x / WARP_SIZE
        // since in this for loop, we retrieve golden sequence word of iteration i in each warp from thread i in that warp, we'd use the following indexing logic
        // (and this explains why NUM_LLRS_PROCESSED_PER_THRD as the maximum number of iterations can't be bigger than warp size)

        const uint32_t NUM_WARPS_PER_TB   = blockDim.x / WARP_SIZE;
        const uint32_t WARP_IDX           = threadIdx.x / WARP_SIZE;
        const uint32_t THREAD_IDX_IN_WARP = threadIdx.x % WARP_SIZE;
        const uint32_t index              = THREAD_IDX_IN_WARP * NUM_WARPS_PER_TB + WARP_IDX;

        // each thread in a warp computes a word of the descrambling sequence that will be used at step "warpLane"
        uint32_t mySeq = gold32n(tbPrms.cinit, cbLLRStartIndex + index * WORD_SIZE);
        //====================================================================================================================================================

        //******** The following parameters are invariant for all CTAs working on the same transport block*******/

        uint32_t adjustedCodeBlockQAMStartIndex = (codeBlockQAMStartIndex / Qm) * QAM_STRIDE;
        // Number of filler bits
        uint32_t F              = tbPrms.F;
        uint32_t nPuncturedBits = 2 * Zc;

        // Within the output buffer, the Ncb circular buffer starts at offset 2*Zc (punctured bits)
        out += nPuncturedBits;

        // Number of systematic bits
        uint32_t K = tbPrms.K;
        // Number of systematic bits in output code block excluding punctured bits
        //uint32_t K_hat = K - nPuncturedBits;
        // Number of payload bits in output code block
        uint32_t Kd = K - nPuncturedBits - F;

        // nZpBitsPerCb is the total number of LLRs i.e. (2 * Zc) + E + F rounded up to a multiple of Zc (rounding needed by LDPC decoder)
        // (2 * Zc) - punctured LLRs
        // E        - rate matched LLRs
        // F        - Filler LLRs
        // uint32_t nZpBitsPerCb = tbPrmsArray[tbIdx].nZpBitsPerCb;

        //number of LLRs belonging to other transport blocks to be skipped before getting to LLRS belonging to current transport block again
        /////////uint32_t cbStep = nBBULayers / Nl;
        // Deinterleave and fill output vector except filler bits

        // detect possibility of more than one thread accessing the same outIdx
        int potentialRaceIfPositive = (E + 2 * F + k0) - Ncb; // if potentialRaceIfPositive > 0, more than one thread can write on the same outIdx
        // Note that before running rate matching, de_rate_matching_reset_buffer should run to reset first potentialRaceIfPositive LLRs in HARQ buffer
        // Resetting those LLRs in this function can potentially lead to intra-CTA race. i.e. while some CTA is writing on the HARQ buffer, another CTA may reset the same element

        // Make sure CTA is not reading beyond input code block size E
        int maxIndex            = round_up_to_next(E, WORD_SIZE); // round up to be multiple of 32
        int maxIndexThisThrdBlk = (fracCbIdx + 1) * blockDim.x * NUM_LLRS_PROCESSED_PER_THRD;
        maxIndex                = (maxIndexThisThrdBlk < maxIndex) ? maxIndexThisThrdBlk : maxIndex;

        // Process two LLRs per iteration
        constexpr int LLR_PER_ITER = 2;

        // --------------------------------------------------------------------------
        // Initial indices & bookkeeping
        uint32_t warpLane         = 0;            // 0,2,4,…,30 (increments by 2)
        uint32_t threadBitOffset  = threadIdx.x;  // bit position for the 1st LLR

        // load the first pair into registers
        int      baseTid = ctaLLROffset + threadIdx.x;

        // current pair (tid0 and tid1)
        int      tid0  = baseTid;
        uint32_t jIdx0 = tid0 / Qm;
        uint32_t kIdx0 = tid0 - jIdx0 * Qm;
        T_OUT    llr0  = static_cast<T_OUT>(0.0f);
        if (jIdx0 < EoverQm) {
            uint32_t idx0 = compute_llr_index(tid0, jIdx0, kIdx0,
                                              codeBlockQAMStartIndex, adjustedCodeBlockQAMStartIndex,
                                              Nl, nBBULayers, layer_map_array, tbPrms.uciOnPuschFlag);
            llr0 = llr_vec_in[idx0];
        }

        int      tid1  = baseTid + blockDim.x;
        uint32_t jIdx1 = (tid1 < maxIndex) ? (tid1 / Qm) : 0u;
        uint32_t kIdx1 = (tid1 < maxIndex) ? (tid1 - jIdx1 * Qm) : 0u;
        T_OUT    llr1  = static_cast<T_OUT>(0.0f);
        if (tid1 < maxIndex && jIdx1 < EoverQm) {
            uint32_t idx1 = compute_llr_index(tid1, jIdx1, kIdx1,
                                              codeBlockQAMStartIndex, adjustedCodeBlockQAMStartIndex,
                                              Nl, nBBULayers, layer_map_array, tbPrms.uciOnPuschFlag);
            llr1 = llr_vec_in[idx1];
        }

        //------------------------------------------------------------------------------
        // main loop

        for (/* baseTid already set */;
             baseTid < maxIndex;
             baseTid += blockDim.x * LLR_PER_ITER,
             warpLane += LLR_PER_ITER,
             threadBitOffset += blockDim.x * LLR_PER_ITER)
        {
            // ------------------- Prefetch/load the next pair into registers ------------------------
            int nextBaseTid = baseTid + blockDim.x * LLR_PER_ITER;

            // next pair
            int      n_tid0  = nextBaseTid;
            uint32_t n_jIdx0 = (n_tid0 < maxIndex) ? (n_tid0 / Qm) : 0u;
            uint32_t n_kIdx0 = (n_tid0 < maxIndex) ? (n_tid0 - n_jIdx0 * Qm) : 0u;
            T_OUT    n_llr0  = static_cast<T_OUT>(0.0f);
            if ((n_tid0 < maxIndex) && (n_jIdx0 < EoverQm)) {
                uint32_t n_idx0 = compute_llr_index(n_tid0, n_jIdx0, n_kIdx0,
                                                    codeBlockQAMStartIndex, adjustedCodeBlockQAMStartIndex,
                                                    Nl, nBBULayers, layer_map_array, tbPrms.uciOnPuschFlag);
                n_llr0 = llr_vec_in[n_idx0];
            }

            int      n_tid1  = nextBaseTid + blockDim.x;
            uint32_t n_jIdx1 = (n_tid1 < maxIndex) ? (n_tid1 / Qm) : 0u;
            uint32_t n_kIdx1 = (n_tid1 < maxIndex) ? (n_tid1 - n_jIdx1 * Qm) : 0u;
            T_OUT    n_llr1  = static_cast<T_OUT>(0.0f);
            if ((n_tid1 < maxIndex) && (n_jIdx1 < EoverQm)) {
                uint32_t n_idx1 = compute_llr_index(n_tid1, n_jIdx1, n_kIdx1,
                                                    codeBlockQAMStartIndex, adjustedCodeBlockQAMStartIndex,
                                                    Nl, nBBULayers, layer_map_array, tbPrms.uciOnPuschFlag);
                n_llr1 = llr_vec_in[n_idx1];
            }

            // -------- vectorize descrambling and clamping -----------------------------------

            // Pack the two preloaded LLRs
            T_OUT_PAIR llr_pair;

            // Conditional logic for initialization
            if constexpr (std::is_same<T_OUT, __half>::value) {
                llr_pair = __halves2half2(llr0, llr1);
            } else {
                llr_pair = {llr0, llr1}; // float2
            }

            // Bit offsets for the two LLRs handled by this thread this iteration
            // Bit offsets select the lane-local bit.
            const uint32_t bitOffset0 = (threadBitOffset & 31);
            const uint32_t bitOffset1 = ((threadBitOffset + blockDim.x) & 31);

            // Fetch scrambling words for the two LLRs in this iter:
            // seqWord0 = current word (warpLane), seqWord1 = next word (warpLane+1)
            uint32_t seqWord0   = __shfl_sync(0xFFFFFFFF, mySeq, warpLane);
            uint32_t seqWord1 = __shfl_sync(0xFFFFFFFF, mySeq, warpLane + 1);

            if (descramblingOn && !tbPrms.uciOnPuschFlag)
            {
                llr_pair = rate_match_xor_sign_pair(seqWord0, seqWord1, bitOffset0, bitOffset1, llr_pair);
            }

            // Clamp both halves together
            if constexpr (std::is_same<T_OUT, __half>::value)
            {
                const __half2 HMIN2 = __halves2half2(LLR_CLAMP_MIN, LLR_CLAMP_MIN);
                const __half2 HMAX2 = __halves2half2(LLR_CLAMP_MAX, LLR_CLAMP_MAX);
                llr_pair = __hmax2(__hmin2(llr_pair, HMAX2), HMIN2);
            } else
            {
                llr_pair.x = max(min(llr_pair.x, LLR_CLAMP_MAX), LLR_CLAMP_MIN);
                llr_pair.y = max(min(llr_pair.y, LLR_CLAMP_MAX), LLR_CLAMP_MIN);
            }

            // -------- process first LLR -----------------------------------------------------
            processOneLLR( jIdx0, kIdx0, llr_pair.x, EoverQm,
                            // rate-matching / combining
                            Kd, F, k0, Ncb, potentialRaceIfPositive, ndi,
                            // misc
                            descramblingOn, LLR_CLAMP_MIN, LLR_CLAMP_MAX,
                            // output
                            out);

            // -------- process second LLR ----------------------------------------------------
            if (tid1 < maxIndex && jIdx1 < EoverQm)                           // guard tail iteration
            {
                processOneLLR(jIdx1, kIdx1, llr_pair.y, EoverQm,
                                // rate-matching / combining
                                Kd, F, k0, Ncb, potentialRaceIfPositive, ndi,
                                // misc
                                descramblingOn, LLR_CLAMP_MIN, LLR_CLAMP_MAX,
                                // output
                                out);
            }

            // ------------------- Rotate prefetched state into current for next iter ----------------
            tid0 = n_tid0; jIdx0 = n_jIdx0; kIdx0 = n_kIdx0; llr0 = n_llr0;
            tid1 = n_tid1; jIdx1 = n_jIdx1; kIdx1 = n_kIdx1; llr1 = n_llr1;
        }

        if(ndi)
        {
            // Use all thread blocks associated with this CB
            uint32_t stride    = nFracCbs * blockDim.x;
            uint32_t globalTid = (fracCbIdx * blockDim.x) + threadIdx.x;

            // Output buffer is of length Ncb_padded

            // 1. Circular buffer initialization (Ncb long circular buffer section of output buffer)
            // NOTE: EITHER 1a, 2a, 2b below needed OR at setup set RM output to zero
            // 1a. Simply write zeros into rest of Ncb long circular buffer (including filler gap)
            if (E + F < Ncb)
            {
                const uint32_t len   = Ncb - (E + F);
                const uint32_t start = (E + F + k0) % Ncb;


                // Helper to exclude the filler gap [Kd, Kd+F) from a single non-wrapping range [a,b)
                auto zeroSpanExcludingFillerGap = [&](uint32_t a, uint32_t b) {
                    // No overlap
                    if (b <= Kd || a >= Kd + F) {
                        zeroRangeVec(out, a, b, globalTid, stride);
                        return;
                    }
                    // Overlap on the left side: zero [a, Kd)
                    if (a < Kd) zeroRangeVec(out, a, Kd, globalTid, stride);
                    // Overlap on the right side: zero [Kd+F, b)
                    if (b > Kd + F) zeroRangeVec(out, Kd + F, b, globalTid, stride);
                    // If [a,b) entirely inside [Kd, Kd+F) : no action needed
                };

                // Our "rest" interval is a physical circular interval of length `len` starting at `start`
                // Convert it to up to two non-wrapping pieces, then subtract the filler once on each piece.
                if (start + len <= Ncb) {
                    // No wrap: [start, start+len)
                    zeroSpanExcludingFillerGap(start, start + len);
                } else {
                    // Wraps: [start, Ncb) ∪ [0, (start+len - Ncb))
                    zeroSpanExcludingFillerGap(start, Ncb);
                    zeroSpanExcludingFillerGap(0, (start + len) - Ncb);
                }
            }

            // 1b. Write filler bits to circular buffer
            for(uint32_t n = Kd + globalTid; n < Kd + F; n += stride)
            {
                // Note: Location of Filler bits is fixed to tail end of systematic bit section of
                // circular buffer and is independent of k0
                uint32_t circBufIdx              = n;
                out[circBufIdx] = LLR_CLAMP_MAX;
            }

            // 2. Initialization of the rest of output buffer: section of length (Ncb_padded - Ncb)
            // 2a. Write zeros into punctured bits (first 2*Zc bits of Ncb_padded output buffer), also out address includes nPuncturedBits offset
            zeroRangeVec(out - nPuncturedBits, 0, nPuncturedBits, globalTid, stride);

            //FixMe the following block results in mismatches for PUSCH derate match standalone unit test, additionally the for loop needs to include tid; disable for now
            // 2b. Write zeros into byte padding section of Ncb_padded output buffer
//            for(uint32_t n = Ncb; n < Ncb_padded; n += stride)
//            {
//                out[n] = 0;
//            }
        }
    }
}

template <typename T_IN, typename T_OUT>
__global__ void __launch_bounds__(96, 10) de_rate_matching_global2(puschRxRateMatchDescr_t* pRmDesc)
{
    // PUSCH kernel descriptor
    puschRxRateMatchDescr_t& rmDesc = *pRmDesc;
    uint32_t tbIdx = blockIdx.z;
    uint16_t ueIdx = rmDesc.schUserIdxs[tbIdx];

    // Array of transport block parameters structs
    const PerTbParams& tbPrms = rmDesc.tbPrmsArray[ueIdx];
    // QAM modulation index
    uint32_t Qm = tbPrms.Qm;
    if(Qm == 1)
    {
        deRateMatchingKernelInner<T_IN, T_OUT, 1>(pRmDesc);
    }
    else if(Qm == 2)
    {
        deRateMatchingKernelInner<T_IN, T_OUT, 2>(pRmDesc);
    }
    else if(Qm == 4)
    {
        deRateMatchingKernelInner<T_IN, T_OUT, 4>(pRmDesc);
    }
    else if(Qm == 6)
    {
        deRateMatchingKernelInner<T_IN, T_OUT, 6>(pRmDesc);
    }
    else if(Qm == 8)
    {
        deRateMatchingKernelInner<T_IN, T_OUT, 8>(pRmDesc);
    }
}

template <typename T_OUT>
__global__ void __launch_bounds__(96, 10) de_rate_matching_reset_buffer(puschRxRateMatchDescr_t* pRmDesc)
{
    // PUSCH kernel descriptor
    puschRxRateMatchDescr_t& rmDesc = *pRmDesc;
    const uint32_t nFracCbs = gridDim.x;
    const uint32_t fracCbIdx = blockIdx.x;
    const uint32_t cbIdx = blockIdx.y;
    const uint32_t tbIdx = blockIdx.z;
    const uint16_t ueIdx = rmDesc.schUserIdxs[tbIdx];

    // Output tensor
    // @todo: rmDesc.out which holds an array of pointers to HARQ buffers (in GPU memory) lives in host pinned memory,
    // check performance impact of accessing this memory
    T_OUT* out = static_cast<T_OUT*>(rmDesc.out[ueIdx]);

    // Array of transport block parameters structs
    const PerTbParams& tbPrms = rmDesc.tbPrmsArray[ueIdx];
    // QAM modulation index
    const uint32_t Qm = tbPrms.Qm;
    // code block index
    uint32_t r = cbIdx + tbPrms.firstCodeBlockIndex;

    // Output code block stride
    uint32_t Ncb_padded = tbPrms.Ncb_padded;
    uint32_t cbStartOffset = r * Ncb_padded;

    // Adjust for codeblock offset
    out += cbStartOffset;

    //******** The following parameters are invariant for all CTAs working on the same transport block*******/
    // They only vary along the y-dimension of the grid, namely across transport blocks

    // Output de-rate matched code block size excluding punctured bits
    uint32_t Ncb = tbPrms.Ncb;
    // number of code blocks in transport block
    uint32_t C = tbPrms.num_CBs;
    // base graph index
    uint32_t bg = tbPrms.bg;
    // redundancy version
    uint32_t rv = tbPrms.rv;
    // new data indicator
    uint32_t ndi = tbPrms.ndi;

    // lifting factor
    uint32_t Zc = tbPrms.Zc;
    // Number of UE layers (Number of layers occupied by transport block tbIdx)
    uint32_t Nl = tbPrms.Nl;

    if (!ndi) return; //early exit, as there is no need to reset any part of the HARQ buffer
    /************/

    if(r < C)
    {
        // Determine input rate matched block size E and start index codeBlockQAMStartIndex

        //******** The following parameters are invariant for all CTAs working on the same transport block*******/

        // Size (number of LLRs) of input rate-matched code block r
        uint32_t E;
        // Number of layers times modulation index: determines how many LLRs are read from each block of NBBULayers
        uint32_t TBLLRsPerNBBULayers = Nl * Qm;
        // total number of LLRs to be read for current transport block
        uint32_t totalNLLRsForTB = TBLLRsPerNBBULayers * C;

        // encodedSize is size (number of LLRs) of current transport block; q1 is number of NBBULayers blocks the transport block is spread over
        uint32_t q1 = tbPrms.uciOnPuschFlag ? tbPrms.G / TBLLRsPerNBBULayers : tbPrms.encodedSize / TBLLRsPerNBBULayers; // exact division
        // number of NBBULayers blocks each code block is spread over
        uint32_t q = q1 / C;

        // This is straight from the spec: compute size E of each code block of current transport block
        uint32_t rr = C - (q1 - q * C) - 1;
        // smaller code blocks size
        uint32_t El = Nl * Qm * q;
        // larger code block size
        //uint32_t Eh = Nl * Qm * ((Ncb + totalNLLRsForTB - 1) / totalNLLRsForTB);
        uint32_t Eh = El + TBLLRsPerNBBULayers * (q * totalNLLRsForTB < tbPrms.encodedSize);

        E = (r <= rr) ? El : Eh;

        // For incremental redundancy transmission: determine k0 based on rv and bg(base graph)
        // Uninitialized scalar variable this changes will treat rv 0 (detected if k0=0): no LLR combining, just write to memory; write
        uint32_t k0 = k0_from_bg_rv(bg, rv, Zc, Ncb);

        //====================================================================================================================================================

        // Number of filler bits
        uint32_t F              = tbPrms.F;
        uint32_t nPuncturedBits = 2 * Zc;

        // Within the output buffer, the Ncb circular buffer starts at offset 2*Zc (punctured bits)
        out += nPuncturedBits;

        // detect possibility of more than one thread accessing the same outIdx
        int potentialRaceIfPositive = (E + 2 * F + k0) - Ncb; // if potentialRaceIfPositive > 0, more than one thread can write on the same outIdx

        if (potentialRaceIfPositive > 0 && ndi)
        {
            int maxOutIdx = Ncb;
            potentialRaceIfPositive = min(potentialRaceIfPositive, maxOutIdx);
            // Use all thread blocks associated with this CB
            uint32_t stride    = nFracCbs * blockDim.x;
            uint32_t globalTid = (fracCbIdx * blockDim.x) + threadIdx.x;
            zeroRangeVec(out, 0, potentialRaceIfPositive, globalTid, stride);
        }
    }
}

void puschRxRateMatch::setup(uint16_t                          nSchUes,                     // number of users with sch data
                             uint16_t*                         pSchUserIdxsCpu,             // indices of users with SCH data
                             const PerTbParams*                pTbPrmsCpu,                  // starting address of transport block parameters (CPU)
                             const PerTbParams*                pTbPrmsGpu,                  // starting address of transport block parameters (GPU)
                             cuphyTensorPrm_t*                 pTPrmRmIn,                   // starting address of input LLR tensor parameters
                             cuphyTensorPrm_t*                 pTPrmCdm1RmIn,
                             void**                            ppRmOut,                     // array of rm outputs (GPU)
                             void*                             pCpuDesc,                    // pointer to descriptor in cpu
                             void*                             pGpuDesc,                    // pointer to descriptor in gpu
                             uint8_t                           enableCpuToGpuDescrAsyncCpy, // option to copy cpu descriptors from cpu to gpu
                             cuphyPuschRxRateMatchLaunchCfg_t* pLaunchCfg,                  // pointer to rate matching launch configuration
                             cudaStream_t                      strm)                        // stream to perform copy
{
    // setup CPU descriptor
    puschRxRateMatchDescr_t& desc    = *(static_cast<puschRxRateMatchDescr_t*>(pCpuDesc));
    uint16_t                 nUciUes = 0;

    for(uint32_t i = 0; i < nSchUes; ++i)
    {
        uint16_t ueIdx      = pSchUserIdxsCpu[i];
        desc.schUserIdxs[i] = ueIdx;
        if(pTbPrmsCpu[ueIdx].uciOnPuschFlag)
        {
            desc.llr_vec_in[i] = pTbPrmsCpu[ueIdx].d_schAndCsi2LLRs;
            nUciUes++;
        }
        else
        {
            uint32_t ueGrpIdx  = pTbPrmsCpu[ueIdx].userGroupIndex;
            if(pTbPrmsCpu[ueIdx].nDmrsCdmGrpsNoData==1)
            {
                desc.llr_vec_in[i] = pTPrmCdm1RmIn[ueGrpIdx].pAddr;
            }
            else
            {
                desc.llr_vec_in[i] = pTPrmRmIn[ueGrpIdx].pAddr;
            }
        }
    }
    desc.out            = ppRmOut;
    desc.tbPrmsArray    = pTbPrmsGpu;
    desc.descramblingOn = m_descramblingOn;

    // optional CPU->GPU copy
    if(enableCpuToGpuDescrAsyncCpy)
    {
        // added Unchecked return value
        CUDA_CHECK(cudaMemcpyAsync(pGpuDesc, pCpuDesc, sizeof(puschRxRateMatchDescr_t), cudaMemcpyHostToDevice, strm));
    }

    // Setup Launch Geometry
    uint32_t EMax = 0; // max number of encoded bits per CB
    uint32_t CMax = 0; // max number of CBs per TB
    for(uint32_t i = 0; i < nSchUes; ++i)
    {
        uint16_t ueIdx = pSchUserIdxsCpu[i];
        CMax           = CMax < pTbPrmsCpu[ueIdx].num_CBs ? pTbPrmsCpu[ueIdx].num_CBs : CMax;
        uint32_t Eh    = pTbPrmsCpu[ueIdx].Nl * pTbPrmsCpu[ueIdx].Qm * ceilf(float(pTbPrmsCpu[ueIdx].encodedSize) / float(pTbPrmsCpu[ueIdx].Nl * pTbPrmsCpu[ueIdx].Qm * pTbPrmsCpu[ueIdx].num_CBs));
        EMax           = EMax < Eh ? Eh : EMax;
    }

    // using larger block size could result in load imbalance in some cases and lower occupancy
    constexpr uint32_t threadBlkDim = 96;

    dim3 gridDim(div_round_up(EMax, threadBlkDim * NUM_LLRS_PROCESSED_PER_THRD), CMax, nSchUes);
    dim3 blockDim(threadBlkDim, 1, 1);
    // printf("gridDim(%d %d %d) blockDim(%d %d %d)\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    pLaunchCfg->desc                                  = pGpuDesc;
    pLaunchCfg->kernelArgs[0]                         = &(pLaunchCfg->desc);
    
    // Configure main de_rate_matching_global2 kernel
    pLaunchCfg->kernelNodeParamsDriver.gridDimX       = gridDim.x;
    pLaunchCfg->kernelNodeParamsDriver.gridDimY       = gridDim.y;
    pLaunchCfg->kernelNodeParamsDriver.gridDimZ       = gridDim.z;
    pLaunchCfg->kernelNodeParamsDriver.blockDimX      = blockDim.x;
    pLaunchCfg->kernelNodeParamsDriver.blockDimY      = blockDim.y;
    pLaunchCfg->kernelNodeParamsDriver.blockDimZ      = blockDim.z;
    pLaunchCfg->kernelNodeParamsDriver.func           = m_kernelFunc;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams   = &(pLaunchCfg->kernelArgs[0]);
    pLaunchCfg->kernelNodeParamsDriver.sharedMemBytes = 0;
    pLaunchCfg->kernelNodeParamsDriver.extra          = nullptr;
    
    // Configure reset buffer kernel (same grid/block dimensions)
    pLaunchCfg->resetKernelNodeParamsDriver.gridDimX       = gridDim.x;
    pLaunchCfg->resetKernelNodeParamsDriver.gridDimY       = gridDim.y;
    pLaunchCfg->resetKernelNodeParamsDriver.gridDimZ       = gridDim.z;
    pLaunchCfg->resetKernelNodeParamsDriver.blockDimX      = blockDim.x;
    pLaunchCfg->resetKernelNodeParamsDriver.blockDimY      = blockDim.y;
    pLaunchCfg->resetKernelNodeParamsDriver.blockDimZ      = blockDim.z;
    pLaunchCfg->resetKernelNodeParamsDriver.func           = m_resetBufferKernelFunc;
    pLaunchCfg->resetKernelNodeParamsDriver.kernelParams   = &(pLaunchCfg->kernelArgs[0]);
    pLaunchCfg->resetKernelNodeParamsDriver.sharedMemBytes = 0;
    pLaunchCfg->resetKernelNodeParamsDriver.extra          = nullptr;
}

void puschRxRateMatch::init(int rmFPconfig,     // 0: FP32 in, FP32 out; 1: FP16 in, FP32 out; 2: FP32 in, FP16 out; 3: FP16 in, FP16 out; other values: don't run
                            int descramblingOn) // enable/disable descrambling
{
    // Save configurations
    m_descramblingOn = descramblingOn;
    m_rmFPconfig = rmFPconfig;

    // Select Main Kernel and Reset Buffer Kernel
    switch(rmFPconfig)
    {
    case 0:
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_kernelFunc, reinterpret_cast<void*>(de_rate_matching_global2<float, float>)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_resetBufferKernelFunc, reinterpret_cast<void*>(de_rate_matching_reset_buffer<float>)));}
        break;

    case 1:
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_kernelFunc, reinterpret_cast<void*>(de_rate_matching_global2<__half, float>)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_resetBufferKernelFunc, reinterpret_cast<void*>(de_rate_matching_reset_buffer<float>)));}
        break;

    case 2:
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_kernelFunc, reinterpret_cast<void*>(de_rate_matching_global2<float, __half>)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_resetBufferKernelFunc, reinterpret_cast<void*>(de_rate_matching_reset_buffer<__half>)));}
        break;

    case 3:
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_kernelFunc, reinterpret_cast<void*>(de_rate_matching_global2<__half, __half>)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_resetBufferKernelFunc, reinterpret_cast<void*>(de_rate_matching_reset_buffer<__half>)));}
        break;
    default:
        break;
    }
}

void puschRxRateMatch::getDescrInfo(size_t& descrSizeBytes, size_t& descrAlignBytes)
{
    descrSizeBytes  = sizeof(puschRxRateMatchDescr_t);
    descrAlignBytes = alignof(puschRxRateMatchDescr_t);
}
