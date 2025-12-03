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

#include "cuComplex.h"
#include "cuda_fp16.h"
#include "cuphy_kernel_util.cuh"

namespace channel_eq {

//=============================================================================
// Constants and Configuration
//=============================================================================

static constexpr uint32_t CUDA_MAX_N_THRDS_PER_BLK = 1024;
static constexpr uint32_t N_THREADS_PER_WARP = 32; // cudaDeviceProp::warpSize;

// FP16 - largest normal number
static constexpr float LLR_LOW_LIM  = -65504.0f; // std::numeric_limits<__half>::lowest();
static constexpr float LLR_HIGH_LIM =  65504.0f; // std::numeric_limits<__half>::max();

// Inverse of zero-forcing regularizer. Equivalent to diagonal MMSE with SNR = 10^(3.6)
static constexpr float INV_ZF_REGULARIZER = 3981.071705534973f;

//=============================================================================
// Debug Functions
//=============================================================================
//#define CUPHY_DEBUG 1

#if CUPHY_DEBUG
__device__ float debug_LLR_get_elem(const float4& Llr, int idx)
{
    switch(idx)
    {
    default:
    case 0: return Llr.x;
    case 1: return Llr.y;
    case 2: return Llr.z;
    case 3: return Llr.w;
    }
}

__device__ float debug_LLR_get_elem(const float2& Llr, int idx)
{
    cuphy_i::word_t w0, w1;
    w0.f32 = Llr.x;
    w1.f32 = Llr.y;
    switch(idx)
    {
    default:
    case 0: return __low2float(w0.f16x2);
    case 1: return __high2float(w0.f16x2);
    case 2: return __low2float(w1.f16x2);
    case 3: return __high2float(w1.f16x2);
    }
}
#endif

//=============================================================================
// Tensor Reference Template
//=============================================================================

template <typename TElem>
struct tensor_ref
{
    TElem*     addr{};
    const int32_t* strides{};

    CUDA_BOTH
    tensor_ref(void* pAddr, const int32_t* pStrides) :
        addr(static_cast<TElem*>(pAddr)),
        strides(pStrides)
    {
    }
    
    CUDA_BOTH int offset(int i0) const
    {
        return (strides[0] * i0);
    }
    
    CUDA_BOTH int offset(int i0, int i1) const
    {
        return (strides[0] * i0) + (strides[1] * i1);
    }
    
    CUDA_BOTH int offset(int i0, int i1, int i2) const
    {
        return (strides[0] * i0) + (strides[1] * i1) + (strides[2] * i2);
    };
    
    CUDA_BOTH int offset(int i0, int i1, int i2, int i3) const
    {
        return (strides[0] * i0) + (strides[1] * i1) + (strides[2] * i2) + (strides[3] * i3);
    };
    
    CUDA_BOTH int offset(int i0, int i1, int i2, int i3, int i4) const
    {
        return (strides[0] * i0) + (strides[1] * i1) + (strides[2] * i2) + (strides[3] * i3) + (strides[4] * i4);
    };
    
    // clang-format off
    CUDA_BOTH TElem&       operator()(int i0)                                   { return *(addr + offset(i0));         }
    CUDA_BOTH TElem&       operator()(int i0, int i1)                           { return *(addr + offset(i0, i1));     }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2)                   { return *(addr + offset(i0, i1, i2)); }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3)           { return *(addr + offset(i0, i1, i2, i3)); }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3, int i4)   { return *(addr + offset(i0, i1, i2, i3, i4)); }

    CUDA_BOTH const TElem& operator()(int i0) const                                 { return *(addr + offset(i0));         }
    CUDA_BOTH const TElem& operator()(int i0, int i1) const                         { return *(addr + offset(i0, i1));     }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const                 { return *(addr + offset(i0, i1, i2)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const         { return *(addr + offset(i0, i1, i2, i3)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3, int i4) const { return *(addr + offset(i0, i1, i2, i3, i4)); }

    // clang-format on
};

//=============================================================================
// Block Templates 
//=============================================================================

template <typename T, int M>
struct block_1D
{
    T         data[M]{};
    CUDA_BOTH T& operator()(int idx) { return data[idx]; }
};

template <typename T, int M, int N>
struct block_2D
{
    T         data[M * N]{};
    CUDA_BOTH T& operator()(int m, int n) { return data[(n * M) + m]; }
};

template <typename T, int L, int M, int N>
struct block_3D
{
    T         data[L * M * N]{};
    CUDA_BOTH T& operator()(int l, int m, int n) { return data[((n * M) + m) * L + l]; }
};

// Partial specialization of block_1D to use shared memory pointers
template <typename T, int M>
struct block_1D<T*, M>
{
    CUDA_BOTH block_1D(T* pData) :
        m_pData(pData) {}; // static_assert(std::is_pointer<T>::value, "Must be a pointer type")
    block_1D()                    = delete;
    block_1D(block_1D const& blk) = delete;
    CUDA_BOTH block_1D& operator  =(block_1D const& block) { m_pData = block.m_pData; };
    ~block_1D()                   = default;

    CUDA_BOTH T&               operator()(int idx) { return m_pData[idx]; }
    static constexpr CUDA_BOTH size_t num_elem() { return M; }

private:
    T* m_pData = nullptr;
};

// Partial specialization of block_2D to use shared memory pointers
template <typename T, int M, int N>
struct block_2D<T*, M, N>
{
    CUDA_BOTH block_2D(T* pData) :
        m_pData(pData){};
    block_2D()                    = delete;
    block_2D(block_2D const& blk) = delete;
    CUDA_BOTH block_2D& operator  =(block_2D const& block) { m_pData = block.m_pData; };
    ~block_2D()                   = default;

    CUDA_BOTH T&               operator()(int m, int n) { return m_pData[(n * M) + m]; }
    static constexpr CUDA_BOTH size_t num_elem() { return M * N; }

private:
    T* m_pData = nullptr;
};

// Partial specialization of block_3D to use shared memory pointers
template <typename T, int L, int M, int N>
struct block_3D<T*, L, M, N>
{
    CUDA_BOTH block_3D(T* pData) :
        m_pData(pData){};
    block_3D()                    = delete;
    block_3D(block_3D const& blk) = delete;
    CUDA_BOTH block_3D& operator  =(block_3D const& block) { m_pData = block.m_pData; };
    ~block_3D()                   = default;

    CUDA_BOTH T&               operator()(int l, int m, int n) { return m_pData[((n * M) + m) * L + l]; }
    static constexpr CUDA_BOTH size_t num_elem() { return L * M * N; }

private:
    T* m_pData = nullptr;
};

//=============================================================================
// Type Conversion Templates
//=============================================================================

// clang-format off
template <typename T> CUDA_BOTH_INLINE T         cuGet(int);
template<>            CUDA_BOTH_INLINE float     cuGet(int x) { return(float(x)); }
template<>            CUDA_BOTH_INLINE __half    cuGet(int x) { return(__half(x)); }
template<>            CUDA_BOTH_INLINE cuComplex cuGet(int x) { return(make_cuComplex(float(x), 0.0f)); }
template<>            CUDA_BOTH_INLINE __half2   cuGet(int x) { return(make_half2(__half(x), 0.0));}

template <typename T> CUDA_BOTH_INLINE T         cuGet(float);
template<>            CUDA_BOTH_INLINE float     cuGet(float x) { return(float(x)); }
template<>            CUDA_BOTH_INLINE cuComplex cuGet(float x) { return(make_cuComplex(x, 0.0f)); }
template<>            CUDA_BOTH_INLINE __half2   cuGet(float x) { return(make_half2(x, 0.0f)); }

template <typename T> CUDA_BOTH_INLINE T         cuGet(__half);
template<>            CUDA_BOTH_INLINE __half    cuGet(__half x) { return x; }
template<>            CUDA_BOTH_INLINE __half2   cuGet(__half x) { return(make_half2(x, 0.0f)); }

template <typename T> CUDA_BOTH_INLINE T         cuAbs(T);
template<>            CUDA_BOTH_INLINE float     cuAbs(float x) { return(fabsf(x)); }

//=============================================================================
// Complex Number Operations
//=============================================================================

static CUDA_BOTH_INLINE float     cuReal(cuComplex x)                          { return cuCrealf(x); }
static CUDA_BOTH_INLINE float     cuImag(cuComplex x)                          { return cuCimagf(x); }
static CUDA_BOTH_INLINE cuComplex cuConj(cuComplex x)                          { return cuConjf(x); }
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, float y)              { return make_cuComplex(cuCrealf(x) * y, cuCimagf(x) * y); }
static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, float y)            { x = make_cuComplex(cuCrealf(x) + y, cuCimagf(x)); return x; }
static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex &x, float y)            { x = make_cuComplex(cuCrealf(x) * y, cuCimagf(x) * y); return x; }
static CUDA_BOTH_INLINE cuComplex cuCma(cuComplex x, cuComplex y, cuComplex a) { return cuCfmaf(x, y,a); } // a = (x*y) + a;
static CUDA_BOTH_INLINE cuComplex cuCmul(cuComplex x, cuComplex y)             { return cuCmulf(x, y); } // complex mul
static CUDA_BOTH_INLINE cuComplex operator+(cuComplex x, cuComplex y)          { return cuCaddf(x, y); }
static CUDA_BOTH_INLINE cuComplex operator-(cuComplex x, cuComplex y)          { return cuCsubf(x, y); }
static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, cuComplex y)        { x = cuCaddf(x, y); return x; };

// Arithmetic functions for half-precision complex
static CUDA_INLINE __half  cuReal(__half2 x)                        { return x.x; }
static CUDA_INLINE __half  cuImag(__half2 x)                        { return x.y; }
static CUDA_INLINE __half2 cuConj(__half2 x)                        { return conj_fast(x); }
static CUDA_INLINE __half2 operator*(__half2 x, __half y)           { return __hmul2(x, make_half2(y, y)); }
static CUDA_INLINE __half2 operator+=(__half2 &x, __half y)         { x = make_half2(cuReal(x) + y, cuImag(x)); return x; }
static CUDA_INLINE __half2 operator*=(__half2 &x, __half y)         { x = __hmul2(x, make_half2(y, y)); return x; }
static CUDA_INLINE __half2 cuCma(__half2 x, __half2 y, __half2 a)   { return __hcmadd(x,y,a); } // acc = (x*y) + a;
static CUDA_INLINE __half2 cuCmul(__half2 x, __half2 y)             { return __hcmadd(x,y, __float2half2_rn(0.f)); } // complex mul

// clang-format on

//=============================================================================
// QAM Tag Mapping Templates
//=============================================================================

// Use tag dispatching to invoke different behaviours (different functions) for each QAM
template <QAM_t>
struct QAMEnumToTagMap;

// Only even QAMs supported by 3GPP
template <>
struct QAMEnumToTagMap<QAM_t::QAM_4>
{
    struct QAM4Tag {};
};

template <>
struct QAMEnumToTagMap<QAM_t::QAM_16>
{
    struct QAM16Tag {};
};

template <>
struct QAMEnumToTagMap<QAM_t::QAM_64>
{
    struct QAM64Tag {};
};

template <>
struct QAMEnumToTagMap<QAM_t::QAM_256>
{
    struct QAM256Tag {};
};

} // namespace channel_eq
