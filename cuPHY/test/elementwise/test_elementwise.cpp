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

#include <gtest/gtest.h>
#include "cuphy.hpp"

namespace
{

////////////////////////////////////////////////////////////////////////
// magnitude_squared()
// Returns the magnitude squared for a complex value
template <typename T> double magnitude_squared(T& a)
{
    return (a.x * a.x) + (a.y * a.y);
}

////////////////////////////////////////////////////////////////////////
// host_add
// Template function for host addition of real values
template <typename T> struct host_add { static T operate(T a, T b) { return (a + b); } };

// Disable narrowing warnings here, since we are using inputs with a
// limited range.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
////////////////////////////////////////////////////////////////////////
// host_complex_add
// Template function for host addition of complex values
template <typename T> struct host_complex_add { static T operate(T a, T b) { return T{a.x + b.x, a.y + b.y}; } };
#pragma GCC diagnostic pop

////////////////////////////////////////////////////////////////////////
// host_xor
struct host_xor { static uint32_t operate(uint32_t a, uint32_t b) { return (a ^ b); } };

////////////////////////////////////////////////////////////////////////
// host_cmp_eq
// Host functor to compare real values
template <typename T> struct host_cmp_eq
{
    static void cmp(T a, T b)
    {
        EXPECT_EQ(a, b);
    }
};

////////////////////////////////////////////////////////////////////////
// host_cmp_abs_error
// Host functor to compare real values
template <typename T> struct host_cmp_abs_error
{
    template <typename TTol>
    static void cmp(T a, T b, TTol tol)
    {
        EXPECT_LE(std::abs(static_cast<double>(a) - static_cast<double>(b)),
                  tol);
    }
};
////////////////////////////////////////////////////////////////////////
// host_cmp_mag_squared
// Host functor to compare the magnitude squared of complex values
template <typename T> struct host_cmp_mag_squared
{
    template <typename TTol>
    static void cmp(T a, T b, TTol tol)
    {
        TTol mag_a = magnitude_squared(a);
        TTol mag_b = magnitude_squared(b);
        //printf("mag_a = %f, mag_b = %f\n", mag_a, mag_b);
        EXPECT_LT(std::abs(mag_a - mag_b), tol);
    }
};

////////////////////////////////////////////////////////////////////////
// range_gen_bits
// Range generator for bit tensors. (Values are ignored by the library)
struct range_gen_bits
{
    static int min() { return 0; }
    static int max() { return 1; }
};

////////////////////////////////////////////////////////////////////////
// range_gen_real
// Uniform distribution range generator for real values
struct range_gen_real
{
    static int min() { return 1; }
    static int max() { return 10; }
};

////////////////////////////////////////////////////////////////////////
// range_gen_complex
// Uniform distribution range generator for complex values
struct range_gen_complex
{
    static int2 min() { return int2{1, 1}; }
    static int2 max() { return int2{10, 10}; }
};

////////////////////////////////////////////////////////////////////////
// tensor_op_add
// Calls dst.add(A, B)
struct tensor_op_add
{
    template <class TDst, class TSrc>
    static void do_operation(TDst& dst, TSrc& srcA, TSrc& srcB)
    {
        dst.add(srcA, srcB);
    }
};

////////////////////////////////////////////////////////////////////////
// tensor_op_xor
// Calls dst.xor_op(A, B)
struct tensor_op_xor
{
    template <class TDst, class TSrcA, class TSrcB>
    static void do_operation(TDst& dst, TSrcA& srcA, TSrcB& srcB)
    {
        cuphy::tensor_xor(dst, srcA, srcB);
    }
};

////////////////////////////////////////////////////////////////////////
// do_elementwise_test()
template <cuphyDataType_t           TType,
          class                     TRangeGen,
          class                     TOp,
          template <typename> class THostOp,
          template <typename> class THostCompare>
void do_elementwise_test(int NUM_ROWS, int NUM_COLS)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    typedef typename tensor_p::element_t                    element_t;
    typedef THostOp<element_t>                              host_op_t;
    typedef THostCompare<element_t>                         host_cmp_t;
    typedef TRangeGen                                       range_gen_t;
    //------------------------------------------------------------------
    // Allocate source and destination tensors
    tensor_p                 tSrcA(NUM_ROWS, NUM_COLS);
    tensor_p                 tSrcB(NUM_ROWS, NUM_COLS);
    tensor_p                 tDst(NUM_ROWS, NUM_COLS);
    //------------------------------------------------------------------
    // Initialize the source tensors with random values
    cuphy::rng rng;
    rng.uniform(tSrcA, range_gen_t::min(), range_gen_t::max());
    rng.uniform(tSrcB, range_gen_t::min(), range_gen_t::max());
    //------------------------------------------------------------------
    // Perform the requested operation
    TOp::do_operation(tDst, tSrcA, tSrcB);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare results
    for(int i = 0; i < NUM_ROWS; ++i)
    {
        for(int j = 0; j < NUM_COLS; ++j)
        {
            host_cmp_t::cmp(tDst(i, j),
                            host_op_t::operate(tSrcA(i, j), tSrcB(i, j)),
                            0.01f);
        }
    }
}

////////////////////////////////////////////////////////////////////////
// do_elementwise_bit_test()
template <class TOp,
          class THostOp>
void do_elementwise_bit_test(int NUM_ROWS, int NUM_COLS)
{
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_p;
    typedef THostOp                                             host_op_t;
    typedef host_cmp_eq<uint32_t>                               host_cmp_t;
    //------------------------------------------------------------------
    // Allocate source and destination tensors
    const std::array<int, 2> SRC_DIMS = {{NUM_ROWS, NUM_COLS}};
    tensor_p                 tSrcA(NUM_ROWS, NUM_COLS);
    tensor_p                 tSrcB(NUM_ROWS, NUM_COLS);
    tensor_p                 tDst(NUM_ROWS, NUM_COLS);
    //------------------------------------------------------------------
    // Initialize the source tensors with random values
    cuphy::rng rng;
    rng.uniform(tSrcA, 0, 1);
    rng.uniform(tSrcB, 0, 1);
    //------------------------------------------------------------------
    // Perform the requested operation
    TOp::do_operation(tDst, tSrcA, tSrcB);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare results
    int NUM_WORDS = (NUM_ROWS + 31) / 32;
    for(int i = 0; i < NUM_WORDS; ++i)
    {
        for(int j = 0; j < SRC_DIMS[1]; ++j)
        {
            host_cmp_t::cmp(tDst(i, j),
                            host_op_t::operate(tSrcA(i, j), tSrcB(i, j)));
            //printf("[%i, %i]: A: 0x%X, B: 0x%X, Output: 0x%X\n", i, j, tSrcA(i, j), tSrcB(i, j), tDst(i, j));
        }
    }
}

////////////////////////////////////////////////////////////////////////
// do_elementwise_bit_partial_word_test()
// Test for correct handling of elementwise operations at the end of
// columns for tensors of type CUPHY_BIT.
template <class TOp,
          class THostOp>
void do_elementwise_bit_partial_word_test(int NUM_ROWS_FULL,
                                          int NUM_ROWS_PARTIAL,
                                          int NUM_COLS)
{
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_p;
    typedef tensor_p::tensor_ref_t                              tensor_ref_p;
    typedef THostOp                                             host_op_t;
    typedef host_cmp_eq<uint32_t>                               host_cmp_t;
    //------------------------------------------------------------------
    // Allocate source and destination tensors
    tensor_p                 tSrcA(NUM_ROWS_FULL,    NUM_COLS);
    tensor_p                 tSrcB(NUM_ROWS_PARTIAL, NUM_COLS);
    tensor_p                 tDst(NUM_ROWS_PARTIAL,  NUM_COLS);
    //------------------------------------------------------------------
    // Initialize the source tensors with random values
    cuphy::rng rng;
    rng.uniform(tSrcA, 0, 1);
    rng.uniform(tSrcB, 0, 1);
    //------------------------------------------------------------------
    // Generate a tensor ref for a subset of tensor A. We want the
    // subset to contain a partial word at the end of the column.
    cuphy::index_group grp(cuphy::index_range(0, NUM_ROWS_PARTIAL),
                           cuphy::dim_all());
    tensor_ref_p       tSrcA_s = tSrcA.subset(grp);
    //------------------------------------------------------------------
    // Perform the requested operation
    TOp::do_operation(tDst, tSrcA_s, tSrcB);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare results
    int NUM_WORDS = (NUM_ROWS_PARTIAL + 31) / 32;
    for(int i = 0; i < NUM_WORDS; ++i)
    {
        uint32_t  A_mask    = 0xFFFFFFFF;
        const int ROW_COUNT = (i+1) * 32;
        if(ROW_COUNT > NUM_ROWS_PARTIAL)
        {
            const int NUM_BITS = NUM_ROWS_PARTIAL - (i * 32);
            A_mask = (1 << NUM_BITS) - 1;
        }
        for(int j = 0; j < NUM_COLS; ++j)
        {
            uint32_t A = tSrcA(i, j) & A_mask;
            uint32_t B = tSrcB(i, j);
            host_cmp_t::cmp(tDst(i, j),
                            host_op_t::operate(A, B));
            //printf("[%i, %i]: A: 0x%X, mask: 0x%X, B: 0x%X, Ref: 0x%X, Output: 0x%X\n",
            //       i,
            //       j,
            //       A,
            //       A_mask,
            //       B,
            //       host_op_t::operate(A, B),
            //       tDst(i, j));
        }
    }
}

////////////////////////////////////////////////////////////////////////
// do_elementwise_broadcast_test()
template <cuphyDataType_t           TType,
          class                     TRangeGen,
          class                     TOp,
          template <typename> class THostOp,
          template <typename> class THostCompare>
void do_elementwise_broadcast_test(int NUM_ROWS, int NUM_COLS)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    typedef typename tensor_p::element_t                    element_t;
    typedef THostOp<element_t>                              host_op_t;
    typedef THostCompare<element_t>                         host_cmp_t;
    typedef TRangeGen                                       range_gen_t;
    //------------------------------------------------------------------
    // Allocate source and destination tensors
    tensor_p                 tSrcA(NUM_ROWS, NUM_COLS);
    tensor_p                 tSrcB(NUM_ROWS);
    tensor_p                 tDst(NUM_ROWS, NUM_COLS);
    //------------------------------------------------------------------
    // Initialize the source tensors with random values
    cuphy::rng rng;
    rng.uniform(tSrcA, range_gen_t::min(), range_gen_t::max());
    rng.uniform(tSrcB, range_gen_t::min(), range_gen_t::max());
    //------------------------------------------------------------------
    // Perform the requested operation
    TOp::do_operation(tDst, tSrcA, tSrcB);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare results
    for(int i = 0; i < NUM_ROWS; ++i)
    {
        for(int j = 0; j < NUM_COLS; ++j)
        {
            host_cmp_t::cmp(tDst(i, j),
                            host_op_t::operate(tSrcA(i, j), tSrcB(i)),
                            0.01f);
            //printf("[%i, %i]: A: %f, B: %f, Output: %f, Compare: %f\n",
            //       i,
            //       j,
            //       tSrcA(i, j),
            //       tSrcB(i),
            //       tDst(i, j),
            //       host_op_t::operate(tSrcA(i, j), tSrcB(i)));
        }
    }
}

////////////////////////////////////////////////////////////////////////
// do_elementwise_broadcast_bit_test()
template <class TOp,
          class THostOp>
void do_elementwise_broadcast_bit_test(int NUM_ROWS, int NUM_COLS)
{
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_p;
    typedef THostOp                                             host_op_t;
    typedef host_cmp_eq<uint32_t>                               host_cmp_t;
    //------------------------------------------------------------------
    // Allocate source and destination tensors
    tensor_p                 tSrcA(NUM_ROWS, NUM_COLS);
    tensor_p                 tSrcB(NUM_ROWS);
    tensor_p                 tDst(NUM_ROWS, NUM_COLS);
    //------------------------------------------------------------------
    // Initialize the source tensors with random values
    cuphy::rng rng;
    rng.uniform(tSrcA, 0, 1);
    rng.uniform(tSrcB, 0, 1);
    //------------------------------------------------------------------
    // Perform the requested operation
    TOp::do_operation(tDst, tSrcA, tSrcB);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare results
    int NUM_WORDS = (NUM_ROWS + 31) / 32;
    for(int i = 0; i < NUM_WORDS; ++i)
    {
        for(int j = 0; j < NUM_COLS; ++j)
        {
            host_cmp_t::cmp(tDst(i, j),
                            host_op_t::operate(tSrcA(i, j), tSrcB(i)));
            //printf("[%i, %i]: A: 0x%X, B: 0x%X, Output: 0x%X\n", i, j, tSrcA(i, j), tSrcB(i), tDst(i, j));
        }
    }
}

// Helper function to create variant based on type
template<cuphyDataType_t Type>
cuphyVariant_t createTypedVariant(float value = 1.0f) {
    cuphyVariant_t variant;
    variant.type = Type;
    if constexpr (Type == CUPHY_R_32F) {
        variant.value.r32f = value;
    } else if constexpr (Type == CUPHY_C_32F) {
        variant.value.c32f = make_cuComplex(value, 0.0f);
    } else if constexpr (Type == CUPHY_BIT) {
        variant.value.b1 = static_cast<uint32_t>(value);
    }
    return variant;
}

// Helper function for testing elementwise operations
template<cuphyDataType_t SrcAType, cuphyDataType_t DstType, cuphyDataType_t SrcBType>
void testElementwiseOperation(
    const int* srcDims,
    const int* dstDims,
    cuphyElementWiseOp_t operation,
    cuphyStatus_t expectedStatus,
    const std::string& errorMessage,
    bool useSecondSource = false,
    bool initializeRandom = false,
    double randMin = -1.0,
    double randMax = 1.0,
    bool useInvalidSrcB = false)
{
    cuphy::typed_tensor<SrcAType, cuphy::device_alloc> tSrcA(srcDims[0], srcDims[1], srcDims[2]);
    cuphy::typed_tensor<DstType, cuphy::device_alloc> tDst(dstDims[0], dstDims[1], dstDims[2]);

    // Initialize tensors with random values if requested
    if (initializeRandom) {
        cuphy::rng rng;
        if constexpr (SrcAType == CUPHY_C_32F) {
            rng.uniform(tSrcA, make_cuComplex(float(randMin), float(randMin)),
                             make_cuComplex(float(randMax), float(randMax)));
        } else if constexpr (SrcAType == CUPHY_C_64F) {
            // For complex double, initialize real and imaginary parts separately
            cuphy::typed_tensor<CUPHY_R_64F, cuphy::device_alloc> real(srcDims[0], srcDims[1], srcDims[2]);
            cuphy::typed_tensor<CUPHY_R_64F, cuphy::device_alloc> imag(srcDims[0], srcDims[1], srcDims[2]);
            rng.uniform(real, randMin, randMax);
            rng.uniform(imag, randMin, randMax);
            // TODO: Copy real and imag to tSrcA as complex values
            // This might need a separate kernel or different initialization approach
            // For now, we'll skip random initialization for C64F
        } else {
            rng.uniform(tSrcA, randMin, randMax);
        }
    }

    auto alpha = createTypedVariant<SrcAType>();
    cudaStream_t stream = nullptr;

    std::unique_ptr<cuphy::typed_tensor<SrcBType, cuphy::device_alloc>> tSrcB;
    cuphyVariant_t beta;

    if (useSecondSource && !useInvalidSrcB) {
        tSrcB = std::make_unique<cuphy::typed_tensor<SrcBType, cuphy::device_alloc>>(
            srcDims[0], srcDims[1], srcDims[2]);

        if (initializeRandom) {
            cuphy::rng rng;
            rng.uniform(*tSrcB, randMin, randMax);
        }
    }

    beta = createTypedVariant<SrcBType>();

    EXPECT_EQ(cuphyTensorElementWiseOperation(
            tDst.desc().handle(),
            tDst.addr(),
            tSrcA.desc().handle(),
            tSrcA.addr(),
            &alpha,
            tSrcB ? tSrcB->desc().handle() : nullptr,
            tSrcB ? tSrcB->addr() : nullptr,
            useSecondSource ? &beta : nullptr,
            operation,
            stream),
        expectedStatus)
        << errorMessage;
}

} // namespace


////////////////////////////////////////////////////////////////////////
// ElementWise.Add
TEST(ElementWise, Add)
{
    do_elementwise_test<CUPHY_R_8I,  range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64,  16);
    do_elementwise_test<CUPHY_C_8I,  range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(128, 1);
    do_elementwise_test<CUPHY_R_8U,  range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (37,  19);
    do_elementwise_test<CUPHY_C_8U,  range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(8,   8);
    do_elementwise_test<CUPHY_R_16I, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_16I, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
    do_elementwise_test<CUPHY_R_16U, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_16U, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
    do_elementwise_test<CUPHY_R_32I, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_32I, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
    do_elementwise_test<CUPHY_R_32U, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_32U, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
    do_elementwise_test<CUPHY_R_16F, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_16F, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
    do_elementwise_test<CUPHY_R_32F, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_32F, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
    do_elementwise_test<CUPHY_R_64F, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_64F, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
}

////////////////////////////////////////////////////////////////////////
// ElementWise.XOR
TEST(ElementWise, XOR)
{
    do_elementwise_bit_test<tensor_op_xor, host_xor>(1024, 16);
}

////////////////////////////////////////////////////////////////////////
// ElementWise.XOR_Partial
TEST(ElementWise, XOR_Partial)
{
    do_elementwise_bit_partial_word_test<tensor_op_xor, host_xor>(64,  48,  16);
    do_elementwise_bit_partial_word_test<tensor_op_xor, host_xor>(264, 164, 8);
}

////////////////////////////////////////////////////////////////////////
// ElementWise.BroadcastBit
TEST(ElementWise, BroadcastBit)
{
    do_elementwise_broadcast_bit_test<tensor_op_xor, host_xor>(1024, 16);
}

////////////////////////////////////////////////////////////////////////
// ElementWise.Broadcast
TEST(ElementWise, Broadcast)
{
    do_elementwise_broadcast_test<CUPHY_R_32F, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64,   11);
    do_elementwise_broadcast_test<CUPHY_C_32F, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(113,  66);
    do_elementwise_broadcast_test<CUPHY_R_16F, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (97,   3);
    do_elementwise_broadcast_test<CUPHY_C_16F, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(1024, 83);
}

TEST(TensorElementwiseUnary, UnaryOperations) {
    // Common dimensions
    const int dims_normal[CUPHY_DIM_MAX] = {64, 32, 16, 1, 1};
    const int dims_small[CUPHY_DIM_MAX] = {32, 16, 8, 1, 1};
    const int dims_mismatch1[CUPHY_DIM_MAX] = {32, 32, 16, 1, 1};
    const int dims_mismatch2[CUPHY_DIM_MAX] = {32, 16, 4, 1, 1};
    const int dims_mismatch3[CUPHY_DIM_MAX] = {32, 8, 8, 1, 1};

    // Test case 1: Real ABS operation with null destination
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_R_32F>(
        dims_normal, dims_normal,
        CUPHY_ELEMWISE_ABS,
        CUPHY_STATUS_INVALID_ARGUMENT,
        "Should handle null destination pointer"
    );

    // Test case 2: Mismatched data types
    testElementwiseOperation<CUPHY_R_32F, CUPHY_C_32F, CUPHY_R_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_ABS,
        CUPHY_STATUS_INVALID_ARGUMENT,
        "Should fail with mismatched data types"
    );

    // Test case 3: Mismatched dimensions
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_R_32F>(
        dims_normal, dims_mismatch1,
        CUPHY_ELEMWISE_ABS,
        CUPHY_STATUS_INTERNAL_ERROR,
        "Should succeed with mismatched dimensions since tensor descriptors handle the dimension mapping"
    );

    // Test case 4: Invalid operation for data type
    testElementwiseOperation<CUPHY_BIT, CUPHY_BIT, CUPHY_BIT>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_ABS,
        CUPHY_STATUS_INVALID_ARGUMENT,
        "Should fail with unsupported operation for data type"
    );

    // Test case 5: Binary operation with mismatched source types
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_C_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_ADD,
        CUPHY_STATUS_UNSUPPORTED_TYPE,
        "Should fail with mismatched source types in binary operation",
        true
    );

    // Test case 6: MIN operation
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_R_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_MIN,
        CUPHY_STATUS_INTERNAL_ERROR,
        "Should succeed with MIN operation",
        true  // useSecondSource = true because MIN is a binary operation
    );

    // Test case 7: MAX operation with random values
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_R_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_MAX,
        CUPHY_STATUS_INTERNAL_ERROR,
        "Should succeed with MAX operation",
        true,   // useSecondSource = true because MAX is a binary operation
        true,   // initializeRandom = true
        -1.0f,  // randMin
        1.0f    // randMax
    );

    // Test case 8: MUL operation with random values
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_R_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_MUL,
        CUPHY_STATUS_INTERNAL_ERROR,
        "Should succeed with MUL operation",
        true,   // useSecondSource = true because MUL is a binary operation
        true,   // initializeRandom = true
        -1.0f,  // randMin
        1.0f    // randMax
    );

    // Test case 9: ABS operation with real float
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_R_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_ABS,
        CUPHY_STATUS_INVALID_ARGUMENT,
        "ABS operation with real float",
        false,  // useSecondSource = false for unary operation
        true,   // initializeRandom = true
        -1.0f,  // randMin
        1.0f    // randMax
    );

    // Test case 10: ABS operation with complex float
    testElementwiseOperation<CUPHY_C_32F, CUPHY_C_32F, CUPHY_C_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_ABS,
        CUPHY_STATUS_UNSUPPORTED_TYPE,
        "ABS operation with complex float",
        false,  // useSecondSource = false for unary operation
        true,   // initializeRandom = true
        -1.0f,  // randMin (will be used for both real and imaginary parts)
        1.0f    // randMax (will be used for both real and imaginary parts)
    );

    // Test case 11: ABS operation with double
    testElementwiseOperation<CUPHY_R_64F, CUPHY_R_64F, CUPHY_R_64F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_ABS,
        CUPHY_STATUS_INVALID_ARGUMENT,
        "ABS operation with double",
        false,  // useSecondSource = false for unary operation
        true,   // initializeRandom = true
        -1.0,   // randMin
        1.0     // randMax
    );

    // Test case 12: ABS operation with complex double
    testElementwiseOperation<CUPHY_C_64F, CUPHY_C_64F, CUPHY_C_64F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_ABS,
        CUPHY_STATUS_UNSUPPORTED_TYPE,
        "ABS operation with complex double",
        false,  // useSecondSource = false for unary operation
        true,   // initializeRandom = true
        -1.0,   // randMin
        1.0     // randMax
    );

    // Test case 13: Binary ABS operation (should fail)
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_R_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_ABS,
        CUPHY_STATUS_INVALID_ARGUMENT,
        "Binary ABS operation should fail as ABS is unary operation",
        true,   // useSecondSource = true to test binary operation
        true,   // initializeRandom = true
        -1.0f,  // randMin
        1.0f    // randMax
    );

    // Test case 14: Binary ABS operation with complex types (should fail)
    testElementwiseOperation<CUPHY_C_32F, CUPHY_C_32F, CUPHY_C_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_ABS,
        CUPHY_STATUS_INVALID_ARGUMENT,
        "Binary ABS operation with complex types should fail",
        true,   // useSecondSource = true to test binary operation
        false,
        -1.0f,  // randMin
        1.0f    // randMax
    );

    // Test case 15: MUL operation with mismatched types
    testElementwiseOperation<CUPHY_R_32F, CUPHY_C_32F, CUPHY_R_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_MUL,
        CUPHY_STATUS_UNSUPPORTED_TYPE,
        "MUL operation should fail with mismatched destination and source types",
        true,   // useSecondSource = true
        true,   // initializeRandom = true
        -1.0f,  // randMin
        1.0f    // randMax
    );

    // Test case 16: MUL operation with mismatched dimensions
    const int dims1[CUPHY_DIM_MAX] = {64, 32, 16, 1, 1};
    const int dims2[CUPHY_DIM_MAX] = {32, 32, 16, 1, 1}; // Different first dimension
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_R_32F>(
        dims1, dims2,
        CUPHY_ELEMWISE_MUL,
        CUPHY_STATUS_SIZE_MISMATCH,
        "MUL operation should fail with mismatched dimensions",
        true,   // useSecondSource = true
        true,   // initializeRandom = true
        -1.0f,  // randMin
        1.0f    // randMax
    );

    // Test case 17: MAX operation with invalid second source tensor
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_R_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_MAX,
        CUPHY_STATUS_INVALID_ARGUMENT,
        "MAX operation should fail with invalid second source tensor",
        true,    // useSecondSource = true
        true,    // initializeRandom = true
        -1.0f,   // randMin
        1.0f,    // randMax
        true     // useInvalidSrcB = true
    );

    // Test case 18: MAX operation with mismatched types
    testElementwiseOperation<CUPHY_R_32F, CUPHY_C_32F, CUPHY_R_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_MAX,
        CUPHY_STATUS_UNSUPPORTED_TYPE,
        "MAX operation should fail with mismatched tensor types",
        true,   // useSecondSource = true
        true,   // initializeRandom = true
        -1.0f,  // randMin
        1.0f    // randMax
    );

    // Test case 19: MAX operation with mismatched second source type
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_C_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_MAX,
        CUPHY_STATUS_UNSUPPORTED_TYPE,
        "MAX operation should fail with mismatched second source type",
        true,   // useSecondSource = true
        false,
        -1.0f,  // randMin
        1.0f    // randMax
    );

    // Test case 20: MAX operation with mismatched dimensions (different third dimension)
    const int dims_mismatched[CUPHY_DIM_MAX] = {32, 16, 4, 1, 1};
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_R_32F>(
        dims_small, dims_mismatched,
        CUPHY_ELEMWISE_MAX,
        CUPHY_STATUS_SIZE_MISMATCH,
        "MAX operation should fail when destination and first source dimensions don't match",
        true,   // useSecondSource = true
        true,   // initializeRandom = true
        -1.0f,  // randMin
        1.0f    // randMax
    );

    // Test case 21: MUL operation with mismatched dimensions
    // Test case with different dimensions between destination and first source
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_R_32F>(
        dims_small, dims_mismatch2,
        CUPHY_ELEMWISE_MUL,
        CUPHY_STATUS_SIZE_MISMATCH,
        "MUL operation should fail when destination and first source dimensions don't match",
        true,   // useSecondSource = true
        true,   // initializeRandom = true
        -1.0f,  // randMin
        1.0f    // randMax
    );

    // Test case 22: test with different second dimension
    testElementwiseOperation<CUPHY_R_32F, CUPHY_R_32F, CUPHY_R_32F>(
        dims_small, dims_mismatch3,
        CUPHY_ELEMWISE_MUL,
        CUPHY_STATUS_SIZE_MISMATCH,
        "MUL operation should fail when destination and first source have different second dimension",
        true,   // useSecondSource = true
        true,   // initializeRandom = true
        -1.0f,  // randMin
        1.0f    // randMax
    );

    // Test case 23: XOR operation with non-bit types (should fail)
    testElementwiseOperation<CUPHY_R_32F, CUPHY_BIT, CUPHY_BIT>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_BIT_XOR,
        CUPHY_STATUS_UNSUPPORTED_TYPE,
        "XOR operation should fail when source A is not a bit type",
        true,   // useSecondSource = true
        true,   // initializeRandom = true
        0.0f,   // randMin
        1.0f    // randMax
    );

    // Test case 24: test with different destination type
    testElementwiseOperation<CUPHY_BIT, CUPHY_R_32F, CUPHY_BIT>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_BIT_XOR,
        CUPHY_STATUS_UNSUPPORTED_TYPE,
        "XOR operation should fail when destination is not a bit type",
        true,   // useSecondSource = true
        true,   // initializeRandom = true
        0.0f,   // randMin
        1.0f    // randMax
    );

    // Test case 25: test with different source B type
    testElementwiseOperation<CUPHY_BIT, CUPHY_BIT, CUPHY_R_32F>(
        dims_small, dims_small,
        CUPHY_ELEMWISE_BIT_XOR,
        CUPHY_STATUS_UNSUPPORTED_TYPE,
        "XOR operation should fail when source B is not a bit type",
        true,   // useSecondSource = true
        true,   // initializeRandom = true
        0.0f,   // randMin
        1.0f    // randMax
    );

    // Test case 26: XOR operation with mismatched dimensions for broadcast
    const int dims_broadcast1[CUPHY_DIM_MAX] = {64, 32, 1, 1, 1};  // Broadcastable dimension
    const int dims_broadcast2[CUPHY_DIM_MAX] = {64, 1, 16, 1, 1};  // Broadcastable dimension
    const int dims_no_broadcast[CUPHY_DIM_MAX] = {64, 48, 16, 1, 1};  // Non-broadcastable dimension
    // Test with non-broadcastable dimensions
    testElementwiseOperation<CUPHY_BIT, CUPHY_BIT, CUPHY_BIT>(
        dims_no_broadcast, dims_small,
        CUPHY_ELEMWISE_BIT_XOR,
        CUPHY_STATUS_SIZE_MISMATCH,
        "XOR operation should fail when dimensions don't match and can't be broadcast",
        true,   // useSecondSource = true
        true,   // initializeRandom = true
        0.0f,   // randMin
        1.0f    // randMax
    );

    // Test case 27: test with valid broadcast dimensions (should pass)
    testElementwiseOperation<CUPHY_BIT, CUPHY_BIT, CUPHY_BIT>(
        dims_broadcast1, dims_broadcast2,
        CUPHY_ELEMWISE_BIT_XOR,
        CUPHY_STATUS_SIZE_MISMATCH,
        "XOR operation should succeed with broadcastable dimensions",
        true,   // useSecondSource = true
        true,   // initializeRandom = true
        0.0f,   // randMin
        1.0f    // randMax
    );

    // Test case 28: test with mismatched broadcast dimensions
    testElementwiseOperation<CUPHY_BIT, CUPHY_BIT, CUPHY_BIT>(
        dims_broadcast1, dims_no_broadcast,
        CUPHY_ELEMWISE_BIT_XOR,
        CUPHY_STATUS_SIZE_MISMATCH,
        "XOR operation should fail with mismatched broadcast dimensions",
        true,   // useSecondSource = true
        true,   // initializeRandom = true
        0.0f,   // randMin
        1.0f    // randMax
    );
}


////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}
