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

// Uncomment to show values being compared
//#define TEST_REDUCTION_PRINT_VALUES 1


namespace
{

template <typename T> class host_op_sum
{
public:
    host_op_sum() : accum_(0)  {}
    void apply(const T& value) { accum_ += value; }
    T    get_result()          { return accum_; }
private:
    T accum_;
};

////////////////////////////////////////////////////////////////////////
// do_reduction_test_2D()
// Performs a reduction test using a 2-D input. The result is 1-D.
template <cuphyDataType_t TType,
          template <typename> class THostOp,
          typename                  TTol>
void do_reduction_test_2D(int             NUM_ROWS,
                          int             NUM_COLS,
                          int             dim,
                          TTol            error_tol)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    typedef typename cuphy::type_traits<TType>::type        value_t;
    typedef THostOp<value_t>                                host_op_t;
    //------------------------------------------------------------------
    // Allocate tensors
    std::array<int, 2> SRC_DIM = {NUM_ROWS, NUM_COLS};
    std::array<int, 2> DST_DIM = SRC_DIM;
    DST_DIM[dim] = 1;
    tensor_p  tSrc(cuphy::tensor_layout(SRC_DIM.size(), SRC_DIM.data(), nullptr));
    tensor_p  tDst(cuphy::tensor_layout(DST_DIM.size(), DST_DIM.data(), nullptr));
    //printf("tSrc: %s\n", tSrc.desc().get_info().to_string().c_str());
    //printf("tDst: %s\n", tDst.desc().get_info().to_string().c_str());
    //------------------------------------------------------------------
    // Initialize the source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 1, 10);
    //------------------------------------------------------------------
    // Perform the reduction operation using the library function
    tSrc.sum(tDst, dim);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Perform the operation on the host
    if(0 == dim)
    {
        for(int j = 0; j < NUM_COLS; ++j)
        {
            host_op_t s;
            for(int i = 0; i < NUM_ROWS; ++i)
            {
                s.apply(tSrc(i, j));
            }
            EXPECT_LT(std::abs(s.get_result() - tDst(0, j)), error_tol)
                << "NUM_ROWS = "   << NUM_ROWS
                << ", NUM_COLS = " << NUM_COLS
                << ", column "     << j
                << ", host = "     << s.get_result()
                << ", library = "  << tDst(0, j)
                << std::endl;
#if TEST_REDUCTION_PRINT_VALUES
            std::cout << "host result: "      << s.get_result()
                      << ", library result: " << tDst(0, j)
                      << ", error = "         << std::abs(s.get_result() - tDst(0, j))
                      << std::endl;
#endif
        }
    }
    else
    {
        for(int i = 0; i < NUM_ROWS; ++i)
        {
            host_op_t s;
            for(int j = 0; j < NUM_COLS; ++j)
            {
                s.apply(tSrc(i, j));
            }
            EXPECT_LT(std::abs(s.get_result() - tDst(i, 0)), error_tol)
                << "NUM_ROWS = "   << NUM_ROWS
                << ", NUM_COLS = " << NUM_COLS
                << ", row "        << i
                << ", host = "     << s.get_result()
                << ", library = "  << tDst(i, 0)
                << std::endl;
#if TEST_REDUCTION_PRINT_VALUES
            std::cout << "host result: " << s.get_result()
                      << ", library result: " << tDst(i, 0)
                      << ", error = " << std::abs(s.get_result() - tDst(i, 0))
                      << std::endl;
#endif
        }
    }
}

////////////////////////////////////////////////////////////////////////
// do_reduction_test_2D_bits()
// Performs a reduction test using a 2-D input of type CUPHY_BIT. The
// result is 1-D tensor with the count of the number of bits in that
// column.
void do_reduction_test_2D_bits(int NUM_ROWS,
                               int NUM_COLS)
{

    typedef cuphy::typed_tensor<CUPHY_BIT,   cuphy::pinned_alloc> tensor_bits_p;
    typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tensor_sum_p;

    //typedef typename cuphy::type_traits<TType>::type        value_t;
    //typedef THostOp<value_t>                                host_op_t;
    //------------------------------------------------------------------
    // Allocate tensors
    std::array<int, 2> SRC_DIM = {NUM_ROWS, NUM_COLS};
    std::array<int, 2> DST_DIM = {1,        NUM_COLS};
    tensor_bits_p  tSrc(cuphy::tensor_layout(SRC_DIM.size(), SRC_DIM.data(), nullptr),
                        cuphy::tensor_flags::align_coalesce);
    tensor_sum_p   tDst(cuphy::tensor_layout(DST_DIM.size(), DST_DIM.data(), nullptr));
    //printf("tSrc: %s\n", tSrc.desc().get_info().to_string().c_str());
    //printf("tDst: %s\n", tDst.desc().get_info().to_string().c_str());
    //------------------------------------------------------------------
    // Initialize the source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 0, 1);
    //------------------------------------------------------------------
    // Perform the reduction operation using the library function
    tSrc.sum(tDst, 0);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Perform the operation on the host
    const int NUM_WORDS = (NUM_ROWS + 31) / 32;
    for(int j = 0; j < NUM_COLS; ++j)
    {
        uint32_t host_count = 0;
        for(int i = 0; i < NUM_WORDS; ++i)
        {
            //printf("[%i]: 0x%X (%u)\n", i, tSrc(i, j), __builtin_popcount(tSrc(i, j)));
            host_count += __builtin_popcount(tSrc(i, j));
        }
        uint32_t lib_count = tDst(0,j);
        //printf("host = %u, library = %u\n", host_count, lib_count);
        EXPECT_EQ(host_count, lib_count)
            << "NUM_ROWS = "   << NUM_ROWS
            << ", NUM_COLS = " << NUM_COLS
            << ", column "     << j
            << ", host = "     << host_count
            << ", library = "  << lib_count
            << std::endl;
#if TEST_REDUCTION_PRINT_VALUES
        std::cout << "host result: "      << host_count
                  << ", library result: " << lib_count
                  << ", error = "         << std::abs(static_cast<int64_t>(host_count) - static_cast<int64_t>(lib_count))
                  << std::endl;
#endif
    }
}

////////////////////////////////////////////////////////////////////////
// do_reduction_test_2D_bits_partial_word()
// Performs a reduction test using a 2-D input of type CUPHY_BIT. The
// result is 1-D tensor with the count of the number of bits in that
// column.
void do_reduction_test_2D_bits_partial_word(int NUM_FULL_ROWS,
                                            int NUM_PARTIAL_ROWS,
                                            int NUM_COLS)
{

    typedef cuphy::typed_tensor<CUPHY_BIT,   cuphy::pinned_alloc> tensor_bits_p;
    typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tensor_sum_p;
    typedef tensor_bits_p::tensor_ref_t                           tensor_bits_p_ref;

    //typedef typename cuphy::type_traits<TType>::type        value_t;
    //typedef THostOp<value_t>                                host_op_t;
    //------------------------------------------------------------------
    // Allocate tensors
    tensor_bits_p  tSrc(NUM_FULL_ROWS, NUM_COLS);
    tensor_sum_p   tDst(1,             NUM_COLS);
    //printf("tSrc: %s\n", tSrc.desc().get_info().to_string().c_str());
    //printf("tDst: %s\n", tDst.desc().get_info().to_string().c_str());
    //------------------------------------------------------------------
    // Initialize the source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 0, 1);
    //------------------------------------------------------------------
    // Generate a tensor ref for a subset of the source tensor. We want
    // the subset to contain a partial word at the end of each column.
    cuphy::index_group grp(cuphy::index_range(0, NUM_PARTIAL_ROWS),
                           cuphy::dim_all());
    tensor_bits_p_ref  tSrc_s = tSrc.subset(grp);
    //------------------------------------------------------------------
    // Perform the reduction operation using the library function
    cuphy::tensor_reduction_sum(tDst, tSrc_s);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Perform the operation on the host
    const int NUM_WORDS = (NUM_PARTIAL_ROWS + 31) / 32;
    for(int j = 0; j < NUM_COLS; ++j)
    {
        uint32_t host_count = 0;
        for(int i = 0; i < NUM_WORDS; ++i)
        {
            uint32_t  src_mask    = 0xFFFFFFFF;
            const int ROW_COUNT = (i+1) * 32;
            if(ROW_COUNT > NUM_PARTIAL_ROWS)
            {
                const int NUM_BITS = NUM_PARTIAL_ROWS - (i * 32);
                src_mask = (1 << NUM_BITS) - 1;
            }
            uint32_t value = tSrc(i, j) & src_mask;
            //printf("[%i]: src = 0x%X, mask = 0x%X, value = 0x%X, popcount = %u\n",
            //       i,
            //       tSrc(i, j),
            //       src_mask,
            //       value,
            //       __builtin_popcount(value));
            host_count += __builtin_popcount(value);
        }
        uint32_t lib_count = tDst(0,j);
        //printf("host = %u, library = %u\n", host_count, lib_count);
        EXPECT_EQ(host_count, lib_count)
            << "NUM_FULL_ROWS = "      << NUM_FULL_ROWS
            << ", NUM_PARTIAL_ROWS = " << NUM_PARTIAL_ROWS
            << ", NUM_COLS = "         << NUM_COLS
            << ", column "             << j
            << ", host = "             << host_count
            << ", library = "          << lib_count
            << std::endl;
    }

}

// Test parameters structure for validation tests
struct ValidationTestParams {
    int src_rows;
    int src_cols;
    int dst_rows;
    int dst_cols;
    int reduction_dim;
    cuphyReductionOp_t reduction_op;
    cuphyStatus_t expected_status;
    const char* test_description;
    float min_val;
    float max_val;
};

// Helper function to run a single validation test
template<typename TSrc, typename TDst>
void run_validation_test(const ValidationTestParams& params) {
    TSrc tSrc(params.src_rows, params.src_cols);
    TDst tDst(params.dst_rows, params.dst_cols);

    // Initialize source with random values
    cuphy::rng rng;
    rng.uniform(tSrc, params.min_val, params.max_val);

    // Try reduction operation
    cuphyStatus_t status = cuphyTensorReduction(
        tDst.desc().handle(), tDst.addr(),
        tSrc.desc().handle(), tSrc.addr(),
        params.reduction_op,
        params.reduction_dim,
        0,  // workspaceSize
        nullptr,  // workspace
        0   // stream
    );

    EXPECT_EQ(status, params.expected_status)
        << "Test failed: " << params.test_description;
}

// Helper function to run a validation test with void descriptors
template<typename TSrc, typename TDst>
void run_void_validation_test(const ValidationTestParams& params,
                            bool void_src,
                            bool void_dst) {
    TSrc tSrc(params.src_rows, params.src_cols);
    TDst tDst(params.dst_rows, params.dst_cols);

    // Initialize source with random values if needed
    cuphy::rng rng;
    rng.uniform(tSrc, params.min_val, params.max_val);

    // Create void descriptor
    cuphy::tensor_desc voidDesc;  // Default constructor sets type to CUPHY_VOID

    // Try reduction operation with appropriate descriptors
    cuphyStatus_t status = cuphyTensorReduction(
        void_dst ? voidDesc.handle() : tDst.desc().handle(), tDst.addr(),
        void_src ? voidDesc.handle() : tSrc.desc().handle(), tSrc.addr(),
        params.reduction_op,
        params.reduction_dim,
        0,  // workspaceSize
        nullptr,  // workspace
        0   // stream
    );

    EXPECT_EQ(status, params.expected_status)
        << "Test failed: " << params.test_description;
}

// Helper function to run a validation test with result verification
template<typename TSrc, typename TDst>
void run_reduction_test_with_verification(const ValidationTestParams& params) {
    TSrc tSrc(params.src_rows, params.src_cols);
    TDst tDst(params.dst_rows, params.dst_cols);

    // Initialize source with random values
    cuphy::rng rng;
    rng.uniform(tSrc, params.min_val, params.max_val);

    // Try reduction operation
    cuphyStatus_t status = cuphyTensorReduction(
        tDst.desc().handle(), tDst.addr(),
        tSrc.desc().handle(), tSrc.addr(),
        params.reduction_op,
        params.reduction_dim,
        0,  // workspaceSize
        nullptr,  // workspace
        0   // stream
    );

    EXPECT_EQ(status, params.expected_status)
        << "Test failed: " << params.test_description;

    // Verify results if the operation was successful
    if (status == CUPHY_STATUS_SUCCESS) {
        cudaStreamSynchronize(0);

        if constexpr (std::is_same_v<TSrc, cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc>>) {
            // Float verification
            for(int j = 0; j < params.dst_cols; ++j) {
                float expected = 0;
                for(int i = 0; i < params.src_rows; ++i) {
                    expected += tSrc(i, j);
                }
                EXPECT_NEAR(tDst(0, j), expected, 0.1f)
                    << params.test_description << ": result mismatch at column " << j;
            }
        } else if constexpr (std::is_same_v<TSrc, cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc>>) {
            // Bit tensor verification
            const int NUM_WORDS = (params.src_rows + 31) / 32;
            for(int j = 0; j < params.dst_cols; ++j) {
                uint32_t expected = 0;
                for(int i = 0; i < NUM_WORDS; ++i) {
                    expected += __builtin_popcount(tSrc(i, j));
                }
                EXPECT_EQ(tDst(0, j), expected)
                    << params.test_description << ": result mismatch at column " << j;
            }
        }
    }
}

} // namespace


////////////////////////////////////////////////////////////////////////
// Reduction.Sum
TEST(Reduction, Sum)
{
    do_reduction_test_2D<CUPHY_R_32F, host_op_sum>(64, 8, 0, 0.1f);
    do_reduction_test_2D<CUPHY_R_32F, host_op_sum>(80, 8, 0, 0.1f);
    do_reduction_test_2D<CUPHY_R_32F, host_op_sum>(8, 64, 1, 0.1f);
    do_reduction_test_2D<CUPHY_R_32F, host_op_sum>(1,  1, 0, 0.1f);
}

////////////////////////////////////////////////////////////////////////
// Reduction.Sum_Bits
TEST(Reduction, Sum_Bits)
{
    do_reduction_test_2D_bits(64,    48);
    do_reduction_test_2D_bits(35000, 11);
}

////////////////////////////////////////////////////////////////////////
// Reduction.Sum_Bits_Partial_Word
TEST(Reduction, Sum_Bits_Partial_Word)
{
    do_reduction_test_2D_bits_partial_word(64,  48, 16);
    do_reduction_test_2D_bits_partial_word(128, 97, 3);
}


// Reduction.ValidationExtended
TEST(Reduction, ValidationExtended) {
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_bit_p;
    typedef cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tensor_f32_p;
    typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tensor_u32_p;

    const ValidationTestParams test_cases[] = {
        // Test case 1: Bit tensor reduction in non-zero dimension
        {
            8, 64,    // src dimensions
            8, 1,     // dst dimensions
            1,        // reduction_dim
            CUPHY_REDUCTION_SUM,
            CUPHY_STATUS_UNSUPPORTED_TYPE,
            "Bit reduction in non-zero dimension",
            0, 1      // min_val, max_val
        },

        // Test case 2: Dimension mismatch - wrong output dimensions
        {
            64, 8,    // src dimensions
            2, 8,     // dst dimensions (wrong, should be 1x8)
            0,        // reduction_dim
            CUPHY_REDUCTION_SUM,
            CUPHY_STATUS_SIZE_MISMATCH,
            "Incorrect output dimensions",
            1, 10     // min_val, max_val
        },

        // Test case 3: Dimension mismatch - wrong number of dimensions
        {
            64, 8,    // src dimensions
            1, 1,     // dst dimensions (wrong number of dimensions)
            0,        // reduction_dim
            CUPHY_REDUCTION_SUM,
            CUPHY_STATUS_SIZE_MISMATCH,
            "Incorrect number of dimensions",
            1, 10     // min_val, max_val
        },

        // Test case 4: Invalid reduction dimension
        {
            64, 8,    // src dimensions
            64, 1,    // dst dimensions
            5,        // reduction_dim (invalid)
            CUPHY_REDUCTION_SUM,
            CUPHY_STATUS_INVALID_ARGUMENT,
            "Invalid reduction dimension",
            1, 10     // min_val, max_val
        }
    };

    // Run test cases with appropriate tensor types
    run_validation_test<tensor_bit_p, tensor_u32_p>(test_cases[0]);  // Bit tensor test
    run_validation_test<tensor_f32_p, tensor_f32_p>(test_cases[1]);  // Wrong dimensions
    run_validation_test<tensor_f32_p, tensor_f32_p>(test_cases[2]);  // Wrong number of dimensions
    run_validation_test<tensor_f32_p, tensor_f32_p>(test_cases[3]);  // Invalid dimension
}

////////////////////////////////////////////////////////////////////////
// Reduction.MinMaxValidation
TEST(Reduction, MinMaxValidation) {
    typedef cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tensor_f32_p;

    const ValidationTestParams test_cases[] = {
        // Test case 1: MIN reduction
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            CUPHY_REDUCTION_MIN,
            CUPHY_STATUS_NOT_SUPPORTED,
            "MIN reduction",
            1, 10     // min_val, max_val
        },

        // Test case 2: MAX reduction
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            CUPHY_REDUCTION_MAX,
            CUPHY_STATUS_NOT_SUPPORTED,
            "MAX reduction",
            1, 10     // min_val, max_val
        },

        // Test case 3: MIN reduction with different types
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            CUPHY_REDUCTION_MIN,
            CUPHY_STATUS_NOT_SUPPORTED,
            "MIN reduction with different types",
            1, 10     // min_val, max_val
        },

        // Test case 4: MAX reduction with different dimensions
        {
            64, 8,    // src dimensions
            64, 1,    // dst dimensions (different dimensions)
            1,        // reduction_dim
            CUPHY_REDUCTION_MAX,
            CUPHY_STATUS_NOT_SUPPORTED,
            "MAX reduction with different dimensions",
            1, 10     // min_val, max_val
        }
    };

    // Run all test cases with float tensors
    for (const auto& test_case : test_cases) {
        run_validation_test<tensor_f32_p, tensor_f32_p>(test_case);
    }
}

////////////////////////////////////////////////////////////////////////
// Reduction.VoidTypeValidation
TEST(Reduction, VoidTypeValidation) {
    typedef cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tensor_f32_p;

    const ValidationTestParams test_cases[] = {
        // Test case 1: void source tensor
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            CUPHY_REDUCTION_SUM,
            CUPHY_STATUS_UNSUPPORTED_TYPE,
            "Void source tensor",
            1, 10     // min_val, max_val
        },

        // Test case 2: void destination tensor
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            CUPHY_REDUCTION_SUM,
            CUPHY_STATUS_UNSUPPORTED_TYPE,
            "Void destination tensor",
            1, 10     // min_val, max_val
        },

        // Test case 3: both tensors void
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            CUPHY_REDUCTION_SUM,
            CUPHY_STATUS_UNSUPPORTED_TYPE,
            "Both tensors void",
            1, 10     // min_val, max_val
        }
    };

    // Run test cases with appropriate void descriptors
    run_void_validation_test<tensor_f32_p, tensor_f32_p>(test_cases[0], true, false);   // void source
    run_void_validation_test<tensor_f32_p, tensor_f32_p>(test_cases[1], false, true);   // void destination
    run_void_validation_test<tensor_f32_p, tensor_f32_p>(test_cases[2], true, true);    // both void
}

////////////////////////////////////////////////////////////////////////
// Reduction.InvalidReductionOp
TEST(Reduction, InvalidReductionOp) {
    typedef cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tensor_f32_p;

    const ValidationTestParams test_cases[] = {
        // Test case 1: Invalid operation value
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            static_cast<cuphyReductionOp_t>(999),  // Invalid operation
            CUPHY_STATUS_INTERNAL_ERROR,
            "Invalid reduction operation",
            1, 10     // min_val, max_val
        },

        // Test case 2: Negative operation value
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            static_cast<cuphyReductionOp_t>(-1),  // Negative operation
            CUPHY_STATUS_INTERNAL_ERROR,
            "Negative reduction operation",
            1, 10     // min_val, max_val
        },

        // Test case 3: Operation value just beyond valid range
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            static_cast<cuphyReductionOp_t>(CUPHY_REDUCTION_MAX + 1),  // Beyond valid range
            CUPHY_STATUS_INTERNAL_ERROR,
            "Beyond valid range reduction operation",
            1, 10     // min_val, max_val
        }
    };

    // Run all test cases
    for (const auto& test_case : test_cases) {
        run_validation_test<tensor_f32_p, tensor_f32_p>(test_case);
    }
}

// Reduction.SupportedTypeCombinations
TEST(Reduction, SupportedTypeCombinations) {
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_bit_p;
    typedef cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tensor_f32_p;
    typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tensor_u32_p;

    const ValidationTestParams test_cases[] = {
        // Test case 1: F32 to F32 reduction
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            CUPHY_REDUCTION_SUM,
            CUPHY_STATUS_SUCCESS,
            "F32 to F32 reduction",
            1, 10     // min_val, max_val
        },

        // Test case 2: BIT to U32 reduction
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            CUPHY_REDUCTION_SUM,
            CUPHY_STATUS_SUCCESS,
            "BIT to U32 reduction",
            0, 1      // min_val, max_val
        },

        // Test case 3: Invalid type combination - F32 to U32
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            CUPHY_REDUCTION_SUM,
            CUPHY_STATUS_NOT_SUPPORTED,
            "F32 to U32 reduction (invalid type combination)",
            1, 10     // min_val, max_val
        },

        // Test case 4: Invalid type combination - U32 to F32
        {
            64, 8,    // src dimensions
            1, 8,     // dst dimensions
            0,        // reduction_dim
            CUPHY_REDUCTION_SUM,
            CUPHY_STATUS_NOT_SUPPORTED,
            "U32 to F32 reduction (invalid type combination)",
            1, 10     // min_val, max_val
        }
    };

    // Run test cases with appropriate tensor types and verification
    run_reduction_test_with_verification<tensor_f32_p, tensor_f32_p>(test_cases[0]);  // F32 to F32
    run_reduction_test_with_verification<tensor_bit_p, tensor_u32_p>(test_cases[1]);  // BIT to U32
    run_validation_test<tensor_f32_p, tensor_u32_p>(test_cases[2]);  // Invalid: F32 to U32
    run_validation_test<tensor_u32_p, tensor_f32_p>(test_cases[3]);  // Invalid: U32 to F32
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}
