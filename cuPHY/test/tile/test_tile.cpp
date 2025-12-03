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

// Define printer for float2 and double2 for Google Test
namespace testing {
namespace internal {
void PrintTo(const float2& value, ::std::ostream* os) {
    *os << "{" << value.x << ", " << value.y << "}";
}

void PrintTo(const double2& value, ::std::ostream* os) {
    *os << "{" << value.x << ", " << value.y << "}";
}
} // namespace internal
} // namespace testing

// Define comparison operators in global namespace
inline bool operator==(const float2& lhs, const float2& rhs) {
    return (lhs.x == rhs.x) && (lhs.y == rhs.y);
}

inline bool operator!=(const float2& lhs, const float2& rhs) {
    return !(lhs == rhs);
}

inline bool operator==(const double2& lhs, const double2& rhs) {
    return (lhs.x == rhs.x) && (lhs.y == rhs.y);
}

inline bool operator!=(const double2& lhs, const double2& rhs) {
    return !(lhs == rhs);
}

namespace
{

// comparison operator for complex floats. "Exact" comparison is OK here
// because we are just copying values.
//bool operator==(const cuFloatComplex& a, const cuFloatComplex& b)
//{
//    return (a.x == b.x) && (a.y == b.y);
//}

template <cuphyDataType_t TType>
void do_tile_test_2D(int             NUM_ROWS,
                     int             NUM_COLS,
                     int             NUM_TILE_ROWS,
                     int             NUM_TILE_COLS)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    //------------------------------------------------------------------
    // Allocate a source tensor
    const std::array<int, 2> SRC_DIMS = {{NUM_ROWS, NUM_COLS}};
    tensor_p                 tSrc(cuphy::tensor_layout(SRC_DIMS.size(), SRC_DIMS.data(), nullptr));
    //------------------------------------------------------------------
    // Initialize the source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 1, 10);
    //------------------------------------------------------------------
    // Allocate a destination tensor with appropriate dimensions
    const std::array<int, 2> TILE = {{NUM_TILE_ROWS,  NUM_TILE_COLS}};
    std::array<int, 2>       dst_dims;
    for(size_t i = 0; i < SRC_DIMS.size(); ++i)
    {
        dst_dims[i] = SRC_DIMS[i] * TILE[i];
    }
    tensor_p                 tDst(cuphy::tensor_layout(dst_dims.size(), dst_dims.data(), nullptr));
    //------------------------------------------------------------------
    // Perform the tile operation
    tSrc.tile(tDst, TILE[0], TILE[1]);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //printf("Input:\n");
    //for(int i = 0; i < SRC_DIMS[0]; ++i)
    //{
    //    printf("[%2i]: ", i);
    //    for(int j = 0; j < SRC_DIMS[1]; ++j)
    //    {
    //        printf("%f ", tSrc(i, j));
    //    }
    //    printf("\n");
    //}
    //printf("Output:\n");
    //for(int i = 0; i < dst_dims[0]; ++i)
    //{
    //    printf("[%2i]: ", i);
    //    for(int j = 0; j < dst_dims[1]; ++j)
    //    {
    //        printf("%f ", tDst(i, j));
    //    }
    //    printf("\n");
    //}
    for(int i = 0; i < dst_dims[0]; ++i)
    {
        for(int j = 0; j < dst_dims[1]; ++j)
        {
            EXPECT_EQ(tDst(i, j), tSrc(i % SRC_DIMS[0], j % SRC_DIMS[1]));
        }
    }
}

// Test parameters struct for tile operations
struct TileTestParams {
    int src_rows;
    int src_cols;
    int tile_rows;
    int tile_cols;
    const char* description;

    TileTestParams(int sr, int sc, int tr, int tc, const char* desc)
        : src_rows(sr), src_cols(sc), tile_rows(tr), tile_cols(tc), description(desc) {}
};

// Helper template function for complex tensor tests
template<cuphyDataType_t ComplexType, cuphyDataType_t RealType>
void run_complex_tile_test(const TileTestParams& test, const char* test_type) {
    const std::array<int, 2> SRC_DIMS = {{test.src_rows, test.src_cols}};
    std::array<int, 2> DST_DIMS = {{test.src_rows * test.tile_rows,
                                   test.src_cols * test.tile_cols}};

    cuphy::typed_tensor<ComplexType, cuphy::pinned_alloc> tSrc(
        cuphy::tensor_layout(SRC_DIMS.size(), SRC_DIMS.data(), nullptr));
    cuphy::typed_tensor<ComplexType, cuphy::pinned_alloc> tDst(
        cuphy::tensor_layout(DST_DIMS.size(), DST_DIMS.data(), nullptr));

    // Initialize source tensor with random values for real and imaginary parts
    cuphy::typed_tensor<RealType, cuphy::pinned_alloc> tReal(
        cuphy::tensor_layout(SRC_DIMS.size(), SRC_DIMS.data(), nullptr));
    cuphy::typed_tensor<RealType, cuphy::pinned_alloc> tImag(
        cuphy::tensor_layout(SRC_DIMS.size(), SRC_DIMS.data(), nullptr));

    cuphy::rng rng;
    rng.uniform(tReal, 1, 10);
    rng.uniform(tImag, 1, 10);

    // Copy real and imaginary parts to complex tensor
    using value_t = typename std::conditional<RealType == CUPHY_R_32F,
                                            float2, double2>::type;
    for(int i = 0; i < SRC_DIMS[0]; ++i) {
        for(int j = 0; j < SRC_DIMS[1]; ++j) {
            value_t value;
            value.x = tReal(i, j);
            value.y = tImag(i, j);
            tSrc(i, j) = value;
        }
    }

    // Perform tiling
    tSrc.tile(tDst, test.tile_rows, test.tile_cols);
    cudaStreamSynchronize(0);

    // Verify results
    for(int i = 0; i < DST_DIMS[0]; ++i) {
        for(int j = 0; j < DST_DIMS[1]; ++j) {
            EXPECT_EQ(tDst(i, j), tSrc(i % SRC_DIMS[0], j % SRC_DIMS[1]))
                << test_type << " test failed: " << test.description
                << " at position (" << i << "," << j << ")";
        }
    }
}

// Helper function for bit tensor tests
void run_bit_tile_test(const TileTestParams& test) {
    const std::array<int, 2> SRC_DIMS = {{test.src_rows, test.src_cols}};
    std::array<int, 2> DST_DIMS = {{test.src_rows * test.tile_rows,
                                   test.src_cols * test.tile_cols}};

    // Create source and destination bit tensors
    cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tSrc(
        cuphy::tensor_layout(SRC_DIMS.size(), SRC_DIMS.data(), nullptr));
    cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tDst(
        cuphy::tensor_layout(DST_DIMS.size(), DST_DIMS.data(), nullptr));

    // Initialize source tensor with random bits
    cuphy::rng rng;
    rng.uniform(tSrc, 0, 1);

    // Attempt tiling - should return CUPHY_STATUS_UNSUPPORTED_TYPE for bit tensors
    const int tileExtents[2] = {test.tile_rows, test.tile_cols};
    uint32_t* src_data = static_cast<uint32_t*>(tSrc.addr());
    uint32_t* dst_data = static_cast<uint32_t*>(tDst.addr());

    cuphyStatus_t status = cuphyTileTensor(tDst.desc().handle(),
                                          dst_data,
                                          tSrc.desc().handle(),
                                          src_data,
                                          2,  // tileRank
                                          tileExtents,
                                          0); // stream
    EXPECT_EQ(status, CUPHY_STATUS_UNSUPPORTED_TYPE)
        << "Expected CUPHY_STATUS_UNSUPPORTED_TYPE for bit tensor tiling: "
        << test.description << ", got: " << status;
}

// Helper function for size mismatch tests
template<cuphyDataType_t TType>
void run_size_mismatch_test(const TileTestParams& test, const std::array<int, 2>& dst_dims) {
    const std::array<int, 2> SRC_DIMS = {{test.src_rows, test.src_cols}};

    // Create source tensor
    cuphy::typed_tensor<TType, cuphy::pinned_alloc> tSrc(
        cuphy::tensor_layout(SRC_DIMS.size(), SRC_DIMS.data(), nullptr));

    // Initialize source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 1, 10);

    // Create destination tensor with mismatched dimensions
    cuphy::typed_tensor<TType, cuphy::pinned_alloc> tDst(
        cuphy::tensor_layout(dst_dims.size(), dst_dims.data(), nullptr));

    // Try to tile with given tile extents
    const int tileRank = 2;
    const int tileExtents[2] = {test.tile_rows, test.tile_cols};

    cuphyStatus_t status = cuphyTileTensor(tDst.desc().handle(),
                                          tDst.addr(),
                                          tSrc.desc().handle(),
                                          tSrc.addr(),
                                          tileRank,
                                          tileExtents,
                                          0); // stream

    // Expected dimensions would be src_dims * tile_extents
    const std::array<int, 2> expected_dims = {{
        test.src_rows * test.tile_rows,
        test.src_cols * test.tile_cols
    }};

    EXPECT_EQ(status, CUPHY_STATUS_SIZE_MISMATCH)
        << "Expected CUPHY_STATUS_SIZE_MISMATCH for mismatched destination dimensions. "
        << "Source: " << test.src_rows << "x" << test.src_cols
        << ", Tile: " << test.tile_rows << "x" << test.tile_cols
        << ", Expected: " << expected_dims[0] << "x" << expected_dims[1]
        << ", Got: " << dst_dims[0] << "x" << dst_dims[1];
}

// Helper function for internal error test
void run_internal_error_test() {
    // Create source tensor with dimensions 4x4
    const std::array<int, 2> SRC_DIMS = {{4, 4}};
    
    // Create source tensor with CUPHY_R_32F type
    cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tSrc(
        cuphy::tensor_layout(SRC_DIMS.size(), SRC_DIMS.data(), nullptr));

    // Initialize source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 1, 10);

    // Create destination tensor with valid dimensions
    const std::array<int, 2> DST_DIMS = {{8, 8}};
    cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tDst(
        cuphy::tensor_layout(DST_DIMS.size(), DST_DIMS.data(), nullptr));

    // Try to tile with 2x2 tile extents
    const int tileExtents[2] = {2, 2};

    // Create an invalid CUDA stream
    cudaStream_t invalid_stream;
    ASSERT_EQ(cudaStreamCreate(&invalid_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(invalid_stream), cudaSuccess);
    // Now invalid_stream is a dangling handle

    // This should return CUPHY_STATUS_INTERNAL_ERROR due to invalid stream
    cuphyStatus_t status = cuphyTileTensor(tDst.desc().handle(),
                                          tDst.addr(),
                                          tSrc.desc().handle(),
                                          tSrc.addr(),
                                          2,  // tileRank
                                          tileExtents,
                                          invalid_stream); // Invalid stream

    EXPECT_EQ(status, CUPHY_STATUS_INTERNAL_ERROR)
        << "Expected CUPHY_STATUS_INTERNAL_ERROR for invalid CUDA stream";
}

} // namespace


////////////////////////////////////////////////////////////////////////
// Tile.Basic2D
TEST(Tile, Basic2D)
{
    do_tile_test_2D<CUPHY_R_32F>(32, 8, 2, 3);
    do_tile_test_2D<CUPHY_R_32F>(32, 8, 2, 3);
    do_tile_test_2D<CUPHY_R_16F>(32, 8, 2, 3);
    do_tile_test_2D<CUPHY_R_8U> (32, 8, 2, 3);
    //do_tile_test_2D<CUPHY_C_64F>(32, 8, 2, 3);
}

// Tile.LargeElementSizes
TEST(Tile, LargeElementSizes) {
    const TileTestParams test_cases[] = {
        {16, 8, 2, 2, "Small dimensions"},
        {32, 16, 3, 2, "Medium dimensions, uneven tiling"},
        {64, 32, 4, 4, "Large dimensions"},
        {48, 24, 2, 4, "Mixed dimensions"}
    };

    for (const auto& test : test_cases) {
        // Test 8-byte elements (CUPHY_C_32F - complex float)
        run_complex_tile_test<CUPHY_C_32F, CUPHY_R_32F>(test, "8-byte");

        // Test 16-byte elements (CUPHY_C_64F - complex double)
        run_complex_tile_test<CUPHY_C_64F, CUPHY_R_64F>(test, "16-byte");
    }
}

////////////////////////////////////////////////////////////////////////
// Tile.BitTensor
TEST(Tile, BitTensor) {
    const TileTestParams test_cases[] = {
        {32, 8, 2, 2, "Small dimensions"},
        {64, 16, 3, 2, "Medium dimensions, uneven tiling"},
        {128, 32, 4, 4, "Large dimensions"},
        {96, 24, 2, 4, "Mixed dimensions"}
    };

    for (const auto& test : test_cases) {
        run_bit_tile_test(test);
    }
}

////////////////////////////////////////////////////////////////////////
// Tile.SizeMismatch
TEST(Tile, SizeMismatch) {
    // Test cases with source dimensions and tile factors
    const TileTestParams test_cases[] = {
        {4, 4, 2, 2, "Basic size mismatch"},  // Should produce 8x8 output
        {8, 4, 2, 3, "Rectangular size mismatch"},  // Should produce 16x12 output
        {6, 6, 3, 2, "Uneven size mismatch"}   // Should produce 18x12 output
    };

    // Corresponding mismatched destination dimensions to test
    const std::array<std::array<int, 2>, 3> mismatched_dst_dims = {{
        {{8, 4}},   // Missing columns (should be 8x8)
        {{16, 8}},  // Missing columns (should be 16x12)
        {{18, 6}}   // Missing columns (should be 18x12)
    }};

    for (size_t i = 0; i < std::size(test_cases); ++i) {
        run_size_mismatch_test<CUPHY_R_32F>(test_cases[i], mismatched_dst_dims[i]);
    }
}

////////////////////////////////////////////////////////////////////////
// Tile.InternalError
TEST(Tile, InternalError) {
    run_internal_error_test();
}

////////////////////////////////////////////////////////////////////////
// Tile.UnsupportedElementSize
TEST(Tile, UnsupportedElementSize) {
    // Create a struct that has a non-standard size (3 bytes)
    struct CustomType {
        uint8_t a;
        uint16_t b;  // Due to alignment, total size will be 4 bytes
        // Force no padding to get 3 bytes
        CustomType() : a(0), b(0) {}
    } __attribute__((packed));  // Force no padding

    // Create source tensor with dimensions 4x4
    const std::array<int, 2> SRC_DIMS = {{4, 4}};

    // Create tensor descriptors manually since we can't use typed_tensor for custom types
    cuphyTensorDescriptor_t srcDesc;
    cuphyCreateTensorDescriptor(&srcDesc);
    cuphySetTensorDescriptor(srcDesc,
                            CUPHY_VOID,  // Use void type since it's custom
                            2,           // 2D tensor
                            SRC_DIMS.data(),
                            nullptr,     // Let library compute strides
                            0);          // No flags

    cuphyTensorDescriptor_t dstDesc;
    cuphyCreateTensorDescriptor(&dstDesc);

    // Destination dimensions for 2x2 tiling
    const std::array<int, 2> DST_DIMS = {{8, 8}};
    cuphySetTensorDescriptor(dstDesc,
                            CUPHY_VOID,  // Use void type since it's custom
                            2,           // 2D tensor
                            DST_DIMS.data(),
                            nullptr,     // Let library compute strides
                            0);          // No flags

    // Allocate memory for source and destination
    CustomType* srcData = new CustomType[SRC_DIMS[0] * SRC_DIMS[1]];
    CustomType* dstData = new CustomType[DST_DIMS[0] * DST_DIMS[1]];

    // Initialize source data
    for(int i = 0; i < SRC_DIMS[0] * SRC_DIMS[1]; i++) {
        srcData[i].a = static_cast<uint8_t>(i % 256);
        srcData[i].b = static_cast<uint16_t>(i % 65536);
    }

    // Try to tile with 2x2 tile extents
    const int tileRank = 2;
    const int tileExtents[2] = {2, 2};

    // This should return CUPHY_STATUS_UNSUPPORTED_TYPE since element size is 3 bytes
    cuphyStatus_t status = cuphyTileTensor(dstDesc,
                                          dstData,
                                          srcDesc,
                                          srcData,
                                          tileRank,
                                          tileExtents,
                                          0); // stream

    EXPECT_EQ(status, CUPHY_STATUS_UNSUPPORTED_TYPE)
        << "Expected CUPHY_STATUS_UNSUPPORTED_TYPE for 3-byte element size";

    // Cleanup
    delete[] srcData;
    delete[] dstData;
    cuphyDestroyTensorDescriptor(srcDesc);
    cuphyDestroyTensorDescriptor(dstDesc);
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}
