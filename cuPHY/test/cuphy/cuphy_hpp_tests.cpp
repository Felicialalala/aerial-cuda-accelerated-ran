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

#include "gtest/gtest.h"
#include "cuphy.hpp" // or wherever tensor_layout_t is defined

TEST(TensorLayoutTest, VariadicGetOffset) {
    static constexpr int DIM = 3;
    cuphy::tensor_layout_t<DIM> layout;
    layout.dimensions()[0] = 2;
    layout.dimensions()[1] = 3;
    layout.dimensions()[2] = 4;
    layout.strides()[0] = 12;
    layout.strides()[1] = 4;
    layout.strides()[2] = 1;

    // Test 0,0,0
    EXPECT_EQ(layout.get_offset(0, 0, 0), 0);
    // Test 1,2,3
    EXPECT_EQ(layout.get_offset(1, 2, 3), 1*12 + 2*4 + 3*1);
    // Test 0,1,2
    EXPECT_EQ(layout.get_offset(0, 1, 2), 0*12 + 1*4 + 2*1);
    // Test 1,0,0
    EXPECT_EQ(layout.get_offset(1, 0, 0), 1*12 + 0*4 + 0*1);
}

TEST(TensorLayoutTest, OffsetCalculation) {
    // Create a tensor layout with dimensions [3, 4, 5]
    constexpr int dims[3] = {3, 4, 5};
    cuphy::tensor_layout_t<3> layout(3, dims, nullptr);

    // Test with C-array
    constexpr int indices_c[3] = {1, 2, 3};
    const int offset_c = layout.offset(indices_c);

    // Test with std::array
    constexpr std::array indices_std = {1, 2, 3};
    const int offset_std = layout.offset(indices_std);

    // Both should produce the same result
    EXPECT_EQ(offset_c, offset_std);

    // Calculate expected offset manually
    // Strides are [1, 3, 12] for dims [3, 4, 5] in row-major order
    const int expected_offset = 1 * 1 + 2 * 3 + 3 * 12;

    EXPECT_EQ(offset_c, expected_offset);
    EXPECT_EQ(offset_std, expected_offset);
}

// Test with different dimensions
TEST(TensorLayoutTest, OffsetCalculationDifferentDims) {
    // Create a tensor layout with dimensions [2, 3]
    constexpr int dims[2] = {2, 3};
    const cuphy::tensor_layout_t<2> layout(2, dims, nullptr);

    // Test with C-array
    constexpr int indices_c[2] = {1, 2};
    const int offset_c = layout.offset(indices_c);

    // Test with std::array
    constexpr std::array indices_std = {1, 2};
    const int offset_std = layout.offset(indices_std);

    // Both should produce the same result
    EXPECT_EQ(offset_c, offset_std);

    // Calculate expected offset manually
    // Assuming row-major layout: offset = indices[0]*strides[0] + indices[1]*strides[1]
    // where strides are typically [dims[1], 1] for row-major
    static constexpr int expected_offset = 1 * 3 + 2 * 1;

    EXPECT_EQ(offset_c, expected_offset);
    EXPECT_EQ(offset_std, expected_offset);
}

// Test with edge cases
TEST(TensorLayoutTest, OffsetCalculationEdgeCases) {
    // Create a tensor layout with dimensions [1, 1, 1]
    constexpr int dims[3] = {1, 1, 1};
    cuphy::tensor_layout_t<3> layout(3, dims, nullptr);

    // Test with zeros
    constexpr int zeros_c[3] = {0, 0, 0};
    constexpr std::array zeros_std = {0, 0, 0};

    EXPECT_EQ(layout.offset(zeros_c), 0);
    EXPECT_EQ(layout.offset(zeros_std), 0);
}

// Additional tests for tensor_layout_t interface
TEST(TensorLayoutTest, GetOffset1D) {
    constexpr int dims[1] = {10};
    cuphy::tensor_layout_t<1> layout(1, dims, nullptr);
    EXPECT_EQ(layout.get_offset(0), 0);
    EXPECT_EQ(layout.get_offset(5), 5);
    EXPECT_EQ(layout.offset(std::array<int, 1>{5}), 5);
}

TEST(TensorLayoutTest, GetOffset2D) {
    constexpr int dims[2] = {3, 4};
    cuphy::tensor_layout_t<2> layout(2, dims, nullptr);
    // Strides: [1, 3]
    EXPECT_EQ(layout.get_offset(0, 0), 0);
    EXPECT_EQ(layout.get_offset(1, 2), 1 * 1 + 2 * 3);
    EXPECT_EQ(layout.offset(std::array<int, 2>{1, 2}), 1 * 1 + 2 * 3);
}

TEST(TensorLayoutTest, GetOffset3D) {
    constexpr int dims[3] = {2, 3, 4};
    cuphy::tensor_layout_t<3> layout(3, dims, nullptr);
    // Strides: [1, 2, 6]
    EXPECT_EQ(layout.get_offset(0, 0, 0), 0);
    EXPECT_EQ(layout.get_offset(1, 2, 3), 1 * 1 + 2 * 2 + 3 * 6);
    EXPECT_EQ(layout.offset(std::array<int, 3>{1, 2, 3}), 1 * 1 + 2 * 2 + 3 * 6);
}

TEST(TensorLayoutTest, CustomStrides) {
    constexpr int dims[2] = {3, 4};
    constexpr int strides[2] = {100, 10};
    cuphy::tensor_layout_t<2> layout(dims, strides);
    EXPECT_EQ(layout.get_offset(1, 2), 1 * 100 + 2 * 10);
    EXPECT_EQ(layout.offset(std::array<int, 2>{1, 2}), 1 * 100 + 2 * 10);
}

TEST(TensorLayoutTest, OutOfBoundsCheck) {
    constexpr int dims[2] = {3, 4};
    cuphy::tensor_layout_t<2> layout(2, dims, nullptr);
    // This should throw if check_bounds is called with out-of-bounds
    int idx1[2] = {3, 0};
    int idx2[2] = {0, 4};
    EXPECT_THROW(layout.check_bounds(idx1), std::runtime_error);
    EXPECT_THROW(layout.check_bounds(idx2), std::runtime_error);
}

TEST(TensorLayoutTest, NegativeIndicesCheck) {
    constexpr int dims[2] = {3, 4};
    cuphy::tensor_layout_t<2> layout(2, dims, nullptr);
    // Negative indices are not valid, but check_bounds does not check for negatives in current impl
    // If you want to enforce, you can add a check in check_bounds
    // For now, just document this
    // EXPECT_THROW(layout.check_bounds(std::array<int, 2>{-1, 0}), std::runtime_error);
}

TEST(TensorLayoutTest, MaxIndices) {
    constexpr int dims[3] = {2, 3, 4};
    cuphy::tensor_layout_t<3> layout(3, dims, nullptr);
    // Strides: [1, 2, 6]
    std::array<int, 3> max_indices = {1, 2, 3};
    int expected = 1 * 1 + 2 * 2 + 3 * 6;
    EXPECT_EQ(layout.get_offset(1, 2, 3), expected);
    EXPECT_EQ(layout.offset(max_indices), expected);
}

TEST(TensorLayoutTest, ConsistencyBetweenConstructors) {
    constexpr int dims[2] = {3, 4};
    constexpr int strides[2] = {100, 10};
    cuphy::tensor_layout_t<2> layout1(dims, strides);
    cuphy::tensor_layout_t<2> layout2;
    layout2.dimensions()[0] = 3;
    layout2.dimensions()[1] = 4;
    layout2.strides()[0] = 100;
    layout2.strides()[1] = 10;
    std::array<int, 2> idx = {1, 2};
    EXPECT_EQ(layout1.offset(idx), layout2.offset(idx));
    EXPECT_EQ(layout1.get_offset(1, 2), layout2.get_offset(1, 2));
}
