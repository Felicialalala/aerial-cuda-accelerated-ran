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
#include "cuphy.h"
#include "cuphy_internal.h"
#include "tensor_desc.hpp"

class VecConstructorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(VecConstructorTest, DefaultConstructor) {
    // Test default constructor
    cuphy::vec<int, 3> v;
    // Default constructor should not initialize elements, but we can check they're accessible
    v[0] = 1;
    v[1] = 2;
    v[2] = 3;
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);
}

TEST_F(VecConstructorTest, ArrayConstructor) {
    // Test constructor from array
    constexpr int arr[3] = {10, 20, 30};
    const cuphy::vec<int, 3> v(arr);

    EXPECT_EQ(v[0], 10);
    EXPECT_EQ(v[1], 20);
    EXPECT_EQ(v[2], 30);
}

TEST_F(VecConstructorTest, FillMethod) {
    // Test fill method
    cuphy::vec<int, 4> v;
    v.fill(42);

    EXPECT_EQ(v[0], 42);
    EXPECT_EQ(v[1], 42);
    EXPECT_EQ(v[2], 42);
    EXPECT_EQ(v[3], 42);
}

TEST_F(VecConstructorTest, IteratorAccess) {
    // Test iterator access
    constexpr int arr[3] = {5, 10, 15};
    const cuphy::vec<int, 3> v(arr);

    int sum = 0;
    for (const auto& elem : v) {
        sum += elem;
    }

    EXPECT_EQ(sum, 30);
}

TEST_F(VecConstructorTest, EqualityOperator) {
    // Test equality operator
    constexpr int arr1[3] = {1, 2, 3};
    constexpr int arr2[3] = {1, 2, 3};
    constexpr int arr3[3] = {1, 2, 4};

    const cuphy::vec<int, 3> v1(arr1);
    const cuphy::vec<int, 3> v2(arr2);
    const cuphy::vec<int, 3> v3(arr3);

    EXPECT_TRUE(v1 == v2);
    EXPECT_FALSE(v1 == v3);
}

TEST_F(VecConstructorTest, InequalityOperator) {
    // Test inequality operator
    constexpr int arr1[3] = {1, 2, 3};
    constexpr int arr2[3] = {1, 2, 3};
    constexpr int arr3[3] = {1, 2, 4};

    const cuphy::vec<int, 3> v1(arr1);
    const cuphy::vec<int, 3> v2(arr2);
    const cuphy::vec<int, 3> v3(arr3);

    EXPECT_FALSE(v1 != v2);
    EXPECT_TRUE(v1 != v3);
}

TEST_F(VecConstructorTest, DifferentTypes) {
    // Test with different types
    constexpr float arr[2] = {1.5f, 2.5f};
    const cuphy::vec<float, 2> v(arr);

    EXPECT_FLOAT_EQ(v[0], 1.5f);
    EXPECT_FLOAT_EQ(v[1], 2.5f);
}

TEST_F(VecConstructorTest, DotProduct) {
    // Test dot product function that works with vec
    constexpr int arr1[4] = {1, 2, 3, 4};
    constexpr int arr2[4] = {5, 6, 7, 8};

    const cuphy::vec<int, 4> v1(arr1);
    const cuphy::vec<int, 4> v2(arr2);

    const int result = dot<int, int, 4>(v1, v2);
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    EXPECT_EQ(result, 70);
}

// This test should fail at compile time due to static_assert
// Uncomment to test the static assertion
// This is verified as part of code development.
/*
TEST_F(VecConstructorTest, StaticAssertTest) {
    constexpr int arr[2] = {1, 2}; // Array with only 2 elements
    const cuphy::vec<int, 3> v(arr);  // Trying to initialize a cuphy::vec<int, 3> with 2 elements
    // This should trigger the static_assert: "Initializer list does not have enough dimensions"
}
*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
