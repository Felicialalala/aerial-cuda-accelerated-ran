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
#include "cuphy_api.h"
#include <vector>
#include <memory>

class PrachValidationTest : public ::testing::Test 
{
protected:
    void SetUp() override 
    {
        // Create valid baseline parameters
        setupValidParams();
    }

    void setupValidParams()
    {
        // Valid cell parameters (Format 0 - ZERO)
        validCellPrms = {
            .occaStartIdx = 0,
            .nFdmOccasions = 1,
            .N_ant = 4,
            .FR = 1,
            .duplex = 0,
            .mu = 0,
            .configurationIndex = 0,
            .restrictedSet = 0
        };

        // Valid occasion parameters
        validOccaPrms = {
            .cellPrmStatIdx = 0,
            .prachRootSequenceIndex = 0,
            .prachZeroCorrConf = 0
        };

        // Valid debug parameters
        validDbgPrms = {
            .pOutFileName = nullptr,
            .enableApiLogging = 0
        };

        // Valid static parameters
        validStatPrms = {
            .pOutInfo = nullptr,
            .nMaxCells = 1,
            .pCellPrms = &validCellPrms,
            .pOccaPrms = &validOccaPrms,
            .nMaxOccaProc = 1,
            .pDbg = &validDbgPrms,
            .enableUlRxBf = 0
        };
    }

    cuphyPrachStatPrms_t validStatPrms{};
    cuphyPrachCellStatPrms_t validCellPrms{};
    cuphyPrachOccaStatPrms_t validOccaPrms{};
    cuphyPrachStatDbgPrms_t validDbgPrms{};
};

// Test cases for null pointer validation
TEST_F(PrachValidationTest, NullPointerValidation) 
{
    // Test null pStatPrms
    EXPECT_EQ(cuphyValidatePrachParams(nullptr), CUPHY_STATUS_INVALID_ARGUMENT);

    // Test null pCellPrms
    cuphyPrachStatPrms_t nullCellPrms = validStatPrms;
    nullCellPrms.pCellPrms = nullptr;
    EXPECT_EQ(cuphyValidatePrachParams(&nullCellPrms), CUPHY_STATUS_INVALID_ARGUMENT);

    // Test null pOccaPrms
    cuphyPrachStatPrms_t nullOccaPrms = validStatPrms;
    nullOccaPrms.pOccaPrms = nullptr;
    EXPECT_EQ(cuphyValidatePrachParams(&nullOccaPrms), CUPHY_STATUS_INVALID_ARGUMENT);
}

// Test case for valid parameters
TEST_F(PrachValidationTest, ValidParameters) 
{
    EXPECT_EQ(cuphyValidatePrachParams(&validStatPrms), CUPHY_STATUS_SUCCESS);
}

// Test cases for cell parameter validation
TEST_F(PrachValidationTest, CellParameterValidation) 
{
    // Test invalid restrictedSet
    cuphyPrachStatPrms_t invalidRestrictedSet = validStatPrms;
    cuphyPrachCellStatPrms_t invalidCellPrms = validCellPrms;
    invalidCellPrms.restrictedSet = 1;
    invalidRestrictedSet.pCellPrms = &invalidCellPrms;
    invalidRestrictedSet.nMaxCells = 1;
    EXPECT_EQ(cuphyValidatePrachParams(&invalidRestrictedSet), CUPHY_STATUS_NOT_SUPPORTED);

    // Test invalid mu (greater than 1)
    cuphyPrachStatPrms_t invalidMu = validStatPrms;
    cuphyPrachCellStatPrms_t invalidCellPrmsMu = validCellPrms;
    invalidCellPrmsMu.mu = 2;
    invalidMu.pCellPrms = &invalidCellPrmsMu;
    invalidMu.nMaxCells = 1;
    EXPECT_EQ(cuphyValidatePrachParams(&invalidMu), CUPHY_STATUS_NOT_SUPPORTED);

    // Test valid mu values (0 and 1)
    cuphyPrachStatPrms_t validMu0 = validStatPrms;
    cuphyPrachCellStatPrms_t validCellPrmsMu0 = validCellPrms;
    validCellPrmsMu0.mu = 0;
    validMu0.pCellPrms = &validCellPrmsMu0;
    validMu0.nMaxCells = 1;
    EXPECT_EQ(cuphyValidatePrachParams(&validMu0), CUPHY_STATUS_SUCCESS);

    cuphyPrachStatPrms_t validMu1 = validStatPrms;
    cuphyPrachCellStatPrms_t validCellPrmsMu1 = validCellPrms;
    validCellPrmsMu1.mu = 1;
    validMu1.pCellPrms = &validCellPrmsMu1;
    validMu1.nMaxCells = 1;
    EXPECT_EQ(cuphyValidatePrachParams(&validMu1), CUPHY_STATUS_SUCCESS);
}

// Test cases for unsupported preamble formats
TEST_F(PrachValidationTest, UnsupportedPreambleFormats) 
{
    // Test unsupported preamble format (ONE)
    cuphyPrachStatPrms_t unsupportedFormat = validStatPrms;
    cuphyPrachCellStatPrms_t unsupportedCellPrms = validCellPrms;
    unsupportedCellPrms.configurationIndex = 28;  // Maps to PreambleFormat::ONE
    unsupportedFormat.pCellPrms = &unsupportedCellPrms;
    unsupportedFormat.nMaxCells = 1;
    EXPECT_EQ(cuphyValidatePrachParams(&unsupportedFormat), CUPHY_STATUS_NOT_SUPPORTED);

    // Test unsupported preamble format (TWO)
    unsupportedCellPrms.configurationIndex = 34;  // Maps to PreambleFormat::TWO
    EXPECT_EQ(cuphyValidatePrachParams(&unsupportedFormat), CUPHY_STATUS_NOT_SUPPORTED);

    // Test unsupported preamble format (THREE)
    unsupportedCellPrms.configurationIndex = 39;  // Maps to PreambleFormat::THREE
    EXPECT_EQ(cuphyValidatePrachParams(&unsupportedFormat), CUPHY_STATUS_NOT_SUPPORTED);
}

// Test cases for occasion parameter validation
TEST_F(PrachValidationTest, OccasionParameterValidation) 
{
    // Test invalid prachZeroCorrConf (>= 16)
    cuphyPrachStatPrms_t invalidZeroCorr = validStatPrms;
    cuphyPrachOccaStatPrms_t invalidOccaPrms = validOccaPrms;
    invalidOccaPrms.prachZeroCorrConf = 16;
    invalidZeroCorr.pOccaPrms = &invalidOccaPrms;
    EXPECT_EQ(cuphyValidatePrachParams(&invalidZeroCorr), CUPHY_STATUS_INVALID_ARGUMENT);

    // Test prachZeroCorrConf at boundary (15 should be valid)
    invalidOccaPrms.prachZeroCorrConf = 15;
    EXPECT_EQ(cuphyValidatePrachParams(&invalidZeroCorr), CUPHY_STATUS_SUCCESS);

    // Test invalid prachRootSequenceIndex for L_RA=839 (format ZERO)
    cuphyPrachStatPrms_t invalidRootSeq = validStatPrms;
    cuphyPrachOccaStatPrms_t invalidOccaPrmsRoot = validOccaPrms;
    invalidOccaPrmsRoot.prachRootSequenceIndex = 838;  // >= (839-1)
    invalidRootSeq.pOccaPrms = &invalidOccaPrmsRoot;
    EXPECT_EQ(cuphyValidatePrachParams(&invalidRootSeq), CUPHY_STATUS_INVALID_ARGUMENT);

    // Test invalid prachRootSequenceIndex for L_RA=139 (format B4)
    cuphyPrachStatPrms_t invalidRootSeqB4 = validStatPrms;
    cuphyPrachCellStatPrms_t cellPrmsB4 = validCellPrms;
    cellPrmsB4.configurationIndex = 200;  // Maps to PreambleFormat::B4 in FR1 FDD table
    cuphyPrachOccaStatPrms_t invalidOccaPrmsRootB4 = validOccaPrms;
    invalidOccaPrmsRootB4.prachRootSequenceIndex = 138;  // >= (139-1)
    invalidRootSeqB4.pCellPrms = &cellPrmsB4;
    invalidRootSeqB4.nMaxCells = 1;
    invalidRootSeqB4.pOccaPrms = &invalidOccaPrmsRootB4;
    EXPECT_EQ(cuphyValidatePrachParams(&invalidRootSeqB4), CUPHY_STATUS_INVALID_ARGUMENT);
}

// Test cases for occasion index bounds
TEST_F(PrachValidationTest, OccasionIndexBounds) 
{
    // Test occasion index exceeding nMaxOccasions
    cuphyPrachStatPrms_t invalidOccaIdx = validStatPrms;
    cuphyPrachCellStatPrms_t invalidCellPrms = validCellPrms;
    invalidCellPrms.occaStartIdx = 2;  // This would exceed nMaxOccaProc=1
    invalidOccaIdx.pCellPrms = &invalidCellPrms;
    EXPECT_EQ(cuphyValidatePrachParams(&invalidOccaIdx), CUPHY_STATUS_INVALID_ARGUMENT);
}

// Test cases for total occasions exceeding max
TEST_F(PrachValidationTest, TotalOccasionsExceedMax) 
{
    // Test total cell occasions exceeding nMaxOccasions
    cuphyPrachStatPrms_t exceedMax = validStatPrms;
    cuphyPrachCellStatPrms_t exceedCellPrms = validCellPrms;
    exceedCellPrms.nFdmOccasions = 2;  // This would exceed nMaxOccaProc=1
    exceedMax.pCellPrms = &exceedCellPrms;
    EXPECT_EQ(cuphyValidatePrachParams(&exceedMax), CUPHY_STATUS_INVALID_ARGUMENT);
}

// Test cases for edge cases
TEST_F(PrachValidationTest, EdgeCases) 
{
    // Test with nMaxCells = 0 (should be valid)
    cuphyPrachStatPrms_t zeroCells = validStatPrms;
    zeroCells.nMaxCells = 0;
    EXPECT_EQ(cuphyValidatePrachParams(&zeroCells), CUPHY_STATUS_SUCCESS);

    // Test with nMaxOccaProc = 0 (should be invalid as nFdmOccasions=1 > 0)
    cuphyPrachStatPrms_t zeroOccas = validStatPrms;
    zeroOccas.nMaxOccaProc = 0;
    EXPECT_EQ(cuphyValidatePrachParams(&zeroOccas), CUPHY_STATUS_INVALID_ARGUMENT);

    // Test minimum valid configuration
    cuphyPrachCellStatPrms_t minCellPrms = {
        .occaStartIdx = 0,
        .nFdmOccasions = 1,
        .N_ant = 1,
        .FR = 1,
        .duplex = 0,
        .mu = 0,
        .configurationIndex = 0,
        .restrictedSet = 0
    };
    cuphyPrachOccaStatPrms_t minOccaPrms = {
        .cellPrmStatIdx = 0,
        .prachRootSequenceIndex = 0,
        .prachZeroCorrConf = 0
    };
    cuphyPrachStatDbgPrms_t minDbgPrms = {
        .pOutFileName = nullptr,
        .enableApiLogging = 0
    };
    cuphyPrachStatPrms_t minValid = {
        .pOutInfo = nullptr,
        .nMaxCells = 1,
        .pCellPrms = &minCellPrms,
        .pOccaPrms = &minOccaPrms,
        .nMaxOccaProc = 1,
        .pDbg = &minDbgPrms,
        .enableUlRxBf = 0
    };
    EXPECT_EQ(cuphyValidatePrachParams(&minValid), CUPHY_STATUS_SUCCESS);
}

// Test cases for different FR and duplex combinations
TEST_F(PrachValidationTest, FRDuplexCombinations) 
{
    // Test FR1 FDD (supported - Format 0)
    cuphyPrachCellStatPrms_t fr1fddCellPrms = {
        .occaStartIdx = 0,
        .nFdmOccasions = 1,
        .N_ant = 4,
        .FR = 1,
        .duplex = 0,
        .mu = 0,
        .configurationIndex = 0,
        .restrictedSet = 0
    };
    cuphyPrachStatPrms_t fr1fdd = validStatPrms;
    fr1fdd.pCellPrms = &fr1fddCellPrms;
    EXPECT_EQ(cuphyValidatePrachParams(&fr1fdd), CUPHY_STATUS_SUCCESS);

    // Test FR1 TDD (supported - Format 0)
    cuphyPrachCellStatPrms_t fr1tddCellPrms = {
        .occaStartIdx = 0,
        .nFdmOccasions = 1,
        .N_ant = 4,
        .FR = 1,
        .duplex = 1,
        .mu = 0,
        .configurationIndex = 0,
        .restrictedSet = 0
    };
    cuphyPrachStatPrms_t fr1tdd = validStatPrms;
    fr1tdd.pCellPrms = &fr1tddCellPrms;
    EXPECT_EQ(cuphyValidatePrachParams(&fr1tdd), CUPHY_STATUS_SUCCESS);

    // Test FR2 (unsupported format for config index 0)
    cuphyPrachCellStatPrms_t fr2CellPrms = {
        .occaStartIdx = 0,
        .nFdmOccasions = 1,
        .N_ant = 4,
        .FR = 2,
        .duplex = 1,
        .mu = 1,
        .configurationIndex = 0,
        .restrictedSet = 0
    };
    cuphyPrachStatPrms_t fr2 = validStatPrms;
    fr2.pCellPrms = &fr2CellPrms;
    EXPECT_EQ(cuphyValidatePrachParams(&fr2), CUPHY_STATUS_NOT_SUPPORTED);
} 