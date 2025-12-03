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

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_internal.h"
#include "descrambling.cuh"
#include "tensor_desc.hpp"
#include "srstx.hpp"

static __device__ __constant__ uint16_t SRS_BW_TABLE[64][8] =
   {{4,1,4,1,4,1,4,1},
    {8,1,4,2,4,1,4,1},
    {12,1,4,3,4,1,4,1},
    {16,1,4,4,4,1,4,1},
    {16,1,8,2,4,2,4,1},
    {20,1,4,5,4,1,4,1},
    {24,1,4,6,4,1,4,1},
    {24,1,12,2,4,3,4,1},
    {28,1,4,7,4,1,4,1},
    {32,1,16,2,8,2,4,2},
    {36,1,12,3,4,3,4,1},
    {40,1,20,2,4,5,4,1},
    {48,1,16,3,8,2,4,2},
    {48,1,24,2,12,2,4,3},
    {52,1,4,13,4,1,4,1},
    {56,1,28,2,4,7,4,1},
    {60,1,20,3,4,5,4,1},
    {64,1,32,2,16,2,4,4},
    {72,1,24,3,12,2,4,3},
    {72,1,36,2,12,3,4,3},
    {76,1,4,19,4,1,4,1},
    {80,1,40,2,20,2,4,5},
    {88,1,44,2,4,11,4,1},
    {96,1,32,3,16,2,4,4},
    {96,1,48,2,24,2,4,6},
    {104,1,52,2,4,13,4,1},
    {112,1,56,2,28,2,4,7},
    {120,1,60,2,20,3,4,5},
    {120,1,40,3,8,5,4,2},
    {120,1,24,5,12,2,4,3},
    {128,1,64,2,32,2,4,8},
    {128,1,64,2,16,4,4,4},
    {128,1,16,8,8,2,4,2},
    {132,1,44,3,4,11,4,1},
    {136,1,68,2,4,17,4,1},
    {144,1,72,2,36,2,4,9},
    {144,1,48,3,24,2,12,2},
    {144,1,48,3,16,3,4,4},
    {144,1,16,9,8,2,4,2},
    {152,1,76,2,4,19,4,1},
    {160,1,80,2,40,2,4,10},
    {160,1,80,2,20,4,4,5},
    {160,1,32,5,16,2,4,4},
    {168,1,84,2,28,3,4,7},
    {176,1,88,2,44,2,4,11},
    {184,1,92,2,4,23,4,1},
    {192,1,96,2,48,2,4,12},
    {192,1,96,2,24,4,4,6},
    {192,1,64,3,16,4,4,4},
    {192,1,24,8,8,3,4,2},
    {208,1,104,2,52,2,4,13},
    {216,1,108,2,36,3,4,9},
    {224,1,112,2,56,2,4,14},
    {240,1,120,2,60,2,4,15},
    {240,1,80,3,20,4,4,5},
    {240,1,48,5,16,3,8,2},
    {240,1,24,10,12,2,4,3},
    {256,1,128,2,64,2,4,16},
    {256,1,128,2,32,4,4,8},
    {256,1,16,16,8,2,4,2},
    {264,1,132,2,44,3,4,11},
    {272,1,136,2,68,2,4,17},
    {272,1,68,4,4,17,4,1},
    {272,1,16,17,8,2,4,2}};
    
static __device__ __constant__ int8_t d_phi_6[30][6] = 
                              {{-3, -1, 3, 3, -1, -3},
                              {-3, 3, -1, -1, 3, -3},
                              {-3, -3, -3, 3, 1, -3},
                              {1, 1, 1, 3, -1, -3},
                              {1, 1, 1, -3, -1, 3},
                              {-3, 1, -1, -3, -3, -3},
                              {-3, 1, 3, -3, -3, -3},
                              {-3, -1, 1, -3, 1, -1},
                              {-3, -1, -3, 1, -3, -3},
                              {-3, -3, 1, -3, 3, -3},
                              {-3, 1, 3, 1, -3, -3},
                              {-3, -1, -3, 1, 1, -3},
                              {1, 1, 3, -1, -3, 3},
                              {1, 1, 3, 3, -1, 3},
                              {1, 1, 1, -3, 3, -1},
                              {1, 1, 1, -1, 3, -3},
                              {-3, -1, -1, -1, 3, -1},
                              {-3, -3, -1, 1, -1, -3},
                              {-3, -3, -3, 1, -3, -1},
                              {-3, 1, 1, -3, -1, -3},
                              {-3, 3, -3, 1, 1, -3},
                              {-3, 1, -3, -3, -3, -1},
                              {1, 1, -3, 3, 1, 3},
                              {1, 1, -3, -3, 1, -3},
                              {1, 1, 3, -1, 3, 3},
                              {1, 1, -3, 1, 3, 3},
                              {1, 1, -1, -1, 3, -1},
                              {1, 1, -1, 3, -1, -1},
                              {1, 1, -1, 3, -3, -1},
                              {1, 1, -3, 1, -1, -1}};

static __device__ __constant__ int8_t d_phi_12[30][12] = 
                                {{-3, 1, -3, -3, -3, 3, -3, -1, 1, 1, 1, -3},
                                {-3, 3, 1, -3, 1, 3, -1, -1, 1, 3, 3, 3},
                                {-3, 3, 3, 1, -3, 3, -1, 1, 3, -3, 3, -3},
                                {-3, -3, -1, 3, 3, 3, -3, 3, -3, 1, -1, -3},
                                {-3, -1, -1, 1, 3, 1, 1, -1, 1, -1, -3, 1},
                                {-3, -3, 3, 1, -3, -3, -3, -1, 3, -1, 1, 3},
                                {1, -1, 3, -1, -1, -1, -3, -1, 1, 1, 1, -3},
                                {-1, -3, 3, -1, -3, -3, -3, -1, 1, -1, 1, -3},
                                {-3, -1, 3, 1, -3, -1, -3, 3, 1, 3, 3, 1},
                                {-3, -1, -1, -3, -3, -1, -3, 3, 1, 3, -1, -3},
                                {-3, 3, -3, 3, 3, -3, -1, -1, 3, 3, 1, -3},
                                {-3, -1, -3, -1, -1, -3, 3, 3, -1, -1, 1, -3},
                                {-3, -1, 3, -3, -3, -1, -3, 1, -1, -3, 3, 3},
                                {-3, 1, -1, -1, 3, 3, -3, -1, -1, -3, -1, -3},
                                {1, 3, -3, 1, 3, 3, 3, 1, -1, 1, -1, 3},
                                {-3, 1, 3, -1, -1, -3, -3, -1, -1, 3, 1, -3},
                                {-1, -1, -1, -1, 1, -3, -1, 3, 3, -1, -3, 1},
                                {-1, 1, 1, -1, 1, 3, 3, -1, -1, -3, 1, -3},
                                {-3, 1, 3, 3, -1, -1, -3, 3, 3, -3, 3, -3},
                                {-3, -3, 3, -3, -1, 3, 3, 3, -1, -3, 1, -3},
                                {3, 1, 3, 1, 3, -3, -1, 1, 3, 1, -1, -3},
                                {-3, 3, 1, 3, -3, 1, 1, 1, 1, 3, -3, 3},
                                {-3, 3, 3, 3, -1, -3, -3, -1, -3, 1, 3, -3},
                                {3, -1, -3, 3, -3, -1, 3, 3, 3, -3, -1, -3},
                                {-3, -1, 1, -3, 1, 3, 3, 3, -1, -3, 3, 3},
                                {-3, 3, 1, -1, 3, 3, -3, 1, -1, 1, -1, 1},
                                {-1, 1, 3, -3, 1, -1, 1, -1, -1, -3, 1, -1},
                                {-3, -3, 3, 3, 3, -3, -1, 1, -3, 3, 1, -3},
                                {1, -1, 3, 1, 1, -1, -1, -1, 1, 3, -3, 1},
                                {-3, 3, -3, 3, -3, -3, 3, -1, -1, 1, 3, -3}};

static __device__ __constant__ int8_t d_phi_18[30][18] = 
                                {{-1, 3, -1, -3, 3, 1, -3, -1, 3, -3, -1, -1, 1, 1, 1, -1, -1, -1},
                                {3, -3, 3, -1, 1, 3, -3, -1, -3, -3, -1, -3, 3, 1, -1, 3, -3, 3},
                                {-3, 3, 1, -1, -1, 3, -3, -1, 1, 1, 1, 1, 1, -1, 3, -1, -3, -1},
                                {-3, -3, 3, 3, 3, 1, -3, 1, 3, 3, 1, -3, -3, 3, -1, -3, -1, 1},
                                {1, 1, -1, -1, -3, -1, 1, -3, -3, -3, 1, -3, -1, -1, 1, -1, 3, 1},
                                {3, -3, 1, 1, 3, -1, 1, -1, -1, -3, 1, 1, -1, 3, 3, -3, 3, -1},
                                {-3, 3, -1, 1, 3, 1, -3, -1, 1, 1, -3, 1, 3, 3, -1, -3, -3, -3},
                                {1, 1, -3, 3, 3, 1, 3, -3, 3, -1, 1, 1, -1, 1, -3, -3, -1, 3},
                                {-3, 1, -3, -3, 1, -3, -3, 3, 1, -3, -1, -3, -3, -3, -1, 1, 1, 3},
                                {3, -1, 3, 1, -3, -3, -1, 1, -3, -3, 3, 3, 3, 1, 3, -3, 3, -3},
                                {-3, -3, -3, 1, -3, 3, 1, 1, 3, -3, -3, 1, 3, -1, 3, -3, -3, 3},
                                {-3, -3, 3, 3, 3, -1, -1, -3, -1, -1, -1, 3, 1, -3, -3, -1, 3, -1},
                                {-3, -1, -3, -3, 1, 1, -1, -3, -1, -3, -1, -1, 3, 3, -1, 3, 1, 3},
                                {1, 1, -3, -3, -3, -3, 1, 3, -3, 3, 3, 1, -3, -1, 3, -1, -3, 1},
                                {-3, 3, -1, -3, -1, -3, 1, 1, -3, -3, -1, -1, 3, -3, 1, 3, 1, 1},
                                {3, 1, -3, 1, -3, 3, 3, -1, -3, -3, -1, -3, -3, 3, -3, -1, 1, 3},
                                {-3, -1, -3, -1, -3, 1, 3, -3, -1, 3, 3, 3, 1, -1, -3, 3, -1, -3},
                                {-3, -1, 3, 3, -1, 3, -1, -3, -1, 1, -1, -3, -1, -1, -1, 3, 3, 1},
                                {-3, 1, -3, -1, -1, 3, 1, -3, -3, -3, -1, -3, -3, 1, 1, 1, -1, -1},
                                {3, 3, 3, -3, -1, -3, -1, 3, -1, 1, -1, -3, 1, -3, -3, -1, 3, 3},
                                {-3, 1, 1, -3, 1, 1, 3, -3, -1, -3, -1, 3, -3, 3, -1, -1, -1, -3},
                                {1, -3, -1, -3, 3, 3, -1, -3, 1, -3, -3, -1, -3, -1, 1, 3, 3, 3},
                                {-3, -3, 1, -1, -1, 1, 1, -3, -1, 3, 3, 3, 3, -1, 3, 1, 3, 1},
                                {3, -1, -3, 1, -3, -3, -3, 3, 3, -1, 1, -3, -1, 3, 1, 1, 3, 3},
                                {3, -1, -1, 1, -3, -1, -3, -1, -3, -3, -1, -3, 1, 1, 1, -3, -3, 3},
                                {-3, -3, 1, -3, 3, 3, 3, -1, 3, 1, 1, -3, -3, -3, 3, -3, -1, -1},
                                {-3, -1, -1, -3, 1, -3, 3, -1, -1, -3, 3, 3, -3, -1, 3, -1, -1, -1},
                                {-3, -3, 3, 3, -3, 1, 3, -1, -3, 1, -1, -3, 3, -3, -1, -1, -1, 3},
                                {-1, -3, 1, -3, -3, -3, 1, 1, 3, 3, -3, 3, 3, -3, -1, 3, -3, 1},
                                {-3, 3, 1, -1, -1, -1, -1, 1, -1, 3, 3, -3, -1, 1, 3, -1, 3, -1}};

static __device__ __constant__ int8_t d_phi_24[30][24] = 
                                {{-1, -3, 3, -1, 3, 1, 3, -1, 1, -3, -1, -3, -1, 1, 3, -3, -1, -3, 3, 3, 3, -3, -3, -3},
                                {-1, -3, 3, 1, 1, -3, 1, -3, -3, 1, -3, -1, -1, 3, -3, 3, 3, 3, -3, 1, 3, 3, -3, -3},
                                {-1, -3, -3, 1, -1, -1, -3, 1, 3, -1, -3, -1, -1, -3, 1, 1, 3, 1, -3, -1, -1, 3, -3, -3},
                                {1, -3, 3, -1, -3, -1, 3, 3, 1, -1, 1, 1, 3, -3, -1, -3, -3, -3, -1, 3, -3, -1, -3, -3},
                                {-1, 3, -3, -3, -1, 3, -1, -1, 1, 3, 1, 3, -1, -1, -3, 1, 3, 1, -1, -3, 1, -1, -3, -3},
                                {-3, -1, 1, -3, -3, 1, 1, -3, 3, -1, -1, -3, 1, 3, 1, -1, -3, -1, -3, 1, -3, -3, -3, -3},
                                {-3, 3, 1, 3, -1, 1, -3, 1, -3, 1, -1, -3, -1, -3, -3, -3, -3, -1, -1, -1, 1, 1, -3, -3},
                                {-3, 1, 3, -1, 1, -1, 3, -3, 3, -1, -3, -1, -3, 3, -1, -1, -1, -3, -1, -1, -3, 3, 3, -3},
                                {-3, 1, -3, 3, -1, -1, -1, -3, 3, 1, -1, -3, -1, 1, 3, -1, 1, -1, 1, -3, -3, -3, -3, -3},
                                {1, 1, -1, -3, -1, 1, 1, -3, 1, -1, 1, -3, 3, -3, -3, 3, -1, -3, 1, 3, -3, 1, -3, -3},
                                {-3, -3, -3, -1, 3, -3, 3, 1, 3, 1, -3, -1, -1, -3, 1, 1, 3, 1, -1, -3, 3, 1, 3, -3},
                                {-3, 3, -1, 3, 1, -1, -1, -1, 3, 3, 1, 1, 1, 3, 3, 1, -3, -3, -1, 1, -3, 1, 3, -3},
                                {3, -3, 3, -1, -3, 1, 3, 1, -1, -1, -3, -1, 3, -3, 3, -1, -1, 3, 3, -3, -3, 3, -3, -3},
                                {-3, 3, -1, 3, -1, 3, 3, 1, 1, -3, 1, 3, -3, 3, -3, -3, -1, 1, 3, -3, -1, -1, -3, -3},
                                {-3, 1, -3, -1, -1, 3, 1, 3, -3, 1, -1, 3, 3, -1, -3, 3, -3, -1, -1, -3, -3, -3, 3, -3},
                                {-3, -1, -1, -3, 1, -3, -3, -1, -1, 3, -1, 1, -1, 3, 1, -3, -1, 3, 1, 1, -1, -1, -3, -3},
                                {-3, -3, 1, -1, 3, 3, -3, -1, 1, -1, -1, 1, 1, -1, -1, 3, -3, 1, -3, 1, -1, -1, -1, -3},
                                {3, -1, 3, -1, 1, -3, 1, 1, -3, -3, 3, -3, -1, -1, -1, -1, -1, -3, -3, -1, 1, 1, -3, -3},
                                {-3, 1, -3, 1, -3, -3, 1, -3, 1, -3, -3, -3, -3, -3, 1, -3, -3, 1, 1, -3, 1, 1, -3, -3},
                                {-3, -3, 3, 3, 1, -1, -1, -1, 1, -3, -1, 1, -1, 3, -3, -1, -3, -1, -1, 1, -3, 3, -1, -3},
                                {-3, -3, -1, -1, -1, -3, 1, -1, -3, -1, 3, -3, 1, -3, 3, -3, 3, 3, 1, -1, -1, 1, -3, -3},
                                {3, -1, 1, -1, 3, -3, 1, 1, 3, -1, -3, 3, 1, -3, 3, -1, -1, -1, -1, 1, -3, -3, -3, -3},
                                {-3, 1, -3, 3, -3, 1, -3, 3, 1, -1, -3, -1, -3, -3, -3, -3, 1, 3, -1, 1, 3, 3, 3, -3},
                                {-3, -1, 1, -3, -1, -1, 1, 1, 1, 3, 3, -1, 1, -1, 1, -1, -1, -3, -3, -3, 3, 1, -1, -3},
                                {-3, 3, -1, -3, -1, -1, -1, 3, -1, -1, 3, -3, -1, 3, -3, 3, -3, -1, 3, 1, 1, -1, -3, -3},
                                {-3, 1, -1, -3, -3, -1, 1, -3, -1, -3, 1, 1, -1, 1, 1, 3, 3, 3, -1, 1, -1, 1, -1, -3},
                                {-1, 3, -1, -1, 3, 3, -1, -1, -1, 3, -1, -3, 1, 3, 1, 1, -3, -3, -3, -1, -3, -1, -3, -3},
                                {3, -3, -3, -1, 3, 3, -3, -1, 3, 1, 1, 1, 3, -1, 3, -3, -1, 3, -1, 3, 1, -1, -3, -3},
                                {-3, 1, -3, 1, -3, 1, 1, 3, 1, -3, -3, -1, 1, 3, -1, -3, 3, 1, -1, -3, -3, -3, -3, -3},
                                {3, -3, -1, 1, 3, -1, -1, -3, -1, 3, -1, -3, -1, -3, 3, -1, 3, 1, 1, -3, 3, -3, -3, -3}};

static __device__ __constant__ uint16_t d_primeNums[303] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987,1993,1997,1999};
    
static inline __device__ float2 gen_descrcode(uint16_t M_ZC, uint16_t rIdx, int u, int v)
{
    float2 descrCode;
    if(M_ZC < 36)
    {
        if(rIdx < M_ZC)
        {
            switch(M_ZC)
            {
            case 6: {
                descrCode.x =(float)cos(M_PI * (d_phi_6[u][rIdx]) / 4.0f);
                descrCode.y= (float)sin(M_PI * (d_phi_6[u][rIdx]) / 4.0f);
                break;
            }
            case 12: {
                descrCode.x =(float)cos(M_PI * (d_phi_12[u][rIdx]) / 4.0f);
                descrCode.y= (float)sin(M_PI * (d_phi_12[u][rIdx]) / 4.0f);
                break;
            }
            case 18: {
                descrCode.x =(float)cos(M_PI * (d_phi_18[u][rIdx]) / 4.0f);
                descrCode.y= (float)sin(M_PI * (d_phi_18[u][rIdx]) / 4.0f);
                break;
            }
            case 24: {
                descrCode.x =(float)cos(M_PI * (d_phi_24[u][rIdx]) / 4.0f);
                descrCode.y= (float)sin(M_PI * (d_phi_24[u][rIdx]) / 4.0f);
                break;
            }
            case 30: {
                descrCode.x =(float)cos(M_PI * (u + 1) * (rIdx + 1) * (rIdx + 2) / 31.0f);
                descrCode.y= (float)(-sin(M_PI * (u + 1) * (rIdx + 1) * (rIdx + 2) / 31.0f));
                break;
            }  
            } 
        }
    }
    else 
    {
        int idx = 0;
        while(M_ZC > d_primeNums[idx])
        {
            idx++;
        }
        idx--;
        uint16_t d_primeNum = d_primeNums[idx];
        float qbar = d_primeNum * (u + 1) / 31.0f;
        float q    = (int)(qbar + 0.5f) + (v * (((int)(2 * qbar) & 1) * -2 + 1));
        uint32_t m = rIdx % d_primeNum;
        descrCode.x =(float)cos(M_PI * q * m * (m + 1) / d_primeNum);
        descrCode.y= (float)(-sin(M_PI * q * m * (m + 1) / d_primeNum));
    }
    return descrCode;
}

__global__ void genSrsTx(SrsTxParams* srstx_params, __half2** tfSignalArray)
{
    
    uint8_t  nAntPorts               = srstx_params[blockIdx.x].nAntPorts;            //N_ap_SRS
    if(threadIdx.y >= nAntPorts)
    {
        return;
    }
    uint8_t  nSyms                   = srstx_params[blockIdx.x].nSyms;                //N_symb_SRS
    if(threadIdx.x >= nSyms)
    {
        return;
    }
    __half2* tfSignal                = tfSignalArray[blockIdx.x];
    uint8_t  nRepetitions            = srstx_params[blockIdx.x].nRepetitions;         //R
    uint8_t  combSize                = srstx_params[blockIdx.x].combSize;             //K_TC
    uint8_t  startSym                = srstx_params[blockIdx.x].startSym;             //l0
    uint16_t sequenceId              = srstx_params[blockIdx.x].sequenceId;           //n_ID_SRS
    uint8_t  configIdx               = srstx_params[blockIdx.x].configIdx;            //C_SRS
    uint8_t  bandwidthIdx            = srstx_params[blockIdx.x].bandwidthIdx;         //B_SRS
    uint8_t  combOffset              = srstx_params[blockIdx.x].combOffset;           //k_TC_bar
    uint8_t  cyclicShift             = srstx_params[blockIdx.x].cyclicShift;          //n_SRS_cs
    uint8_t  frequencyPosition       = srstx_params[blockIdx.x].frequencyPosition;    //n_RRC
    uint16_t frequencyShift          = srstx_params[blockIdx.x].frequencyShift;       //n_shift
    uint8_t  frequencyHopping        = srstx_params[blockIdx.x].frequencyHopping;     //b_hop
    uint8_t  resourceType            = srstx_params[blockIdx.x].resourceType;         //resourceType
    uint16_t Tsrs                    = srstx_params[blockIdx.x].Tsrs;                 //Tsrs
    uint16_t Toffset                 = srstx_params[blockIdx.x].Toffset;              //Toffset
    uint8_t  groupOrSequenceHopping  = srstx_params[blockIdx.x].groupOrSequenceHopping;//groupOrSequenceHopping
    uint16_t idxSlotInFrame          = srstx_params[blockIdx.x].idxSlotInFrame;
    uint16_t idxFrame                = srstx_params[blockIdx.x].idxFrame;
    uint16_t nSlotsPerFrame          = srstx_params[blockIdx.x].nSlotsPerFrame;
    uint16_t nSymbsPerSlot           = srstx_params[blockIdx.x].nSymbsPerSlot;
       
    uint8_t csMax = 8;
    if(combSize == 4)
    {
        csMax = 12;
    }
    uint8_t idxSym     = threadIdx.x;
    uint8_t idxAntPort = threadIdx.y;
    
    // compute phase shift alpha
    float alpha = 2 * M_PI * static_cast<float>((cyclicShift + (csMax * idxAntPort)/nAntPorts) % csMax) / static_cast<float>(csMax);

    // compute SRS sequence group u and sequence number v
    uint16_t M_sc_b_SRS = SRS_BW_TABLE[configIdx][2*bandwidthIdx]*CUPHY_N_TONES_PER_PRB/combSize;
    int u    = 0;
    int v    = 0;
    int f_gh = 0;
    if(groupOrSequenceHopping==1)
    {
        for(int m = 0; m < 8; m++)
        {
            uint32_t idxSeq = 8 * (idxSlotInFrame * nSymbsPerSlot + startSym + idxSym) + m;
            f_gh = f_gh + ((descrambling::gold32(sequenceId, idxSeq) >> (idxSeq % 32)) & 0x1) * (1 << m);
        }
        f_gh = f_gh % 30;

    }
    else if(groupOrSequenceHopping==2)
    {
        if(M_sc_b_SRS >= 6 * CUPHY_N_TONES_PER_PRB)
        {
            uint32_t idxSeq = idxSlotInFrame * nSymbsPerSlot + startSym + idxSym;
            v = (descrambling::gold32(sequenceId, idxSeq) >> (idxSeq % 32)) & 0x1;

        }
        else
        {
            v = 0;
        }
    }
    u = (f_gh + sequenceId)%30;
    
    // compute freq domain starting position k0
    uint16_t k_TC = 0;
    if ((cyclicShift >= csMax/2) && (nAntPorts == 4) && (idxAntPort == 1 || idxAntPort == 3))
    {
        k_TC = (combOffset + combSize/2) % combSize;
    }
    else
    {
        k_TC = combOffset;
    }
    uint16_t k0_bar = frequencyShift * CUPHY_N_TONES_PER_PRB + k_TC;
    uint16_t k0 = k0_bar;
    for(uint8_t b = 0; b <= bandwidthIdx; b++)
    {
        uint16_t m_SRS_b = 0;
        uint16_t N_b = 0;
        uint16_t nb = 0;
        if(frequencyHopping >= bandwidthIdx)
        {
            N_b = SRS_BW_TABLE[configIdx][2*b+1];
            m_SRS_b = SRS_BW_TABLE[configIdx][2*b];
            nb = static_cast<uint16_t>(floor(4*frequencyPosition/m_SRS_b)) % N_b;
        }
        else
        {
            N_b = SRS_BW_TABLE[configIdx][2*b+1];
            m_SRS_b = SRS_BW_TABLE[configIdx][2*b];
            if(b <= frequencyHopping)
            {
                nb = static_cast<uint16_t>(floor(4*frequencyPosition/m_SRS_b)) % N_b;
            }
            else
            {   
                uint16_t n_SRS = 0;
                if(resourceType == 0)
                {
                    n_SRS = floor(idxSym/nRepetitions);
                }
                else
                {
                    uint16_t slotIdx = nSlotsPerFrame * idxFrame + idxSlotInFrame - Toffset;
                    if((slotIdx % Tsrs) == 0)
                    {
                        n_SRS = (slotIdx/Tsrs) * (nSyms/nRepetitions) + static_cast<uint16_t>(floor(idxSym/nRepetitions));
                    }
                    else
                    {
                        //warning('Not a SRS slot ...\n');
                        n_SRS = 0;
                    }
                }
                uint16_t PI_bm1 = 1;
                for(uint8_t b_prime = frequencyHopping+1; b_prime <= b-1; b_prime++)
                {
                    PI_bm1 = PI_bm1*SRS_BW_TABLE[configIdx][2*b_prime+1];
                }
                uint16_t PI_b = PI_bm1 * N_b;
                uint16_t Fb = 0;
                if((N_b % 2) == 0)
                {
                    Fb = (N_b/2)*floor((n_SRS % PI_b)/PI_bm1) + floor((n_SRS % PI_b)/(2*PI_bm1));
                }
                else
                {
                    Fb = floor(N_b/2)*floor(n_SRS/PI_bm1);
                }
                nb = (Fb + static_cast<uint16_t>(floor(4*frequencyPosition/m_SRS_b))) % N_b;
            }
        }
        k0 +=  (combSize * (m_SRS_b*CUPHY_N_TONES_PER_PRB/combSize) * nb);
    }
    
    float2 rot = {0, 0};
    float2 descrCode = {0, 0};
    float2 val = {0, 0};
    // compute r_bar and map ZC sequence to REs
    for(uint16_t k = threadIdx.z; k < M_sc_b_SRS; k+=blockDim.z)
    {
        descrCode = gen_descrcode(M_sc_b_SRS, k, u, v);
        __sincosf(k*alpha, &rot.y, &rot.x);
        val.x = descrCode.x*rot.x - descrCode.y*rot.y;
        val.y = descrCode.x*rot.y + descrCode.y*rot.x;
        tfSignal[k0+k*combSize + (idxSym+startSym)*3276 + idxAntPort*OFDM_SYMBOLS_PER_SLOT*3276] = __float22half2_rn(val);
    }
}


cuphyStatus_t cuphySrsTxKernelSelect(cuphyGenSrsTxLaunchCfg_t* pGenSrsTxLaunchCfg, uint32_t numParams)
{
    if(!pGenSrsTxLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    kernelSelectGenSrsTx(pGenSrsTxLaunchCfg, numParams);
    return CUPHY_STATUS_SUCCESS;
}

void kernelSelectGenSrsTx(cuphyGenSrsTxLaunchCfg_t* pLaunchCfg, uint32_t numParams)
{
    if (pLaunchCfg->kernelNodeParamsDriver.func == nullptr)
    {
        void* kernelFunc = reinterpret_cast<void*>(genSrsTx);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));}
    }

    // launch geometry (can change!)
    dim3 gridDim(numParams);
    dim3 blockDim(OFDM_SYMBOLS_PER_SLOT, 4, 16);

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra    = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}


