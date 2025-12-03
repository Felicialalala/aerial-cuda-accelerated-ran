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

#if !defined(GEN_PUCCH_PERF_CURVE_INCLUDED_)
#define GEN_PUCCH_PERF_CURVE_INCLUDED_

void gen_pucch_perf_curve(std::string pucchInputFilename, std::string pucchDbgFilename, std::string perfOutFilename, uint8_t  formatType, std::vector<float> snrVec, uint32_t nItrPerSnr, uint8_t mode, uint8_t seed);

#endif // GEN_PUCCH_PERF_CURVE_INCLUDED_