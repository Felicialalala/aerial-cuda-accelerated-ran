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

#ifndef UTILS_H__
#define UTILS_H__

#include "nvlog.hpp"

#define TAG_ORDER_TB_BASE 1050
#define TAG_ORDER_TB_INIT 1050 + 1
#define TAG_ORDER_TB_RUN 1050 + 2
#define DOCA_GPUNETIO_BUF_ARR_MAX_DEV 16
#define DOCA_GPUNETIO_CQE_CI_MASK 0xFFFF

constexpr int PRACH_MAX_NUM_SEC=4;
constexpr float tolf_iq_comp = 0.1;

#endif

