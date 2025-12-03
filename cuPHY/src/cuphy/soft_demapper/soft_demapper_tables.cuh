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

#if !defined(SOFT_DEMAPPER_TABLES_CUH_INCLUDED_)

#define SOFT_DEMAPPER_TABLES_CUH_INCLUDED_

#include <cuda_fp16.h>

namespace soft_demapper
{


struct mod_symbol_half_to_tex_coord_t
{
    __half2_raw m;
    __half2_raw b;
};

__constant__ mod_symbol_half_to_tex_coord_t sym_transform_h[9] =
{
    {  {0x0000 /* = 0.0000 */, 0x0000 /* = 0.0000 */} , {0x0000 /* = 0.0000 */, 0x0000 /* = 0.0000 */} }, // 0
    {  {0x31A8 /* = 0.1768 */, 0x31A8 /* = 0.1768 */} , {0x3900 /* = 0.6250 */, 0x3900 /* = 0.6250 */} }, // 1
    {  {0x31A8 /* = 0.1768 */, 0x31A8 /* = 0.1768 */} , {0x3900 /* = 0.6250 */, 0x3900 /* = 0.6250 */} }, // 2
    {  {0x0000 /* = 0.0000 */, 0x0000 /* = 0.0000 */} , {0x0000 /* = 0.0000 */, 0x0000 /* = 0.0000 */} }, // 3
    {  {0x3253 /* = 0.1976 */, 0x3253 /* = 0.1976 */} , {0x3880 /* = 0.5625 */, 0x3880 /* = 0.5625 */} }, // 4
    {  {0x0000 /* = 0.0000 */, 0x0000 /* = 0.0000 */} , {0x0000 /* = 0.0000 */, 0x0000 /* = 0.0000 */} }, // 5
    {  {0x327B /* = 0.2025 */, 0x327B /* = 0.2025 */} , {0x3840 /* = 0.5312 */, 0x3840 /* = 0.5312 */} }, // 6
    {  {0x0000 /* = 0.0000 */, 0x0000 /* = 0.0000 */} , {0x0000 /* = 0.0000 */, 0x0000 /* = 0.0000 */} }, // 7
    {  {0x3285 /* = 0.2037 */, 0x3285 /* = 0.2037 */} , {0x3820 /* = 0.5156 */, 0x3820 /* = 0.5156 */} }, // 8
};

} // namespace soft_demapper


#endif // !defined(SOFT_DEMAPPER_TABLES_CUH_INCLUDED_)

