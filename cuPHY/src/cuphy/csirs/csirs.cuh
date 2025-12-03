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

#if !defined(CSIRS_CUH_INCLUDED_)
#define CSIRS_CUH_INCLUDED_

__constant__  CsirsSymbLocRow constRowDataCsirs[CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH] =
                                 {{1, 3, 1, 1, /* rowData[0] */
                                  {0, 0, 0},
                                  {0, 4, 8},
                                  {0, 0, 0},
                                  {0, 0, 0},
                                  {0, 0, 0}},
                                 /* rowData[1]*/
                                 {1, 1, 1, 1,
                                  {0},
                                  {0},
                                  {0},
                                  {0},
                                  {0}},
                                 /* rowData[2]*/
                                 {2, 1, 2, 1,
                                  {0},
                                  {0},
                                  {0},
                                  {0},
                                  {0}},
                                 /* rowData[3]*/
                                 {4, 2, 2, 1,
                                  {0, 0},
                                  {0, 2},
                                  {0, 0},
                                  {0, 0},
                                  {0, 1}},
                                 /* rowData[4]*/
                                 {4, 2, 2, 1,
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 1},
                                  {0, 1}},
                                 /* rowData[5]*/
                                 {8, 4, 2, 1,
                                  {0, 1, 2, 3},
                                  {0, 0, 0, 0},
                                  {0, 0, 0, 0},
                                  {0, 0, 0, 0},
                                  {0, 1, 2, 3}},
                                 /* rowData[6]*/
                                 {8, 4, 2, 1,
                                  {0, 1, 0, 1},
                                  {0, 0, 0, 0},
                                  {0, 0, 0, 0},
                                  {0, 0, 1, 1},
                                  {0, 1, 2, 3}},
                                 /* rowData[7]*/
                                 {8, 2, 2, 2,
                                  {0, 1},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 1}},
                                 /* rowData[8]*/
                                 {12, 6, 2, 1,
                                  {0, 1, 2, 3, 4, 5},
                                  {0, 0, 0, 0, 0},
                                  {0, 0, 0, 0, 0},
                                  {0, 0, 0, 0, 0},
                                  {0, 1, 2, 3, 4, 5}},
                                 /* rowData[9] */
                                 {12, 3, 2, 2,
                                  {0, 1, 2},
                                  {0, 0, 0},
                                  {0, 0, 0},
                                  {0, 0, 0},
                                  {0, 1, 2}},
                                 /* rowData[10] */
                                 {16, 8, 2, 1,
                                   {0, 1, 2, 3, 0, 1, 2, 3},
                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                   {0, 0, 0, 0, 1, 1, 1, 1},
                                   {0, 1, 2, 3, 4, 5, 6, 7}},
                                 /* rowData[11] */
                                 {16, 4, 2, 2,
                                   {0, 1, 2, 3},
                                   {0, 0, 0, 0},
                                   {0, 0, 0, 0},
                                   {0, 0, 0, 0},
                                   {0, 1, 2, 3}},
                                 /* rowData[12] */
                                 {24, 12, 2, 1,
                                   {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2},
                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                   {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1},
                                   {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1},
                                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
                                 /* rowData[13] */
                                 {24, 6, 2, 2,
                                   {0, 1, 2, 0, 1, 2},
                                   {0, 0, 0, 0, 0, 0},
                                   {0, 0, 0, 1, 1, 1},
                                   {0, 0, 0, 0, 0, 0},
                                   {0, 1, 2, 3, 4, 5}},
                                 /* rowData[14] */
                                 {24, 3, 2, 4,
                                   {0, 1, 2},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 1, 2}},
                                 /* rowData[15] */
                                 {32, 16, 2, 1,
                                   {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                   {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                                   {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
                                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
                                 /* rowData[16] */
                                 {32, 8, 2, 2,
                                   {0, 1, 2, 3, 0, 1, 2, 3},
                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                   {0, 0, 0, 0, 1, 1, 1, 1},
                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                   {0, 1, 2, 3, 4, 5, 6, 7}},
                                 /* rowData[17] */
                                 {32, 4, 2, 4,
                                   {0, 1, 2, 3},
                                   {0, 0, 0, 0},
                                   {0, 0, 0, 0},
                                   {0, 0, 0, 0},
                                   {0, 1, 2, 3}}};


__constant__ int8_t constSeqTableCsirs[MAX_CDM_TYPE][CUPHY_CSIRS_MAX_SEQ_INDEX_COUNT][2][4] = /*!< wf/wt seq table layout: 2- Wf,Wt; 4 max(maxkprimelen, maxlprimelen) */
                                                                           {{{{1}, {1}}}, // NO_CDM
                                                                            {{{1, 1}, {1}}, // CDM2_FD
                                                                            {{1, -1}, {1}}},
                                                                            {{{1, 1}, {1, 1}}, // CDM4_FD2_TD2
                                                                            {{1, -1}, {1, 1}},
                                                                            {{1, 1}, {1, -1}},
                                                                            {{1, -1}, {1, -1}}},
                                                                            {{{1, 1}, {1, 1, 1, 1}}, // CDM8_FD2_TD4
                                                                            {{1, -1}, {1, 1, 1, 1}},
                                                                            {{1, 1}, {1, -1, 1, -1}},
                                                                            {{1, -1}, {1, -1, 1, -1}},
                                                                            {{1, 1}, {1, 1, -1, -1}},
                                                                            {{1, -1}, {1, 1, -1, -1}},
                                                                            {{1, 1}, {1, -1, -1, 1}},
                                                                            {{1, -1}, {1, -1, -1, 1}}}};

#endif
