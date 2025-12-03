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

struct crcEncodeDescr
{
    uint32_t*     d_cbCRCs;
    uint32_t*     d_tbCRCs;
    const uint32_t* d_inputTransportBlocks;
    uint8_t*           d_codeBlocks;
    const PdschPerTbParams* d_tbPrmsArray;
    bool            reverseBytes;
    bool            codeBlocksOnly;
};
typedef struct crcEncodeDescr crcEncodeDescr_t;

struct prepareCrcEncodeDescr
{
    uint32_t                offset[PDSCH_MAX_UES_PER_CELL_GROUP]; //FIXME keep first field; used only for TBs in testing mode (TM)
    const uint32_t*         d_inputOrigTBs;
    uint32_t*               d_inputTBs;
    uint32_t*               d_inputTBsTM;
    const PdschPerTbParams* d_tbPrmsArray;
};
typedef struct prepareCrcEncodeDescr prepareCrcEncodeDescr_t;
