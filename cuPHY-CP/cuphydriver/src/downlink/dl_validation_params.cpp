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

#include "dl_validation_params.hpp"

DLValidationParams::DLValidationParams(phydriver_handle _pdh, int _start_cell, int _num_cells) :
    pdh(_pdh),
    start_cell(_start_cell),
    num_cells(_num_cells)
{

}
DLValidationParams::~DLValidationParams()
{
}

void DLValidationParams::setPhyDriverHandler(phydriver_handle pdh_)
{
    pdh = pdh_;
}

phydriver_handle DLValidationParams::getPhyDriverHandler(void) const
{
    return pdh;
}