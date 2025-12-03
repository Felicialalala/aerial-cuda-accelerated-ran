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

#ifndef YANG_MODEL_MANAGER_H
#define YANG_MODEL_MANAGER_H

#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#ifdef ENABLE_YANG_PARSER
#include <libyang-cpp/Context.hpp>
#include <libyang-cpp/DataNode.hpp>
#include <libyang-cpp/Utils.hpp>
#include <libyang-cpp/Context.hpp>
#include <libyang-cpp/Type.hpp>
#include <libyang-cpp/Utils.hpp>
#include <libyang-cpp/Set.hpp>
#endif // ENABLE_YANG_PARSER

using namespace std::literals;

#define CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM 4 // Set CUBB_HOME to N level parent directory of this process. Example: 4 means "../../../../"

static const std::string  YANG_MODEL_DIR = "/opt/modeling/data-model/yang/published/o-ran/ru-fh";

//Yang Models
static const std::string YANG_MODEL_O_RAN_INTERFACES = "o-ran-interfaces";
static const std::string YANG_MODEL_O_RAN_DELAY_MANAGEMENT = "o-ran-delay-management";
static const std::string YANG_MODEL_O_RAN_UPLANE_CONF = "o-ran-uplane-conf";
static const std::string YANG_MODEL_O_RAN_MODULE_CAP = "o-ran-module-cap";
static const std::string YANG_MODEL_IANA_IF_TYPE = "iana-if-type";


//Node path
static const std::string PATH_DELAY_MANAGEMENT = "/o-ran-delay-management:delay-management";


static constexpr char CELL_PARAM_RU_TYPE[] = "ru_type";
static constexpr char CELL_PARAM_DST_MAC_ADDR[] = "dst_mac_addr";
static constexpr char CELL_PARAM_VLAN_ID[] = "vlan_id";
static constexpr char CELL_PARAM_PCP[] = "pcp";
static constexpr char CELL_PARAM_COMPRESSION_BITS[] = "compression_bits";
static constexpr char CELL_PARAM_DECOMPRESSION_BITS[] = "decompression_bits";
static constexpr char CELL_PARAM_EXPONENT_DL[] = "exponent_dl";
static constexpr char CELL_PARAM_EXPONENT_UL[] = "exponent_ul";
static constexpr char CELL_PARAM_DL_COMP_METH[] = "dl_comp_meth";
static constexpr char CELL_PARAM_UL_COMP_METH[] = "ul_comp_meth";
static constexpr char CELL_PARAM_DL_BIT_WIDTH[] = "dl_bit_width";
static constexpr char CELL_PARAM_UL_BIT_WIDTH[] = "ul_bit_width";
static constexpr char CELL_PARAM_MAX_AMP_UL[] = "max_amp_ul";
static constexpr char CELL_PARAM_PUSCH_PRB_STRIDE[] = "pusch_prb_stride";
static constexpr char CELL_PARAM_PRACH_PRB_STRIDE[] = "prach_prb_stride";
static constexpr char CELL_PARAM_SECTION_3_TIME_OFFSET[] = "section_3_time_offset";
static constexpr char CELL_PARAM_FH_DISTANCE_RANGE[] = "fh_distance_range";
static constexpr char CELL_PARAM_UL_GAIN_CALIBRATION[] = "ul_gain_calibration";
static constexpr char CELL_PARAM_LOWER_GUARD_BW[] = "lower_guard_bw";
static constexpr char CELL_PARAM_REF_DL[] = "ref_dl";
static constexpr char CELL_PARAM_NIC[] = "nic";

class YangModelManager
{
public:
    static YangModelManager &getInstance();
    int parseEditConfig(uint16_t phy_id, std::string& cfg_str);
    std::pair<int, std::string> xGet(uint16_t phy_id, std::string& xpath);

private:
    YangModelManager() {}
    YangModelManager(const YangModelManager &other) = delete;
    YangModelManager &operator=(const YangModelManager &other) = delete;

#ifdef ENABLE_YANG_PARSER
    std::unordered_map<int, std::optional<libyang::Context>> ctx_mp;
    std::unordered_map<int, std::optional<libyang::DataNode>> data_mp;
#endif // ENABLE_YANG_PARSER
};

#endif // YANG_MODEL_MANAGER_H
