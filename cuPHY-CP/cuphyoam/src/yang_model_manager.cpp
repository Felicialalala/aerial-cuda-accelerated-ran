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

#include "yang_model_manager.hpp"
#include "cuphyoam.hpp"
#include "nvlog.hpp"
#include "utils.hpp"
#include <charconv>
#include <filesystem>

#define TAG (NVLOG_TAG_BASE_CUPHY_OAM + 1) // "OAM.YMMGR"

YangModelManager &YangModelManager::getInstance()
{
    static YangModelManager instance;
    return instance;
}

#ifdef ENABLE_YANG_PARSER

std::pair<int, std::string> YangModelManager::xGet(uint16_t phy_id, std::string &xpath)
{
    std::pair<int, std::string> ret{-1, ""};
    std::optional<std::string> parsed_res = std::nullopt;
    NVLOGC_FMT(TAG, "xpath: {} ", xpath);
    if (data_mp.find(phy_id) != data_mp.end())
    {
        try
        {
            auto node = data_mp[phy_id]->findPath(xpath);
            if (node.has_value())
            {
                parsed_res = *node->printStr(libyang::DataFormat::XML, libyang::PrintFlags::KeepEmptyCont);
                // res = *node->printStr(libyang::DataFormat::XML, libyang::PrintFlags::WithSiblings | libyang::PrintFlags::KeepEmptyCont);
            }
            if (parsed_res == std::nullopt)
            {
                ret.second = "Empty node";
                NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Empty node");
            }
            else
            {
                ret.first = 0;
                ret.second = parsed_res.value();
                NVLOGD_FMT(TAG, "Node tree: {} ", parsed_res.value());
            }
        }
        catch (libyang::Error err)
        {
            ret.second = "Runtime_error exception: " + std::string(err.what());
            NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Runtime_error exception: {} ", err.what());
        }
    }
    return ret;
}

int YangModelManager::parseEditConfig(uint16_t phy_id, std::string &cfg_str)
{
    int ret = 0;
    auto new_cell_config = new CuphyOAMCellConfig();
    try
    {
        if (ctx_mp.find(phy_id) == ctx_mp.end())
        {
            std::string yang_model_path = YANG_MODEL_DIR;

            std::optional<libyang::Context> ctx{std::in_place, std::nullopt, libyang::ContextOptions::NoYangLibrary | libyang::ContextOptions::DisableSearchCwd};
            ctx->setSearchDir(yang_model_path);
            ctx->loadModule(YANG_MODEL_O_RAN_INTERFACES, std::nullopt);
            ctx->loadModule(YANG_MODEL_O_RAN_DELAY_MANAGEMENT, std::nullopt);
            ctx->loadModule(YANG_MODEL_O_RAN_UPLANE_CONF, std::nullopt);
            ctx->loadModule(YANG_MODEL_O_RAN_MODULE_CAP, std::nullopt);
            ctx->loadModule(YANG_MODEL_IANA_IF_TYPE, std::nullopt);

            ctx_mp[phy_id] = ctx;

#if 0
            for (auto &m : ctx->modules())
            {
                NVLOGD_FMT(TAG, "Model name: {} ", m.name());
                for (auto &i : m.identities())
                {
                    NVLOGD_FMT(TAG, "Identity name: {} ", i.name());
                    for (auto &di : i.derived())
                    {
                        NVLOGD_FMT(TAG, "derived Identity name: {} ", di.name());
                    }
                    for (auto &dri : i.derivedRecursive())
                    {
                        NVLOGD_FMT(TAG, "derivedRecursive Identity name: {} ", dri.name());
                    }
                }
            }
#endif
        }

        auto root = ctx_mp[phy_id]->parseData(cfg_str, libyang::DataFormat::XML, libyang::ParseOptions::ParseOnly);
        if (root == std::nullopt)
        {
            ret = -1;
            goto end;
        }
        auto parsed_tree = *root->printStr(libyang::DataFormat::XML, libyang::PrintFlags::WithSiblings | libyang::PrintFlags::KeepEmptyCont);
        NVLOGC_FMT(TAG, "===============================Received config=============================");
        NVLOGC_FMT(TAG, " {} ", parsed_tree);

        std::unordered_map<std::string, std::optional<libyang::DataNode>> refs;
        std::unordered_set<std::string> node_to_del;
        auto siblings = root->firstSibling().siblings();
        for (const auto &sibling : siblings)
        {
            for (const auto &it : sibling.childrenDfs())
            {
                if (it.isTerm())
                {
                    std::string path_str = it.path();
                    NVLOGI_FMT(TAG, "xpath: {} ,  term value: {} ", it.path(), it.asTerm().valueStr());
                    std::filesystem::path p(path_str);
                    auto var_name = p.filename();
                    std::string str_value{it.asTerm().valueStr()};
                    std::string key;
                    double d_value;
                    auto result = std::from_chars(str_value.data(), str_value.data() + str_value.size(), d_value);
                    if (path_str.find("ru-elements") != std::string::npos)
                    {
                        if (var_name == "o-du-mac-address")
                        {
                            key = CELL_PARAM_NIC;
                            auto nic_pcie_addr = convertMacAddrToPCIeAddr(str_value);
                            if (nic_pcie_addr != std::nullopt)
                            {
                                uint64_t pcie_addr = encodePCIeAddr(nic_pcie_addr.value());
                                d_value = pcie_addr;
                            }
                            else
                            {
                                node_to_del.insert(key);
                                NVLOGC_FMT(TAG, "Nic PCIe addre not found for mac: {} ... ", str_value);
                            }
                        }
                        else if (var_name == "ru-mac-address")
                        {
                            uint64_t mac = encodeMACAddr(str_value);
                            key = CELL_PARAM_DST_MAC_ADDR;
                            d_value = mac;
                        }
                        else if (var_name == "vlan-id")
                        {
                            key = CELL_PARAM_VLAN_ID;
                        }
                    }
                    else if (path_str.find("interface") != std::string::npos)
                    {
                        if (var_name == "u-plane-marking")
                        {
                            key = CELL_PARAM_PCP;
                        }
                        else if (path_str.find("l2-mtu") != std::string::npos)
                        {
                            key = "mtu";
                        }
                    }
                    else if (path_str.find("low-level-rx-endpoints") != std::string::npos)
                    {
                        if (var_name == "iq-bitwidth")
                        {
                            key = CELL_PARAM_DL_BIT_WIDTH;
                        }
                        else if (var_name == "compression-method")
                        {
                            key = CELL_PARAM_DL_COMP_METH;
                            if(str_value == "BLOCK_FLOATING_POINT")
                            {
                                d_value = 1;
                            }
                            else if(str_value == "NO_COMPRESSION")
                            {
                                d_value = 0;
                            }
                        }
                        else if (var_name == "exponent")
                        {
                            key = CELL_PARAM_EXPONENT_DL;
                        }
                    }
                    else if (path_str.find("low-level-tx-endpoints") != std::string::npos)
                    {
                        if (var_name == "iq-bitwidth")
                        {
                            key = CELL_PARAM_UL_BIT_WIDTH;
                        }
                        else if (var_name == "compression-method")
                        {
                            key = CELL_PARAM_UL_COMP_METH;
                            if(str_value == "BLOCK_FLOATING_POINT")
                            {
                                d_value = 1;
                            }
                            else if(str_value == "NO_COMPRESSION")
                            {
                                d_value = 0;
                            }
                        }
                        else if (var_name == "exponent")
                        {
                            key = CELL_PARAM_EXPONENT_UL;
                        }
                    }
                    else
                    {
                        // double d_value;
                        // auto result = std::from_chars(str_value.data(), str_value.data() + str_value.size(), d_value);
                        // new_cell_config->attrs[var_name] = d_value;
                        // NVLOGI_FMT(TAG, "Node: {} set to : {} ", var_name, d_value);
                    }
                    if (!key.empty())
                    {
                        if(!node_to_del.count(key))
                        {
                            new_cell_config->attrs[key] = d_value;
                        }
                        refs[key] = it;
                    }
                }
            }
        }


        if (new_cell_config->attrs.size() > 0)
        {
            CuphyOAM *oam = CuphyOAM::getInstance();
            new_cell_config->multi_attrs_cfg = true;
            new_cell_config->cell_id = phy_id;
            oam->cell_multi_attri_update_callback(new_cell_config->cell_id, new_cell_config->attrs, new_cell_config->res);
        }

        for(auto &[key, op_res] : new_cell_config->res)
        {
            if(op_res != 0)
            {
                node_to_del.insert(key);
            }
        }

        for(auto & d : node_to_del)
        {
            if(refs.find(d) != refs.end())
            {
                refs[d]->unlink();
            }
        }

        //Merge into existing data tree
        if (data_mp.find(phy_id) == data_mp.end())
        {
            data_mp[phy_id] = root;
        }
        else
        {
            data_mp[phy_id]->merge(*root);
        }

#if 1
        auto tree = *data_mp[phy_id]->printStr(libyang::DataFormat::XML, libyang::PrintFlags::WithSiblings | libyang::PrintFlags::KeepEmptyCont);
        NVLOGD_FMT(TAG, "==================Cell YANG Model data tree==================== ");
        NVLOGD_FMT(TAG, "Node tree: {} ", tree);

        for (const auto &sibling : data_mp[phy_id]->siblings())
        {
            NVLOGD_FMT(TAG, "==================== ");
            for (const auto &it : sibling.childrenDfs())
            {
                if (it.isTerm())
                {
                    NVLOGD_FMT(TAG, "xpath: {} ,  term value: {} ", it.path(), it.asTerm().valueStr());
                }
            }
        }
#endif
    }
    catch (libyang::Error err)
    {
        ret = -1;
        NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Runtime_error exception: {} ", err.what());
    }

end:
    delete new_cell_config;
    return ret;
}

#else // !ENABLE_YANG_PARSER

std::pair<int, std::string> YangModelManager::xGet(uint16_t phy_id, std::string &xpath)
{
    NVLOGW_FMT(TAG, "YANG Model Manager is disabled. xGet operation not supported for phy_id: {}, xpath: {}", phy_id, xpath);
    return std::make_pair(-1, "YANG Model Manager is disabled");
}

int YangModelManager::parseEditConfig(uint16_t phy_id, std::string &cfg_str)
{
    NVLOGW_FMT(TAG, "YANG Model Manager is disabled. parseEditConfig operation not supported for phy_id: {}", phy_id);
    return -1;
}

#endif // ENABLE_YANG_PARSER
