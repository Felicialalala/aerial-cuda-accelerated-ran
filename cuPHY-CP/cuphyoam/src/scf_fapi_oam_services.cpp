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

#include <grpcpp/grpcpp.h>
#include "scf_fapi_oam_services.hpp"
#include "yang_model_manager.hpp"
#include "cuphyoam.hpp"
#include "app_config.hpp"
#include "nvlog.hpp"

#define TAG (NVLOG_TAG_BASE_CUPHY_OAM + 2) // "OAM.YMSVC"

grpc::Status P9MessagesService::HandleMsg(ServerContext *context, const Msg *request, Msg *response)
{
    // Handle the request message
    NVLOGD_FMT(TAG, "Received request message with ID: {} ", request->header().msg_id());
    auto phy_id = request->header().phy_id();

    auto& appConfig = AppConfig::getInstance();
    auto cell_group_num = appConfig.getCellGroupNum();

    bool valid_phy_id = true;
    if(phy_id < 1 || phy_id > cell_group_num)
    {
        valid_phy_id = false;
    }

    // Handle the request type
    if (request->body().has_request())
    {
        response->mutable_header()->set_msg_id(request->header().msg_id());
        if (request->body().request().has_get())
        {
            // Handle the Get request
            NVLOGC_FMT(TAG, "Received Get request for phy id: {} ", phy_id);
            //auto ret = YangModelManager::getInstance().parseEditConfig(1, test_data);

            auto status = p9_messages::v1::Status::OK;
            std::string res = "";
            if (valid_phy_id)
            {
                auto &get = request->body().request().get();
                // auto sz = get.filter_size();
                auto xpath = get.filter(0);
                auto [status, res_str] = YangModelManager::getInstance().xGet(phy_id, xpath);
                if (!status)
                {
                    status = p9_messages::v1::Status::ERROR_GENERAL;
                }
                res = res_str;
            }
            else
            {
                res = "Invalid M Plane ID";
                status = p9_messages::v1::Status::ERROR_GENERAL;
                NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Invalid M Plane ID: {} , Valid range: [ 1 ~ {}]", phy_id, cell_group_num);
            }
            // Create a Get response message
            GetResp *get_resp = new GetResp();
            get_resp->mutable_status_resp()->set_status_code(status);
            get_resp->set_data(res);

            // Set the response message
            response->mutable_body()->mutable_response()->set_allocated_get_resp(get_resp);
        }
        else if (request->body().request().has_edit_config())
        {
            // Handle the EditConfig request
            NVLOGC_FMT(TAG, "Received EditConfig request for phy id: {} ", phy_id);

            auto status = p9_messages::v1::Status::OK;
            //if (valid_phy_id && !appConfig.isCellActive(phy_id))
            if (valid_phy_id)
            {
                auto cfg_str = request->body().request().edit_config().delta_config();
                // std::cout << "delta_config: " << cfg_str << std::endl;
                auto res = YangModelManager::getInstance().parseEditConfig(phy_id, cfg_str);
                if(res)
                {
                    status = p9_messages::v1::Status::ERROR_GENERAL;
                }
            }
            else
            {
                status = p9_messages::v1::Status::ERROR_GENERAL;
                if (appConfig.isCellActive(phy_id))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Cell with M Plane ID: {} is active, cannot perform the operation... ]", phy_id);
                }
                else
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYOAM_EVENT, "Invalid M Plane ID: {} , Valid range: [ 1 ~ {}]", phy_id, cell_group_num);
                }
            }

            // Create an EditConfig response message
            EditConfigResp *edit_config_resp = new EditConfigResp();
            edit_config_resp->mutable_status_resp()->set_status_code(status);

            // Set the response message
            response->mutable_body()->mutable_response()->set_allocated_edit_config_resp(edit_config_resp);
        }
    }

    return grpc::Status::OK;
}
