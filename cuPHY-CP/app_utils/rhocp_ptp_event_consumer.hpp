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

#ifndef rhocp_ptp_event_consumer_H
#define rhocp_ptp_event_consumer_H

#include <map>
#include <string>
#include <httplib.h>

namespace AppUtils
{
    const std::string rhocpApiPath = "/api/ocloudNotifications/v2/";
    
    bool publisherHealthCheck(const std::string& PTP_PUBLISHER, int retryDelaySeconds = 4);
    int pullEvents(const std::string& PTP_PUBLISHER, const std::string& PTP_NODE_NAME);

    const std::map<std::string, std::string> initPTPSubscriptions();
    void deleteAllSubscriptions(const std::string& PTP_PUBLISHER);
    void subscribeToEvents(const std::string& PTP_PUBLISHER, const std::string& PTP_NODE_NAME, const std::string& PTP_CONSUMER, int retryCount = 0);
    std::shared_ptr<httplib::Server> startEventServer(const std::string& PTP_CONSUMER);
}

#endif 
