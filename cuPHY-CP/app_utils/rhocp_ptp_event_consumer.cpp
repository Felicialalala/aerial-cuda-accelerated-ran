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

#include <string>
#include <stdexcept>
#include <cstdlib>
#include <map>
#include <curl/curl.h>
#include <chrono>
#include <thread>

#include "nvlog.hpp"
#include "rhocp_ptp_event_consumer.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>

using namespace AppUtils;
using json = nlohmann::json;
#define TAG (NVLOG_TAG_BASE_APP_CFG_UTILS + 2) // "APP.UTILS"

/**
 * pullPtpEvents() requires the subscription(s) to be already created.
 * Can use subscribeToEvents() to create the subscription(s).
 * pullEvents() returns 0 only when both system clock and NIC clock are locked. 
 * 
 */
int AppUtils::pullEvents(const std::string& PTP_PUBLISHER, const std::string& PTP_NODE_NAME) {
    try {
        const auto& subscriptions = initPTPSubscriptions();
        if (subscriptions.empty()) {
            NVLOGW_FMT(TAG, "Nothing to pull, initPTPSubscriptions() returned empty map.");
            return 0;
        }
        for (const auto& [eventType, eventResource] : subscriptions) {
            std::string resourcePath = "cluster/node/" + PTP_NODE_NAME + eventResource;
            std::string fullURL = "http://" + PTP_PUBLISHER + rhocpApiPath + resourcePath + "/CurrentState";

            NVLOGI_FMT(TAG, "Pulling event status from: {}", fullURL);

            auto curl = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(
                curl_easy_init(), &curl_easy_cleanup);
            if (!curl) {
                NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "pullEvents() Error: CURL initialization failed");
                throw std::runtime_error("pullEvents() Error: CURL initialization failed");
            }

            curl_easy_setopt(curl.get(), CURLOPT_URL, fullURL.c_str());
            curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 5L);
            curl_easy_setopt(curl.get(), CURLOPT_FAILONERROR, 1L); 
            
            std::string response_string;
            curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION,
                +[](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
                    auto* data = static_cast<std::string*>(userdata);
                    data->append(ptr, size * nmemb);
                    return size * nmemb;
            });
            curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &response_string);

            CURLcode res = curl_easy_perform(curl.get());
            long http_code = 0;
            curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);

            if (res == CURLE_OK && http_code == 200) {
                NVLOGI_FMT(TAG, "pullEvents() Received current state for: {}", resourcePath);
                try {
                    json currentState = json::parse(response_string);
                    // NVLOGI_FMT(TAG, "Current state for {}: {}", resourcePath, currentState.dump(4));
                    bool allLocked = true;
                    if (!currentState.contains("data") || !currentState["data"].contains("values")) {
                        NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "pullEvents() Error: JSON return missing or invalid 'data.values' in JSON {}: {}", resourcePath, response_string);
                        return -1;
                    }
                    for (const auto& val : currentState["data"]["values"]) {
                        //TODO: only verified for clock class, os and nic clock state output, when enable more events, need to verify the JSON structure
                        if(!val.contains("ResourceAddress") || !val.contains("data_type") || !val.contains("value_type") || !val.contains("value")) {
                            NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "pullEvents() Error: JSON value missing required fields in {}: {}", resourcePath, val.dump());
                            return -1;
                        }
                        if (val["data_type"] == "notification" && val["value_type"] == "enumeration" && val["value"] != "LOCKED") {
                            allLocked = false;
                            const std::string dumpStr = val.dump();
                            NVLOGW_FMT(TAG, "{} is not LOCKED, pullEvents returned {} ", eventResource, dumpStr);
                        }
                    }
                    if (!allLocked) return -1;
                } catch (const json::parse_error& e) {
                    NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "pullEvents() Error: Failed to parse JSON response for {}: {}", resourcePath, e.what());
                    return -1;
                }
            } else {
                NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "pullEvents() Error -- Failed to get current state for {}: {} (HTTP {})", resourcePath, curl_easy_strerror(res), http_code);
                if (http_code == 404) {
                    NVLOGW_FMT(TAG, "Did you subscribe first?");
                    return -1; 
                } 
            }
        }
        return 0; 

    } catch (const std::exception& e) {
        NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "pullEvents() Error: {}", e.what());
        return -1;
    }
}


/*
 * PTP_CONSUMER should be in the format of host:port such as aerial-r750-03.nvidia.com:9050
 * In aerial rhocp deployment, we only have one rhocp node
 * both cuphycontroller container and PTP publisher which runs in openshift-ptp namespace
 * run on this only rhocp node so here PTP_CONSUMER is also aerail-smc-03.nvidia.com:9050
*/ 
std::shared_ptr<httplib::Server> AppUtils::startEventServer(const std::string& PTP_CONSUMER) {
    // this event server does not have core pin by itself, we call it from cuphydriver thread (rhocp_ptp_events_monitoring_func) 
    // that has core pin (affinity set to low priority core)
    // so the event server thread inherits this low priority core pin
    std::shared_ptr<httplib::Server> svr = std::make_shared<httplib::Server>();
    // Set thread pool count to 1, though httplib likely uses the same core affinity as the calling thread.
    svr->new_task_queue = []() -> httplib::TaskQueue* {
        return new httplib::ThreadPool(1); 
    };
    svr->Post("/event", [](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                return;
            }

            NVLOGI_FMT(TAG, "An event received and is kept in req.body" );
            NVLOGD_FMT(TAG, "Received body: {}", req.body);
            // TODO: Currently, We are pulling the events in a forever loop, instead of processing the event changes pushed by the publisher.
            // We can implement the logic to parse req.body and report PTP status accordingly.
            // But use pullEvents(), we are more confident that we do not miss any events.
            res.status = 204;
        } catch (const std::exception& e) {
            NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "Event server error at handling /event: {}", e.what());
            res.status = 500;
        }
    });

    // Split http consumer server host and port
    std::string host = "0.0.0.0";
    int port = 9050; // Default port we use

    auto pos = PTP_CONSUMER.find(':');
    if (pos != std::string::npos && pos != 0 && pos != PTP_CONSUMER.length() - 1) {
        host = PTP_CONSUMER.substr(0, pos);
        try {
            port = std::stoi(PTP_CONSUMER.substr(pos + 1));
        } catch (const std::exception& e) {
            NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "Invalid port in PTP_CONSUMER '{}': {}", PTP_CONSUMER, e.what());
            return nullptr;
        }  
    } else {
        NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "Invalid PTP_CONSUMER format: {}", PTP_CONSUMER);
        return nullptr; 
    }

    NVLOGC_FMT(TAG, "Starting RHOCP PTP event consumer server on {}:{}", host, port);
    std::thread([svr, host, port]() {
        // svr->listen is a blocking call that never returns. 
        if (!svr->listen(host.c_str(), port)) {
            NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "Failed to start server on {}:{}", host, port);
        }
    }).detach();
    return svr;
}

// example healthURL: "http://10.105.38.34:9043/api/ocloudNotifications/v2/health
// curl_global_init(CURL_GLOBAL_DEFAULT) and  curl_global_cleanup() should be called once per process
// typically at the start and end of your application.
bool AppUtils::publisherHealthCheck(const std::string& PTP_PUBLISHER, int retryDelaySeconds) {
    try {
        std::string healthURL = "http://" + PTP_PUBLISHER + rhocpApiPath + "health";
            bool healthy = false;
            for (int i = 0; i < 4; ++i) {
                NVLOGI_FMT(TAG, "Health check attempt {} for {}", i + 1, healthURL);
                auto curl = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(
                    curl_easy_init(), &curl_easy_cleanup);  
                if (!curl) {
                    NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "RHOCP curl failed at curl_easy_init()");
                    throw std::runtime_error("Error: CURL initialization failed");
                }

                curl_easy_setopt(curl.get(), CURLOPT_URL, healthURL.c_str());
                curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 5L);
                curl_easy_setopt(curl.get(), CURLOPT_FAILONERROR, 1L);   // fail if response code >= 400

                CURLcode res = curl_easy_perform(curl.get());
                long http_code = 0;
                curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);

                if (res == CURLE_OK && http_code == 200) {
                    NVLOGI_FMT(TAG, "publisherHealthCheck() returned healthy status from {}", healthURL);
                    healthy = true;
                    break;
                } else {
                    NVLOGW_FMT(TAG, "Attempt {} RHOCP PTP Health check failed for {}: {} (HTTP {})",
                            i, healthURL, curl_easy_strerror(res), http_code);
                    std::this_thread::sleep_for(std::chrono::seconds(retryDelaySeconds));
                }
            }
        return healthy;
    } catch (const std::exception& e) {
        NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "RHOCP PTP publisherHealthCheck() Error: {}", e.what());
        return false;
    }
}


/**
 * @brief return PTP subscriptions map[event name] -> [event resource path]
 * More supported PTP events can be found here:
 * https://github.com/redhat-cne/sdk-go/blob/main/pkg/event/ptp/types.go
 * https://github.com/redhat-cne/sdk-go/blob/main/pkg/event/ptp/resource.go
 */
const std::map<std::string, std::string> AppUtils::initPTPSubscriptions() {
    return {
        { "event.sync.sync-status.os-clock-sync-state-change",    "/sync/sync-status/os-clock-sync-state" },
        // { "event.sync.ptp-status.ptp-clock-class-change",         "/sync/ptp-status/clock-class" },
        { "event.sync.ptp-status.ptp-state-change",               "/sync/ptp-status/lock-state" },
        // { "event.sync.gnss-status.gnss-state-change",           "/sync/gnss-status/gnss-sync-status" },
        // { "event.sync.sync-status.synchronization-state-change",  "/sync/sync-status/sync-state" },  // this is the overall synchronization health of the node, including the OS System Clock
    };
}

/**
 * @brief Subscribe to events from the publisher. Before subscribe, the consumer server must be running and listening on /event endpoint. Can be done by startEventServer().
 * It uses this endpoint created by startEventServer() as payload["EndpointUri"] in the subscription request.
 * @param PTP_PUBLISHER The publisher address and port such as aerial-smc-03.nvidia.com:9043
 * @param PTP_NODE_NAME The openshift node name such as aerial-smc-03.nvidia.com gotten by `oc get node`
 * @param PTP_CONSUMER The consumer address and port such as aerial-r750-03.nvidia.com:9050
 *  example lock-state event subscription payload:
 * {
 *  "EndpointUri": "http://consumer.nvidia.com:9050/event", 
 *  "ResourceAddress": "/cluster/node/producer.nvidia.com/sync/ptp-status/lock-state"
 * }
 */
void AppUtils::subscribeToEvents(const std::string& PTP_PUBLISHER, const std::string& PTP_NODE_NAME, const std::string& PTP_CONSUMER, int retryCount) {
    try {
        if (retryCount > 1) {
            NVLOGW_FMT(TAG, "subscribeToEvents() retried more than once, skipping retry");
            return;
        }

        NVLOGI_FMT(TAG, "{} will subscribe to events from {}", PTP_CONSUMER, PTP_PUBLISHER);
        bool healthStatus = publisherHealthCheck(PTP_PUBLISHER);
        if (!healthStatus) {
            NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "Abort subscribeToEvents() because publisherHealthCheck() returned false for {}.", PTP_PUBLISHER);
            throw std::runtime_error("publisherHealthCheck() returned false");   
        }
        // NVLOGI_FMT(TAG, "Before subscribe, for simplicity, deleting all existing subscriptions for the publisher");
        // deleteAllSubscriptions(PTP_PUBLISHER); 
        const auto& subscriptions = initPTPSubscriptions();
        bool allSubscribed = true;

        std::string subscribeURL = "http://" + PTP_PUBLISHER + rhocpApiPath + "subscriptions";
        for (const auto& [eventType, eventResource] : subscriptions) {
            NVLOGI_FMT(TAG, "Subscribing to event: {}, resource: {}", eventType, eventResource);
            json payload = {
                {"EndpointUri", "http://" + PTP_CONSUMER + "/event"},
                {"ResourceAddress",  "/cluster/node/" + PTP_NODE_NAME + eventResource}
            };
            std::string payloadStr = payload.dump();
            auto curl = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(
                curl_easy_init(), &curl_easy_cleanup);
            if (!curl) {
                NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "RHOCP curl failed at curl_easy_init()");
                throw std::runtime_error("subscribeToEvents() Error: CURL initialization failed");
            }

            curl_easy_setopt(curl.get(), CURLOPT_URL, subscribeURL.c_str());
            curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE, payloadStr.size());
            curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS, payloadStr.c_str());
            curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 6L);
            curl_easy_setopt(curl.get(), CURLOPT_FAILONERROR, 1L);

            struct curl_slist* headers = curl_slist_append(nullptr, "Content-Type: application/json");
            curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers);
            CURLcode res = curl_easy_perform(curl.get());
            long http_code = 0;
            curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);
            curl_slist_free_all(headers);

            if (res == CURLE_OK && http_code == 201) {
                NVLOGI_FMT(TAG, "Successfully subscribed to event: {}", eventType);
            } 
            // else if (http_code == 409) {
            //     NVLOGW_FMT(TAG, "HTTP code {}, Subscription already exists for event: {}. Now delete all subscriptions and retry ", http_code, eventType);
            //     try {
            //         deleteAllSubscriptions(PTP_PUBLISHER);
            //     } catch (const std::exception& e) {
            //          NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT,
            //             "subscribeToEvents() Retry aborted: deleteAllSubscriptions() failed for {}: {}", PTP_PUBLISHER, e.what());
            //         allSubscribed = false;
            //         continue;
            //     }
            //     subscribeToEvents(PTP_PUBLISHER, PTP_NODE_NAME, PTP_CONSUMER, retryCount + 1);
            //     return; // prevent caller from continuing the subscripion for loop after retry
            // } 
            else if (http_code == 400) {
                NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "Failed to subscribe to event: {}, resource: {}. Error: {} (HTTP {})",
                           eventType, eventResource, curl_easy_strerror(res), http_code);
                NVLOGW_FMT(TAG, "Please make sure the PTP_CONSUMER sever {} is running and listening on /event endpoint", PTP_CONSUMER);
                allSubscribed = false;
            } else {
                NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "Failed to subscribe to event: {}, resource: {}. Error: {} (HTTP {})",
                           eventType, eventResource, curl_easy_strerror(res), http_code);
                allSubscribed = false;
            }
        }

        if (allSubscribed) {
            NVLOGC_FMT(TAG, "All subscriptions created successfully!");
        } else {
            NVLOGW_FMT(TAG, "Subscription(s) failed. Please check the logs above for event(s) failed.");
            throw std::runtime_error("subscribeToEvents() Error: subscription(s) failed");
        }

    } catch (const std::exception& e) {
        NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "subscribeToEvents() Error: {}", e.what());
    }            
}


void AppUtils::deleteAllSubscriptions(const std::string& PTP_PUBLISHER) {
    try {
        std::string deleteURL = "http://" + PTP_PUBLISHER + rhocpApiPath + "subscriptions";
        NVLOGC_FMT(TAG, "deleteAllSubscriptions() will use deleteURL: {}", deleteURL);
        bool healthStatus = publisherHealthCheck(PTP_PUBLISHER);
        if (!healthStatus) {
            NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "publisherHealthCheck() returned false for {}. Abort deleteAllSubscriptions()", PTP_PUBLISHER);
            throw std::runtime_error("publisherHealthCheck() returned false");
        }
        NVLOGC_FMT(TAG, "Deleting all subscriptions for publisher: {}", PTP_PUBLISHER);
        auto curl = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(
            curl_easy_init(), &curl_easy_cleanup);
        if (!curl) {
            NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "deleteAllSubscriptions() Error: Failed to initialize CURL");
            throw std::runtime_error("CURL init failed");
        }
        curl_easy_setopt(curl.get(), CURLOPT_URL, deleteURL.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_CUSTOMREQUEST, "DELETE");
        curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 6L); 
        curl_easy_setopt(curl.get(), CURLOPT_FAILONERROR, 1L);

        struct curl_slist* headers = curl_slist_append(nullptr, "Content-Type: application/json");
        curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers);
        CURLcode res = curl_easy_perform(curl.get());
        long http_code = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);
        curl_slist_free_all(headers);
        if (res != CURLE_OK) {
            NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "deleteAllSubscriptions() Error: Failed to delete subscriptions {}", curl_easy_strerror(res));
            throw std::runtime_error("deleteAllSubscriptions() Error: " + std::string(curl_easy_strerror(res)));
        }  else
            NVLOGC_FMT(TAG, "All subscriptions deleted successfully for publisher: {}", PTP_PUBLISHER);
    } catch (const std::exception& e) {
        NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "deleteAllSubscriptions() Error: {}", e.what());
        throw; 
    }
}