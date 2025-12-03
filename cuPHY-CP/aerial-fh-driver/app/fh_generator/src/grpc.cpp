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

#include "fh_generator.hpp"

#include <grpcpp/grpcpp.h>
#include "aerial_common.grpc.pb.h"
#include <chrono>

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using namespace std::chrono;

void* FhGenerator::du_sfn_slot_sync_cmd_thread_func(void* arg)
{
    // Switch to a low_priority_core to avoid blocking time critical thread
    // auto& appConfig = AppConfig::getInstance();
    // auto low_priority_core = appConfig.getLowPriorityCore();
    // NVLOGD_FMT(TAG, "{}: OAM thread affinity set to cpu core {}", __func__, low_priority_core);
    // nv_assign_thread_cpu_core(low_priority_core);
    nanoseconds ts_now;

    if(pthread_setname_np(pthread_self(), "du_sync") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }

    sleep(4);//wait for readiness of OAM server

    while(!synced_with_peer.load())
    {
        CuphyOAM *oam = CuphyOAM::getInstance();
        CuphyOAMSfnSlotSyncCmd* sfn_slot_sync_cmd;
        while ((sfn_slot_sync_cmd = oam->get_sfn_slot_sync_cmd()) != nullptr)
        {
            ts_now = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
            NVLOGC_FMT(TAG, "{} time_anchor {} received time_anchor {}",__func__, time_anchor_.load(), sfn_slot_sync_cmd->time_anchor);
            if(sfn_slot_sync_cmd->time_anchor == time_anchor_.load())
            {
                synced_with_peer.store(true);
                NVLOGC_FMT(TAG, "{} Synchronized!",__func__);
                break;
            }
            oam->free_sfn_slot_sync_cmd(sfn_slot_sync_cmd);
        }
        usleep(100000);
    }
    NVLOGC_FMT(TAG, "du_sfn_slot_sync_cmd_thread_func exit");
    return nullptr;
}

void* FhGenerator::ru_sfn_slot_sync_cmd_thread_func(void* arg)
{
    // Switch to a low_priority_core to avoid blocking time critical thread
    // auto& appConfig = AppConfig::getInstance();
    // auto low_priority_core = appConfig.getLowPriorityCore();
    // NVLOGD_FMT(TAG, "{}: OAM thread affinity set to cpu core {}", __func__, low_priority_core);
    // nv_assign_thread_cpu_core(low_priority_core);
    nanoseconds ts_now;

    if(pthread_setname_np(pthread_self(), "ru_sync") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }

    sleep(4);//wait for readiness of OAM server

    while(!synced_with_peer.load())
    {
        CuphyOAM *oam = CuphyOAM::getInstance();
        CuphyOAMSfnSlotSyncCmd* sfn_slot_sync_cmd;
        while ((sfn_slot_sync_cmd = oam->get_sfn_slot_sync_cmd()) != nullptr)
        {
            ts_now = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
            NVLOGC_FMT(TAG, "{} time_anchor {} received time_anchor {}",__func__, time_anchor_.load(), sfn_slot_sync_cmd->time_anchor);

            if(sfn_slot_sync_cmd->time_anchor == time_anchor_.load())
            {
                synced_with_peer.store(true);
                NVLOGC_FMT(TAG, "{} Synchronized!",__func__);
                break;
            }

            if(sfn_slot_sync_cmd->time_anchor > ts_now.count() + 2 * 1000 * 1000 * 1000)
            {
                NVLOGC_FMT(TAG, "{} received request, time_anchor {} is acceptable, send ack",__func__, sfn_slot_sync_cmd->time_anchor);
                ru_send_ack.store(true);
                time_anchor_.store(sfn_slot_sync_cmd->time_anchor);
            }
            else
            {
                NVLOGC_FMT(TAG, "{} received request, time_anchor {} is too soon, ignore request",__func__, sfn_slot_sync_cmd->time_anchor);
            }

            oam->free_sfn_slot_sync_cmd(sfn_slot_sync_cmd);
        }
        usleep(100000);
    }
    NVLOGC_FMT(TAG, "ru_sfn_slot_sync_cmd_thread_func exit");
    return nullptr;
}

int FhGenerator::send_sfn_slot_sync_grpc_command()
{
    int ret=0;
    // NVLOGC_FMT(TAG,"{} : peer_oam_addr {}",__func__, peer_oam_addr);
    aerial::Common::Stub stub(grpc::CreateChannel(peer_oam_addr, grpc::InsecureChannelCredentials()));
    nanoseconds ts_now = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
    uint64_t curr_time=ts_now.count();
    int32_t sync_done=2;

    aerial::SfnSlotSyncRequest request;
    request.set_sync_done(sync_done);
    request.set_time_anchor(time_anchor_);

    aerial::DummyReply reply;
    ClientContext context;

    Status status = stub.SendSfnSlotSyncCmd(&context, request, &reply);
    if (status.ok())
    {
        NVLOGC_FMT(TAG,"gRPC message sent successfully for SFN/slot sync at time {} for anchor {}",curr_time, time_anchor_.load());
        ret=0;
    }
    else
    {
        NVLOGC_FMT(TAG,"gRPC channel not active yet at {}, waiting for receiving application to run", peer_oam_addr);
        ret=1;
    }
    return ret;
}