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

#pragma once

#include "aerial_common.grpc.pb.h"
#include "aerial_push_notification.grpc.pb.h"

class CuphyCommonServiceImpl final : public aerial::Common::Service
{
  std::atomic<int> active_clients{0};

  grpc::Status WarmUp(grpc::ServerContext* context, const aerial::GenericRequest* request, aerial::DummyReply* reply) override;

  grpc::Status TerminateCuphycontroller(grpc::ServerContext* context, const aerial::GenericRequest* request,
                  aerial::DummyReply* reply) override;

  grpc::Status GetSFN(grpc::ServerContext* context, const aerial::GenericRequest* request,
                  aerial::SFNReply* reply) override;

  grpc::Status GetCpuUtilization(grpc::ServerContext* context, const aerial::GenericRequest* request,
                  aerial::CpuUtilizationReply* reply) override;

  grpc::Status SetPuschH5DumpNextCrc(grpc::ServerContext* context, const aerial::GenericRequest* request, aerial::DummyReply* reply) override;

  grpc::Status GetFAPIStream(grpc::ServerContext* context, const aerial::FAPIStreamRequest* request, grpc::ServerWriter<aerial::FAPIStreamReply>* writer) override;

  grpc::Status UpdateCellParamsSyncCall(grpc::ServerContext* context, const aerial::CellParamUpdateRequest* request, aerial::CellParamUpdateReply* reply) override;

  grpc::Status UpdateCellParams(grpc::ServerContext* context, const aerial::CellParamUpdateRequest* request, aerial::DummyReply* reply) override;

  grpc::Status SendCellCtrlCmd(grpc::ServerContext* context, const aerial::CellCtrlCmdRequest* request, aerial::DummyReply* reply) override;

  grpc::Status SimulateCPUStall(grpc::ServerContext* context, const aerial::SimulatedCPUStallRequest* request, aerial::DummyReply* reply) override;

  grpc::Status SendFapiDelayCmd(grpc::ServerContext* context, const aerial::FapiDelayCmdRequest* request, aerial::DummyReply* reply) override;

  grpc::Status SimulateULUPlaneDrop(grpc::ServerContext* context, const aerial::SimulateULUPlaneDropRequest* request, aerial::DummyReply* reply) override;

  grpc::Status SendSfnSlotSyncCmd(grpc::ServerContext* context, const aerial::SfnSlotSyncRequest* request,aerial::DummyReply* reply) override;

  grpc::Status SendGenericAsyncCmd(grpc::ServerContext* context, const aerial::GenericAsyncRequest* request, aerial::DummyReply* reply) override;

  grpc::Status SendCellUlPcapCmd(grpc::ServerContext* context, const aerial::PcapRequest* request, aerial::DummyReply* reply) override;

  grpc::Status SendZeroUplane(grpc::ServerContext* context, const aerial::ZeroUplaneRequest* request, aerial::DummyReply* reply) override;

  grpc::Status FlushUlPcap(grpc::ServerContext* context, const aerial::FlushUlPcapRequest* request, aerial::DummyReply* reply) override;
};

struct subscriber
{
  grpc::ServerWriter<::aerial::PushResponse>* writer;
  bool finished;
};

class CuphyPushNotificationServiceImpl final : public aerial::PushNotification::Service
{
public:
  grpc::Status Subscribe(grpc::ServerContext *context, const aerial::PushRequest *request,
                         grpc::ServerWriter<::aerial::PushResponse> *writer) override;
  grpc::Status Unsubscribe(grpc::ServerContext *context, const aerial::PushRequest *request,
                           ::aerial::PushResponse *response) override;
  void notifyClient(int clientId, std::string msg);
  void notifyAll(std::string msg);

private:
  std::mutex _streamMutex;
  std::unordered_map<int, struct subscriber> _subscribers;
};
