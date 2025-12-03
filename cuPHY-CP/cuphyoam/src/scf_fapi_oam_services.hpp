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
#include <grpcpp/grpcpp.h>

#include "scf_fapi_p9.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
//using grpc::Status;

using p9_messages::v1::Msg;
using p9_messages::v1::Header;
using p9_messages::v1::Body;
using p9_messages::v1::Request;
using p9_messages::v1::Response;
using p9_messages::v1::Get;
using p9_messages::v1::GetResp;
using p9_messages::v1::EditConfig;
using p9_messages::v1::EditConfigResp;
using p9_messages::v1::Error;
//using p9_messages::v1::Status;

// gRPC server implementation
class P9MessagesService final : public p9_messages::v1::P9Messages::Service
{
  grpc::Status HandleMsg(ServerContext *context, const Msg *request, Msg *response) override;
};