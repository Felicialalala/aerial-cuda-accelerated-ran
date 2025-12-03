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

#include <iostream>
#include <string>
#include <grpcpp/grpcpp.h>
#include "aerial_common.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using aerial::CellParamUpdateRequest;
using aerial::DummyReply;

int main(int argc, char** argv)
{
  std::string server_address("0.0.0.0:50051");
  std::string mac_addr("26:04:9D:9E:29:DD");
  int cell_id = 1;
  int vlan_tci = 0xE002;

  for (int i = 1; i < argc; i++)
  {
    if (strcmp(argv[i], "--server_address") == 0)
    {
      server_address = argv[++i];
    }
    else if (strcmp(argv[i], "--cell_id") == 0)
    {
      cell_id = std::stoi(argv[++i]);
    }
    else if (strcmp(argv[i], "--mac_addr") == 0)
    {
      mac_addr = argv[++i];
    }
    else if (strcmp(argv[i], "--vlan_tci") == 0)
    {
      vlan_tci = std::stoi(argv[++i], nullptr, 16);
    }
  }

  aerial::Common::Stub stub(grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials()));

  CellParamUpdateRequest request;
  request.set_cell_id(cell_id);
  request.set_dst_mac_addr(mac_addr);
  request.set_update_network_cfg(true);
  request.set_vlan_tci(vlan_tci);

  DummyReply reply;
  ClientContext context;

  Status status = stub.UpdateCellParams(&context, request, &reply);
  if (status.ok())
  {
    std::cout << "Message successfully sent to server side" << std::endl;
  }
  else
  {
    std::cout << "RPC failed: " << status.error_code() << ": " << status.error_message() << std::endl;
  }

  return 0;
}