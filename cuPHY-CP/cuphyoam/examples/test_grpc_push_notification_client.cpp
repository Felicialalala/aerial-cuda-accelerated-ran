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
#include <memory>
#include <string>
#include <thread>
#include <grpcpp/grpcpp.h>
#include "aerial_push_notification.grpc.pb.h"

using aerial::PushNotification;
using aerial::PushRequest;
using aerial::PushResponse;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

// Implementation of the SubscribeStreamClient class
class SubscribeStreamClient final
{
public:
  SubscribeStreamClient(std::shared_ptr<Channel> channel)
      : stub_(PushNotification::NewStub(channel)) {}

  void Subscribe(int id)
  {
    ClientContext context;
    PushRequest request;
    request.set_id(id);

    std::unique_ptr<grpc::ClientReader<PushResponse>> reader(
        stub_->Subscribe(&context, request));

    while (reader->Read(&response_))
    {
      std::cout << "Received message: " << response_.data() << std::endl;
    }

    Status status = reader->Finish();
    if (!status.ok())
    {
      std::cout << "Error: " << status.error_message() << std::endl;
    }
  }

  void Unsubscribe(int id)
  {
    ClientContext context;
    PushRequest request;
    PushResponse reply;
    request.set_id(id);

    Status status = stub_->Unsubscribe(&context, request, &reply);
    if (status.ok())
    {
      std::cout << "Successfully Unsubscribed" << std::endl;
    }
    else
    {
      std::cout << "Unsubscribing RPC failed: " << status.error_code() << ": " << status.error_message() << std::endl;
    }
  }

private:
  std::unique_ptr<PushNotification::Stub> stub_;
  PushResponse response_;
};

int main(int argc, char **argv)
{
  int client_id = 1;
  for (int i = 1; i < argc; i++)
  {
    if (strcmp(argv[i], "--client_id") == 0)
    {
      client_id = std::stoi(argv[++i]);
    }
  }

  std::string server_address("localhost:50051");
  std::shared_ptr<Channel> channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  SubscribeStreamClient client(channel);

  std::thread thread([&]()
                     {
    std::cout << "Unsubscribe in 10 seconds..." << std::endl;
    sleep(10);
    client.Unsubscribe(client_id); });

  // Subscribe to notifications
  client.Subscribe(client_id);

  thread.join();
  return 0;
}