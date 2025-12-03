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
#include <unistd.h>
#include <mutex>
#include <csignal>
#include "cuphyoam.hpp"
#include "oam_services.hpp"

void CuphyPushNotificationServiceImpl::notifyClient(int clientId, std::string msg)
{
    std::lock_guard<std::mutex> lock(_streamMutex);
    auto it = _subscribers.find(clientId);
    if (it != _subscribers.end())
    {
        aerial::PushResponse message;
        message.set_data(msg);
        // Asynchronously write to the stored writer
        it->second.writer->Write(message);
    }
}

void CuphyPushNotificationServiceImpl::notifyAll(std::string msg)
{
    std::lock_guard<std::mutex> lock(_streamMutex);
    for(auto & subscriber : _subscribers)
    {
        aerial::PushResponse message;
        message.set_data(msg);
        // Asynchronously write to the stored writer
        subscriber.second.writer->Write(message);
    }
}

grpc::Status CuphyPushNotificationServiceImpl::Subscribe(grpc::ServerContext *context, const aerial::PushRequest *request,
                                                         grpc::ServerWriter<::aerial::PushResponse> *writer)
{
    // Send a message to the client
    aerial::PushResponse message;
    message.set_data("Hello, client. You are connected to Aerial OAM push notification service!");
    writer->Write(message);

    //std::lock_guard<std::mutex> lock(_streamMutex);
    subscriber client;
    client.writer = writer;
    client.finished = false;

    _streamMutex.lock();
    _subscribers[request->id()] = client;
    _streamMutex.unlock();

    while(!context->IsCancelled() && !_subscribers[request->id()].finished);
    if(context->IsCancelled())
    {
        std::cout << "Lost connection with client: " << request->id() << std::endl;
    }
    else
    {
        std::cout << "Client " << request->id() << " unsubscribed " << std::endl;
    }

    _streamMutex.lock();
    _subscribers.erase(request->id());
    _streamMutex.unlock();

    return grpc::Status::OK;
}

grpc::Status CuphyPushNotificationServiceImpl::Unsubscribe(grpc::ServerContext *context, const aerial::PushRequest *request,
                                                           ::aerial::PushResponse *response)
{
    std::lock_guard<std::mutex> lock(_streamMutex);
    auto it = _subscribers.find(request->id());
    if (it != _subscribers.end())
    {
        it->second.finished = true;
    }
    return grpc::Status::OK;
}