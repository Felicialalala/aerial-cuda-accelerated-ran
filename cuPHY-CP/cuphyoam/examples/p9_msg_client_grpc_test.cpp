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
#include <fstream>
#include <sstream>

#include <grpcpp/grpcpp.h>

#include "scf_fapi_p9.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

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

using namespace std::literals;

static const std::string PATH_DELAY_MANAGEMENT = "/o-ran-delay-management:delay-management";

static std::string ru_1_delay_mgmt = R"(<delay-management xmlns="urn:o-ran:delay:1.0">
    <bandwidth-scs-delay-state>
      <bandwidth>100000</bandwidth>
      <subcarrier-spacing>1000</subcarrier-spacing>
      <ru-delay-profile>
        <t2a-min-up>1000000000</t2a-min-up>
        <t2a-max-up>1000000000</t2a-max-up>
        <t2a-min-cp-dl>1000000000</t2a-min-cp-dl>
        <t2a-max-cp-dl>1000000000</t2a-max-cp-dl>
        <tcp-adv-dl>1000000000</tcp-adv-dl>
        <ta3-min>1000000000</ta3-min>
        <ta3-max>1000000000</ta3-max>
        <t2a-min-cp-ul>1000000000</t2a-min-cp-ul>
        <t2a-max-cp-ul>1000000000</t2a-max-cp-ul>
        <ta3-min-ack>1000000000</ta3-min-ack>
        <ta3-max-ack>1000000000</ta3-max-ack>
      </ru-delay-profile>
    </bandwidth-scs-delay-state>
    <adaptive-delay-configuration>
      <bandwidth-scs-delay-state>
        <bandwidth>100000</bandwidth>
        <subcarrier-spacing>1000</subcarrier-spacing>
        <o-du-delay-profile>
          <t1a-max-up>1000000000</t1a-max-up>
          <tx-max>1000000000</tx-max>
          <ta4-max>1000000000</ta4-max>
          <rx-max>1000000000</rx-max>
          <t1a-max-cp-dl>1000000000</t1a-max-cp-dl>
        </o-du-delay-profile>
      </bandwidth-scs-delay-state>
      <transport-delay>
        <t12-min>1000000000</t12-min>
        <t12-max>1000000000</t12-max>
        <t34-min>1000000000</t34-min>
        <t34-max>1000000000</t34-max>
      </transport-delay>
    </adaptive-delay-configuration>
    <beam-context-gap-period>333</beam-context-gap-period>
  </delay-management>)"s;

static std::string ru_2_delay_mgmt = R"(<delay-management xmlns="urn:o-ran:delay:1.0">
    <bandwidth-scs-delay-state>
      <bandwidth>100000</bandwidth>
      <subcarrier-spacing>1000</subcarrier-spacing>
      <ru-delay-profile>
        <t2a-min-up>1111111</t2a-min-up>
        <t2a-max-up>2222222</t2a-max-up>
        <t2a-min-cp-dl>3333333</t2a-min-cp-dl>
        <t2a-max-cp-dl>4444444</t2a-max-cp-dl>
        <tcp-adv-dl>125000</tcp-adv-dl>
        <ta3-min>6666666</ta3-min>
        <ta3-max>5555555</ta3-max>
        <t2a-min-cp-ul>8888888</t2a-min-cp-ul>
        <t2a-max-cp-ul>777777</t2a-max-cp-ul>
        <ta3-min-ack>99999</ta3-min-ack>
        <ta3-max-ack>888888</ta3-max-ack>
      </ru-delay-profile>
    </bandwidth-scs-delay-state>
    <adaptive-delay-configuration>
      <bandwidth-scs-delay-state>
        <bandwidth>100000</bandwidth>
        <subcarrier-spacing>1000</subcarrier-spacing>
        <o-du-delay-profile>
          <t1a-max-up>1000000000</t1a-max-up>
          <tx-max>1000000000</tx-max>
          <ta4-max>1000000000</ta4-max>
          <rx-max>1000000000</rx-max>
          <t1a-max-cp-dl>1000000000</t1a-max-cp-dl>
        </o-du-delay-profile>
      </bandwidth-scs-delay-state>
      <transport-delay>
        <t12-min>1000000000</t12-min>
        <t12-max>1000000000</t12-max>
        <t34-min>1000000000</t34-min>
        <t34-max>1000000000</t34-max>
      </transport-delay>
    </adaptive-delay-configuration>
    <beam-context-gap-period>333</beam-context-gap-period>
  </delay-management>)"s;

// gRPC client implementation
class P9MessagesClient {
public:
    P9MessagesClient(std::shared_ptr<Channel> channel) : stub_(p9_messages::v1::P9Messages::NewStub(channel)) {}

    // gRPC client method implementation
    void Get(int32_t phy_id, const std::string& filter) {
        // Create a Get request message
        p9_messages::v1::Get* get_req = new p9_messages::v1::Get();
        get_req->add_filter(filter);

        // Create a request message
        Msg* request = new Msg();
        request->mutable_header()->set_msg_id("1");
        request->mutable_header()->set_oru_name("ORU");
        request->mutable_header()->set_vf_id(1);
        request->mutable_header()->set_phy_id(phy_id);
        request->mutable_header()->set_trp_id(3);
        request->mutable_body()->mutable_request()->set_allocated_get(get_req);

        // Create a response message
        Msg* response = new Msg();

        // Send the request message to the server and get the response message
        ClientContext context;
        grpc::Status status = stub_->HandleMsg(&context, *request, response);
        if (status.ok()) {
            // Handle the response message
            std::cout << "Received response message with ID: " << response->header().msg_id() << std::endl;
            if (response->body().has_response()) {
                if (response->body().response().has_get_resp()) {
                    std::cout << "Received Get response" << std::endl;
                    std::cout << "Status: " << response->body().response().get_resp().status_resp().status_code() << std::endl;
                    std::cout << "Data: " << response->body().response().get_resp().data() << std::endl;
                }
            }
        } else {
            std::cout << "RPC failed: " << status.error_message() << std::endl;
        }

        // Cleanup
        delete request;
        delete response;
    }

    // gRPC client method implementation
    void EditConfig(int32_t phy_id, std::string& cfg) {
        // Create a EditConfig request message
        p9_messages::v1::EditConfig* edit_config_req = new p9_messages::v1::EditConfig();
        std::string *data = new std::string(cfg);
        edit_config_req->set_allocated_delta_config(data);

        // Create a request message
        Msg* request = new Msg();
        request->mutable_header()->set_msg_id("1");
        request->mutable_header()->set_oru_name("ORU");
        request->mutable_header()->set_vf_id(1);
        request->mutable_header()->set_phy_id(phy_id);
        request->mutable_header()->set_trp_id(3);
        request->mutable_body()->mutable_request()->set_allocated_edit_config(edit_config_req);

        // Create a response message
        Msg* response = new Msg();

        // Send the request message to the server and get the response message
        ClientContext context;
        grpc::Status status = stub_->HandleMsg(&context, *request, response);
        if (status.ok()) {
            // Handle the response message
            std::cout << "Received response message with ID: " << response->header().msg_id() << std::endl;
            if (response->body().has_response()) {
                if (response->body().response().has_edit_config_resp()) {
                    std::cout << "Received EditConfig response" << std::endl;
                    std::cout << "Status: " << response->body().response().edit_config_resp().status_resp().status_code() << std::endl;
                }
            }
        } else {
            std::cout << "RPC failed: " << status.error_message() << std::endl;
        }

        auto res = edit_config_req->release_delta_config();

        // Cleanup
        delete request;
        delete response;
    }

private:
    std::unique_ptr<p9_messages::v1::P9Messages::Stub> stub_;
};

int main(int argc, char** argv) {
    std::string server_address("localhost:50051");
    std::shared_ptr<Channel> channel = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
    P9MessagesClient client(channel);

    int phy_id = -1;
    std::string cmd = "";
    std::string xml_file = "";
    std::string xpath = "";
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--phy_id") == 0)
        {
            phy_id = std::stoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--cmd") == 0)
        {
            cmd = argv[++i];
        }
        else if (strcmp(argv[i], "--xml_file") == 0)
        {
            xml_file = argv[++i];
        }
        else if (strcmp(argv[i], "--xpath") == 0)
        {
            xpath = argv[++i];
        }
    }

    if (argc > 1)
    {
        if (phy_id == -1)
        {
            std::cout << "No phy_id specified, exit.. " << std::endl;
        }

        if (cmd == "edit_config")
        {
            std::ifstream file(xml_file);
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string xmlString = buffer.str();
            // std::cout << xmlString << std::endl;
            client.EditConfig(phy_id, xmlString);
        }
        else if (cmd == "get")
        {
            // std::cout << xpath << std::endl;
            client.Get(phy_id, xpath);
        }
    }
    else
    {
        // Send EditConfig requests to the server
        client.EditConfig(1, ru_1_delay_mgmt);
        sleep(2);
        client.EditConfig(2, ru_2_delay_mgmt);
        sleep(5);
        // Send Get requests to the server
        client.Get(1, PATH_DELAY_MANAGEMENT);
        sleep(2);
        client.Get(2, PATH_DELAY_MANAGEMENT);
    }

    return 0;
}
