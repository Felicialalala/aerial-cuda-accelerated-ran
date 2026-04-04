/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "OnlineProtocol.h"

namespace cumac::online {

struct StepAction {
    int32_t stepIdx = 0;
    std::vector<int32_t> ueAction;
    std::vector<int16_t> prgAction;
};

struct StepState {
    StepStateHeader header;
    std::vector<float> obsCellFeatures;
    std::vector<float> obsUeFeatures;
    std::vector<float> obsPrgFeatures;
    std::vector<int16_t> obsEdgeIndex;
    std::vector<float> obsEdgeAttr;
    std::vector<uint8_t> actionMaskUe;
    std::vector<uint8_t> actionMaskCellUe;
    std::vector<uint8_t> actionMaskPrgCell;
};

class OnlineBridgeServer {
public:
    explicit OnlineBridgeServer(std::string socketPath);
    ~OnlineBridgeServer();

    bool initialize();

    bool recvResetReq(ResetReqPayload* req);
    bool sendResetRsp(const StepState& state);

    bool recvStepReq(StepAction* req, bool* closeReq = nullptr);
    bool sendStepRsp(const StepState& state);

    bool recvCloseReq();
    bool sendCloseRsp();

    bool sendError(int32_t code, const std::string& message);

private:
    std::string m_socketPath;
    int m_listenFd = -1;
    int m_clientFd = -1;

    bool readExact(int fd, void* buf, size_t len) const;
    bool writeExact(int fd, const void* buf, size_t len) const;

    bool recvMessage(MsgType expectedType, std::vector<uint8_t>* payload);
    bool sendMessage(MsgType type, const std::vector<uint8_t>& payload);

    std::vector<uint8_t> buildStatePayload(const StepState& state) const;

    void closeAll();
};

} // namespace cumac::online
