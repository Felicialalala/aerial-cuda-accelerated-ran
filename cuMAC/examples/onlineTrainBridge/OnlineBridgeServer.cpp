/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OnlineBridgeServer.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace cumac::online {
namespace {

template <typename T>
void appendPod(std::vector<uint8_t>& out, const T& value)
{
    const size_t pos = out.size();
    out.resize(pos + sizeof(T));
    std::memcpy(out.data() + pos, &value, sizeof(T));
}

template <typename T>
bool readPod(const uint8_t* data, size_t size, size_t* off, T* out)
{
    if (*off + sizeof(T) > size) {
        return false;
    }
    std::memcpy(out, data + *off, sizeof(T));
    *off += sizeof(T);
    return true;
}

} // namespace

OnlineBridgeServer::OnlineBridgeServer(std::string socketPath) : m_socketPath(std::move(socketPath))
{
}

OnlineBridgeServer::~OnlineBridgeServer()
{
    closeAll();
}

void OnlineBridgeServer::closeAll()
{
    if (m_clientFd >= 0) {
        close(m_clientFd);
        m_clientFd = -1;
    }
    if (m_listenFd >= 0) {
        close(m_listenFd);
        m_listenFd = -1;
    }
    if (!m_socketPath.empty()) {
        unlink(m_socketPath.c_str());
    }
}

bool OnlineBridgeServer::readExact(int fd, void* buf, size_t len) const
{
    uint8_t* p = reinterpret_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < len) {
        const ssize_t n = recv(fd, p + done, len - done, 0);
        if (n == 0) {
            return false;
        }
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool OnlineBridgeServer::writeExact(int fd, const void* buf, size_t len) const
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(buf);
    size_t done = 0;
    while (done < len) {
        const ssize_t n = send(fd, p + done, len - done, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool OnlineBridgeServer::initialize()
{
    if (m_socketPath.empty()) {
        std::cerr << "[ONLINE_BRIDGE] empty socket path\n";
        return false;
    }

    closeAll();

    m_listenFd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (m_listenFd < 0) {
        std::cerr << "[ONLINE_BRIDGE] socket() failed\n";
        return false;
    }

    unlink(m_socketPath.c_str());

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    if (m_socketPath.size() >= sizeof(addr.sun_path)) {
        std::cerr << "[ONLINE_BRIDGE] socket path too long: " << m_socketPath << "\n";
        return false;
    }
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", m_socketPath.c_str());

    if (bind(m_listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::cerr << "[ONLINE_BRIDGE] bind() failed: " << std::strerror(errno) << "\n";
        return false;
    }
    if (listen(m_listenFd, 1) != 0) {
        std::cerr << "[ONLINE_BRIDGE] listen() failed\n";
        return false;
    }

    std::cerr << "[ONLINE_BRIDGE] waiting client on " << m_socketPath << "\n";
    m_clientFd = accept(m_listenFd, nullptr, nullptr);
    if (m_clientFd < 0) {
        std::cerr << "[ONLINE_BRIDGE] accept() failed\n";
        return false;
    }
    std::cerr << "[ONLINE_BRIDGE] client connected\n";
    return true;
}

bool OnlineBridgeServer::recvMessage(MsgType expectedType, std::vector<uint8_t>* payload)
{
    if (payload == nullptr || m_clientFd < 0) {
        return false;
    }

    MsgHeader h {};
    if (!readExact(m_clientFd, &h, sizeof(h))) {
        return false;
    }
    if (h.magic != kMagic || h.version != kVersion) {
        std::cerr << "[ONLINE_BRIDGE] invalid header magic/version\n";
        return false;
    }
    if (static_cast<MsgType>(h.type) != expectedType) {
        std::cerr << "[ONLINE_BRIDGE] unexpected message type=" << h.type
                  << " expected=" << static_cast<uint16_t>(expectedType) << "\n";
        return false;
    }

    payload->assign(h.payloadBytes, 0U);
    if (h.payloadBytes > 0U) {
        if (!readExact(m_clientFd, payload->data(), payload->size())) {
            return false;
        }
    }
    return true;
}

bool OnlineBridgeServer::sendMessage(MsgType type, const std::vector<uint8_t>& payload)
{
    if (m_clientFd < 0) {
        return false;
    }
    MsgHeader h {};
    h.type = static_cast<uint16_t>(type);
    h.payloadBytes = static_cast<uint32_t>(payload.size());

    if (!writeExact(m_clientFd, &h, sizeof(h))) {
        return false;
    }
    if (!payload.empty()) {
        if (!writeExact(m_clientFd, payload.data(), payload.size())) {
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> OnlineBridgeServer::buildStatePayload(const StepState& state) const
{
    std::vector<uint8_t> out;
    out.reserve(
        sizeof(StepStateHeader) +
        sizeof(float) * state.obsCellFeatures.size() +
        sizeof(float) * state.obsUeFeatures.size() +
        sizeof(int16_t) * state.obsEdgeIndex.size() +
        sizeof(float) * state.obsEdgeAttr.size() +
        sizeof(uint8_t) * state.actionMaskUe.size() +
        sizeof(uint8_t) * state.actionMaskCellUe.size() +
        sizeof(uint8_t) * state.actionMaskPrgCell.size());

    appendPod(out, state.header);

    const auto appendRaw = [&](const void* p, size_t bytes) {
        const size_t pos = out.size();
        out.resize(pos + bytes);
        std::memcpy(out.data() + pos, p, bytes);
    };

    if (!state.obsCellFeatures.empty()) {
        appendRaw(state.obsCellFeatures.data(), sizeof(float) * state.obsCellFeatures.size());
    }
    if (!state.obsUeFeatures.empty()) {
        appendRaw(state.obsUeFeatures.data(), sizeof(float) * state.obsUeFeatures.size());
    }
    if (!state.obsEdgeIndex.empty()) {
        appendRaw(state.obsEdgeIndex.data(), sizeof(int16_t) * state.obsEdgeIndex.size());
    }
    if (!state.obsEdgeAttr.empty()) {
        appendRaw(state.obsEdgeAttr.data(), sizeof(float) * state.obsEdgeAttr.size());
    }
    if (!state.actionMaskUe.empty()) {
        appendRaw(state.actionMaskUe.data(), sizeof(uint8_t) * state.actionMaskUe.size());
    }
    if (!state.actionMaskCellUe.empty()) {
        appendRaw(state.actionMaskCellUe.data(), sizeof(uint8_t) * state.actionMaskCellUe.size());
    }
    if (!state.actionMaskPrgCell.empty()) {
        appendRaw(state.actionMaskPrgCell.data(), sizeof(uint8_t) * state.actionMaskPrgCell.size());
    }

    return out;
}

bool OnlineBridgeServer::recvResetReq(ResetReqPayload* req)
{
    if (req == nullptr) {
        return false;
    }
    std::vector<uint8_t> payload;
    if (!recvMessage(MsgType::ResetReq, &payload)) {
        return false;
    }

    size_t off = 0U;
    if (!readPod(payload.data(), payload.size(), &off, req)) {
        std::cerr << "[ONLINE_BRIDGE] invalid ResetReq payload\n";
        return false;
    }
    return true;
}

bool OnlineBridgeServer::sendResetRsp(const StepState& state)
{
    return sendMessage(MsgType::ResetRsp, buildStatePayload(state));
}

bool OnlineBridgeServer::recvStepReq(StepAction* req, bool* closeReq)
{
    if (req == nullptr || m_clientFd < 0) {
        return false;
    }
    if (closeReq != nullptr) {
        *closeReq = false;
    }

    MsgHeader h {};
    if (!readExact(m_clientFd, &h, sizeof(h))) {
        return false;
    }
    if (h.magic != kMagic || h.version != kVersion) {
        std::cerr << "[ONLINE_BRIDGE] invalid header magic/version\n";
        return false;
    }

    std::vector<uint8_t> payload(h.payloadBytes, 0U);
    if (h.payloadBytes > 0U) {
        if (!readExact(m_clientFd, payload.data(), payload.size())) {
            return false;
        }
    }

    const MsgType type = static_cast<MsgType>(h.type);
    if (type == MsgType::CloseReq) {
        if (!payload.empty()) {
            std::cerr << "[ONLINE_BRIDGE] invalid CloseReq payload size\n";
            return false;
        }
        if (closeReq != nullptr) {
            *closeReq = true;
        }
        return true;
    }
    if (type != MsgType::StepReq) {
        std::cerr << "[ONLINE_BRIDGE] unexpected message type=" << h.type
                  << " expected=" << static_cast<uint16_t>(MsgType::StepReq) << "\n";
        return false;
    }

    size_t off = 0U;
    StepReqHeader hdr {};
    if (!readPod(payload.data(), payload.size(), &off, &hdr)) {
        std::cerr << "[ONLINE_BRIDGE] invalid StepReq header\n";
        return false;
    }

    req->stepIdx = hdr.stepIdx;
    req->ueAction.assign(hdr.ueActionLen, -1);
    req->prgAction.assign(hdr.prgActionLen, static_cast<int16_t>(-1));

    const size_t ueBytes = sizeof(int32_t) * static_cast<size_t>(hdr.ueActionLen);
    const size_t prgBytes = sizeof(int16_t) * static_cast<size_t>(hdr.prgActionLen);
    if (off + ueBytes + prgBytes != payload.size()) {
        std::cerr << "[ONLINE_BRIDGE] invalid StepReq payload size\n";
        return false;
    }

    if (ueBytes > 0U) {
        std::memcpy(req->ueAction.data(), payload.data() + off, ueBytes);
        off += ueBytes;
    }
    if (prgBytes > 0U) {
        std::memcpy(req->prgAction.data(), payload.data() + off, prgBytes);
    }
    return true;
}

bool OnlineBridgeServer::sendStepRsp(const StepState& state)
{
    return sendMessage(MsgType::StepRsp, buildStatePayload(state));
}

bool OnlineBridgeServer::recvCloseReq()
{
    std::vector<uint8_t> payload;
    if (!recvMessage(MsgType::CloseReq, &payload)) {
        return false;
    }
    return payload.empty();
}

bool OnlineBridgeServer::sendCloseRsp()
{
    std::vector<uint8_t> empty;
    return sendMessage(MsgType::CloseRsp, empty);
}

bool OnlineBridgeServer::sendError(int32_t code, const std::string& message)
{
    std::vector<uint8_t> payload;
    ErrorPayload e {};
    e.code = code;
    appendPod(payload, e);

    const uint32_t msgLen = static_cast<uint32_t>(message.size());
    appendPod(payload, msgLen);
    if (msgLen > 0U) {
        const size_t pos = payload.size();
        payload.resize(pos + msgLen);
        std::memcpy(payload.data() + pos, message.data(), msgLen);
    }
    return sendMessage(MsgType::ErrorRsp, payload);
}

} // namespace cumac::online
