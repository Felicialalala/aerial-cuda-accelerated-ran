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

#ifndef _FD_SHARE_H_
#define _FD_SHARE_H_

#include <sys/socket.h>
#include <sys/un.h>

int send_fd(int fd, int fd_to_send);

int recv_fd(int fd);

void unix_sock_address_init(struct sockaddr_un* addr, char* path);

int unix_sock_create(struct sockaddr_un* addr);

int unix_sock_listen_and_accept(int listen_fd, struct sockaddr_un* client_addr);

int unix_sock_connect(int sock_fd, struct sockaddr_un* server_addr);

#endif /* _FD_SHARE_H_ */
