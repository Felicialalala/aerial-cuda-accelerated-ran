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

# if !defined(NV_IPC_ENDPOINT_HPP_)
#define NV_IPC_ENDPOINT_HPP_

#include "nv_phy_utils.hpp"
#include "nv_phy_base_common.hpp"
#include <sys/epoll.h>
#include <map>
#include <atomic>
#include <sstream>

namespace nv {

    /**
     * @brief UDP transport configuration
     *
     * Configuration parameters for UDP-based IPC communication.
     */
    struct udp_config: public ipc_config {

        /**
         * @brief Constructor
         * @param addr IP address
         * @param remote Remote port number
         * @param local Local port number
         */
        udp_config(std::string addr, int remote, int local):
            ip(addr), remote_port(remote), local_port(local) {
        }
        
        std::string ip;         ///< IP address
        int remote_port;        ///< Remote port number
        int local_port;         ///< Local port number
    };

    /**
     * @brief Shared memory configuration
     *
     * Configuration parameters for shared memory based IPC.
     */
    struct shm_config: public ipc_config {
        int cell_index;  ///< Cell index for this shared memory region
        
        /**
         * @brief Constructor
         * @param cell_id Cell identifier
         */
        shm_config(int cell_id): cell_index(cell_id) {
            max_msg_buf_size_ = 152;
            max_data_buf_size_ = 150 * 1024;
        }
    };

    /**
     * @brief Base IPC endpoint class
     *
     * Abstract base class for inter-process communication endpoints.
     * Supports reader/writer callback pattern for message handling.
     */
    struct ipc_endpoint: public ipc_base {
        
        /**
         * @brief Get configuration
         * @return Pointer to IPC configuration (nullptr in base class)
         */
        ipc_config* get_config() { return nullptr;}
        
        /**
         * @brief Get message buffer
         * @return Pointer to message buffer (nullptr in base class)
         */
        virtual void* get_msg_buf() {return nullptr; }
        
        /**
         * @brief Get data buffer
         * @return Pointer to data buffer (nullptr in base class)
         */
        virtual void* get_data_buf() { return nullptr; }
        
        /**
         * @brief Constructor
         * @param reader Pointer to reader callback function
         * @param writer Pointer to writer callback function
         */
        ipc_endpoint(reader * reader, writer* writer): 
            _reader(reader), _writer(writer) {}

        /**
         * @brief Get reader callback
         * @return Pointer to reader function
         */
        virtual reader* get_reader() { return _reader;}
        
        /**
         * @brief Get writer callback
         * @return Pointer to writer function
         */
        virtual writer* get_writer() { return _writer;}

        /**
         * @brief Set writer callback
         * @param writer Pointer to new writer function
         */
        virtual void set_writer(writer * writer) {
            if (writer != _writer) {
                _writer = writer;
            }
        }

        /**
         * @brief Set reader callback
         * @param reader Pointer to new reader function
         */
        virtual void set_reader(reader* reader) {
            if (reader != _reader) {
                _reader = reader;
            }
        }

        /**
         * @brief Write message by ID
         * @param msg_id Message identifier
         */
        virtual void write(uint16_t msg_id) {}
        
        /**
         * @brief Write raw buffer
         * @param buf Buffer pointer
         * @param length Buffer length in bytes
         */
        virtual void write(void* buf, std::size_t length) {}
     
        reader * _reader;  ///< Reader callback function pointer
        writer * _writer;  ///< Writer callback function pointer
    };
    
    /**
     * @brief UDP socket based IPC endpoint
     *
     * Implements IPC communication over UDP sockets.
     */
    struct udp_endpoint: public ipc_endpoint {

        public:
        /**
         * @brief Constructor
         * @param config UDP configuration parameters
         * @param reader Reader callback function
         * @param writer Writer callback function
         */
        udp_endpoint(udp_config* config, reader* reader, writer * writer):
            ipc_endpoint(reader, writer),
            config_(config),
            buffer_(new uint8_t[ipc_config::MAX_MSG_BUF_SIZE]) {
            std::cout<<"Creating udp channel"<<std::endl;
            sockfd_ = create_udp_socket(::ntohl(::inet_addr(config_->ip.c_str())), config_->local_port);
            target_ = new sockaddr_in();
            set_sockaddr_in(target_, config_->ip.c_str(), config_->remote_port);
            std::memset(buffer_.get(), 0 , ipc_config::MAX_DATA_BUF_SIZE);
        }

        /**
         * @brief Destructor - closes socket and releases resources
         */
        ~udp_endpoint() {
            std::cout<<"Closing udp channel"<<std::endl;
            close(sockfd_);
            config_.reset();
            buffer_.release();
        }

        /**
         * @brief Get socket file descriptor
         * @return Socket file descriptor
         */
        int get_fd() { return sockfd_;}
        
        /**
         * @brief Get IPC configuration
         * @return Pointer to configuration
         */
        ipc_config * get_config() { return config_.get(); }
        
        /**
         * @brief Get message buffer
         * @return Pointer to message buffer
         */
        void* get_msg_buf() { return buffer_.get();}
        
        /**
         * @brief Get data buffer (same as message buffer for UDP)
         * @return Pointer to data buffer
         */
        void* get_data_buf() { return get_msg_buf(); }
        
        /**
         * @brief Set socket blocking mode
         * @param blocking true for blocking mode, false for non-blocking
         */
        void set_blocking_fd(bool blocking) {
            nv::set_blocking_fd(sockfd_, blocking);
        }
        
        /**
         * @brief Read data from UDP socket
         *
         * Receives data and invokes the reader callback if data is available.
         */
        void read() {
            sockaddr_in clientAddr;
            socklen_t sock_size = sizeof(clientAddr);
            std::size_t bytes_read = ::recvfrom(sockfd_, buffer_.get(),
               ipc_config::MAX_MSG_BUF_SIZE, 0, reinterpret_cast<sockaddr*>(&clientAddr), &sock_size);
            if (bytes_read > 0) {
                std::cout<<"Bytes read:"<<bytes_read<<std::endl;
                reader* readFn = get_reader();
                if(readFn != nullptr) {
                    (*readFn)(buffer_.get(), bytes_read, this);
                }   
            }
        }
        void write (uint16_t msg_id) {
            std::memset(buffer_.get(), 0 , ipc_config::MAX_DATA_BUF_SIZE);
            writer* writeFn = get_writer();
            if(writeFn != nullptr) {
                std::size_t numwrite = (*writeFn)(buffer_.get(), msg_id, this);
                if (numwrite > 0) {
                    std::cout<<"Bytes Written: "<<numwrite<<std::endl;
                    ::sendto(sockfd_, buffer_.get(),
                        numwrite, 0, reinterpret_cast<sockaddr*>(target_), sizeof(*target_));
                }
            }
        }

        void write(void* buf, std::size_t length) {
            if (buf == nullptr || length < 1 ) {
                std::runtime_error("Cannot write to interface");
                
            }

            // std::ostringstream ss;
            // for(int i = 0; i < length ; i++) {
            //     ss << "0x" << std::hex << static_cast<unsigned>(*(static_cast<uint8_t*>(buf) + i)) << ", ";
            // }
            
            // std::cout<<ss.str()<<std::endl;
             ::sendto(sockfd_, buf,
                        length , 0, reinterpret_cast<sockaddr*>(target_), sizeof(*target_));
        }
        private:

        int sockfd_;
        std::unique_ptr<udp_config> config_;

        std::unique_ptr<uint8_t> buffer_;
        sockaddr_in* target_;
    };

    struct shm_endpoint: public ipc_endpoint {
        public:
        shm_endpoint(shm_config* config):
            ipc_endpoint(nullptr, nullptr),
            config_(config) {
        }
        ipc_config * get_config() { return config_.get(); }

        private:
        std::unique_ptr<shm_config> config_;
    };

    struct io_muxer {
        virtual void add_endpoint(ipc_base* ipc, uint32_t options) {}
        virtual void remove_endpoint(ipc_base* ipc) {}
    };

    struct epoll_muxer : public io_muxer {
        static constexpr std::size_t MAX_FD_EVENTS = 1024;
        epoll_muxer():is_mux_(false),
        epoll_events_(new epoll_event[MAX_FD_EVENTS]) {
            epoll_fd_ = ::epoll_create1(0);
            if (epoll_fd_ == -1) {
                throw std::runtime_error("Cannot create epoll_fd "
                                        + std::string(__PRETTY_FUNCTION__));
            }
        }

        void add_endpoint(ipc_base* ipc, uint32_t options) {
            if (epoll_fd_ == ipc_base::INVALID_FD
            || ipc == nullptr 
            || ipc->get_fd() == ipc_base::INVALID_FD) {
                throw std::runtime_error("Invalid ipc_endpoint"
                                        + std::string(__PRETTY_FUNCTION__));
            }
            int fd = ipc->get_fd();
            ipc->set_blocking_fd(false);
            
            epoll_event event;
            std::memset(&event, 0, sizeof(event));
            event.data.fd = fd;
            event.events = options;

            if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &event) == -1) {
                throw std::runtime_error("Cannot add ipc_endpoint for epoll_ctl:"
                                        + std::to_string(fd));
            }
            ep_map_.insert(std::make_pair(fd, ipc));
        }

        void remove_endpoint(ipc_base* ipc) {
            if (epoll_fd_ == ipc_base::INVALID_FD
            || ipc == nullptr 
            || ipc->get_fd() == ipc_base::INVALID_FD) {
                throw std::runtime_error("Invalid  ipc_endpoint"
                                        + std::string(__PRETTY_FUNCTION__));
            }
            int fd = ipc->get_fd();
            if (ep_map_.count(fd) < 1) {
                throw std::runtime_error("No matching ipc_endpoint "
                                        + std::to_string(fd));
            }

            ep_map_.erase(fd);
            if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, NULL) == -1) {
                throw std::runtime_error("Cannot delete ipc_endpoint for epoll_ctl:"
                                        + std::to_string(fd));
            }
            ipc->set_blocking_fd(true);
        }

      void mux() {
            epoll_event * events = epoll_events_.get();
            int fd_events;
            do {
                // epoll_wait() may return EINTR when get unexpected signal SIGSTOP from system
                fd_events = epoll_wait(epoll_fd_, events, MAX_FD_EVENTS, -1);
            } while (fd_events == -1 && errno == EINTR);
            if (fd_events == -1) {
                throw std::runtime_error("Cannot mux epoll_wait:"
                                        + std::to_string(fd_events));
            }

          for (int i = 0 ; i < fd_events; i++) {
                ipc_base* ep =  ep_map_[events[i].data.fd];
                epoll_event fd_event = events[i];
                if (fd_event.events & EPOLLIN) {
                    ep->read();
                }
          }
      }
      private:
       int epoll_fd_;
       std::map<int, ipc_base*> ep_map_;
       std::atomic_bool is_mux_;
       std::unique_ptr<epoll_event> epoll_events_;
    };
    
} 
#endif
