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

#ifndef AERIAL_FH_OUTPUT_FORMATTING__
#define AERIAL_FH_OUTPUT_FORMATTING__

#include "aerial-fh-driver/api.hpp"
#include "dpdk.hpp"

#include <iomanip>

/**
 * Stream output operator for DPDK Ethernet address
 *
 * Formats MAC address as XX:XX:XX:XX:XX:XX in hexadecimal.
 * Restores previous stream flags after formatting.
 *
 * @param os Output stream
 * @param eth_addr Ethernet address to format
 * @return Reference to output stream
 */
inline std::ostream& operator<<(std::ostream& os, rte_ether_addr const& eth_addr)
{
    auto f{os.flags()};
    os << std::setfill('0') << std::setw(2) << std::hex << (unsigned int)eth_addr.addr_bytes[0];
    for(size_t i = 1; i < 6; ++i)
    {
        os << ":" << std::setfill('0') << std::setw(2) << std::hex << (unsigned int)eth_addr.addr_bytes[i];
    }
    os.flags(f);
    return os;
}

/**
 * Stream output operator for aerial_fh::MacAddr
 *
 * Formats MAC address as XX:XX:XX:XX:XX:XX in hexadecimal.
 * Restores previous stream flags after formatting.
 *
 * @param os Output stream
 * @param eth_addr MAC address to format
 * @return Reference to output stream
 */
inline std::ostream& operator<<(std::ostream& os, aerial_fh::MacAddr const& eth_addr)
{
    auto f{os.flags()};
    os << std::setfill('0') << std::setw(2) << std::hex << (unsigned int)eth_addr.bytes[0];
    for(size_t i = 1; i < 6; ++i)
    {
        os << ":" << std::setfill('0') << std::setw(2) << std::hex << (unsigned int)eth_addr.bytes[i];
    }
    os.flags(f);
    return os;
}

/**
 * Hexadecimal formatter for stream output
 *
 * Pretty prints integers in hexadecimal format with "0x" prefix and zero padding.
 * Restores previous stream settings after formatting.
 *
 * Usage example:
 * \code
 *     std::cout << "Value: " << 42 << ", in hex: " << Hex(42, 8) << "\n";
 *     // Output: Value: 42, in hex: 0x0000002a
 * \endcode
 *
 * \note Not suitable for high-performance printing in tight loops
 */
class Hex {
public:
    /**
     * Constructor
     * @param v Value to format
     * @param w Field width (number of hex digits)
     */
    template <class T>
    Hex(T const& v, size_t w) :
        value{v},
        width{w}
    {}

    /**
     * Stream output operator
     * @param os Output stream
     * @param h Hex formatter object
     * @return Reference to output stream
     */
    friend std::ostream& operator<<(std::ostream& os, Hex const& h)
    {
        std::ios old{nullptr};
        old.copyfmt(os);
        os << "0x" << std::setw(h.width) << std::setfill('0') << std::hex << h.value;
        os.copyfmt(old);
        return os;
    }

protected:
    int64_t value;  //!< Value to format
    size_t  width;  //!< Field width
};

#endif //ifndef AERIAL_FH_OUTPUT_FORMATTING__
