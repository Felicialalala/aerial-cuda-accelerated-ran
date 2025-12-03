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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <iomanip>
#include <sstream>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include "aerial-fh-driver/oran.hpp"
#include "aerial-fh-driver/api.hpp"

// Structure to hold difference information
struct SectionDifference {
    std::string field_name;
    std::string ref_value;
    std::string test_value;
};

// Structure to hold extension information
struct CPlaneExtensionPktInfo {
    std::vector<uint8_t> data;  // Extension data
};

struct OranEtherAddr {
    struct oran_ether_addr eth_hdr; 

    OranEtherAddr (struct oran_ether_addr &hdr) : eth_hdr (hdr) {}

    bool operator<(const OranEtherAddr &other) const {
        return std::tie(eth_hdr.addr_bytes[0], eth_hdr.addr_bytes[1], eth_hdr.addr_bytes[2], eth_hdr.addr_bytes[3], eth_hdr.addr_bytes[4], eth_hdr.addr_bytes[5]) <
               std::tie(other.eth_hdr.addr_bytes[0], other.eth_hdr.addr_bytes[1], other.eth_hdr.addr_bytes[2], other.eth_hdr.addr_bytes[3], other.eth_hdr.addr_bytes[4], other.eth_hdr.addr_bytes[5]); 

    }
}; 

static std::map<OranEtherAddr, int> ReferenceMacMap; 

// Returns the unique Mac Identifier constructed in the Reference set's MacMap.
// If the queried map is not found, it returns -1. 
static int GetMacIdentifier (OranEtherAddr &ip_mac) 
{
    auto it = ReferenceMacMap.find(ip_mac); 

    if (it == ReferenceMacMap.end()) {
        return -1; 
    } else {
        return it->second;
    }
}

static int macId = 0; 
static int GenerateUniqMacIdentifier (OranEtherAddr &ip_mac) 
{
    int mid = GetMacIdentifier (ip_mac); 
    
    if (mid == -1) {
        // MAC ID not found - insert an unique identifier
        // and return it
        ReferenceMacMap [ip_mac] = macId++; 
        return ReferenceMacMap [ip_mac];  
    } else {
        return mid; 
    }
}

struct CPlaneSectionPktInfo
{
    union
    {
        struct oran_cmsg_sect1_common_hdr sect1_cmn; 
        struct oran_cmsg_sect3_common_hdr sect3_cmn; 
    };

    union
    {
        struct oran_cmsg_sect0 sect_0;
        struct oran_cmsg_sect1 sect_1;
        struct oran_cmsg_sect3 sect_3;
        struct oran_cmsg_sect5 sect_5;
        // TODO implement Section Type 6 & 7
    };

    std::vector<CPlaneExtensionPktInfo> ext4;
    std::vector<CPlaneExtensionPktInfo> ext5;
    std::vector<CPlaneExtensionPktInfo> ext11;

    void dumpSect1() const {
        printf (
            "sect1_cmn.radioAppHdr.dataDirection    : 0x%x \n" 
            "sect1_cmn.radioAppHdr.payloadVersion   : 0x%x \n" 
            "sect1_cmn.radioAppHdr.filterIndex      : 0x%x \n" 
            "sect1_cmn.radioAppHdr.frameId          : 0x%x \n" 
            "sect1_cmn.radioAppHdr.subframeId       : 0x%x \n" 
            "sect1_cmn.radioAppHdr.slotId           : 0x%x \n" 
            "sect1_cmn.radioAppHdr.startSymbolId    : 0x%x \n" 
            "sect1_cmn.radioAppHdr.numberOfSections : 0x%x \n" 
            "sect1_cmn.radioAppHdr.sectionType      : 0x%x \n" 
            "sect1_cmn.udCompHdr                    : 0x%x \n" 
            "sect1.sectionId                        : 0x%x \n" 
            "sect1.rb                               : 0x%x \n" 
            "sect1.symInc                           : 0x%x \n" 
            "sect1.startPrbc                        : 0x%x \n" 
            "sect1.numPrbc                          : 0x%x \n" 
            "sect1.reMask                           : 0x%x \n" 
            "sect1.numSymbol                        : 0x%x \n" 
            "sect1.ef                               : 0x%x \n" 
            "sect1.beamId                           : 0x%x \n"
            "----------------------------------------------\n",
            sect1_cmn.radioAppHdr.dataDirection.get(),    
            sect1_cmn.radioAppHdr.payloadVersion.get(),  
            sect1_cmn.radioAppHdr.filterIndex.get(),      
            sect1_cmn.radioAppHdr.frameId,          
            sect1_cmn.radioAppHdr.subframeId.get(),       
            sect1_cmn.radioAppHdr.slotId.get(),           
            sect1_cmn.radioAppHdr.startSymbolId.get(),   
            sect1_cmn.radioAppHdr.numberOfSections, 
            sect1_cmn.radioAppHdr.sectionType,      
            sect1_cmn.udCompHdr,                    
            sect_1.sectionId.get(),                       
            sect_1.rb.get(),                               
            sect_1.symInc.get(),                           
            sect_1.startPrbc.get(),                        
            sect_1.numPrbc.get(),                          
            sect_1.reMask.get(),                           
            sect_1.numSymbol.get(),                        
            sect_1.ef.get(),                               
            sect_1.beamId.get()
            );
    }

    void dumpExt () const {
        for (const CPlaneExtensionPktInfo & info : ext4) {
            printf ("ext4.data:");
            int index = 0; 
            for (const uint8_t &d: info.data) {
                printf ("[%03d]:0x%02x ", index++, d);  
            }
            printf ("----------------------------------------------\n");
        }
        for (const CPlaneExtensionPktInfo & info : ext5) {
            printf ("ext5.data:");
            int index = 0; 
            for (const uint8_t &d: info.data) {
                printf ("[%03d]:0x%02x ", index++, d);  
            }
            printf ("----------------------------------------------\n");
        }
        for (const CPlaneExtensionPktInfo & info : ext11) {
            printf ("ext11.data:");
            int index = 0; 
            for (const uint8_t &d: info.data) {
                printf ("[%03d]:0x%02x ", index++, d);  
            }
            printf ("----------------------------------------------\n");
        }
    }
};

// Key structure to uniquely identify a C-plane message
struct CPlaneMessageKey {
    uint8_t frameId;
    uint8_t subframeId;
    uint8_t slotId;
    uint8_t startSym;
    uint16_t eaxcId;
    uint8_t section_type;
    uint8_t sectionId;
    uint8_t macId; 

    bool operator<(const CPlaneMessageKey& other) const {
        return std::tie(frameId, subframeId, slotId, startSym, eaxcId, section_type, sectionId, macId) <
               std::tie(other.frameId, other.subframeId, other.slotId, other.startSym, other.eaxcId, other.section_type, other.sectionId, other.macId);
    }

    void print_key() const {
        std::cout << "Frame: " << static_cast<int>(frameId)
                  << ", Subframe: " << static_cast<int>(subframeId)
                  << ", Slot: " << static_cast<int>(slotId)
                  << ", Symbol: " << static_cast<int>(startSym)
                  << ", eAxC ID: " << eaxcId
                  << ", Section Type: " << static_cast<int>(section_type)
                  << ", Section ID: " << static_cast<int>(sectionId)
                  << ", Mac ID: " << static_cast<int>(macId)
                  << std::endl;
    }
};

// PCAP file format structures
struct PcapFileHeader {
    uint32_t magic_number;   // magic number
    uint16_t version_major;  // major version number
    uint16_t version_minor;  // minor version number
    int32_t  thiszone;      // GMT to local correction
    uint32_t sigfigs;       // accuracy of timestamps
    uint32_t snaplen;       // max length of captured packets
    uint32_t network;       // data link type
};

struct PcapPacketHeader {
    uint32_t ts_sec;        // timestamp seconds
    uint32_t ts_usec;       // timestamp microseconds
    uint32_t incl_len;      // number of octets of packet saved in file
    uint32_t orig_len;      // actual length of packet
};

static CPlaneMessageKey create_key(const struct oran_cmsg_radio_app_hdr &info, uint16_t section_id, int mac_idx, int flow_id) {
        CPlaneMessageKey key = {
            info.frameId,
            (uint8_t) info.subframeId.get(),
            (uint8_t) info.slotId.get(),
            (uint8_t) info.startSymbolId.get(),
            (uint8_t) flow_id,
            (uint8_t) info.sectionType,
            (uint8_t) section_id,
            (uint8_t) mac_idx 
        }; 

        return key; 
}

class OranPcapParser {
private:
    std::ifstream file;
    std::string filename;
    bool is_little_endian;
    
    // Helper function to check endianness of PCAP file
    bool check_pcap_endianness(uint32_t magic) {
        if (magic == 0xa1b2c3d4) return true;        // Little endian
        else if (magic == 0xd4c3b2a1) return false;  // Big endian
        throw std::runtime_error("Invalid PCAP magic number");
    }

    // Helper function to swap endianness if needed
    template<typename T>
    T fix_endianness(T value) {
        if (!is_little_endian) {
            T result = 0;
            for (size_t i = 0; i < sizeof(T); ++i) {
                result = (result << 8) | ((value >> (i * 8)) & 0xFF);
            }
            return result;
        }
        return value;
    }

#define GET_EXT_LEN(len) ((len&0xFF)<<8 | ((len >> 8)&0xFF))
    // Helper function to parse extensions
    bool parse_extensions(const uint8_t* data, CPlaneSectionPktInfo &sectionInfo, uint32_t &offset) {

        bool ef = 0; 
        do {
            const oran_cmsg_ext_hdr* ext_hdr = reinterpret_cast<const oran_cmsg_ext_hdr*>(data);
            // ef denotes if this is the last section or if there's more.
            ef = ext_hdr->ef.get(); 
            int ext_len_bytes = 0; 
            if (ext_hdr->extType.get() == ORAN_CMSG_SECTION_EXT_TYPE_4) {
                const struct  oran_cmsg_sect_ext_type_4* ext4 = 
                    reinterpret_cast<const struct  oran_cmsg_sect_ext_type_4*>(data+ sizeof (oran_cmsg_ext_hdr)); 
                ext_len_bytes = GET_EXT_LEN(ext4->extLen) * 4; 
                CPlaneExtensionPktInfo ext4v;
                ext4v.data.assign(data, data + ext_len_bytes); 
                sectionInfo.ext4.push_back (std::move(ext4v)); 
            }
            
            if (ext_hdr->extType.get() == ORAN_CMSG_SECTION_EXT_TYPE_5) {
                const struct  oran_cmsg_sect_ext_type_5* ext5 = 
                    reinterpret_cast<const struct  oran_cmsg_sect_ext_type_5*>(data+ sizeof (oran_cmsg_ext_hdr)); 
                ext_len_bytes = GET_EXT_LEN(ext5->extLen) * 4; 
                CPlaneExtensionPktInfo ext5v;
                ext5v.data.assign(data, data + ext_len_bytes); 
                sectionInfo.ext5.push_back (std::move(ext5v)); 
            }
 
            if (ext_hdr->extType.get() == ORAN_CMSG_SECTION_EXT_TYPE_11) {
                const struct  oran_cmsg_sect_ext_type_11* ext11 = 
                    reinterpret_cast<const struct  oran_cmsg_sect_ext_type_11*>(data+ sizeof (oran_cmsg_ext_hdr)); 
                ext_len_bytes = GET_EXT_LEN(ext11->extLen) * 4; 
                CPlaneExtensionPktInfo ext11v;
                ext11v.data.assign(data, data + ext_len_bytes); 
                sectionInfo.ext11.push_back (std::move(ext11v)); 
            }

            // Advance to inspect the next set of bytes
            data = data + ext_len_bytes; 
            offset += ext_len_bytes; 
        } while (ef); 

        return true;
    }

    // Helper function to parse ORAN header and payload
    bool parse_oran_packet(const std::vector<uint8_t>& packet_data, std::map<CPlaneMessageKey, std::vector<CPlaneSectionPktInfo>> &messages) {

        if (packet_data.size() < sizeof(struct oran_eth_hdr)) { return false;}

        // Skip Ethernet header
        struct oran_eth_hdr eth_hdr = {0}; 
        size_t offset = sizeof(struct oran_eth_hdr);
        memcpy (&eth_hdr, packet_data.data(), sizeof(struct oran_eth_hdr)); 

        uint16_t flow_id = oran_msg_get_flowid((uint8_t *) packet_data.data()); 
       
        OranEtherAddr eth (eth_hdr.eth_hdr.dst_addr);
        int mac_id = GenerateUniqMacIdentifier (eth); 

        // Parse ORAN ECPRI header
        if (packet_data.size() < offset + sizeof(struct oran_ecpri_hdr)) {return false;}
        const struct oran_ecpri_hdr* ecpri_header = reinterpret_cast<const struct oran_ecpri_hdr*>(packet_data.data() + offset);
        offset += sizeof(struct oran_ecpri_hdr);

        const uint8_t* oran_payload = packet_data.data() + offset;
        
        // Parse ORAN payload radio app hdr structure
        struct oran_cmsg_radio_app_hdr hdr; 
        memcpy(&hdr, oran_payload, sizeof(hdr));

        for (int i = 0; i < hdr.numberOfSections; ++i) {

            CPlaneSectionPktInfo sectionInfo;

            if (hdr.sectionType == ORAN_CMSG_SECTION_TYPE_1) {
                if (i == 0) {
                    memcpy (&sectionInfo.sect1_cmn, oran_payload, sizeof (sectionInfo.sect1_cmn));
                    oran_payload += sizeof (struct oran_cmsg_sect1_common_hdr); 
                }

                memcpy (&sectionInfo.sect_1, oran_payload, sizeof (struct oran_cmsg_sect1)); 
                oran_payload += sizeof (struct oran_cmsg_sect1); 

                if (sectionInfo.sect_1.ef.get()) {
                    // Section indicates presence of one/more extensions
                    uint32_t offset = 0; 
                    parse_extensions(oran_payload, sectionInfo, offset); 
                    oran_payload += offset;
                }
                              
                // Create a unique key for the map with the radio app header and the section ID
                CPlaneMessageKey key = create_key(hdr, sectionInfo.sect_1.sectionId.get(),mac_id, flow_id); 
                messages [key].push_back(sectionInfo); 
            }
        }

        return true;
    }

public:
    OranPcapParser() : is_little_endian(true) {}
    
    ~OranPcapParser() {
        close();
    }
    
    bool open(const std::string& fname) {
        filename = fname;
        file.open(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            std::cerr << "Error code: " << strerror(errno) << std::endl;
            return false;
        }

        // Read and validate PCAP file header
        PcapFileHeader header;
        if (!file.read(reinterpret_cast<char*>(&header), sizeof(header))) {
            std::cerr << "Error reading PCAP header" << std::endl;
            close(); 
            return false;
        }

        try {
            is_little_endian = check_pcap_endianness(header.magic_number);
        } catch (const std::runtime_error& e) {
            std::cerr << e.what() << std::endl;
            return false;
        }

        return true;
    }
    
    bool parse_next_packet(std::map<CPlaneMessageKey, std::vector<CPlaneSectionPktInfo>> &messages, size_t &count) {
        if (!file.is_open()) {
            return false;
        }
        
        // Read packet header
        PcapPacketHeader pkt_header;
        if (!file.read(reinterpret_cast<char*>(&pkt_header), sizeof(pkt_header))) {
            return false;  // End of file or error
        }

        // Fix endianness
        uint32_t packet_size = fix_endianness(pkt_header.incl_len);

        // Read packet data
        std::vector<uint8_t> packet_data(packet_size);
        if (!file.read(reinterpret_cast<char*>(packet_data.data()), packet_size)) {
            return false;
        }

        // Seems like a valid packet, increment counter to account for a new valid packet
        ++count; 
        return parse_oran_packet(packet_data, messages);
    }
    
    void close() {
        if (file.is_open()) {
            file.close();
        }
    }
};

class OranPcapComparator {
private:
    // Maps to store messages and their extensions from reference and test files
    std::map<CPlaneMessageKey, std::vector<CPlaneSectionPktInfo>> ref_messages;
    std::map<CPlaneMessageKey, std::vector<CPlaneSectionPktInfo>> test_messages;

    std::map<CPlaneMessageKey, std::vector<SectionDifference>> differences; 

    
    // Statistics
    size_t total_messages_ref = 0;
    size_t total_messages_test = 0;
    size_t matching_messages = 0;
    size_t mismatched_messages = 0;
    size_t missing_messages = 0;
    size_t ref_packet_count = 0, test_packet_count = 0; 
    
    // Helper function to create message key
    // CPlaneMessageKey create_key(const oran_c_plane_info_t& info, int section_idx = 0) {
   
    // Compare individual section fields and collect differences
    std::vector<SectionDifference> compare_sections(
        const std::vector<CPlaneSectionPktInfo>& ref_sections,
        const std::vector<CPlaneSectionPktInfo>& test_sections) {

        std::vector<SectionDifference> differences;

        if (ref_sections.size() != test_sections.size()) {
            differences.push_back ({
                "Mismatched Len of Sections", 
                std::to_string(ref_sections.size()),
                std::to_string(test_sections.size())
            }); 
        }

        if (ref_sections.size() == test_sections.size()) {
            // TODO Enhance this to account for random ordered sections within a key
            uint32_t matching_sections = 0; 
            for (int i = 0; i < ref_sections.size(); ++i) {
                is_section_equal(ref_sections[i], test_sections[i], differences); 
            }
        }

        return differences; 
    }


public:
    // Load messages from a pcap file
    bool load_pcap(const std::string& filename, bool is_reference) {
        OranPcapParser parser;
        if (!parser.open(filename)) {
            std::cerr << "Failed to open " << filename << std::endl;
            return false;
        }

        std::cout << "Loading " << (is_reference ? "reference file:" : "test file:") << filename << "\n"; 

        auto& messages = is_reference ? ref_messages : test_messages;
        size_t count = 0; 
        while (parser.parse_next_packet(messages, count)); 

        auto& pkt_count = is_reference ? ref_packet_count : test_packet_count; 
        pkt_count += count; 

        parser.close();
        if (is_reference) {
            total_messages_ref = ref_messages.size();
        } else {
            total_messages_test = test_messages.size();
        }
        return true;
    }

#define COMPARE(REF,TEST,FIELD,DIFF) \
                                   do {\
                                        if (REF.FIELD!=TEST.FIELD) {\
                                            DIFF.push_back({\
                                                #FIELD,\
                                                std::to_string(REF.FIELD),\
                                                std::to_string(TEST.FIELD),\
                                            });\
                                        }\
                                   } while(0) 

#define COMPARE_EXT(REF,TEST,FIELD,TYPE,DIFF) \
                                   do {\
                                        if (REF!=TEST) {\
                                            DIFF.push_back({\
                                                "Ext"+std::to_string(TYPE)+"["+std::to_string(FIELD)+"]",\
                                                std::to_string(REF),\
                                                std::to_string(TEST),\
                                            });\
                                        }\
                                   } while(0) 

    void is_extension_equal (const CPlaneExtensionPktInfo& ref, const CPlaneExtensionPktInfo &test, uint8_t ext_type, std::vector<SectionDifference> &differences) {
        for (int i = 0; i < ref.data.size(); ++i) {
             COMPARE_EXT (ref.data[i], test.data[i], i, ext_type, differences); 
        }
    }

    void is_section_equal(const CPlaneSectionPktInfo& ref, const CPlaneSectionPktInfo &test, std::vector<SectionDifference> &differences) {
        // Section common header compare
        COMPARE (ref, test, sect1_cmn.radioAppHdr.dataDirection.get(), differences); 
        COMPARE (ref, test, sect1_cmn.radioAppHdr.payloadVersion.get(), differences); 
        COMPARE (ref, test, sect1_cmn.radioAppHdr.filterIndex.get(), differences); 
        COMPARE (ref, test, sect1_cmn.radioAppHdr.frameId, differences); 
        COMPARE (ref, test, sect1_cmn.radioAppHdr.subframeId.get(), differences); 
        COMPARE (ref, test, sect1_cmn.radioAppHdr.slotId.get(), differences); 
        COMPARE (ref, test, sect1_cmn.radioAppHdr.startSymbolId.get(), differences); 
        COMPARE (ref, test, sect1_cmn.radioAppHdr.numberOfSections, differences); 
        COMPARE (ref, test, sect1_cmn.radioAppHdr.sectionType, differences); 
        COMPARE (ref, test, sect1_cmn.udCompHdr, differences); 
        
        // Section-1 body compare.        
        COMPARE (ref, test, sect_1.sectionId.get(), differences); 
        COMPARE (ref, test, sect_1.rb.get(), differences); 
        COMPARE (ref, test, sect_1.symInc.get(), differences); 
        COMPARE (ref, test, sect_1.startPrbc.get(), differences); 
        COMPARE (ref, test, sect_1.numPrbc.get(), differences); 
        COMPARE (ref, test, sect_1.reMask.get(), differences); 
        COMPARE (ref, test, sect_1.numSymbol.get(), differences); 
        COMPARE (ref, test, sect_1.ef.get(), differences); 
        COMPARE (ref, test, sect_1.beamId.get(), differences); 
        
        // TODO - Enhance to interpret the mismatch and provide more useful diagnostics
        // than simple byte mismatch locations.
        if (ref.ext4.size () != test.ext4.size()) {
            differences.push_back ( {
                "# of Ext4 length Mismatch",
                std::to_string(ref.ext4.size()),
                std::to_string(test.ext4.size())}); 
        }

        if (ref.ext4.size() < ref.ext4.size()) {
            std::cout << "Extra extension(4)s in ref file!\n"; 
        } else if (ref.ext4.size() > ref.ext4.size()) {
            std::cout << "Extra extension(4)s in test file!\n"; 
        } else {
            for (int i = 0; i < ref.ext4.size(); ++i) {
                is_extension_equal(ref.ext4[i], test.ext4[i], 4, differences); 
            }
        }

        if (ref.ext5.size() < ref.ext5.size()) {
            std::cout << "Extra extension(5)s in ref file!\n"; 
        } else if (ref.ext5.size() > ref.ext5.size()) {
            std::cout << "Extra extension(5)s in test file!\n"; 
        } else {
            for (int i = 0; i < ref.ext5.size(); ++i) {
                is_extension_equal(ref.ext5[i], test.ext5[i], 5, differences); 
            }
        }

        if (ref.ext11.size() < ref.ext11.size()) {
            std::cout << "Extra extension(11)s in ref file!\n"; 
        } else if (ref.ext4.size() > ref.ext5.size()) {
            std::cout << "Extra extension(11)s in test file!\n"; 
        } else {
            for (int i = 0; i < ref.ext11.size(); ++i) {
                is_extension_equal(ref.ext11[i], test.ext11[i], 11, differences); 
            }
        }
    }

    // Compare loaded messages and report differences
    void compare() {

        for (const auto& [key, val] : ref_messages) {
            auto test_it = test_messages.find(key); 

            if (test_it == test_messages.end()) {
                continue;
            }

            bool message_matches = true; 

            const auto &ref = val; 
            const auto &test = test_it->second;

            auto section_differences = compare_sections(ref, test); 

            if (!section_differences.empty()) {
                std::cout << "Differences found in message section:" << std::endl;
                key.print_key();
                for (const auto& diff : section_differences) {
                    std::cout << "  Field: " << diff.field_name
                        << ", Reference: " << diff.ref_value
                        << ", Test: " << diff.test_value << std::endl;
                }
                message_matches = false;
            }

            if (message_matches) {
                matching_messages++;
            } else {
                mismatched_messages++;
            }
        }

        if (total_messages_ref < total_messages_test) {
            for (const auto& [key, _] : test_messages) {
                if (ref_messages.find(key) == ref_messages.end()) {
                    std::cout << "Extra message in Test file\n";
                    key.print_key();
                    missing_messages++;
                }
            }
        } else if (total_messages_ref > total_messages_test) {
            for (const auto& [key, _] : ref_messages) {
                if (test_messages.find(key) == test_messages.end()) {
                    std::cout << "Extra message in Ref file\n";
                    key.print_key();
                    missing_messages++;
                }
            }
        }
    }

    // Print statistics about the comparison
    void print_stats() {
        std::cout << "\nComparison Statistics:" << std::endl;
        std::cout << "Total packets in reference: " << ref_packet_count << std::endl;
        std::cout << "Total packets in test: " << test_packet_count << std::endl;
        std::cout << "Total messages in reference: " << total_messages_ref << std::endl;
        std::cout << "Total messages in test: " << total_messages_test << std::endl;
        std::cout << "Matching messages: " << matching_messages << std::endl;
        std::cout << "Mismatched messages: " << mismatched_messages << std::endl;
        std::cout << "Missing messages: " << missing_messages << std::endl;
        std::cout << "Mac Map size: " << ReferenceMacMap.size() << std::endl; 
    }
};

