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

#ifndef TESTS_SCF_5G_SLOT_COMMANDS_UTILS_HPP__
#define TESTS_SCF_5G_SLOT_COMMANDS_UTILS_HPP__
#include <cstdint>
#include "hdf5hpp.hpp"
inline constexpr const char* NUMPDU = "nPdu";
inline constexpr const char* PDU = "PDU";
inline constexpr const char* NZP_REMAP = "Xtf_remap_trsnzp";
inline constexpr const char* ZP_REMAP = "Xtf_remap";

struct tvCsirsPdu
{
    uint32_t type{};
    uint32_t BWPSize{};
    uint32_t BWPStart{};
    uint32_t SubcarrierSpacing{};
    uint32_t CyclicPrefix{};
    uint32_t StartRB{};
    uint32_t NrOfRBs{};
    uint32_t CSIType{};
    uint32_t Row{};
    uint32_t FreqDomain{};
    uint32_t SymbL0{};
    uint32_t SymbL1{};
    uint32_t CDMType{};
    uint32_t FreqDensity{};
    uint32_t ScrambId{};
    uint32_t enablePrcdBf{};
    uint32_t idxUE{};
    uint32_t numPRGs{};
    uint32_t prgSize{};
    uint32_t digBFInterfaces{};
    uint32_t PMidx{};
    uint32_t beamIdx{};
    uint32_t powerControlOffset{};
    uint32_t powerControlOffsetSS{};
    uint32_t csirsPduIdx{};
    uint32_t lastCsirsPdu{};

    void printTvCsirsPdu() const {
        printf("CSIRS PDU Contents:\n");
        printf("type: %u\n", type);
        printf("BWPSize: %u\n", BWPSize);
        printf("BWPStart: %u\n", BWPStart);
        printf("SubcarrierSpacing: %u\n", SubcarrierSpacing);
        printf("CyclicPrefix: %u\n", CyclicPrefix);
        printf("StartRB: %u\n", StartRB);
        printf("NrOfRBs: %u\n", NrOfRBs);
        printf("CSIType: %u\n", CSIType);
        printf("Row: %u\n", Row);
        printf("FreqDomain: %u\n", FreqDomain);
        printf("SymbL0: %u\n", SymbL0);
        printf("SymbL1: %u\n", SymbL1);
        printf("CDMType: %u\n", CDMType);
        printf("FreqDensity: %u\n", FreqDensity);
        printf("ScrambId: %u\n", ScrambId);
        printf("enablePrcdBf: %u\n", enablePrcdBf);
        printf("idxUE: %u\n", idxUE);
        printf("numPRGs: %u\n", numPRGs);
        printf("prgSize: %u\n", prgSize);
        printf("digBFInterfaces: %u\n", digBFInterfaces);
        printf("PMidx: %u\n", PMidx);
        printf("beamIdx: %u\n", beamIdx);
        printf("powerControlOffset: %u\n", powerControlOffset);
        printf("powerControlOffsetSS: %u\n", powerControlOffsetSS);
        printf("csirsPduIdx: %u\n", csirsPduIdx);
        printf("lastCsirsPdu: %u\n", lastCsirsPdu);
    }

    void readCsirsPduFromH5(const hdf5hpp::hdf5_dataset_elem& pdu_ds) {
        type = pdu_ds["type"].as<uint32_t>();
        BWPSize = pdu_ds["BWPSize"].as<uint32_t>();
        BWPStart = pdu_ds["BWPStart"].as<uint32_t>();
        SubcarrierSpacing = pdu_ds["SubcarrierSpacing"].as<uint32_t>();
        CyclicPrefix = pdu_ds["CyclicPrefix"].as<uint32_t>();
        StartRB = pdu_ds["StartRB"].as<uint32_t>();
        NrOfRBs = pdu_ds["NrOfRBs"].as<uint32_t>();
        CSIType = pdu_ds["CSIType"].as<uint32_t>();
        Row = pdu_ds["Row"].as<uint32_t>();
        FreqDomain = pdu_ds["FreqDomain"].as<uint32_t>();
        SymbL0 = pdu_ds["SymbL0"].as<uint32_t>();
        SymbL1 = pdu_ds["SymbL1"].as<uint32_t>();
        CDMType = pdu_ds["CDMType"].as<uint32_t>();
        FreqDensity = pdu_ds["FreqDensity"].as<uint32_t>();
        ScrambId = pdu_ds["ScrambId"].as<uint32_t>();
        enablePrcdBf = pdu_ds["enablePrcdBf"].as<uint32_t>();
        idxUE = pdu_ds["idxUE"].as<uint32_t>();
        numPRGs = pdu_ds["numPRGs"].as<uint32_t>();
        prgSize = pdu_ds["prgSize"].as<uint32_t>();
        digBFInterfaces = pdu_ds["digBFInterfaces"].as<uint32_t>();
        PMidx = pdu_ds["PMidx"].as<uint32_t>();
        // beamIdx = pdu_ds["beamIdx"].as<uint32_t>();
        powerControlOffset = pdu_ds["powerControlOffset"].as<uint32_t>();
        powerControlOffsetSS = pdu_ds["powerControlOffsetSS"].as<uint32_t>();
        csirsPduIdx = pdu_ds["csirsPduIdx"].as<uint32_t>();
        lastCsirsPdu = pdu_ds["lastCsirsPdu"].as<uint32_t>();
    }
};

struct tvPdschPdu
{
    uint32_t type{};
    uint32_t pduBitmap{};
    uint32_t RNTI{};
    uint32_t pduIndex{};
    uint32_t BWPSize{};
    uint32_t BWPStart{};
    uint32_t SubcarrierSpacing{};
    uint32_t CyclicPrefix{};
    uint32_t NrOfCodewords{};
    uint32_t targetCodeRate{};
    uint32_t qamModOrder{};
    uint32_t mcsIndex{};
    uint32_t mcsTable{};
    uint32_t rvIndex{};
    uint32_t TBSize{};
    uint32_t dataScramblingId{};
    uint32_t nrOfLayers{};
    uint32_t transmissionScheme{};
    uint32_t refPoint{};
    uint32_t DmrsSymbPos{};
    uint32_t DmrsMappingType{};
    uint32_t dmrsConfigType{};
    uint32_t DmrsScramblingId{};
    uint32_t SCID{};
    uint32_t numDmrsCdmGrpsNoData{};
    uint32_t dmrsPorts{};
    uint32_t resourceAlloc{};
    uint32_t rbBitmap[36]{};
    uint32_t rbStart{};
    uint32_t rbSize{};
    uint32_t VRBtoPRBMapping{};
    uint32_t StartSymbolIndex{};
    uint32_t NrOfSymbols{};
    uint32_t enablePrcdBf{};
    uint32_t numPRGs{};
    uint32_t prgSize{};
    uint32_t digBFInterfaces{};
    uint32_t PMidx{};
    uint32_t beamIdx[4]{};
    uint32_t powerControlOffset{};
    uint32_t powerControlOffsetSS{};
    uint32_t IsLastCbPresent{};
    uint32_t isInlineTbCrc{};
    uint32_t dlTbCrc{};
    uint32_t I_LBRM{};
    uint32_t maxLayers{};
    uint32_t maxQm{};
    uint32_t n_PRB_LBRM{};
    uint32_t testModel{};
    uint32_t idxUE{};
    uint32_t idxUeg{};
    uint32_t pdschPduIdx{};
    uint32_t lastPdschPdu{};
    bool isNewGrpStart{};

    void printTvPdschPdu() const {
        printf("PDSCH PDU Contents:\n");
        printf("type: %u\n", type);
        printf("pduBitmap: %u\n", pduBitmap);
        printf("RNTI: %u\n", RNTI);
        printf("pduIndex: %u\n", pduIndex);
        printf("BWPSize: %u\n", BWPSize);
        printf("BWPStart: %u\n", BWPStart);
        printf("SubcarrierSpacing: %u\n", SubcarrierSpacing);
        printf("CyclicPrefix: %u\n", CyclicPrefix);
        printf("NrOfCodewords: %u\n", NrOfCodewords);
        printf("targetCodeRate: %u\n", targetCodeRate);
        printf("qamModOrder: %u\n", qamModOrder);
        printf("mcsIndex: %u\n", mcsIndex);
        printf("mcsTable: %u\n", mcsTable);
        printf("rvIndex: %u\n", rvIndex);
        printf("TBSize: %u\n", TBSize);
        printf("dataScramblingId: %u\n", dataScramblingId);
        printf("nrOfLayers: %u\n", nrOfLayers);
        printf("transmissionScheme: %u\n", transmissionScheme);
        printf("refPoint: %u\n", refPoint);
        printf("DmrsSymbPos: %u\n", DmrsSymbPos);
        printf("DmrsMappingType: %u\n", DmrsMappingType);
        printf("dmrsConfigType: %u\n", dmrsConfigType);
        printf("DmrsScramblingId: %u\n", DmrsScramblingId);
        printf("SCID: %u\n", SCID);
        printf("numDmrsCdmGrpsNoData: %u\n", numDmrsCdmGrpsNoData);
        printf("dmrsPorts: %u\n", dmrsPorts);
        printf("resourceAlloc: %u\n", resourceAlloc);
        
        printf("rbBitmap: ");
        for(int i = 0; i < 36; i++) {
            printf("%u ", rbBitmap[i]);
        }
        printf("\n");
        
        printf("rbStart: %u\n", rbStart);
        printf("rbSize: %u\n", rbSize);
        printf("VRBtoPRBMapping: %u\n", VRBtoPRBMapping);
        printf("StartSymbolIndex: %u\n", StartSymbolIndex);
        printf("NrOfSymbols: %u\n", NrOfSymbols);
        printf("enablePrcdBf: %u\n", enablePrcdBf);
        printf("numPRGs: %u\n", numPRGs);
        printf("prgSize: %u\n", prgSize);
        printf("digBFInterfaces: %u\n", digBFInterfaces);
        printf("PMidx: %u\n", PMidx);
        
        printf("beamIdx: ");
        for(int i = 0; i < 4; i++) {
            printf("%u ", beamIdx[i]);
        }
        printf("\n");
        
        printf("powerControlOffset: %u\n", powerControlOffset);
        printf("powerControlOffsetSS: %u\n", powerControlOffsetSS);
        printf("IsLastCbPresent: %u\n", IsLastCbPresent);
        printf("isInlineTbCrc: %u\n", isInlineTbCrc);
        printf("dlTbCrc: %u\n", dlTbCrc);
        printf("I_LBRM: %u\n", I_LBRM);
        printf("maxLayers: %u\n", maxLayers);
        printf("maxQm: %u\n", maxQm);
        printf("n_PRB_LBRM: %u\n", n_PRB_LBRM);
        printf("testModel: %u\n", testModel);
        printf("idxUE: %u\n", idxUE);
        printf("idxUeg: %u\n", idxUeg);
        printf("pdschPduIdx: %u\n", pdschPduIdx);
        printf("lastPdschPdu: %u\n", lastPdschPdu);
    }

    void readPdschPduFromH5(const hdf5hpp::hdf5_dataset_elem& pdu_ds) {
        type = pdu_ds["type"].as<uint32_t>();
        pduBitmap = pdu_ds["pduBitmap"].as<uint32_t>();
        RNTI = pdu_ds["RNTI"].as<uint32_t>();
        pduIndex = pdu_ds["pduIndex"].as<uint32_t>();
        BWPSize = pdu_ds["BWPSize"].as<uint32_t>();
        BWPStart = pdu_ds["BWPStart"].as<uint32_t>();
        SubcarrierSpacing = pdu_ds["SubcarrierSpacing"].as<uint32_t>();
        CyclicPrefix = pdu_ds["CyclicPrefix"].as<uint32_t>();
        NrOfCodewords = pdu_ds["NrOfCodewords"].as<uint32_t>();
        targetCodeRate = pdu_ds["targetCodeRate"].as<uint32_t>();
        qamModOrder = pdu_ds["qamModOrder"].as<uint32_t>();
        mcsIndex = pdu_ds["mcsIndex"].as<uint32_t>();
        mcsTable = pdu_ds["mcsTable"].as<uint32_t>();
        rvIndex = pdu_ds["rvIndex"].as<uint32_t>();
        TBSize = pdu_ds["TBSize"].as<uint32_t>();
        dataScramblingId = pdu_ds["dataScramblingId"].as<uint32_t>();
        nrOfLayers = pdu_ds["nrOfLayers"].as<uint32_t>();
        transmissionScheme = pdu_ds["transmissionScheme"].as<uint32_t>();
        refPoint = pdu_ds["refPoint"].as<uint32_t>();
        DmrsSymbPos = pdu_ds["DmrsSymbPos"].as<uint32_t>();
        DmrsMappingType = pdu_ds["DmrsMappingType"].as<uint32_t>();
        dmrsConfigType = pdu_ds["dmrsConfigType"].as<uint32_t>();
        DmrsScramblingId = pdu_ds["DmrsScramblingId"].as<uint32_t>();
        SCID = pdu_ds["SCID"].as<uint32_t>();
        numDmrsCdmGrpsNoData = pdu_ds["numDmrsCdmGrpsNoData"].as<uint32_t>();
        dmrsPorts = pdu_ds["dmrsPorts"].as<uint32_t>();
        resourceAlloc = pdu_ds["resourceAlloc"].as<uint32_t>();
        
        // Read array members
        std::vector<uint32_t> tmp_rbBitmap = pdu_ds["rbBitmap"].as<std::vector<uint32_t>>();
        
        for(int i = 0; i < 36; ++i)
        {
            rbBitmap[i] = tmp_rbBitmap[i];
        }
        
        rbStart = pdu_ds["rbStart"].as<uint32_t>();
        rbSize = pdu_ds["rbSize"].as<uint32_t>();
        VRBtoPRBMapping = pdu_ds["VRBtoPRBMapping"].as<uint32_t>();
        StartSymbolIndex = pdu_ds["StartSymbolIndex"].as<uint32_t>();
        NrOfSymbols = pdu_ds["NrOfSymbols"].as<uint32_t>();
        enablePrcdBf = pdu_ds["enablePrcdBf"].as<uint32_t>();
        numPRGs = pdu_ds["numPRGs"].as<uint32_t>();
        prgSize = pdu_ds["prgSize"].as<uint32_t>();
        digBFInterfaces = pdu_ds["digBFInterfaces"].as<uint32_t>();
        PMidx = pdu_ds["PMidx"].as<uint32_t>();
        
        // Read beamIdx array
        std::vector<uint32_t> tmp_beamIdx = pdu_ds["beamIdx"].as<std::vector<uint32_t>>();
        
        for(int i = 0; i < 4; ++i)
        {
            beamIdx[i] = tmp_beamIdx[i];
        }
        
        powerControlOffset = pdu_ds["powerControlOffset"].as<uint32_t>();
        powerControlOffsetSS = pdu_ds["powerControlOffsetSS"].as<uint32_t>();
        IsLastCbPresent = pdu_ds["IsLastCbPresent"].as<uint32_t>();
        isInlineTbCrc = pdu_ds["isInlineTbCrc"].as<uint32_t>();
        dlTbCrc = pdu_ds["dlTbCrc"].as<uint32_t>();
        I_LBRM = pdu_ds["I_LBRM"].as<uint32_t>();
        maxLayers = pdu_ds["maxLayers"].as<uint32_t>();
        maxQm = pdu_ds["maxQm"].as<uint32_t>();
        n_PRB_LBRM = pdu_ds["n_PRB_LBRM"].as<uint32_t>();
        testModel = pdu_ds["testModel"].as<uint32_t>();
        idxUE = pdu_ds["idxUE"].as<uint32_t>();
        idxUeg = pdu_ds["idxUeg"].as<uint32_t>();
        pdschPduIdx = pdu_ds["pdschPduIdx"].as<uint32_t>();
        lastPdschPdu = pdu_ds["lastPdschPdu"].as<uint32_t>();
    }
};

enum tv_channel_type_t
{
    TV_NONE     = 0,
    TV_PBCH     = 1,
    TV_PDCCH    = 2,
    TV_PDSCH    = 3,
    TV_CSI_RS   = 4,
    TV_PRACH    = 6,
    TV_PUCCH    = 7,
    TV_PUSCH    = 8,
    TV_SRS      = 9,
    TV_BFW_DL   = 10,
    TV_BFW_UL   = 11, // TV_BFW_DL but bfwUL == 1
    TV_PDCCH_UL = 12 // TV_PDCCH but dciUL == 1

};

struct prb_info_entry
{
    prb_info_entry(uint16_t cell_id_, uint16_t sym_id_,uint16_t startPrbc_,uint16_t numPrbc_,uint16_t reMask_, uint16_t numSym_) : cell_id(cell_id_), sym_id(sym_id_), startPrbc(startPrbc_), numPrbc(numPrbc_), reMask(reMask_), numSym(numSym_) {};
    uint16_t cell_id;
    uint16_t sym_id;
    uint16_t startPrbc;
    uint16_t numPrbc;
    uint16_t reMask;
    uint16_t rbInc;
    uint16_t numApIndices;
    uint16_t portMask;
    uint16_t numSym;
};

using reference_prb_info = std::vector<prb_info_entry>;

typedef std::unique_ptr<void, decltype(&free)> unique_void_ptr;
struct Dataset {
    size_t size{0};
    unique_void_ptr data{nullptr,free};
};

#endif