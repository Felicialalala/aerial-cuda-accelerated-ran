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

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cuphy.h"
#include "cuphy_api.h"
#include "pycuphy_srs_util.hpp"
#include "pycuphy_util.hpp"


namespace py = pybind11;

namespace pycuphy {

void readSrsCellParams(std::vector<cuphySrsCellPrms_t>& srsCellPrms,
                       const std::vector<py::object>& pySrsCellPrms) {
    srsCellPrms.resize(pySrsCellPrms.size());
    for(int cellIdx = 0; cellIdx < srsCellPrms.size(); cellIdx++)
    {
        srsCellPrms[cellIdx].slotNum        = pySrsCellPrms[cellIdx].attr("slot_num").cast<uint16_t>();
        srsCellPrms[cellIdx].frameNum       = pySrsCellPrms[cellIdx].attr("frame_num").cast<uint16_t>();
        srsCellPrms[cellIdx].srsStartSym    = pySrsCellPrms[cellIdx].attr("srs_start_sym").cast<uint8_t>();
        srsCellPrms[cellIdx].nSrsSym        = pySrsCellPrms[cellIdx].attr("num_srs_sym").cast<uint8_t>();
        srsCellPrms[cellIdx].nRxAntSrs      = pySrsCellPrms[cellIdx].attr("num_rx_ant_srs").cast<uint16_t>();
        srsCellPrms[cellIdx].mu             = pySrsCellPrms[cellIdx].attr("mu").cast<uint8_t>();
    }
}


void readSrsCellDynParams(std::vector<cuphySrsCellDynPrm_t>& srsCellPrms,
                          const std::vector<pybind11::object>& pySrsRxCellConfigs) {
    srsCellPrms.resize(pySrsRxCellConfigs.size());
    for(int cellIdx = 0; const auto& pySrsRxCellConfig : pySrsRxCellConfigs) {
        srsCellPrms[cellIdx].slotNum        = pySrsRxCellConfig.attr("slot").cast<uint16_t>();
        srsCellPrms[cellIdx].frameNum       = pySrsRxCellConfig.attr("frame").cast<uint16_t>();
        srsCellPrms[cellIdx].srsStartSym    = pySrsRxCellConfig.attr("srs_start_sym").cast<uint8_t>();
        srsCellPrms[cellIdx].nSrsSym        = pySrsRxCellConfig.attr("num_srs_sym").cast<uint8_t>();
        srsCellPrms[cellIdx].cellPrmStatIdx = cellIdx;
        srsCellPrms[cellIdx].cellPrmDynIdx  = cellIdx;
        cellIdx++;
    }
}


void readUeSrsParams(std::vector<cuphyUeSrsPrm_t>& ueSrsPrms,
                     const std::vector<py::object>& pyUeSrsPrms) {
    ueSrsPrms.resize(pyUeSrsPrms.size());
    for(int ueIdx = 0; ueIdx < pyUeSrsPrms.size(); ueIdx++) {
        ueSrsPrms[ueIdx].cellIdx = pyUeSrsPrms[ueIdx].attr("cell_idx").cast<uint16_t>();
        ueSrsPrms[ueIdx].nAntPorts = pyUeSrsPrms[ueIdx].attr("num_ant_ports").cast<uint8_t>();
        ueSrsPrms[ueIdx].nSyms = pyUeSrsPrms[ueIdx].attr("num_syms").cast<uint8_t>();
        ueSrsPrms[ueIdx].nRepetitions = pyUeSrsPrms[ueIdx].attr("num_repetitions").cast<uint8_t>();
        ueSrsPrms[ueIdx].combSize = pyUeSrsPrms[ueIdx].attr("comb_size").cast<uint8_t>();
        ueSrsPrms[ueIdx].startSym = pyUeSrsPrms[ueIdx].attr("start_sym").cast<uint8_t>();
        ueSrsPrms[ueIdx].sequenceId = pyUeSrsPrms[ueIdx].attr("sequence_id").cast<uint16_t>();
        ueSrsPrms[ueIdx].configIdx = pyUeSrsPrms[ueIdx].attr("config_idx").cast<uint8_t>();
        ueSrsPrms[ueIdx].bandwidthIdx = pyUeSrsPrms[ueIdx].attr("bandwidth_idx").cast<uint8_t>();
        ueSrsPrms[ueIdx].combOffset = pyUeSrsPrms[ueIdx].attr("comb_offset").cast<uint8_t>();
        ueSrsPrms[ueIdx].cyclicShift = pyUeSrsPrms[ueIdx].attr("cyclic_shift").cast<uint8_t>();
        ueSrsPrms[ueIdx].frequencyPosition = pyUeSrsPrms[ueIdx].attr("frequency_position").cast<uint8_t>();
        ueSrsPrms[ueIdx].frequencyShift = pyUeSrsPrms[ueIdx].attr("frequency_shift").cast<uint16_t>();
        ueSrsPrms[ueIdx].frequencyHopping = pyUeSrsPrms[ueIdx].attr("frequency_hopping").cast<uint8_t>();
        ueSrsPrms[ueIdx].resourceType = pyUeSrsPrms[ueIdx].attr("resource_type").cast<uint8_t>();
        ueSrsPrms[ueIdx].Tsrs = pyUeSrsPrms[ueIdx].attr("periodicity").cast<uint16_t>();
        ueSrsPrms[ueIdx].Toffset = pyUeSrsPrms[ueIdx].attr("offset").cast<uint16_t>();
        ueSrsPrms[ueIdx].groupOrSequenceHopping = pyUeSrsPrms[ueIdx].attr("group_or_sequence_hopping").cast<uint8_t>();
        ueSrsPrms[ueIdx].chEstBuffIdx = pyUeSrsPrms[ueIdx].attr("ch_est_buff_idx").cast<uint16_t>();
        const py::array& srs_ant_port_to_ue_ant_map = pyUeSrsPrms[ueIdx].attr("srs_ant_port_to_ue_ant_map");
        const py::buffer_info& buf = srs_ant_port_to_ue_ant_map.request();
        memcpy(&ueSrsPrms[ueIdx].srsAntPortToUeAntMap, buf.ptr, 4);
        ueSrsPrms[ueIdx].rnti = 0;
        ueSrsPrms[ueIdx].handle = 0;
        ueSrsPrms[ueIdx].prgSize = pyUeSrsPrms[ueIdx].attr("prg_size").cast<uint8_t>();
        ueSrsPrms[ueIdx].usage = 0;
    }
}



void readUeSrsRxParams(std::vector<cuphyUeSrsPrm_t>& ueSrsPrms,
                       std::vector<int>& numPrgs,
                       const std::vector<pybind11::object>& pySrsRxUeConfigs) {
    ueSrsPrms.resize(pySrsRxUeConfigs.size());
    numPrgs.resize(pySrsRxUeConfigs.size());

    for(int ueIdx = 0; auto& pySrsRxUeConfig : pySrsRxUeConfigs) {
        ueSrsPrms[ueIdx].cellIdx = pySrsRxUeConfig.attr("cell_idx").cast<uint16_t>();

        const py::object& pySrsConfig = pySrsRxUeConfig.attr("srs_config");
        ueSrsPrms[ueIdx].nAntPorts = pySrsConfig.attr("num_ant_ports").cast<uint8_t>();
        ueSrsPrms[ueIdx].nSyms = pySrsConfig.attr("num_syms").cast<uint8_t>();
        ueSrsPrms[ueIdx].nRepetitions = pySrsConfig.attr("num_repetitions").cast<uint8_t>();
        ueSrsPrms[ueIdx].combSize = pySrsConfig.attr("comb_size").cast<uint8_t>();
        ueSrsPrms[ueIdx].startSym = pySrsConfig.attr("start_sym").cast<uint8_t>();
        ueSrsPrms[ueIdx].sequenceId = pySrsConfig.attr("sequence_id").cast<uint16_t>();
        ueSrsPrms[ueIdx].configIdx = pySrsConfig.attr("config_idx").cast<uint8_t>();
        ueSrsPrms[ueIdx].bandwidthIdx = pySrsConfig.attr("bandwidth_idx").cast<uint8_t>();
        ueSrsPrms[ueIdx].combOffset = pySrsConfig.attr("comb_offset").cast<uint8_t>();
        ueSrsPrms[ueIdx].cyclicShift = pySrsConfig.attr("cyclic_shift").cast<uint8_t>();
        ueSrsPrms[ueIdx].frequencyPosition = pySrsConfig.attr("frequency_position").cast<uint8_t>();
        ueSrsPrms[ueIdx].frequencyShift = pySrsConfig.attr("frequency_shift").cast<uint16_t>();
        ueSrsPrms[ueIdx].frequencyHopping = pySrsConfig.attr("frequency_hopping").cast<uint8_t>();
        ueSrsPrms[ueIdx].resourceType = pySrsConfig.attr("resource_type").cast<uint8_t>();
        ueSrsPrms[ueIdx].Tsrs = pySrsConfig.attr("periodicity").cast<uint16_t>();
        ueSrsPrms[ueIdx].Toffset = pySrsConfig.attr("offset").cast<uint16_t>();
        ueSrsPrms[ueIdx].groupOrSequenceHopping = pySrsConfig.attr("group_or_sequence_hopping").cast<uint8_t>();

        ueSrsPrms[ueIdx].chEstBuffIdx = ueIdx;
        const py::array& srs_ant_port_to_ue_ant_map = pySrsRxUeConfig.attr("srs_ant_port_to_ue_ant_map");
        const py::buffer_info& buf = srs_ant_port_to_ue_ant_map.request();
        memcpy(&ueSrsPrms[ueIdx].srsAntPortToUeAntMap, buf.ptr, 4);
        ueSrsPrms[ueIdx].rnti = 0;
        ueSrsPrms[ueIdx].handle = 0;
        ueSrsPrms[ueIdx].prgSize = pySrsRxUeConfig.attr("prg_size").cast<uint8_t>();
        ueSrsPrms[ueIdx].usage = 0;
        ueSrsPrms[ueIdx].srsStartPrg = pySrsRxUeConfig.attr("start_prg").cast<uint8_t>();
        ueSrsPrms[ueIdx].srsChestBufferIndexL2 = ueIdx;
        numPrgs[ueIdx] = pySrsRxUeConfig.attr("num_prgs").cast<uint16_t>();
        ueIdx++;
    }
}


void readUeSrsTxParams(std::vector<cuphyUeSrsTxPrm_t>& ueSrsPrms,
                       const uint16_t slotIdx,
                       const uint16_t frameIdx,
                       const std::vector<pybind11::object>& pySrsConfigs) {
    ueSrsPrms.resize(pySrsConfigs.size());
    for(int ueIdx = 0; auto& prms : ueSrsPrms) {
        prms.nAntPorts = pySrsConfigs[ueIdx].attr("num_ant_ports").cast<uint8_t>();
        prms.nSyms = pySrsConfigs[ueIdx].attr("num_syms").cast<uint8_t>();
        prms.nRepetitions = pySrsConfigs[ueIdx].attr("num_repetitions").cast<uint8_t>();
        prms.combSize = pySrsConfigs[ueIdx].attr("comb_size").cast<uint8_t>();
        prms.startSym = pySrsConfigs[ueIdx].attr("start_sym").cast<uint8_t>();
        prms.sequenceId = pySrsConfigs[ueIdx].attr("sequence_id").cast<uint16_t>();
        prms.configIdx = pySrsConfigs[ueIdx].attr("config_idx").cast<uint8_t>();
        prms.bandwidthIdx = pySrsConfigs[ueIdx].attr("bandwidth_idx").cast<uint8_t>();
        prms.combOffset = pySrsConfigs[ueIdx].attr("comb_offset").cast<uint8_t>();
        prms.cyclicShift = pySrsConfigs[ueIdx].attr("cyclic_shift").cast<uint8_t>();
        prms.frequencyPosition = pySrsConfigs[ueIdx].attr("frequency_position").cast<uint8_t>();
        prms.frequencyShift = pySrsConfigs[ueIdx].attr("frequency_shift").cast<uint16_t>();
        prms.frequencyHopping = pySrsConfigs[ueIdx].attr("frequency_hopping").cast<uint8_t>();
        prms.resourceType = pySrsConfigs[ueIdx].attr("resource_type").cast<uint8_t>();
        prms.Tsrs = pySrsConfigs[ueIdx].attr("periodicity").cast<uint16_t>();
        prms.Toffset = pySrsConfigs[ueIdx].attr("offset").cast<uint16_t>();
        prms.idxSlotInFrame = slotIdx;
        prms.idxFrame = frameIdx;
        ueIdx++;
    }
}


SrsTensorPrms::SrsTensorPrms(const pybind11::dict& chEstParams, const cudaStream_t cuStream) {

    const auto convert_complex_float{[](const auto& chEstParams, const cudaStream_t cuStream) {
        return deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
            chEstParams,
            CUPHY_C_32F,
            CUPHY_C_16F,
            cuphy::tensor_flags::align_tight,
            cuStream);
    }};

    const auto convert_float{[](const auto& chEstParams, const cudaStream_t cuStream) {
        return deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
            chEstParams,
            CUPHY_R_32F,
            CUPHY_R_16F,
            cuphy::tensor_flags::align_tight,
            cuStream);
    }};

    m_tPrmFocc_table = convert_complex_float(chEstParams["focc_table"], cuStream);
    m_srsFilterPrms.tPrmFocc_table.desc = m_tPrmFocc_table.desc().handle();
    m_srsFilterPrms.tPrmFocc_table.pAddr = m_tPrmFocc_table.addr();

    m_tPrmFocc_comb2_table = convert_complex_float(chEstParams["focc_table_comb2"], cuStream);
    m_srsFilterPrms.tPrmFocc_comb2_table.desc  = m_tPrmFocc_comb2_table.desc().handle();
    m_srsFilterPrms.tPrmFocc_comb2_table.pAddr = m_tPrmFocc_comb2_table.addr();

    m_tPrmFocc_comb4_table = convert_complex_float(chEstParams["focc_table_comb4"], cuStream);
    m_srsFilterPrms.tPrmFocc_comb4_table.desc  = m_tPrmFocc_comb4_table.desc().handle();
    m_srsFilterPrms.tPrmFocc_comb4_table.pAddr = m_tPrmFocc_comb4_table.addr();

    m_tPrmW_comb2_nPorts1_wide = convert_complex_float(chEstParams["W_comb2_nPorts1_wide"], cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts1_wide.desc = m_tPrmW_comb2_nPorts1_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts1_wide.pAddr = m_tPrmW_comb2_nPorts1_wide.addr();

    m_tPrmW_comb2_nPorts2_wide = convert_complex_float(chEstParams["W_comb2_nPorts2_wide"], cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts2_wide.desc = m_tPrmW_comb2_nPorts2_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts2_wide.pAddr = m_tPrmW_comb2_nPorts2_wide.addr();

    m_tPrmW_comb2_nPorts4_wide = convert_complex_float(chEstParams["W_comb2_nPorts4_wide"], cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts4_wide.desc = m_tPrmW_comb2_nPorts4_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts4_wide.pAddr = m_tPrmW_comb2_nPorts4_wide.addr();

    m_tPrmW_comb2_nPorts8_wide = convert_complex_float(chEstParams["W_comb2_nPorts8_wide"], cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts8_wide.desc = m_tPrmW_comb2_nPorts8_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts8_wide.pAddr = m_tPrmW_comb2_nPorts8_wide.addr();

    m_tPrmW_comb4_nPorts1_wide = convert_complex_float(chEstParams["W_comb4_nPorts1_wide"], cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts1_wide.desc = m_tPrmW_comb4_nPorts1_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts1_wide.pAddr = m_tPrmW_comb4_nPorts1_wide.addr();

    m_tPrmW_comb4_nPorts2_wide = convert_complex_float(chEstParams["W_comb4_nPorts2_wide"], cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts2_wide.desc = m_tPrmW_comb4_nPorts2_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts2_wide.pAddr = m_tPrmW_comb4_nPorts2_wide.addr();

    m_tPrmW_comb4_nPorts4_wide = convert_complex_float(chEstParams["W_comb4_nPorts4_wide"], cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts4_wide.desc = m_tPrmW_comb4_nPorts4_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts4_wide.pAddr = m_tPrmW_comb4_nPorts4_wide.addr();

    m_tPrmW_comb4_nPorts6_wide = convert_complex_float(chEstParams["W_comb4_nPorts6_wide"], cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts6_wide.desc = m_tPrmW_comb4_nPorts6_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts6_wide.pAddr = m_tPrmW_comb4_nPorts6_wide.addr();

    m_tPrmW_comb4_nPorts12_wide = convert_complex_float(chEstParams["W_comb4_nPorts12_wide"], cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts12_wide.desc = m_tPrmW_comb4_nPorts12_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts12_wide.pAddr = m_tPrmW_comb4_nPorts12_wide.addr();

    m_tPrmW_comb2_nPorts1_narrow = convert_complex_float(chEstParams["W_comb2_nPorts1_narrow"], cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts1_narrow.desc = m_tPrmW_comb2_nPorts1_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts1_narrow.pAddr = m_tPrmW_comb2_nPorts1_narrow.addr();

    m_tPrmW_comb2_nPorts2_narrow = convert_complex_float(chEstParams["W_comb2_nPorts2_narrow"], cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts2_narrow.desc = m_tPrmW_comb2_nPorts2_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts2_narrow.pAddr = m_tPrmW_comb2_nPorts2_narrow.addr();

    m_tPrmW_comb2_nPorts4_narrow = convert_complex_float(chEstParams["W_comb2_nPorts4_narrow"], cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts4_narrow.desc = m_tPrmW_comb2_nPorts4_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts4_narrow.pAddr = m_tPrmW_comb2_nPorts4_narrow.addr();

    m_tPrmW_comb2_nPorts8_narrow = convert_complex_float(chEstParams["W_comb2_nPorts8_narrow"], cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts8_narrow.desc = m_tPrmW_comb2_nPorts8_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts8_narrow.pAddr = m_tPrmW_comb2_nPorts8_narrow.addr();

    m_tPrmW_comb4_nPorts1_narrow = convert_complex_float(chEstParams["W_comb4_nPorts1_narrow"], cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts1_narrow.desc = m_tPrmW_comb4_nPorts1_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts1_narrow.pAddr = m_tPrmW_comb4_nPorts1_narrow.addr();

    m_tPrmW_comb4_nPorts2_narrow = convert_complex_float(chEstParams["W_comb4_nPorts2_narrow"], cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts2_narrow.desc = m_tPrmW_comb4_nPorts2_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts2_narrow.pAddr = m_tPrmW_comb4_nPorts2_narrow.addr();

    m_tPrmW_comb4_nPorts4_narrow = convert_complex_float(chEstParams["W_comb4_nPorts4_narrow"], cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts4_narrow.desc = m_tPrmW_comb4_nPorts4_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts4_narrow.pAddr = m_tPrmW_comb4_nPorts4_narrow.addr();

    m_tPrmW_comb4_nPorts6_narrow = convert_complex_float(chEstParams["W_comb4_nPorts6_narrow"], cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts6_narrow.desc = m_tPrmW_comb4_nPorts6_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts6_narrow.pAddr = m_tPrmW_comb4_nPorts6_narrow.addr();

    m_tPrmW_comb4_nPorts12_narrow = convert_complex_float(chEstParams["W_comb4_nPorts12_narrow"], cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts12_narrow.desc = m_tPrmW_comb4_nPorts12_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts12_narrow.pAddr = m_tPrmW_comb4_nPorts12_narrow.addr();

    m_srsFilterPrms.noisEstDebias_comb2_nPorts1 = chEstParams["noisEstDebias_comb2_nPorts1"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb2_nPorts2 = chEstParams["noisEstDebias_comb2_nPorts2"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb2_nPorts4 = chEstParams["noisEstDebias_comb2_nPorts4"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb2_nPorts8 = chEstParams["noisEstDebias_comb2_nPorts8"].cast<float>();

    m_srsFilterPrms.noisEstDebias_comb4_nPorts1  = chEstParams["noisEstDebias_comb4_nPorts1"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb4_nPorts2  = chEstParams["noisEstDebias_comb4_nPorts2"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb4_nPorts4  = chEstParams["noisEstDebias_comb4_nPorts4"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb4_nPorts6  = chEstParams["noisEstDebias_comb4_nPorts6"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb4_nPorts12 = chEstParams["noisEstDebias_comb4_nPorts12"].cast<float>();

    m_cpuBufferRkhsGridPrms = std::move(cuphy::buffer<cuphyRkhsGridPrms_t, cuphy::pinned_alloc>(NUM_RKHS_GRIDS));
    m_srsRkhsPrms.pRkhsGridPrms = m_cpuBufferRkhsGridPrms.addr();
    m_srsRkhsPrms.nGridSizes = NUM_RKHS_GRIDS;

    for(int gridIdx = 0; gridIdx < NUM_RKHS_GRIDS; ++gridIdx) {
        std::string str = "srsRkhs_eigenVecs_grid" + std::to_string(gridIdx);
        m_eigenVecTable[gridIdx] = convert_float(chEstParams[str.c_str()], cuStream);
        m_cpuBufferRkhsGridPrms[gridIdx].eigenVecs.desc  = m_eigenVecTable[gridIdx].desc().handle();
        m_cpuBufferRkhsGridPrms[gridIdx].eigenVecs.pAddr = m_eigenVecTable[gridIdx].addr();

        str = "srsRkhs_eigValues_grid" + std::to_string(gridIdx);
        m_eigenValueTable[gridIdx] = convert_float(chEstParams[str.c_str()], cuStream);
        m_cpuBufferRkhsGridPrms[gridIdx].eigenValues.desc  = m_eigenValueTable[gridIdx].desc().handle();
        m_cpuBufferRkhsGridPrms[gridIdx].eigenValues.pAddr = m_eigenValueTable[gridIdx].addr();

        str = "srsRkhs_eigenCorr_grid" + std::to_string(gridIdx);
        m_eigenCorrTable[gridIdx] = convert_complex_float(chEstParams[str.c_str()], cuStream);
        m_cpuBufferRkhsGridPrms[gridIdx].eigenCorr.desc  = m_eigenCorrTable[gridIdx].desc().handle();
        m_cpuBufferRkhsGridPrms[gridIdx].eigenCorr.pAddr = m_eigenCorrTable[gridIdx].addr();
    }

    m_secondStageTwiddleFactorsTable = convert_complex_float(chEstParams["srsRkhs_secondStageTwiddleFactors_grid2"], cuStream);
    m_cpuBufferRkhsGridPrms[2].secondStageTwiddleFactors.desc  = m_secondStageTwiddleFactorsTable.desc().handle();
    m_cpuBufferRkhsGridPrms[2].secondStageTwiddleFactors.pAddr = m_secondStageTwiddleFactorsTable.addr();

    m_secondStageFourierPermTable = deviceFromNumpy<uint8_t>(chEstParams["srsRkhs_secondStageFourierPerm_grid2"],
                                                             CUPHY_R_8U,
                                                             CUPHY_R_8U,
                                                             cuphy::tensor_flags::align_tight,
                                                             cuStream);
    m_cpuBufferRkhsGridPrms[2].secondStageFourierPerm.desc  = m_secondStageFourierPermTable.desc().handle();
    m_cpuBufferRkhsGridPrms[2].secondStageFourierPerm.pAddr = m_secondStageFourierPermTable.addr();
}


}
