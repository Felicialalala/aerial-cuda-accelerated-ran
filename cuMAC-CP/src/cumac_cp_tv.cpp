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

#include <unistd.h>

#include "api.h"
#include "cumac_task.hpp"

#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "nvlog.hpp"

#include "cumac_cp_tv.hpp"

#include <chrono>

#define TAG (NVLOG_TAG_BASE_CUMAC_CP + 2) // "CUMCP.CFG"

#define CHECK_VALUE_EQUAL_ERR(v1, v2)                                                                                              \
    do                                                                                                                             \
    {                                                                                                                              \
        if ((v1) != (v2))                                                                                                          \
        {                                                                                                                          \
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: values doesn't equal: v1={} > v2={}", __func__, __LINE__, v1, v2); \
        }                                                                                                                          \
    } while (0);

#define CONFIG_CUMAC_TV_PATH "testVectors/cumac/"

using namespace std;
using namespace std::chrono;

static cumac_cp_tv_t *tv_ptr = nullptr;
cumac_cp_tv_t *get_cumac_tv_ptr()
{
    return tv_ptr;
}

// Current parsing cell_id, slot_id, channel and TV file name for debug log
static int curr_cell;
static int curr_slot;
static int curr_task;
static std::string curr_tv;

static const char *CUMAC_CP_TV = "CUMAC_CP";
static inline const char *get_task_name(int task_type)
{
    return CUMAC_CP_TV;
}

int check_bytes(const char *name1, const char *name2, void *buf1, void *buf2, size_t nbytes)
{
    int check_result = 0;
    if (buf1 == nullptr || buf2 == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "bytes pointer is null: {}=0x{} {}=0x{}", name1, buf1, name2, buf2);
    }
    else if (memcmp(buf1, buf2, nbytes))
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "bytes differ: byte[0] {}=0x{:02X} {}=0x{:02X}", name1, *(uint8_t *)buf1, name2, *(uint8_t *)buf2);

        char info_str[64];
        snprintf(info_str, 64, "ARRAY DIFF: %s", name1);
        NVLOGI_FMT_ARRAY(TAG, info_str, reinterpret_cast<uint8_t *>(buf1), nbytes);
        snprintf(info_str, 64, "ARRAY DIFF: %s", name2);
        NVLOGI_FMT_ARRAY(TAG, info_str, reinterpret_cast<uint8_t *>(buf2), nbytes);

        uint8_t *v1 = reinterpret_cast<uint8_t *>(buf1);
        uint8_t *v2 = reinterpret_cast<uint8_t *>(buf2);
        uint32_t i;
        for (i = 0; i < nbytes; i++)
        {
            if (*(v1 + i) != *(v2 + i))
            {
                break;
            }
        }
        v1 += i;
        v2 += i;
        snprintf(info_str, 64, "ARRAY DIFF from %s[%u]", name1, i);
        NVLOGI_FMT_ARRAY(TAG, info_str, v1, nbytes - i);
        snprintf(info_str, 64, "ARRAY DIFF from %s[%u]", name2, i);
        NVLOGI_FMT_ARRAY(TAG, info_str, v2, nbytes - i);
        check_result = -1;
    }
    else
    {
        NVLOGI_FMT(TAG, "bytes same: byte[0] {}={}=0x{:02X}", name1, name2, *(uint8_t *)buf1);
    }
    return check_result;
}

template <typename T>
static int yaml_try_parse_list(yaml::node &parent_node, const char *name, std::vector<T> &values)
{
    yaml::node list_nodes = parent_node[name];
    if (list_nodes.type() != YAML_SEQUENCE_NODE)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: failed to parse {}: error type {}\n", __func__, name, list_nodes.type());
        return -1;
    }

    size_t num = list_nodes.length();
    values.resize(num);

    for (size_t i = 0; i < num; i++)
    {
        yaml::node node = list_nodes[i];
        values[i] = node.as<T>();
    }
    return 0;
}

static int h5dset_try_read(hdf5hpp::hdf5_file &file, const char *name, void *buf, size_t size)
{
    if (!file.is_valid_dataset(name))
    {
        NVLOGW_FMT(TAG, "TV cell {} slot {} {} {} dataset {} not exist",
                   curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), name);
        return -1;
    }

    try
    {
        hdf5hpp::hdf5_dataset h5dset = file.open_dataset(name);
        if (h5dset.get_buffer_size_bytes() != size)
        {
            NVLOGW_FMT(TAG, "TV cell {} slot {} {} {} dataset {} size doesn't match: dataset_size={} buf_size={}",
                       curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(),
                       name, h5dset.get_buffer_size_bytes(), size);
            return -1;
        }
        else
        {
            h5dset.read(buf);
            return 0;
        }
    }
    catch (std::exception &e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV cell {} slot {} {} {} dataset {} exception: {}",
                   curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), name, e.what());
    }
    return -1;
}

template <typename Type>
static int h5dset_try_read_array(hdf5hpp::hdf5_file &file, const char *name, Type **buf_ptr, uint32_t elem_num)
{
    NVLOGC_FMT(TAG, "Loading buffer {} num={}", name, elem_num);
    *buf_ptr = new Type[elem_num];
    if (h5dset_try_read(file, name, *buf_ptr, sizeof(Type) * elem_num) < 0)
    {
        delete *buf_ptr;
        return -1;
    }

    return 0;
}

static int h5dset_try_read_complex(hdf5hpp::hdf5_file &file, const char *name_real, const char *name_imag, cuComplex **complex_ptr, uint32_t elem_num)
{
    float *tmp_real = nullptr;
    float *tmp_imag = nullptr;

    if (h5dset_try_read_array(file, name_real, &tmp_real, elem_num) < 0)
    {
        return -1;
    }

    if (h5dset_try_read_array(file, name_imag, &tmp_imag, elem_num) < 0)
    {
        delete tmp_real;
        return -1;
    }

    *complex_ptr = new cuComplex[elem_num];
    for (int i = 0; i < elem_num; i++)
    {
        cuComplex *val = *complex_ptr + i;
        val->x = *(tmp_real + i);
        val->y = *(tmp_imag + i);
    }

    delete tmp_real;
    delete tmp_imag;

    return 0;
}

static int h5dset_try_read_u32_to_bits(hdf5hpp::hdf5_file &file, const char *name, std::vector<uint8_t> &dest, int num_bits)
{
    uint32_t *src = new uint32_t[num_bits];
    int ret = h5dset_try_read(file, name, src, num_bits * sizeof(uint32_t));
    if (ret == 0)
    {
        // Initiate bytes to 0
        int nbytes = (num_bits + 7) / 8;
        dest.resize(nbytes);
        for (int i = 0; i < nbytes; i++)
        {
            dest[i] = 0;
        }
        // Convert bits to bytes
        for (int j = 0; j < num_bits; j++)
        {
            dest[j / 8] |= src[j] == 0 ? 0 : 1 << j % 8;
        }
    }
    delete src;
    return ret;
}

template <typename TypeSrc, typename TypeDst>
static int h5dset_try_read_convert(hdf5hpp::hdf5_file &file, const char *name, TypeDst *dst, uint32_t num)
{
    if (num == 0)
    {
        NVLOGW_FMT(TAG, "TV cell {} slot {} {} {} dataset {} reading with num=0",
                   curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), name);
        return -1;
    }

    TypeSrc *src = new uint32_t[num];
    int ret = h5dset_try_read(file, name, src, num * sizeof(TypeSrc));
    if (ret == 0)
    {
        for (int i = 0; i < num; i++)
        {
            dst[i] = src[i];
        }
    }
    delete src;
    return ret;
}

template <typename T>
static T h5dset_try_parse(const hdf5hpp::hdf5_dataset_elem &dset_elem, const char *name, T default_value, bool miss_warning = true)
{
    T value;
    try
    {
        value = dset_elem[name].as<T>();
    }
    catch (std::exception &e)
    {
        value = default_value;
        if (miss_warning)
        {
            NVLOGW_FMT(TAG, "TV cell {} slot {} {} {} key {} not exist",
                       curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), name);
        }
    }
    return value;
}

template <typename T>
static T h5dset_try_parse(hdf5hpp::hdf5_dataset &h5dset, const char *name, T default_value, bool miss_warning = true)
{
    // return h5dset_try_parse(h5dset[0], name, default_value, miss_warning);
    return h5dset_try_parse(h5dset[0], name, default_value, miss_warning);
}

template <typename T>
static T h5file_try_parse(const char *file_name, const char *dset_name, const char *var_name, T default_value, bool miss_warning = true, int dset_id = 0)
{
    char h5path[MAX_PATH_LEN];
    get_full_path_file(h5path, CONFIG_CUMAC_TV_PATH, file_name, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    if (access(h5path, F_OK) != 0)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "H5 file {} not exist", h5path);
    }

    hdf5hpp::hdf5_file hdf5file;
    try
    {
        hdf5file = hdf5hpp::hdf5_file::open(h5path);
    }
    catch (std::exception &e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Exception: {}: hdf5_file::open({}): {}", __FUNCTION__, h5path, e.what());
        return default_value;
    }

    if (hdf5file.is_valid_dataset(dset_name))
    {
        hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset(dset_name);
        return h5dset_try_parse(dset[dset_id], var_name, default_value, miss_warning);
    }
    else
    {
        return default_value;
    }
}

int calculate_buf_num(struct cumac::cumacSchedulerParam &param, cumac_buf_num_t &buf_num)
{
    uint32_t prdLen = param.nUe * param.nPrbGrp * param.nBsAnt * param.nBsAnt;
    uint32_t detLen = param.nUe * param.nPrbGrp * param.nBsAnt * param.nBsAnt;
    uint32_t hLen = param.nPrbGrp * param.nUe * param.nCell * param.nBsAnt * param.nUeAnt;

    uint32_t pfSize = param.nPrbGrp * param.numUeSchdPerCellTTI;
    uint32_t pow2N = 2;
    while (pow2N < pfSize)
    {
        pow2N = pow2N << 1;
    }
    buf_num.postEqSinr = param.nActiveUe * param.nPrbGrp * param.nUeAnt;
    buf_num.cellId = param.nCell;
    buf_num.cellAssoc = param.nCell * param.nUe;
    buf_num.cellAssocActUe = param.nCell * param.nActiveUe;
    buf_num.wbSinr = param.nActiveUe * param.nUeAnt;
    buf_num.sinVal = param.nUe * param.nPrbGrp * param.nUeAnt;
    buf_num.prdMat = prdLen;
    buf_num.detMat = detLen;
    buf_num.estH_fr = hLen;
    buf_num.setSchdUePerCellTTI = param.nUe;
    buf_num.allocSol = param.allocType == 1 ? param.nUe * 2 : param.nCell * param.nPrbGrp;
    buf_num.layerSelSol = param.nUe;
    buf_num.mcsSelSol = param.nUe;
    buf_num.pfMetricArr = param.nCell * pow2N;
    buf_num.pfIdArr = param.nCell * pow2N;
    buf_num.avgRatesActUe = param.nActiveUe;
    buf_num.avgRates = param.nUe;
    buf_num.newDataActUe = param.nActiveUe;
    buf_num.tbErrLastActUe = param.nActiveUe;
    buf_num.tbErrLast = param.nUe;
    return 0;
}

int parse_tv_file(cumac_cp_tv_t &tv, std::string tv_file)
{
    char file_path[MAX_PATH_LEN];
    get_full_path_file(file_path, CONFIG_CUMAC_TV_PATH, tv_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    if (access(file_path, F_OK) != 0)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV file not exist: {}", file_path);
        return -1;
    }

    hdf5hpp::hdf5_file hdf5file;
    try
    {
        hdf5file = hdf5hpp::hdf5_file::open(file_path);
    }
    catch (std::exception &e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "hdf5_file::open failed. file={}", file_path);
        return -1;
    }

    try
    {
        // Read cumacSchedulerParam
        std::string filepath_str = std::string(file_path);
        H5::H5File file(filepath_str, H5F_ACC_RDONLY);
        // Open the dataset
        H5::DataSet dataset = file.openDataSet("cumacSchedulerParam");
        // Get the compound data type
        H5::CompType compoundType = dataset.getCompType();
        // Read the data from the dataset
        dataset.read(&tv.params, compoundType);

        // nUe=48, nCell=8, totNumCell=8, nPrbGrp=68, nBsAnt=4, nUeAnt=4, W=1.44e+06, sigmaSqrd=1, betaCoeff=1,
        // precodingScheme=1, receiverScheme=1, allocType=1, columnMajor=1, nActiveUe=800, numUeSchdPerCellTTI=6, sinValThr=0.1

        calculate_buf_num(tv.params, tv.buf_num);

        // Create and populate data buffers. Example size: nActiveUe=800, nPrbGrp=68, nBsAnt=4, nUeAnt=4, nMaxSchUePerCell=6
        h5dset_try_read_array(hdf5file, "avgRates", &tv.avgRates, tv.buf_num.avgRates);                   // Dataset {27200}
        h5dset_try_read_array(hdf5file, "cellAssoc", &tv.cellAssoc, tv.buf_num.cellAssoc);                // Dataset {27200}
        h5dset_try_read_array(hdf5file, "cellAssocActUe", &tv.cellAssocActUe, tv.buf_num.cellAssocActUe); // Dataset {27200}
        h5dset_try_read_array(hdf5file, "tbErrLast", &tv.tbErrLast, tv.buf_num.tbErrLast);                // Dataset {27200}

        h5dset_try_read_array(hdf5file, "cellId", &tv.cellId, tv.buf_num.cellId);                           // Dataset {27200}
        h5dset_try_read_array(hdf5file, "postEqSinr", &tv.postEqSinr, tv.buf_num.postEqSinr);               // Dataset {27200}
        h5dset_try_read_array(hdf5file, "wbSinr", &tv.wbSinr, tv.buf_num.wbSinr);                           // Dataset {400}
        h5dset_try_read_complex(hdf5file, "detMat_real", "detMat_imag", &tv.detMat, tv.buf_num.detMat);     // Dataset {6528}
        h5dset_try_read_complex(hdf5file, "estH_fr_real", "estH_fr_imag", &tv.estH_fr, tv.buf_num.estH_fr); // Dataset {52224}
        h5dset_try_read_complex(hdf5file, "prdMat_real", "prdMat_imag", &tv.prdMat, tv.buf_num.prdMat);     // Dataset {6528}
        h5dset_try_read_array(hdf5file, "sinVal", &tv.sinVal, tv.buf_num.sinVal);                           // Dataset {1632}
        h5dset_try_read_array(hdf5file, "avgRatesActUe", &tv.avgRatesActUe, tv.buf_num.avgRatesActUe);      // Dataset {100}
        h5dset_try_read_array(hdf5file, "tbErrLastActUe", &tv.tbErrLastActUe, tv.buf_num.tbErrLastActUe);   // Dataset {100}

        // Parse TV RESPONSE
        h5dset_try_read_array(hdf5file, "setSchdUePerCellTTI", &tv.setSchdUePerCellTTI, tv.buf_num.setSchdUePerCellTTI); // Dataset {6}
        h5dset_try_read_array(hdf5file, "mcsSelSol", &tv.mcsSelSol, tv.buf_num.mcsSelSol);                               // Dataset {6}
        h5dset_try_read_array(hdf5file, "layerSelSol", &tv.layerSelSol, tv.buf_num.layerSelSol);                         // Dataset {6}
        h5dset_try_read_array(hdf5file, "allocSol", &tv.allocSol, tv.buf_num.allocSol);                                  // Dataset {12}

        tv.parsed = 1;
        tv_ptr = &tv;

        struct cumac::cumacSchedulerParam &p = tv.params;
        NVLOGC_FMT(TAG, "Parsed TV: cumacSchedulerParam-1: nUe={} nCell={} totNumCell={} nPrbGrp={} nBsAnt={} nUeAnt={} W={} sigmaSqrd={} maxNumUePerCell={} nMaxSchdUePerRnd={} betaCoeff={}",
                   p.nUe, p.nCell, p.totNumCell, p.nPrbGrp, p.nBsAnt, p.nUeAnt, p.W, p.sigmaSqrd, p.maxNumUePerCell, p.nMaxSchdUePerRnd, p.betaCoeff);
        NVLOGC_FMT(TAG, "Parsed TV: cumacSchedulerParam-2: nActiveUe={} numUeSchdPerCellTTI={} precodingScheme={} receiverScheme={} allocType={} columnMajor={} allocType={} columnMajor={} sinValThr={}",
                   p.nActiveUe, p.numUeSchdPerCellTTI, p.precodingScheme, p.receiverScheme, p.allocType, p.columnMajor, p.allocType, p.columnMajor, p.sinValThr);
    }
    catch (std::exception &e)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV ERR: {}", e.what());
        return -1;
    }

    return 0;
}

bool pfm_load_tv_H5(const std::string& tv_name, std::vector<cumac_pfm_cell_info_t>& pfm_cell_info, std::vector<cumac_pfm_output_cell_info_t>& pfm_output_cell_info)
{
    NVLOGC_FMT(TAG, "Loading PFM sorting TV file {}", tv_name.c_str());

    char file_path[MAX_PATH_LEN];
    get_full_path_file(file_path, CONFIG_CUMAC_TV_PATH, tv_name.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

    if (access(file_path, F_OK) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM TV file not exist: {}", file_path);
        return false;
    }

    try
    {
        H5::H5File file(file_path, H5F_ACC_RDONLY);

        const int num_cell = pfm_cell_info.size();

        if (num_cell != pfm_output_cell_info.size())
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - number of cells in the output cell info array and the input cell info array are different: {} vs {}", num_cell, pfm_output_cell_info.size());
            return false;
        }

        for (int cIdx = 0; cIdx < num_cell; cIdx++)
        {
            const std::string datasetName = "INPUT_CELL_INFO_" + std::to_string(cIdx);
            // see if the dataset exists
            if (H5Lexists(file.getId(), datasetName.c_str(), H5P_DEFAULT) > 0)
            {
                H5::DataSet dataset = file.openDataSet(datasetName);
                dataset.read(reinterpret_cast<uint8_t*>(&pfm_cell_info[cIdx]), H5::PredType::NATIVE_UINT8);
            }
            else
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting TV file {} does not contain input cell info for cell {}", tv_name.c_str(), cIdx);
                return false;
            }
        }

        for (int cIdx = 0; cIdx < num_cell; cIdx++)
        {
            const std::string datasetName = "OUTPUT_CELL_INFO_" + std::to_string(cIdx);
            // see if the dataset exists
            if (H5Lexists(file.getId(), datasetName.c_str(), H5P_DEFAULT) > 0)
            {
                H5::DataSet dataset = file.openDataSet(datasetName);
                dataset.read(reinterpret_cast<uint8_t*>(&pfm_output_cell_info[cIdx]), H5::PredType::NATIVE_UINT8);
            }
            else
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting TV file {} does not contain output cell info for cell {}", tv_name.c_str(), cIdx);
                return false;
            }
        }

        NVLOGC_FMT(TAG, "PFM sorting TV file {} loaded successfully", tv_name.c_str());
        return true;
    }
    catch (const H5::FileIException &e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM TV file exception: {}", e.getDetailMsg());
        return false;
    }
    catch (const std::exception &e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM TV file exception: {}", e.what());
        return false;
    }
}

bool pfm_validate_tv_h5(const std::string& tv_name, const std::vector<cumac_pfm_cell_info_t>& pfm_cell_info, const std::vector<cumac_pfm_output_cell_info_t>& pfm_output_cell_info)
{
    // get number of cells from the TV name "PFM_SORT_TV_xxCELLS_SLOT_yyyy.h5"
    const std::size_t tv_pos = tv_name.find("TV_");
    const std::size_t cells_pos = tv_name.find("CELLS");

    if (tv_pos == std::string::npos || cells_pos == std::string::npos)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - invalid TV file name format: {}", tv_name.c_str());
        return false;
    }

    const std::size_t num_start = tv_pos + 3;  // Position after "TV_"
    if (num_start >= cells_pos)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - invalid TV file name format: {}", tv_name.c_str());
        return false;
    }

    int num_cell{};
    try
    {
        num_cell = std::stoi(tv_name.substr(num_start, cells_pos - num_start));
    }
    catch (const std::invalid_argument& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - invalid number format in TV file name: {}", tv_name.c_str());
        return false;
    }
    catch (const std::out_of_range& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - number out of range in TV file name: {}", tv_name.c_str());
        return false;
    }

    if (num_cell != pfm_cell_info.size())
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - number of cells in the TV file does not match: {} vs {}", num_cell, pfm_cell_info.size());
        return false;
    }

    std::vector<cumac_pfm_cell_info_t> temp_cell_info(num_cell);
    std::vector<cumac_pfm_output_cell_info_t> temp_output_cell_info(num_cell);

    if (!pfm_load_tv_H5(tv_name, temp_cell_info, temp_output_cell_info))
    {
        return false;
    }

    for (int cIdx = 0; cIdx < num_cell; cIdx++)
    {
        // check if temp_cell_info[cIdx] matches pfm_cell_info[cIdx]
        if (temp_cell_info[cIdx].num_ue != pfm_cell_info[cIdx].num_ue ||
            temp_cell_info[cIdx].num_lc_per_ue != pfm_cell_info[cIdx].num_lc_per_ue ||
            temp_cell_info[cIdx].num_lcg_per_ue != pfm_cell_info[cIdx].num_lcg_per_ue)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - cell info in the TV file does not match for cell {}", cIdx);
            return false;
        }

        for (int idx = 0; idx < (CUMAC_PFM_NUM_QOS_TYPES_UL + CUMAC_PFM_NUM_QOS_TYPES_DL); idx++)
        {
            if (temp_cell_info[cIdx].num_output_sorted_lc[idx] != pfm_cell_info[cIdx].num_output_sorted_lc[idx])
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - number of output sorted LCs in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        // check if temp_cell_info[cIdx].ue_info matches pfm_cell_info[cIdx].ue_info
        for (int ueIdx = 0; ueIdx < temp_cell_info[cIdx].num_ue; ueIdx++)
        {
            if (temp_cell_info[cIdx].ue_info[ueIdx].rcurrent_dl != pfm_cell_info[cIdx].ue_info[ueIdx].rcurrent_dl ||
                temp_cell_info[cIdx].ue_info[ueIdx].rcurrent_ul != pfm_cell_info[cIdx].ue_info[ueIdx].rcurrent_ul ||
                temp_cell_info[cIdx].ue_info[ueIdx].rnti != pfm_cell_info[cIdx].ue_info[ueIdx].rnti ||
                temp_cell_info[cIdx].ue_info[ueIdx].id != pfm_cell_info[cIdx].ue_info[ueIdx].id ||
                temp_cell_info[cIdx].ue_info[ueIdx].num_layers_dl != pfm_cell_info[cIdx].ue_info[ueIdx].num_layers_dl ||
                temp_cell_info[cIdx].ue_info[ueIdx].num_layers_ul != pfm_cell_info[cIdx].ue_info[ueIdx].num_layers_ul ||
                temp_cell_info[cIdx].ue_info[ueIdx].flags != pfm_cell_info[cIdx].ue_info[ueIdx].flags)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UE info in the TV file does not match for cell {}", cIdx);
                return false;
            }
            else
            {
                // check dl_lc_info
                for (int lcIdx = 0; lcIdx < temp_cell_info[cIdx].num_lc_per_ue; lcIdx++)
                {
                    if (temp_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].tbs_scheduled != pfm_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].tbs_scheduled ||
                        temp_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].flags != pfm_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].flags ||
                        temp_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].qos_type != pfm_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].qos_type)
                    {
                        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - DL LC info in the TV file does not match for cell {}", cIdx);
                        return false;
                    }
                }

                // check ul_lcg_info
                for (int lcgIdx = 0; lcgIdx < temp_cell_info[cIdx].num_lcg_per_ue; lcgIdx++)
                {
                    if (temp_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].tbs_scheduled != pfm_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].tbs_scheduled ||
                        temp_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].flags != pfm_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].flags ||
                        temp_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].qos_type != pfm_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].qos_type)
                    {
                        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UL LCG info in the TV file does not match for cell {}", cIdx);
                        return false;
                    }
                }
            }
        }
    }

    // check if temp_output_cell_info matches pfm_output_cell_info
    for (int cIdx = 0; cIdx < num_cell; cIdx++)
    {
        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[0]; idx++)
        {
            if (temp_output_cell_info[cIdx].dl_gbr_critical[idx].rnti != pfm_output_cell_info[cIdx].dl_gbr_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_gbr_critical[idx].lc_id != pfm_output_cell_info[cIdx].dl_gbr_critical[idx].lc_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - DL GBR critical LC info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[1]; idx++)
        {
            if (temp_output_cell_info[cIdx].dl_gbr_non_critical[idx].rnti != pfm_output_cell_info[cIdx].dl_gbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_gbr_non_critical[idx].lc_id != pfm_output_cell_info[cIdx].dl_gbr_non_critical[idx].lc_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - DL GBR non-critical LC info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[2]; idx++)
        {
            if (temp_output_cell_info[cIdx].dl_ngbr_critical[idx].rnti != pfm_output_cell_info[cIdx].dl_ngbr_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_ngbr_critical[idx].lc_id != pfm_output_cell_info[cIdx].dl_ngbr_critical[idx].lc_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - DL NGBR critical LC info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[3]; idx++)
        {
            if (temp_output_cell_info[cIdx].dl_ngbr_non_critical[idx].rnti != pfm_output_cell_info[cIdx].dl_ngbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_ngbr_non_critical[idx].lc_id != pfm_output_cell_info[cIdx].dl_ngbr_non_critical[idx].lc_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - DL NGBR non-critical LC info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[4]; idx++)
        {
            if (temp_output_cell_info[cIdx].dl_mbr_non_critical[idx].rnti != pfm_output_cell_info[cIdx].dl_mbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_mbr_non_critical[idx].lc_id != pfm_output_cell_info[cIdx].dl_mbr_non_critical[idx].lc_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - DL MBR non-critical LC info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[5]; idx++)
        {
            if (temp_output_cell_info[cIdx].ul_gbr_critical[idx].rnti != pfm_output_cell_info[cIdx].ul_gbr_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_gbr_critical[idx].lcg_id != pfm_output_cell_info[cIdx].ul_gbr_critical[idx].lcg_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UL GBR critical LCG info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[6]; idx++)
        {
            if (temp_output_cell_info[cIdx].ul_gbr_non_critical[idx].rnti != pfm_output_cell_info[cIdx].ul_gbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_gbr_non_critical[idx].lcg_id != pfm_output_cell_info[cIdx].ul_gbr_non_critical[idx].lcg_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UL GBR non-critical LCG info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[7]; idx++)
        {
            if (temp_output_cell_info[cIdx].ul_ngbr_critical[idx].rnti != pfm_output_cell_info[cIdx].ul_ngbr_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_ngbr_critical[idx].lcg_id != pfm_output_cell_info[cIdx].ul_ngbr_critical[idx].lcg_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UL NGBR critical LCG info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[8]; idx++)
        {
            if (temp_output_cell_info[cIdx].ul_ngbr_non_critical[idx].rnti != pfm_output_cell_info[cIdx].ul_ngbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_ngbr_non_critical[idx].lcg_id != pfm_output_cell_info[cIdx].ul_ngbr_non_critical[idx].lcg_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UL NGBR non-critical LCG info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[9]; idx++)
        {
            if (temp_output_cell_info[cIdx].ul_mbr_non_critical[idx].rnti != pfm_output_cell_info[cIdx].ul_mbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_mbr_non_critical[idx].lcg_id != pfm_output_cell_info[cIdx].ul_mbr_non_critical[idx].lcg_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UL MBR non-critical LCG info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }
    }

    NVLOGC_FMT(TAG, "PFM sorting TV file {} validated successfully", tv_name.c_str());
    return true;
}

int parse_group_tv(cumac_cp_tv_t &tv, const int cell_num)
{
    // Construct old TV filename based on cell count
    char file_path[MAX_PATH_LEN];
    const std::string old_tv_file = "TV_cumac_F08-MC-CC-" + std::to_string(cell_num) + "PC_DL.h5";
    get_full_path_file(file_path, CONFIG_CUMAC_TV_PATH, old_tv_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

    if (access(file_path, F_OK) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Old TV file not found: {}", old_tv_file.c_str());
        return -1;
    }

    NVLOGC_FMT(TAG, "Found old TV file: {} with {} cells", old_tv_file.c_str(), cell_num);

    // Parse the old TV file
    if (parse_tv_file(tv, old_tv_file) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to parse old TV file: {}", old_tv_file.c_str());
        return -1;
    }

    // Construct PFM TV filename based on cell count
    const std::string pfm_tv_file = "PFM_SORT_TV_" + std::to_string(cell_num) + "CELLS_SLOT_1000.h5";
    get_full_path_file(file_path, CONFIG_CUMAC_TV_PATH, pfm_tv_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

    if (access(file_path, F_OK) == 0)
    {
        NVLOGC_FMT(TAG, "Found PFM TV file: {}", pfm_tv_file.c_str());

        // Initialize PFM vectors
        tv.pfmCellInfo.resize(cell_num);
        std::memset(tv.pfmCellInfo.data(), 0, sizeof(cumac_pfm_cell_info_t) * cell_num);

        tv.pfmSortSol.resize(cell_num);
        std::memset(tv.pfmSortSol.data(), 0, sizeof(cumac_pfm_output_cell_info_t) * cell_num);

        // Load PFM TV data
        if (!pfm_load_tv_H5(pfm_tv_file, tv.pfmCellInfo, tv.pfmSortSol))
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to load PFM TV file: {}", pfm_tv_file.c_str());
            return -1;
        }

        NVLOGC_FMT(TAG, "Successfully loaded PFM TV file: {}", pfm_tv_file.c_str());
    }
    else
    {
        NVLOGW_FMT(TAG, "PFM TV file not found: {}, skipping PFM TV loading", pfm_tv_file.c_str());
    }

    NVLOGC_FMT(TAG, "Successfully parsed group TV files for {} cells", cell_num);
    return 0;
}