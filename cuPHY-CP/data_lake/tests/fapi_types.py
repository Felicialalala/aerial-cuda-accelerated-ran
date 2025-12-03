# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import IntEnum
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd

class FapiGroup(IntEnum):
    """FAPI message groups from fapi_group_t"""
    DL_TTI_REQ = 0
    UL_TTI_REQ = 1
    TX_DATA_REQ = 2
    UL_DCI_REQ = 3
    DL_BFW_CVI_REQ = 4
    UL_BFW_CVI_REQ = 5
    FAPI_REQ_SIZE = 6

class ChannelType(IntEnum):
    """Channel types from channel_type_t"""
    PUSCH = 0
    PDSCH = 1
    PDCCH_UL = 2
    PDCCH_DL = 3
    PBCH = 4
    PUCCH = 5
    PRACH = 6
    CSI_RS = 7
    SRS = 8
    BFW_DL = 9
    BFW_UL = 10
    CHANNEL_MAX = 11

class TvChannelType(IntEnum):
    """Test vector channel types from tv_channel_type_t"""
    TV_NONE = 0
    TV_PBCH = 1
    TV_PDCCH = 2
    TV_PDSCH = 3
    TV_CSI_RS = 4
    TV_PRACH = 6
    TV_PUCCH = 7
    TV_PUSCH = 8
    TV_SRS = 9
    TV_BFW_DL = 10
    TV_BFW_UL = 11
    TV_PDCCH_UL = 12

class TvType(IntEnum):
    """Test vector types from tv_type_t"""
    TV_GENERIC = 0
    TV_PRACH_MSG2 = 2
    TV_PRACH_MSG3 = 3
    TV_PRACH_MSG4 = 4
    TV_HARQ = 5

class BfpType(IntEnum):
    """BFP types from bfp_t"""
    BFP9 = 0
    BFP14 = 1
    BFP16 = 2

class UciPduType(IntEnum):
    """UCI PDU types from uci_pdu_type_t"""
    UCI_PDU_TYPE_PUSCH = 0
    UCI_PDU_TYPE_PF01 = 1
    UCI_PDU_TYPE_PF234 = 2
    UCI_PDU_TYPE_NUM = 3

class IndicationType(IntEnum):
    """Indication types from indication_type_t"""
    IND_PUSCH_UCI = 15
    IND_PRACH = 16
    IND_PUCCH = 17
    IND_PUSCH_DATA = 18

@dataclass
class UlMeasurement:
    """UL measurement structure from ul_measurement_t"""
    UL_CQI: int
    SNR: int
    SNR_ehq: int
    TimingAdvance: int
    TimingAdvanceNs: int
    RSSI: int
    RSSI_ehq: int
    RSRP: int
    RSRP_ehq: int

@dataclass
class ValdTolerance:
    """Validation tolerance structure from vald_tolerance_t"""
    ul_meas: List[UlMeasurement]  # 3 PDU types
    pusch_pe_noiseVardB: int

@dataclass
class CellConfig:
    """Cell configuration structure from cell_configs_t"""
    dlGridSize: int
    ulGridSize: int
    dlBandwidth: int
    ulBandwidth: int
    numTxAnt: int
    numRxAnt: int
    numRxAntSrs: int
    numTxPort: int
    numRxPort: int
    mu: int
    phyCellId: int
    dmrsTypeAPos: int
    FrameDuplexType: int
    enable_fapiv3_csi2_api: int
    ul_gain_calibration: float
    enable_codebook_BF: int
    max_amp_ul: float
    enable_dynamic_BF: int
    enable_static_dynamic_beamforming: int
    negTV_enable: int
    pusch_seg0_Tchan_start_offset: int
    pusch_seg0_Tchan_duration: int
    pusch_seg1_Tchan_start_offset: int
    pusch_seg1_Tchan_duration: int

@dataclass
class PrachFdOccasionConfig:
    """PRACH FD occasion config from prach_fd_occasion_config_t"""
    prachRootSequenceIndex: int
    numRootSequences: int
    k1: int
    prachZeroCorrConf: int
    numUnusedRootSequences: int
    unusedRootSequences: List[int]

@dataclass
class PrachConfig:
    """PRACH configuration from prach_configs_t"""
    prachSequenceLength: int
    prachSubCSpacing: int
    restrictedSetConfig: int
    numPrachFdOccasions: int
    prachConfigIndex: int
    SsbPerRach: int
    prachMultipleCarriersInABand: int
    prachFdOccasions: List[PrachFdOccasionConfig]

@dataclass
class CsiPart:
    """CSI part structure from csi_part_t"""
    Crc: int
    BitLen: int
    DetectionStatus: int
    Payload: List[int]

@dataclass
class FapiUlMeasure1002:
    """FAPI UL measurement 10.02 from fapi_ul_measure_10_02_t"""
    ul_cqi: int
    timing_advance: int
    rssi: int

@dataclass
class PucchUciInd:
    """PUCCH UCI indication from pucch_uci_ind_t"""
    idxInd: int
    idxPdu: int
    PucchFormat: int
    meas: UlMeasurement
    SRindication: int
    SRconfidenceLevel: int
    NumHarq: int
    HarqconfidenceLevel: int
    HarqValue: List[int]
    noiseVardB: int
    SrBitLen: int
    HarqCrc: int
    HarqBitLen: int
    HarqDetectionStatus: int
    csi_parts: List[CsiPart]
    SrPayload: List[int]
    HarqPayload: List[int]

@dataclass
class ComplexInt16:
    """Complex int16 from complex_int16_t"""
    re: int
    im: int

@dataclass
class PrecodingMatrix:
    """Precoding matrix from precoding_matrix_t"""
    PMidx: int
    numLayers: int
    numAntPorts: int
    precoderWeight_v: List[ComplexInt16]

@dataclass
class TxBeamformingData:
    """TX beamforming data from tx_beamforming_data_t"""
    numPRGs: int
    prgSize: int
    digBFInterfaces: int
    PMidx_v: List[int]
    beamIdx_v: List[int]

@dataclass
class RxBeamformingData:
    """RX beamforming data from rx_beamforming_data_t"""
    numPRGs: int
    prgSize: int
    digBFInterfaces: int
    beamIdx_v: List[int]

@dataclass
class RxSrsBeamformingData:
    """RX SRS beamforming data from rx_srs_beamforming_data_t"""
    numPRGs: int
    prgSize: int
    digBFInterfaces: int
    beamIdx_v: List[List[int]]  # 2D array for SCF_FAPI_10_04

@dataclass
class ChannelSegment:
    """Channel segment from channel_segment_t"""
    type: int
    channel_start_offset: int
    channel_duration: int

@dataclass
class PbchTvData:
    """PBCH TV data from pbch_tv_data_t"""
    betaPss: int
    ssbBlockIndex: int
    ssbSubcarrierOffset: int
    bchPayloadFlag: int
    physCellId: int
    SsbOffsetPointA: int
    bchPayload: int
    tx_beam_data: TxBeamformingData

@dataclass
class PbchTv:
    """PBCH TV from pbch_tv_t"""
    data: List[PbchTvData]

@dataclass
class PrachTvRef:
    """PRACH TV reference from prach_tv_ref_t"""
    numPrmb: int
    prmbIdx_v: List[int]
    delay_v: List[float]
    peak_v: List[float]

@dataclass
class PrachInd:
    """PRACH indication from prach_ind_t"""
    idxInd: int
    idxPdu: int
    SymbolIndex: int
    SlotIndex: int
    FreqIndex: int
    avgRssi: int
    avgSnr: int
    avgNoise: int
    numPreamble: int
    preambleIndex_v: List[int]
    TimingAdvance_v: List[int]
    TimingAdvanceNano_v: List[int]
    PreamblePwr_v: List[int]

@dataclass
class PrachTvData:
    """PRACH TV data from prach_tv_data_t"""
    type: int
    physCellID: int
    NumPrachOcas: int
    prachFormat: int
    numRa: int
    prachStartSymbol: int
    numCs: int
    prachPduIdx: int
    ind: PrachInd
    rx_beam_data: RxBeamformingData
    ref: PrachTvRef

@dataclass
class PrachTv:
    """PRACH TV from prach_tv_t"""
    data: List[PrachTvData]

@dataclass
class DciInfo:
    """DCI info from dciinfo_t"""
    RNTI: int
    ScramblingId: int
    ScramblingRNTI: int
    CceIndex: int
    AggregationLevel: int
    beta_PDCCH_1_0: int
    powerControlOffsetSS: int
    PayloadSizeBits: int
    Payload: List[int]
    tx_beam_data: TxBeamformingData

@dataclass
class PuschUciInd:
    """PUSCH UCI indication from pusch_uci_ind_t"""
    meas: UlMeasurement
    idxInd: int
    idxPdu: int
    isEarlyHarq: int
    HarqCrc: int
    HarqBitLen: int
    HarqDetStatus_earlyHarq: int
    HarqDetectionStatus: int
    sinrdB: int
    sinrdB_ehq: int
    postEqSinrdB: int
    noiseVardB: int
    postEqNoiseVardB: int
    csi_parts: List[CsiPart]
    HarqPayload: List[int]
    HarqPayload_earlyHarq: List[int]

@dataclass
class CsiPart2Info:
    """CSI part 2 info from csi_part2_info_t"""
    priority: int
    numPart1Params: int
    paramOffsets: List[int]
    paramSizes: List[int]
    part2SizeMapIndex: int
    part2SizeMapScope: int

@dataclass
class PuschTvData:
    """PUSCH TV data from pusch_tv_data_t"""
    tbpars: Dict  # Placeholder for tb_pars struct
    BWPSize: int
    BWPStart: int
    pduBitmap: int
    harqAckBitLength: int
    csiPart1BitLength: int
    csiPart2BitLength: int
    alphaScaling: int
    betaOffsetHarqAck: int
    betaOffsetCsi1: int
    betaOffsetCsi2: int
    harqProcessID: int
    newDataIndicator: int
    qamModOrder: int
    tbErr: int
    SubcarrierSpacing: int
    CyclicPrefix: int
    targetCodeRate: int
    FrequencyHopping: int
    txDirectCurrentLocation: int
    uplinkFrequencyShift7p5khz: int
    numDmrsCdmGrpsNoData: int
    dmrsSymLocBmsk: int
    BFP: BfpType
    TransformPrecoding: int
    puschIdentity: int
    groupOrSequenceHopping: int
    lowPaprGroupNumber: int
    lowPaprSequenceNumber: int
    numPart2s: int
    csip2_v3_parts: List[CsiPart2Info]
    data_ind: Dict  # Placeholder for pusch_data_ind_t
    uci_ind: PuschUciInd
    tb_size: int
    tb_buf: List[int]
    rx_beam_data: RxBeamformingData

@dataclass
class PuschTv:
    """PUSCH TV from pusch_tv_t"""
    data_size: int
    data_buf: List[int]
    data: List[PuschTvData]
    negTV_enable: int

@dataclass
class SrsInd0:
    """SRS indication 0 from srs_ind0_t"""
    taOffset: int
    taOffsetNs: int
    wideBandSNR: int
    prgSize: int
    numSymbols: int
    numReportedSymbols: int
    numPRGs: int

@dataclass
class SrsIqSample:
    """SRS IQ sample from srs_iq_sample_t"""
    re: float
    im: float

@dataclass
class SrsInd1:
    """SRS indication 1 from srs_ind1_t"""
    numUeSrsAntPorts: int
    numGnbAntennaElements: int
    prgSize: int
    numPRGs: int
    report_iq_data: List[SrsIqSample]

@dataclass
class SrsInd:
    """SRS indication from srs_ind_t"""
    idxInd: int
    idxPdu: int
    ind0: SrsInd0
    ind1: SrsInd1

@dataclass
class SrsFapiv4:
    """SRS FAPI v4 from srs_fapiv4_t"""
    usage: int
    numTotalUeAntennas: int
    ueAntennasInThisSrsResourceSet: int
    sampledUeAntennas: int

@dataclass
class SrsTvData:
    """SRS TV data from srs_tv_data_t"""
    type: int
    RNTI: int
    srsChestBufferIndex: int
    BWPSize: int
    BWPStart: int
    SubcarrierSpacing: int
    CyclicPrefix: int
    numAntPorts: int
    numSymbols: int
    numRepetitions: int
    timeStartPosition: int
    configIndex: int
    sequenceId: int
    bandwidthIndex: int
    combSize: int
    combOffset: int
    cyclicShift: int
    frequencyPosition: int
    frequencyShift: int
    frequencyHopping: int
    groupOrSequenceHopping: int
    resourceType: int
    Tsrs: int
    Toffset: int
    Beamforming: int
    numPRGs: int
    prgSize: int
    digBFInterfaces: int
    beamIdx: List[int]
    srsPduIdx: int
    lastSrsPdu: int
    rx_beam_data: RxSrsBeamformingData
    fapi_v4_params: SrsFapiv4
    ind: SrsInd
    SNRval: List[int]

@dataclass
class SrsTv:
    """SRS TV from srs_tv_t"""
    data: List[SrsTvData]

@dataclass
class CvMembankConfig:
    """CV membank config from cv_membank_config_t"""
    RNTI: int
    reportType: int
    nGnbAnt: int
    nPrbGrps: int
    nUeAnt: int
    startPrbGrp: int
    srsPrbGrpSize: int
    cv_samples: List[int]

@dataclass
class CvMembankConfigReq:
    """CV membank config request from cv_membank_config_req_t"""
    data: List[CvMembankConfig]

@dataclass
class BfwCvUeData:
    """BFW CV UE data from bfw_cv_ue_data_t"""
    RNTI: int
    srsChestBufferIndex: int
    pduIndex: int
    numOfUeAnt: int
    ueAntIndexes: List[int]
    gNbAntIdxStart: int
    gNbAntIdxEnd: int

@dataclass
class BfwCvData:
    """BFW CV data from bfw_cv_data_t"""
    nUes: int
    ue_grp_data: List[List[BfwCvUeData]]
    rbStart: int
    rbSize: int
    numPRGs: int
    prgSize: int
    bfwUl: int

@dataclass
class BfwCvTv:
    """BFW CV TV from bfw_cv_tv_t"""
    data: List[BfwCvData]

@dataclass
class TestVector:
    """Test vector from test_vector_t"""
    pdcch_tv: Dict  # Placeholder for pdcch_tv_t
    pdsch_tv: Dict  # Placeholder for pdsch_tv_t
    pucch_tv: Dict  # Placeholder for pucch_tv_t
    pusch_tv: PuschTv
    prach_tv: PrachTv
    pbch_tv: PbchTv
    csirs_tv: Dict  # Placeholder for csirs_tv_t
    srs_tv: SrsTv
    bfw_tv: BfwCvTv
    dset_tv: Dict[ChannelType, List[str]]

@dataclass
class FapiReq:
    """FAPI request from fapi_req_t"""
    cell_idx: int
    slot_idx: int
    tv_file: str
    channel: ChannelType
    tv_data: TestVector
    h5f: object  # Placeholder for hdf5hpp::hdf5_file*

@dataclass
class Timing:
    """Timing from timing_t"""
    ontime: int
    late: int
    early: int

@dataclass
class PdschStaticParam:
    """PDSCH static parameters from pdsch_static_param_t"""
    pduBitmap: int
    BWPSize: int
    BWPStart: int
    SubCarrierSpacing: int
    CyclicPrefix: int
    NrOfCodeWords: int
    rvIndex: int
    dataScramblingId: int
    transmission: int
    refPoint: int
    dlDmrsScrmablingId: int
    scid: int
    resourceAlloc: int
    VRBtoPRBMapping: int
    powerControlOffset: int
    powerControlOffsetSS: int
    numDmrsCdmGrpsNoData: List[int]
    rbBitmap: List[int]
    tx_beam_data: TxBeamformingData

@dataclass
class PuschStaticParam:
    """PUSCH static parameters from pusch_static_param_t"""
    pduBitmap: int
    BWPSize: int
    BWPStart: int
    SubCarrierSpacing: int
    CyclicPrefix: int
    dataScramblingId: int
    dmrsConfigType: int
    ulDmrsScramblingId: int
    puschIdentity: int
    scid: int
    resourceAlloc: int
    VRBtoPRBMapping: int
    FrequencyHopping: int
    txDirectCurrentLocation: int
    uplinkFrequencyShift7p5khz: int
    rvIndex: int
    harqProcessID: int
    newDataIndicator: int
    numCb: int
    cbPresentAndPosition: int
    numDmrsCdmGrpsNoData: List[int]
    rbBitmap: List[int]
    rx_beam_data: RxBeamformingData

@dataclass
class StaticSlotParam:
    """Static slot parameters from static_slot_param_t"""
    pdsch: PdschStaticParam
    pusch: PuschStaticParam

@dataclass
class PrbInfo:
    """PRB info from prb_info_t"""
    prbStart: int
    prbEnd: int

@dataclass
class DynPduParam:
    """Dynamic PDU parameters from dyn_pdu_param_t"""
    prb: PrbInfo
    rnti: int
    beam: int
    layer: int
    mcs_table: int
    mcs: int
    dmrs_port_bmsk: int
    dmrs_sym_loc_bmsk: int
    nrOfSymbols: int
    modulation_order: int
    target_code_rate: int
    tb_size: int

@dataclass
class DynSlotParam:
    """Dynamic slot parameters from dyn_slot_param_t"""
    ch_type: ChannelType
    pdus: List[DynPduParam]

@dataclass
class DbtMd:
    """DBT metadata from dbt_md_"""
    bf_stat_dyn_enabled: bool
    num_static_beamIdx: int
    num_TRX_beamforming: int
    dbt_data_buf: List[ComplexInt16]

@dataclass
class Csi2MapParams:
    """CSI2 map parameters from csi2_map_params_"""
    numPart1Params: int
    sizePart1Params: List[int]
    mapBitWidth: int
    map: List[int]

@dataclass
class Csi2Maps:
    """CSI2 maps from csi2_maps_"""
    nCsi2Maps: int
    mapParams: List[Csi2MapParams]
    totalSizeInBytes: int

@dataclass
class SFN_iter:
    """SFN iterator that increments modulo 1024 every nCells times it's called"""
    value: int = 0
    nCells: int = 1
    count: int = 0

    def __next__(self) -> int:
        """Return current value and increment every nCells calls"""
        current = self.value
        self.count += 1
        if self.count >= self.nCells:
            self.value = (self.value + 1) % 1024
            self.count = 0
        return current

    def __iter__(self) -> 'SFN_iter':
        """Return self as iterator"""
        return self

    def reset(self) -> None:
        """Reset the SFN value to 0"""
        self.value = 0
        self.count = 0

    def set(self, value: int) -> None:
        """Set the SFN value modulo 1024"""
        self.value = value % 1024
        self.count = 0

    def set_nCells(self, nCells: int) -> None:
        """Set the number of cells before incrementing"""
        self.nCells = nCells

def get_channel_name(channel: ChannelType) -> str:
    """Get channel name from channel type"""
    return {
        ChannelType.PUSCH: "PUSCH",
        ChannelType.PDSCH: "PDSCH",
        ChannelType.PDCCH_UL: "PDCCH_UL",
        ChannelType.PDCCH_DL: "PDCCH_DL",
        ChannelType.PBCH: "PBCH",
        ChannelType.PUCCH: "PUCCH",
        ChannelType.PRACH: "PRACH",
        ChannelType.CSI_RS: "CSI_RS",
        ChannelType.SRS: "SRS",
        ChannelType.BFW_DL: "BFW_DL",
        ChannelType.BFW_UL: "BFW_UL",
        ChannelType.CHANNEL_MAX: "INVALID"
    }.get(channel, "INVALID")

def get_channel_type(channel_name: str) -> ChannelType:
    """Get channel type from channel name"""
    for ch in ChannelType:
        if get_channel_name(ch) == channel_name:
            return ch
    return ChannelType.CHANNEL_MAX

def get_tv_channel_type(channel: ChannelType) -> TvChannelType:
    """Get TV channel type from channel type"""
    return {
        ChannelType.PUSCH: TvChannelType.TV_PUSCH,
        ChannelType.PDSCH: TvChannelType.TV_PDSCH,
        ChannelType.PDCCH_UL: TvChannelType.TV_PDCCH_UL,
        ChannelType.PDCCH_DL: TvChannelType.TV_PDCCH,
        ChannelType.PBCH: TvChannelType.TV_PBCH,
        ChannelType.PUCCH: TvChannelType.TV_PUCCH,
        ChannelType.PRACH: TvChannelType.TV_PRACH,
        ChannelType.CSI_RS: TvChannelType.TV_CSI_RS,
        ChannelType.SRS: TvChannelType.TV_SRS,
        ChannelType.BFW_DL: TvChannelType.TV_BFW_DL,
        ChannelType.BFW_UL: TvChannelType.TV_BFW_UL,
    }.get(channel, TvChannelType.TV_NONE)

@dataclass
class PuschReq:
    RNTI: int
    BWPSize: int
    BWPStart: int
    SubcarrierSpacing: int
    CyclicPrefix: int
    targetCodeRate: int
    qamModOrder: int
    mcsIndex: int
    mcsTable: int
    TransformPrecoding: int
    dataScramblingId: int
    nrOfLayers: int
    DmrsSymbPos: int
    dmrsConfigType: int
    DmrsScramblingId: int
    puschIdentity: int
    SCID: int
    numDmrsCdmGrpsNoData: int
    dmrsPorts: int
    resourceAlloc: int
    rbStart: int
    rbSize: int
    VRBtoPRBMapping: int
    FrequencyHopping: int
    txDirectCurrentLocation: int
    uplinkFrequencyShift7p5khz: int
    StartSymbolIndex: int
    NrOfSymbols: int
    rvIndex: int
    harqProcessID: int
    newDataIndicator: int
    TBSize: int
    numCb: int
    numPRGs: int
    prgSize: int

@dataclass
class PuschInd:
    TbCrcStatus: int
    sinrdB: float
    TimingAdvanceNano: float
    RSSI: float

@dataclass
class PDU:
    """PDU data structure"""
    type: int  # Type identifier
    index: int  # Index of the PDU
    payload: List[int]  # PDU payload data

@dataclass
class IND:
    idxPdu: int
    type: int
    data: Dict[str, Any] 

@dataclass
class CellSlotConfig:
    cell_index: int
    channels: List[str]
    pdus: List[PDU]
    inds: List[IND]

@dataclass
class SlotConfig:
    slot: int
    total_pusch_pdus: int
    dataframe: pd.DataFrame
    x_tf: np.ndarray  # Add field for X_tf data