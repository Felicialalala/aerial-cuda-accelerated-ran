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

import aerial_mcore as NRSimulator
import matlab 
import numpy as np
import pdb

eng = NRSimulator.initialize()

testAlloc = {
    "dl": 0,
    "ul": 1,
    "pusch": 1
}
SysPar = eng.initSysPar(testAlloc)

SysPar['SimCtrl']['fp16AlgoSel'] = 1
SysPar['SimCtrl']['N_frame'] = 1
SysPar["carrier"]["Nant_gNB"] = 2.
SysPar["carrier"]["Nant_UE"] = 1.
SysPar["carrier"]["N_ID_CELL"] = 41

SysPar["pusch"] = [eng.cfgPusch()]
updatePusch = {
  "BWPSize": 273.0,
  "BWPStart": 0.0,
  "RNTI": 5,
  "NrOfCodewords": 1.0,
  "mcsIndex": 16,
  "mcsTable": 0,
  "rvIndex": 0,
  "newDataIndicator": 1,
  "harqProcessID": 0,
  "TransformPrecoding": 0.0,
  "dataScramblingId": 41,
  "nrOfLayers": 1.0,
  "portIdx": 0.0,
  "DmrsSymbPos": matlab.double([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
  "dmrsConfigType": 0.0,
  "DmrsScramblingId": 41,
  "DmrsMappingType": 0.0,
  "puschIdentity": 0.0,
  "SCID": 0.0,
  "numDmrsCdmGrpsNoData": 2.0,
  "resourceAlloc": 1.0,
  "rbStart": 24.,
  "rbSize": 26.,
  "VRBtoPRBMapping": 0.0,
  "FrequencyHopping": 0.0,
  "txDirectCurrentLocation": 0.0,
  "uplinkFrequencyShift7p5khz": 0.0,
  "StartSymbolIndex": 0.,
  "NrOfSymbols": 14.,
  "prcdBf": [],
  "I_LBRM": 0,
  "maxLayers": 4,
  "maxQm": 8,
  "n_PRB_LBRM": 273,
  "seed": 0.0,
  "pduBitmap": 1,
  "payload": [  ], #insert payload here
  "harqPayload": [],
  "idxUE": 0.0,
  "idxUeg": 0.0,
  "harqAckBitLength": 0.0,
  "csiPart1BitLength": 0.0,
  "csiPart1Payload": [],
  "csiPart2Payload": [],
  "alphaScaling": 3.0,
  "betaOffsetHarqAck": 0.0,
  "betaOffsetCsi1": 0.0,
  "betaOffsetCsi2": 0.0,
  "rank": 1.0,
  "rankBitOffset": 0.0,
  "rankBitSize": 2.0
}
SysPar["pusch"][0].update(updatePusch)

SysPar_out, UE, gNB = eng.nrSimulator(SysPar,nargout=3)

#pdb.set_trace()

tx_tensor = np.array(UE[0]['Phy']['tx']['Xtf'])

reStart = int(SysPar_out["pusch"][0]["rbStart"])*12
reEnd = reStart + int(SysPar_out["pusch"][0]["rbSize"])*12
startSymIndex = int(SysPar_out["pusch"][0]["StartSymbolIndex"])
stopSymIndex = startSymIndex + int(SysPar_out["pusch"][0]["NrOfSymbols"])
for sym in range(startSymIndex,stopSymIndex,1):
   if SysPar_out["pusch"][0]["DmrsSymbPos"][0][sym] == 0.0:
       print(f"1st RE of Symbol {sym} DMRS: {tx_tensor[reStart,sym]}")
   else:
       print(f"1st RE of Symbol {sym} DATA: {tx_tensor[reStart,sym]}")
