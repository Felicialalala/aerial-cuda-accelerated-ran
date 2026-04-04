# Aerial 平台候选状态字段核对表

## 1. 文档目的

本文基于当前仓库代码，对你给出的候选状态字段做一次“平台能力 vs 当前 Stage-B 主线”核对，避免把：

- 平台内部确实存在的中间量
- API 里预留但当前主线默认未填充的字段
- 只能后处理构造的派生量

混成同一类。

结论优先以当前仓库代码为准，时间点为 `2026-04-03`。

## 2. 统一 shape 口径

### 2.1 统一记号

| 记号 | 本文统一含义 | 在 Aerial/cuMAC 中的对应字段 |
| --- | --- | --- |
| `B_coord` | 参与协同优化的中心簇小区数 | `cumacCellGrpPrms::nCell` |
| `B_total` | 实际参与干扰/链路计算的总小区数 | `cumacCellGrpPrms::totNumCell` |
| `U_b` | 小区 `b` 的 active UE 数 | 不是原生 ragged 维；通过 `cellAssocActUe` 从 `nActiveUe` 扁平索引恢复 |
| `U_tot` | 所有 coordinated UE 总数 | `cumacCellGrpPrms::nActiveUe` |
| `M_sched` | 调度粒度数 | 当前 Stage-B 为 `PRG/RBG`，对应 `cumacCellGrpPrms::nPrbGrp`，不是 `RB` |
| `Nt` | 发射天线数 | DL 主线下对应 `nBsAnt` |
| `Nr` | 接收天线数 | DL 主线下对应 `nUeAnt` |
| `F_node` | 聚合后小区级特征维度 | 当前 online bridge 的 `cellFeatDim = 5` |
| `F_edge` | 聚合后小区间边特征维度 | 当前 online bridge 的 `edgeFeatDim = 2` |

### 2.2 当前 Stage-B 主线已验证事实

1. `M_sched` 当前是 `PRG/RBG`，不是 `RB`。默认 `68 PRG × 4 PRB`，对应 [`cuMAC/examples/parameters.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/parameters.h) 与 [`Doc/current_stageB_effective_configuration.md`](/home/oai2/aerial-cuda-accelerated-ran/Doc/current_stageB_effective_configuration.md)。
2. 平台 API 支持 `B_total != B_coord`，但当前 online PPO / BC 主线要求 `n_tot_cell == n_cell`，见 [`training/gnnrl/ppo_online_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/ppo_online_train.py) 与 [`training/gnnrl/bc_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/bc_train.py)。
3. 当前 online bridge 实际维度是：
   - `cellFeatDim = 5`
   - `ueFeatDim = 12`
   - `edgeFeatDim = 2`
   - `prgFeatDim = 4`
   见 [`cuMAC/examples/onlineTrainBridge/OnlineObservationTypes.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/onlineTrainBridge/OnlineObservationTypes.h)。
4. 当前 Stage-B 默认 `4T4R`，且 [`cuMAC/examples/parameters.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/parameters.h) 里 `prdSchemeConst = 0`，即默认 `no precoding`；平台能力上仍支持 `SVD precoding`。
5. 当前 `4T4R Type-0` 主线每个 `cell × PRG` 只落一个 slot/UE，不存在“同小区同 PRG 多用户复用”的原生语义。

### 2.3 原生内存布局提醒

很多量在代码里不是 `[B_total, U_b, ...]` 这种 ragged 形式，而是：

- 一个全局扁平 `nActiveUe`
- 再配一个 `cellAssocActUe[cIdx * nActiveUe + uIdx]`

来恢复“某个 UE 属于哪个 cell”。

因此：

- 你的算法文档里写 `[B_total, U_b, ...]` 没问题
- 但接代码时通常要先从扁平 `uIdx` + `cellAssocActUe` 做映射

## 3. 核对结果总览

### 3.1 当前可直接利用的强字段

- `estH_fr_actUe` / `srsEstChan`
- `postEqSinr` / `wbSinr`
- `bufferSize`
- `tbErrLastActUe`
- `newDataActUe`
- `mcsSelSol`
- `cellAssocActUe`

### 3.2 API 里有，但当前 Stage-B 主线默认不填或不稳定的字段

- `cqiActUe`
- `riActUe`
- `pmiActUe`
- `noiseVarActUe`
- `bsTxPow`

### 3.3 当前没有原生字段、需要后处理或重新定义的量

- 显式 `ICI`
- `Intra-cell interference`
- `CoMP indicator`
- `Beam overlap`
- `PHR`

## 4. 节点核对表

说明：

- “是否必须 hook”指的是：如果你想把该量送进当前 online bridge / replay / RL state，是否要新增导出逻辑。
- “结论”使用以下口径：
  - `存在`
  - `可推导`
  - `需 hook`
  - `需重定义`
  - `暂不采用`

| 节点名 | 应有 shape | 实际变量名 | 所在文件 / 类 / 函数 | 实际 shape | 是否必须 hook | 结论 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Channel Coeff | 理想 `[B_total, U_b, M_sched, Nr, Nt]` 或等价布局 | `cellGrpPrms->estH_fr_actUe`； serving-cell 子集为 `estH_fr_actUe_prd`；64TR 为 `cellGrpPrms->srsEstChan` | `cuMAC/src/api.h::cumacCellGrpPrms`；`cuMAC/examples/network.cu::createAPI/genChann4TrKernel`；`cuMAC/examples/multiCellMuMimoScheduler/mMimoNetwork.cu::genChan64TrKernel` | 4TR: 扁平等价 `[nPrbGrp, nActiveUe, totNumCell, nUeAnt, nBsAnt]`，内层 `eIdx = bsAntIdx*nUeAnt + ueAntIdx`； serving-cell-only 为 `[nActiveUe, nPrbGrp, nUeAnt, nBsAnt]`；64TR `srsEstChan[cIdx]` 等价 `[U_local, nPrbGrp, nUeAnt, nBsAnt]` | 是 | 存在 | 这是最关键原始量。平台内部有，若要进入 RL 状态需自己导出。 |
| Tx Power | 常见 `[B_total]` / `[B_total, M_sched]` | 4TR: `Pt_Rbg`、`Pt_rbgAnt`、`netData->bsTxPower_perAntPrg`；64TR API 另有 `bsTxPow` | `cuMAC/src/api.h::cumacCellGrpPrms`；`cuMAC/examples/network.cu::network`；`cuMAC/examples/multiCellMuMimoScheduler/mMimoNetwork.cu` | 当前 4TR Stage-B 实际是全局等功率标量，不是 per-cell/per-PRG 动态数组；64TR API 预留 `bsTxPow[nCell]` | 否 | 存在 | 先区分“总功率”“每 PRG 功率”“每天线每 PRG 功率”。当前 4TR 主线没有 per-PRG power control。 |
| Noise Floor | 标量 `[]` 或 `[M_sched]` | `sigmaSqrdConst`、`netData->noiseFloor`、`cellGrpPrms->sigmaSqrd` | `cuMAC/examples/parameters.h`；`cuMAC/examples/network.cu::network`；`cuMAC/src/api.h::cumacCellGrpPrms` | 物理口径是标量；`sigmaSqrdDBmConst` 已按 `bandwidthRBGConst` 折算到单 PRG 带宽；运行时 `sigmaSqrd` 还会被缩放归一化 | 否 | 存在 | 当前代码里真正用于 SINR kernel 的常是归一化后的 `sigmaSqrd`，不一定等于物理噪声功率。 |
| RSRP | 理想 `[B_total, U_b, M_sched]` | 最接近的现成量是 `netData->rxSigPowDB`；另有 TV 检查名 `rsrpCurrTx/rsrpLastTx` 但主线未见生成器 | `cuMAC/examples/network.cu::genLSFading`；`cuMAC/examples/multiCellMuMimoScheduler/multiCellMuMimoScheduler.h` | 当前主线现成的是 `[totNumCell, nActiveUe]` 的大尺度接收功率 proxy（dBm），无 PRG 维 | 是 | 可推导 | 不是严格 3GPP RSRP。若你只要“服务/邻区信号强度”，`rxSigPowDB` 可作 proxy；若要更严格定义需由信道和功率再推。 |
| Intra-cell Interference | 若存在，多为 `[B_total, U_b, M_sched]` | 无原生字段 | 4TR `type-0` 调度由 `allocSol[prgIdx * stride + cIdx]` 决定，每个 `cell × PRG` 只有一个 slot/UE | 无 | 否 | 暂不采用 | 当前 4TR Stage-B `Type-0` 不支持同 PRG 同小区多用户复用，原生语义下应视为 `0` 或不定义。 |
| ICI | 常见 `[B_coord, U_b, M_sched]` 或聚合 `[B_coord, U_b]` | 无导出字段；内核内部有 interference covariance `CMat` | `cuMAC/src/tools/multiCellSinrCal.cu::multiCellSinrCalKernel_*` | 未导出；内核里是每个 `UE × PRG` 的干扰加噪协方差矩阵，而不是标量数组 | 是 | 需 hook | 若想拿“精确 ICI”，建议在 `multiCellSinrCal` 内核或等价 scheduler kernel 里导出；否则只能由 `SINR + 信号功率` 近似反推。 |
| Composite SINR | 常见 `[B_coord, U_b, M_sched]` / `[B_coord, U_b]` | `cellGrpPrms->postEqSinr`、`cellGrpPrms->wbSinr`；64TR 另有 `srsWbSnr` | `cuMAC/src/api.h::cumacCellGrpPrms`；`cuMAC/src/tools/multiCellSinrCal.cu`；`cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h` | 4TR: `postEqSinr` 为 `[nActiveUe, nPrbGrp, nUeAnt]`，`wbSinr` 为 `[nActiveUe, nUeAnt]`；64TR 常用 wideband `wbSinr[nActiveUe, nUeAnt]` 或 `srsWbSnr[nActiveUe]` | 否 | 存在 | 这是当前最值得直接复用的字段之一。先明确你要的是 PRG 级还是宽带级。 |
| Cross-cell Channel | 理想 `[B_src, B_dst, U_b, M_sched, Nr, Nt]` | 4TR: `estH_fr_actUe` 的 cell 维；64TR: `srsEstChan` | `cuMAC/examples/network.cu::genChann4TrKernel`；`cuMAC/examples/multiCellMuMimoScheduler/mMimoNetwork.cu::genChan64TrKernel` | 4TR 当前例程中等价 `[PRG, UE, totNumCell, nUeAnt, nBsAnt]`，其中 `totNumCell` 就是跨小区链路维；64TR 为每 cell 一份指针数组 | 是 | 存在 | 文档里的 `B·K` 只能是后续聚合特征，真正可用原始量是多小区链路矩阵。 |
| CoMP Indicator | 更合理 `[B_coord, B_coord]` 或边列表 `[E]` | 无原生字段 | 当前 online bridge 只有 `edgeAttr = [src_load_ratio, load_diff_ratio]` | 无 | 否 | 需重定义 | 更适合作为后处理边标签，例如由干扰比、几何邻接、共享 PRG 强度或跨小区信道统计构造。 |
| CQI | 常见 `[B_coord, U_b]` 或 `[B_coord, U_b, M_sched]` | `cellGrpUeStatus->cqiActUe` | `cuMAC/src/api.h::cumacCellGrpUeStatus`；`cuMAC/src/4T4R/mcsSelectionLUT.cu`；`cuMAC/examples/multiCellMuMimoScheduler/mMimoNetwork.cu` | 当前可见语义是 wideband `[nActiveUe]`，不是 PRG 级 | 视路径而定 | 存在 | 4TR 当前 Stage-B 默认未分配；64TR 仅在 `mcsSelCqi == 1` 时分配。若为空，MCS 走 `wbSinr` 路径。 |
| RI | 常见 `[B_coord, U_b]` | `cellGrpUeStatus->riActUe` | `cuMAC/src/api.h::cumacCellGrpUeStatus`；`cuMAC/src/4T4R/multiCellLayerSel.cu`；`cuMAC/examples/multiCellMuMimoScheduler/mMimoNetwork.cu` | `[nActiveUe]` | 视路径而定 | 存在 | 当前 4TR Stage-B 默认未分配；64TR 在 RI-based SU layer selection 打开时分配。不是 PRG 级。 |
| PMI | 文档常写 `[B_coord, U_b]`，但真实依赖完整信道 | API 仅预留 `cellGrpUeStatus->pmiActUe`，当前仓库未见主线分配/使用 | `cuMAC/src/api.h::cumacCellGrpUeStatus` | 若用户自行填充则是 `[nActiveUe]`；当前主线默认不存在有效内容 | 是 | 暂不采用 | 当前平台主线 precoding 是 `no precoding` 或 `SVD precoding`，不是 codebook PMI 驱动。把 PMI 当现成状态会误导。 |
| MCS | 常见 `[B_coord, U_b]` 或 `[B_coord, U_b, M_sched]` | `schdSol->mcsSelSol`；上一发射用 `cellGrpUeStatus->mcsSelSolLastTx` | `cuMAC/src/api.h::cumacSchdSol`；`cuMAC/src/4T4R/mcsSelectionLUT.cu` | 当前 4TR `Type-0` 常见是 `[nUe]`，即“每个 scheduled UE / TB 一个 MCS”；64TR 常见 `[nActiveUe]` | 否 | 存在 | 很容易拿到，但不是每 PRG 一个 MCS。 |
| Beam Overlap | 更合理 `[U_tot, U_tot]` 或边列表 | 无原生字段；可利用 `prdMat_actUe` / `prdMat_asim` / `srsEstChan` / 64TR beamforming weights 派生 | `cuMAC/src/api.h::cumacCellGrpPrms`；`cuMAC/src/4T4R/svdPrecoding.cu`；`cuMAC/src/64T64R/multiCellBeamform.cu` | 无直接数组 | 是 | 需重定义 | 更适合作为 edge feature，不建议首版把它当必备节点。 |
| HARQ | 常见 `[B_coord, U_b, n_harq_proc]`，可聚合到 `[B_coord, U_b]` | `newDataActUe`、`tbErrLastActUe`、`allocSolLastTx`、`mcsSelSolLastTx`、`layerSelSolLastTx` | `cuMAC/src/api.h::cumacCellGrpUeStatus`；`cuMAC/examples/network.cu::createAPI`；`cuMAC/src/4T4R/mcsSelectionLUT.cu` | 当前主线主要是 `[nActiveUe]` 的“上一发射摘要”，加若干 lastTx 数组；不是多 HARQ process 结构 | 否 | 存在 | 但当前 4TR `Type-0` 主线默认 `HARQ disabled`，且 RR 文档也明确 `type-0 + HARQ` 不支持。 |
| BSR | 常见 `[B_coord, U_b]` | `cellGrpUeStatus->bufferSize` | `cuMAC/src/api.h::cumacCellGrpUeStatus`；`cuMAC/examples/network.cu::createAPI`；`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp` | `[nActiveUe]`，单位 `bytes` | 否 | 存在 | 不是 3GPP LCG 级标准 BSR，但作为 backlog / queue bytes proxy 非常实用。 |
| PHR | 文档写 `B·K`，但需先核语义 | 无原生字段；最接近的是系统侧 `Pt_Rbg` / `Pt_rbgAnt` / 64TR `bsTxPow` | `cuMAC/src/api.h::cumacCellGrpPrms`；`cuMAC/examples/network.cu::network` | 无 UE 级 PHR 数组 | 否 | 需重定义 | 当前仓库没有可直接对应 3GPP UE PHR 的字段；若你想表达“剩余功率预算”，那是新定义，不应直接沿用 PHR 名称。 |

## 5. 当前 online bridge 已暴露聚合特征

如果你问的是“当前 online RL 不加 hook 直接能拿到什么”，答案不是上表的全部，而是下面这些聚合特征：

当前 `joint` 模型前向实际接收的是：

- `obs_cell_features: [n_cell, 5]`
- `obs_ue_features: [n_active_ue, 12]`
- `obs_prg_features: [n_cell, n_prg, 4]`
- `obs_edge_index: [n_edges, 2]`
- `obs_edge_attr: [n_edges, 2]`
- `action_mask_cell_ue: [n_cell, n_active_ue]`

其中：

- `obs_*_features` 是数值特征
- `obs_edge_index` 是小区间完全有向图的拓扑索引，不是数值特征
- `action_mask_cell_ue` 是结构输入，用来表达真实 `cell -> UE` 合法关联，并在 UE 融合/分类时屏蔽非法类别

### 5.1 Cell-level

`F_node = 5`

来自 [`cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp)：

1. `cell_load_bytes`
2. `active_ue_count`
3. `mean_wb_sinr_lin`
4. `mean_avg_rate_mbps`
5. `tb_err_rate`

### 5.2 UE-level

`ueFeatDim = 12`

前 8 维来自 [`cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp)：

1. `buffer_bytes`
2. `avg_rate_mbps`
3. `wb_sinr_lin`
4. `cqi`
5. `ri`
6. `tbErrLastActUe`
7. `newDataActUe`
8. `staleSlots`

后 4 维来自 [`cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h)：

9. `hol_delay_ms`
10. `ttl_slack_ms`
11. `recent_scheduled_ratio`
12. `recent_goodput_deficit_norm`

这 4 维已经把 TTL / 近期服务状态 / 同小区相对欠账信息显式接进当前 online 主线。

### 5.3 PRG-level

`prgFeatDim = 4`

来自 [`cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h)：

1. `top1SinrDb`
2. `top2GapDb`
3. `prevPrgAssigned`
4. `reuseRatio`

口径上分别表示：

- 当前 `cell × PRG` 上最强关联 UE 的 subband SINR
- 第 1 名与第 2 名 SINR 的差值
- 上一个 TTI 该 `cell × PRG` 是否实际被分配
- 上一个 TTI 同一 `PRG` 在多少个小区同时被用到，按 `reuse_count / n_cell` 归一化

### 5.4 Edge-level

`F_edge = 2`

1. `src_load_ratio`
2. `load_diff_ratio`

这说明如果你的首版算法只需要“小区图 + PRG 局部质量/复用状态”，其实不一定要第一天就把原始 CFR 全部拉出来。

## 6. 建议的首版取数优先级

### 6.1 首版建议直接使用

- `cellAssocActUe`
- `bufferSize`
- `postEqSinr`
- `wbSinr`
- `mcsSelSol`
- `tbErrLastActUe`
- `newDataActUe`
- `rxSigPowDB`（若你接受它只是 signal-strength proxy）

### 6.2 若算法强依赖，需要尽快 hook

- `estH_fr_actUe` / `estH_fr_actUe_prd`
- `srsEstChan`
- `multiCellSinrCal` 内部的显式 ICI / 干扰协方差

### 6.3 不建议首版硬塞进状态

- `Intra-cell interference`
- `PMI`
- `Beam overlap`
- `CoMP indicator`
- `PHR`

## 7. 一页结论

1. 当前平台里最有价值的原始量不是 `CQI/PMI/PHR` 这一类“名字很像标准字段”的变量，而是：
   - `estH_fr_actUe`
   - `srsEstChan`
   - `postEqSinr`
   - `wbSinr`
   - `bufferSize`
   - `tbErrLastActUe`
   - `newDataActUe`
2. 你表里最关键、也最值得优先核对的两个节点：
   - `Channel Coeff`
   - `Composite SINR`
   当前都能在平台内部拿到。
3. `ICI` 在当前仓库里没有现成导出字段，但确实在 SINR kernel 内部被显式计算；如果你的新算法强依赖显式干扰节点，优先在 `multiCellSinrCal` 附近加 hook，而不是先去找“有没有现成 CSV/struct 字段”。
4. 当前 Stage-B 4T4R `Type-0` 场景下：
   - `M_sched = PRG`
   - `B_total = B_coord`（当前在线主线）
   - 同 PRG 同小区多用户复用不成立
   - 因而 `Intra-cell interference`、`PMI`、`PHR` 这类量不应机械照搬文档公式。
