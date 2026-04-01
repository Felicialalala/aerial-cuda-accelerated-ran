# Aerial 平台天线方向性与 MIMO Precoding 支持梳理

## 1. 文档目的与范围

本文从平台整体能力出发，梳理 Aerial 仿真平台中与以下主题相关的支持情况：

- 天线方向性是否支持
- 方向性支持落在哪一层实现
- 不同信道模型下方向性能力有何差异
- 不同天线配置下，MIMO 的 precoding 或 beamforming 如何实现

这里讨论的平台范围包括：

- `testBenches/chanModels` 中的 3GPP 38.901 信道模型能力
- `cuMAC` 中的 4T4R SU-MIMO 与 64T64R MU-MIMO 调度/PHY 抽象能力
- `cuPHY-CP/ru-emulator` 与 TV/HDF5 路径中的 beamforming/precoding 接口能力

本文刻意不以当前某一条 StageB 基线实验为结论依据，而是以仓库整体代码能力为主线。

## 2. 总体结论

平台整体对天线方向性的支持可以分为三层理解：

1. 扇区级方向性
   `cuMAC/examples/network.cu` 中的大尺度衰落生成会根据小区扇区朝向、水平偏角、下倾角计算天线增益，再叠加 pathloss 和 shadow fading。这意味着即使在内部 Rayleigh 链路里，也不是“完全无方向性”，而是具备扇区级方向性增益。

2. 阵列/面板级方向性
   `testBenches/chanModels` 与 `CDL` 路径支持更完整的阵列几何、极化、间距、方向图和移动方向建模；这才是更接近“阵列方向图”和“波束相关”语义的方向性支持。

3. Precoding / Beamforming
   平台并不是只有一种 MIMO 实现方式：
   - 4T4R SU-MIMO 路径支持 `no precoding` 与 `SVD precoding`
   - 64T64R MU-MIMO 路径支持基于 `SRS` 信道估计的 `RZF` beamforming
   - 另外还支持从外部输入 SVD precoder / detector / singular values，以及 RU 侧 beam ID / DBT 的验证链路

因此，对外表述时更准确的说法应是：

`Aerial 平台整体支持扇区级方向性、3GPP 阵列/面板级方向图建模、4T4R SU-MIMO 的 SVD precoding、64T64R MU-MIMO 的 SRS 驱动 RZF beamforming，以及外部 precoder/beamforming 权重输入与 RU 侧 beamforming 配置验证。`

同时需要单独强调一点：

`平台整体` 并不等于 `cuMAC 当前主调度实现`。从底层信道模型和天线面板配置能力看，平台并不只限于 `4T4R` 和 `64T64R`；但从 `cuMAC` 当前公开的主调度、layer selection、precoding/beamforming 实现看，产品化主链路主要围绕 `4T4R SU-MIMO` 和 `64T64R MU-MIMO` 两档展开。

## 3. 天线方向性支持分层

### 3.1 `cuMAC` 网络层的大尺度方向性

在 `cuMAC/examples/network.cu` 的 `genLSFading()` 中，平台会基于以下量计算 BS 侧方向性增益：

- 小区扇区朝向 `sector orientation`
- UE 相对 BS 的水平角 `phi`
- UE 相对 BS 的俯仰角 `theta`
- BS 下倾角 `down tilt`
- 水平、垂直方向的天线衰减

随后得到：

- `antGain = GEmax + antAtten`
- `rxSigPowDB = TxPower_perAntPrg + antGain - PL - SF`

这说明平台在大尺度层面已经显式考虑了扇区方向性，而不是只做纯距离损耗。

这一层方向性的特点是：

- 属于扇区级/宏站方向性
- 作用于接收功率和链路预算
- 适用于内部 Rayleigh、TDL、CDL 链路前的功率缩放
- 不等价于完整阵列响应、码本波束或逐路径方向图

### 3.2 `chanModels` 的面板/阵列方向性

`testBenches/chanModels` 是平台中更完整的天线方向性建模入口。`AntPanelConfig` 提供：

- `nAnt`
- `antSize`
- `antSpacing`
- `antTheta`
- `antPhi`
- `antPolarAngles`
- `antModel`

其中 `antModel` 的语义是：

- `0`: isotropic
- `1`: directional
- `2`: direct pattern

这意味着平台整体并不局限于“固定的 3GPP 方向性模板”，还支持直接输入方向图数组：

- `antTheta[181]`
- `antPhi[360]`

对应地，`antenna_config_reader.hpp` 会在 `antModel == 2` 时读取显式方向图；否则将方向图数组置零，并走各向同性或内建方向性模型。

### 3.3 TDL 与 CDL 的方向性能力差异

平台在链路级信道模型上至少区分两类：

- `TDL`
- `CDL`

两者的方向性语义并不相同。

#### TDL

`TDL` 支持多天线维度，例如：

- `nBsAnt`
- `nUeAnt`
- delay profile
- Doppler

但 `tdl_chan.cuh` 里明确写了：

`no additional MIMO antenna correlations are added`

这说明 TDL 路径虽然支持多天线矩阵维度，但不具备与 CDL 同等级的阵列方向图/额外天线相关性表达能力。换句话说，TDL 更偏向“带多天线维度的简化链路级快衰落”。

#### CDL

`CDL` 路径会显式接收并使用以下天线面板参数：

- `bsAntSize`
- `bsAntSpacing`
- `bsAntPolarAngles`
- `bsAntPattern`
- `ueAntSize`
- `ueAntSpacing`
- `ueAntPolarAngles`
- `ueAntPattern`
- `vDirection`

因此，若从“完整阵列方向性支持”角度描述，`CDL` 才是平台中最完整、最直接的方向图与阵列几何承载路径。

### 3.4 不同链路下的方向性支持矩阵

| 链路/模型 | 是否有方向性 | 方向性层级 | 主要特点 |
| --- | --- | --- | --- |
| 内部 Rayleigh | 有 | 扇区级大尺度方向性 | 依赖 `genLSFading()` 的扇区/下倾角增益；小尺度快衰落本身不带显式阵列方向图 |
| TDL | 有，但有限 | 大尺度方向性 + 多天线维度 | 支持多天线和多径/Doppler，但未引入额外 MIMO 天线相关性 |
| CDL | 有，最完整 | 阵列/面板级方向性 | 支持阵列尺寸、间距、极化、方向图、移动方向 |
| SLS/统计信道模型 | 有 | 系统级 + 面板级方向性 | `antenna_panels` 可配置 `ant_model`、阵列尺寸、间距、极化等 |

## 4. 不同天线配置下的 MIMO 与 Precoding 实现

### 4.1 4T4R SU-MIMO

### 4.1.1 支持的 precoding 类型

在 `cuMAC/src/api.h` 中，`precodingScheme` 当前定义为：

- `0`: no precoding
- `1`: SVD precoding

这意味着 4T4R 路径下，平台并不是默认只能做 precoding，而是显式支持“不开 precoder”和“使用 SVD precoder”两种模式。

### 4.1.2 SVD precoding 的实现方式

`cuMAC/src/4T4R/svdPrecoding.cu` 的实现逻辑是：

1. 按链路方向确定矩阵尺寸
   - DL: `M = nUeAnt`, `N = nBsAnt`
   - UL: `M = nBsAnt`, `N = nUeAnt`
2. 以 `nActiveUe * nPrbGrp` 为 batch 大小
3. 对每个 `UE x PRG` 的信道矩阵做 batched SVD
4. 调用 `cusolverDnCgesvdjBatched`
5. 输出三类结果：
   - `sinVal_actUe`
   - `detMat_actUe`
   - `prdMat_actUe`

从实现语义上看，4T4R 的 precoding 是：

- 基于瞬时 CFR 信道矩阵
- 逐 UE、逐 PRG 的 SVD
- `V` 矩阵作为 precoder
- `U` 矩阵作为 detector 侧信息
- singular values 用于后续 layer selection 与 SINR 计算

这条路径更接近“基于信道分解的 SU-MIMO precoding”，而不是 NR 码本 PMI 驱动的 precoding。

### 4.1.3 Layer selection 的实现方式

4T4R 的 layer selection 并不是固定层数，而是和信道分解结果绑定。

`cuMAC/src/4T4R/multiCellLayerSel.cu` 中提供了几种典型方式：

- Type-0 分配下，基于每个 PRG 的 singular value 与 `sinValThr` 比较，并取所有已分配 PRG 中的最小层数
- Type-1 分配下，按已分配 PRG 的层数做平均
- 也支持基于 `RI` 的 layer selection 路径

因此，4T4R 下“precoding”和“层数选择”是耦合的：

- 先得到每个 PRG 的信道 SVD
- 再根据 singular values 或 RI 选择层数
- 最后进入 SINR 与 MCS 选择

### 4.1.4 4T4R 的 SINR 计算约束

`cuMAC/src/tools/multiCellSinrCal.cu` 中对 4T4R 的支持边界很明确：

- DL 支持
  - `no precoding + MMSE-IRC`
  - `SVD precoding + MMSE-IRC`
- SVD precoding 的 SINR 计算只支持 column-major 信道访问
- UL 的 SINR 计算仅支持 SVD precoding

所以从平台角度看，4T4R 的完整 SU-MIMO 抽象链路是：

`channel -> SVD -> layer selection -> MMSE-IRC post-eq SINR -> MCS`

### 4.1.5 4T4R 路径的适用表述

如果要对外描述 4T4R 能力，更合适的说法是：

`平台支持 4T4R SU-MIMO，precoding 采用基于 UE/PRG 瞬时信道矩阵的 batched SVD，实现每 PRG 的 precoder、detector 与 singular value 计算，并基于其完成层数选择与 SINR/MCS 抽象。`

### 4.2 64T64R MU-MIMO

### 4.2.1 支持的总体链路

64T64R 路径是独立于 4T4R SVD 路径的一条实现链，典型流程见 `cuMAC/examples/multiCellMuMimoScheduler/main.cpp`：

1. 生成或加载信道
2. `multiCellMuUeSort`
3. `multiCellMuUeGrp`
4. `multiCellBeamform`
5. `mcsSelectionLUT`

这说明 64T64R 的重点不是“单用户信道分解”，而是：

- 先筛 UE
- 再做 MU 分组
- 最后做组级 beamforming

### 4.2.2 64T64R 的方向性与信道配置入口

`cuMAC/examples/multiCellMuMimoScheduler/mMimoNetwork.h` 与 `mMimoNetwork.cu` 表明：

- `fading_type = 0` 使用内部 Rayleigh
- `fading_type = 1` 使用 SLS 统计信道模型

当选择 `fading_type = 1` 时，YAML 中的嵌入式 `channel_config` 会进一步解析：

- `system_level`
- `link_level`
- `simulation`
- `antenna_panels`

其中 `antenna_panels` 可分别配置 BS/UE 的：

- `n_ant`
- `ant_model`
- `ant_size`
- `ant_spacing`
- `ant_polar_angles`

这意味着 64T64R 路径在 `SLS` 模式下，可以把：

- system-level 场景参数
- link-level fast fading 配置
- antenna panel 方向性配置

统一接入同一条仿真链路，而不只是单独替换一个快衰落矩阵。

因此，64T64R 路径不仅支持大阵列本身，也支持通过 SLS 配置把方向性面板参数真正接入仿真链路。

### 4.2.3 MU 用户筛选与分组前提

64T64R 的 beamforming 不是对任意 UE 集合直接做，而是先经过 CSI 可行性筛选。

`multiCellMuUeSort.cu` 中，平台用 `srsWbSnr >= srsSnrThr` 判断 UE 是否具备 MU-MIMO 可行性；若可行，会在权重计算中用 `muCoeff` 提升其调度优先级。

`multiCellMuUeGrp.cu` 中，平台会计算候选 UE 之间的信道相关性：

- 求归一化相关值 `corrVal`
- 若 `corrVal > chanCorrThr`，则认为正交性不足，不适合进入同组 MU 传输

因此，64T64R 的 precoding 前置条件不是“有天线就做”，而是：

- 要有可用的 SRS CSI
- 要满足 SRS SNR 门限
- 要满足组内相关性门限

### 4.2.4 Beamforming 的实现方式

`cuMAC/src/64T64R/multiCellBeamform.cu` 明确了这条路径的边界：

- 不支持 4T4R
- 不支持 type-0 allocation
- 只支持 DL
- `bfPowAllocScheme` 目前只支持 `0` 和 `1`

其核心算法是典型的 `RZF`：

1. 从 `srsEstChan` 中取出组内所有 UE/层的信道，拼成 `stackedChann`
2. 计算 regularized Gram matrix
   - `HH' + zfCoeff * I`
3. 求逆
   - `(HH' + zfCoeff * I)^(-1)`
4. 计算 beamformer
   - `W = H' (HH' + zfCoeff * I)^(-1)`
5. 按 `bfPowAllocScheme` 做功率归一化
6. 将结果写入 `prdMat`

因此，64T64R 路径中的 precoding/beamforming 更准确地应称为：

`基于 SRS CSI 的 RZF beamforming`

而不是 4T4R 路径那种基于单 UE SVD 的 precoding。

### 4.2.5 功率归一化方式

`bfPowAllocScheme` 目前支持：

- `0`: equal power allocation for RX side received per-layer beamforming gain
- `1`: equal power allocation for TX side per-layer beamforming gain

因此，64T64R 的 beamforming 不仅决定空间权重，还把功率归一化策略编码到实现里。

### 4.2.6 无 SRS 时的降级行为

`multiCellBeamform.cu` 还实现了一个重要降级逻辑：

- 如果是 SU-MIMO UE，但没有可用 `SRS` 信道估计
- 则不执行正常的 RZF
- 而是生成对角、等功率的简化 precoder
- 并将 `bfGainPrgCurrTx` 置为无效值

这说明平台在 64T64R 路径下对 CSI 缺失是有显式 fallback 的。

### 4.2.7 64T64R 路径的适用表述

对外描述 64T64R 时，更合适的表述是：

`平台支持 64T64R MU-MIMO，下行采用基于 SRS 估计信道的 UE 排序、相关性分组与 RZF beamforming；beamforming 权重由组级信道矩阵计算得到，并结合功率分配策略完成归一化。`

### 4.3 外部输入的 Precoder / Detector / Beamforming 能力

平台除了内部生成 precoder，还支持外部输入。

### 4.3.1 Aerial Sim / AODT 接口

`cuMAC/src/api.h` 定义了三类 Aerial Sim 格式输入：

- `prdMat_asim`
- `detMat_asim`
- `sinVal_asim`

其语义分别是：

- 外部提供的 SVD precoder
- 外部提供的 SVD detector
- 外部提供的 per-UE / per-PRG singular values

`cuMAC/src/4T4R/multiCellScheduler.cu` 的 Aerial Sim `setup()` 会将这些外部数组接入调度器动态描述符中，说明平台不仅能“自己算 precoding”，也能“消费外部 precoding 结果”。

### 4.3.2 HDF5 / TV 加载路径

`cuMAC/examples/tools/h5TvLoad.cpp` 中会为以下对象分配与加载内存：

- `prdMat_asim`
- `detMat_asim`
- `sinVal_asim`
- `srsEstChan`
- `beamformGainCurrTx`
- `bfGainPrgCurrTx`

这说明平台具备通过 HDF5 / TV 驱动的方式，验证 precoding、beamforming 与调度链路的一致性。

### 4.3.3 RU Emulator 侧 beamforming 接口

`cuPHY-CP/ru-emulator` 进一步体现了平台在接口层面的 beamforming 支持：

- README 中明确写到 `tv_parser` 支持 beamforming weights and configurations
- 下行 C-plane 处理中明确管理 beamforming parameters 和 beam IDs
- `config_parser.cpp` 中可解析：
  - `enable_static_dynamic_beamforming`
  - `num_static_beamIdx`
  - `num_TRX_beamforming`
  - `DBT_real`
  - `DBT_imag`

因此，从平台整体视角看，Aerial 不只是“内部仿真里有 beamforming 算法”，还支持：

- test vector 驱动的 beamforming 验证
- RU/DU 接口中的 beam ID 与 DBT 配置处理

## 5. 支持边界与注意事项

### 5.1 “支持方向性”不代表所有链路都支持同等级方向图

平台中的“方向性支持”至少有三个等级：

- 扇区级大尺度方向性
- 面板/阵列级方向图方向性
- precoder/beamformer 带来的空间成形

这三者不能混为一谈。

### 5.2 内部 Rayleigh 不等于完整阵列方向图

内部 Rayleigh 链路下，方向性主要体现在大尺度增益，而不是 CDL 那种完整阵列方向图与逐路径空间结构。

### 5.3 TDL 的多天线支持强于 SISO，但弱于 CDL 的阵列表达能力

TDL 支持多天线维度，但代码中明确未加入额外 MIMO 天线相关性，因此它不能等同于完整方向图阵列模型。

### 5.4 4T4R 与 64T64R 的 precoding 机制不同

两条链路不能简单混称为“平台采用某一种 precoding”：

- 4T4R：逐 UE、逐 PRG 的 SVD precoding
- 64T64R：基于 SRS 的组级 RZF beamforming

### 5.5 外部输入能力不等于内部算法生成能力

`prdMat_asim`、`detMat_asim`、`sinVal_asim`、RU 侧 DBT/beam ID 支持，说明平台具备外部接入与验证能力；但这与“平台内部一定生成同类算法结果”是两个不同维度。

### 5.6 平台整体能力不只限于 `4T4R` 和 `64T64R`

这一点很容易被误解，需要明确区分两层含义。

第一层是底层信道与天线建模能力。

在 `chanModels`、`SLS`、`TDL`、`CDL` 这些底层模块里，天线配置本身是参数化的，而不是硬编码为 `4` 或 `64`：

- `AntPanelConfig` 使用通用的 `nAnt`、`antSize`、`antSpacing`、`antPolarAngles`、方向图数组描述天线面板
- `TDL` 会从外部 panel 配置中读取 `nBsAnt` 和 `nUeAnt`
- `CDL` 则通过 `bsAntSize`、`ueAntSize` 的乘积得到实际天线数

因此，从“平台是否能表达其他阵列规模”这个角度说，答案是：`能`。

第二层是 `cuMAC` 当前主调度/precoding/beamforming 实现边界。

`cuMAC/src/api.h` 中对当前 API 的说明已经比较直接：

- `nBsAnt`: `Value: 4 or 64`
- `nUeAnt`: `Value: 2, 4`

同时，当前代码结构本身也分成两大主路径：

- `4T4R/`
- `64T64R/`

并且 `64T64R` beamforming 路径还使用了：

- `maxNumBsAnt_ = 64`
- `maxNumUeAnt_ = 4`

这说明从当前 `cuMAC` 的主实现和公开能力看，平台上层调度与 MIMO 空间处理主要覆盖：

- `4T4R SU-MIMO`
- `64T64R MU-MIMO`

因此，更准确的结论应写成：

`Aerial 平台整体并不只支持 4T4R 和 64T64R；但 cuMAC 当前主调度与 precoding/beamforming 产品化实现，主要围绕这两档 BS 天线配置展开。`

## 6. 建议的对外表述

如果需要一个兼顾准确性和完整性的统一表述，可以使用：

`Aerial 平台整体支持多层次天线方向性建模，包括扇区级方向性增益、基于 3GPP 38.901 的阵列/面板级方向图配置，以及与之配套的 MIMO 空间处理能力。在 4T4R SU-MIMO 场景下，平台支持基于瞬时信道矩阵的 batched SVD precoding；在 64T64R MU-MIMO 场景下，平台支持基于 SRS 信道估计的 UE 排序、相关性分组和 RZF beamforming。同时，平台还支持外部 precoder/detector/singular values 输入，以及 RU 侧 beam ID 和动态 beamforming table 的验证链路。`

如果需要进一步突出“平台整体能力”和“当前 cuMAC 主实现边界”的区别，也可以使用下面这版：

`Aerial 平台底层的信道与天线面板建模能力并不只限于 4T4R 和 64T64R，可以参数化描述更一般的阵列规模与方向图；但从 cuMAC 当前公开的主调度、layer selection、precoding 与 beamforming 实现看，产品化主链路主要覆盖 4T4R SU-MIMO 与 64T64R MU-MIMO 两类配置。`

## 7. 一句话结论

若只问“平台是否支持天线方向性和 MIMO precoding”，答案是：

`支持，但支持是分层的：Rayleigh/TDL/CDL/SLS 对方向性的表达能力不同；4T4R 主要是 SVD precoding，64T64R 主要是 SRS 驱动的 RZF beamforming。`

## 8. 关键代码入口

如果后续需要继续做代码级核查，优先看以下入口：

- `cuMAC/examples/network.cu`
  - 大尺度方向性、Rayleigh/TDL/CDL 信道接入、功率缩放
- `cuMAC/examples/parameters.h`
  - 4T4R 示例链路中的天线参数、precoding 开关
- `testBenches/chanModels/src/chanModelsDataset.hpp`
  - `AntPanelConfig` 数据结构
- `testBenches/chanModels/src/sls_chan_src/antenna_config_reader.hpp`
  - `antModel` 与显式方向图读取
- `testBenches/chanModels/src/tdl_chan_src/tdl_chan.cuh`
  - TDL 的多天线支持边界
- `cuMAC/src/4T4R/svdPrecoding.cu`
  - 4T4R SU-MIMO 的 SVD precoding
- `cuMAC/src/4T4R/multiCellLayerSel.cu`
  - 4T4R 的层数选择
- `cuMAC/src/tools/multiCellSinrCal.cu`
  - 不同 precoding 模式下的 SINR 计算支持边界
- `cuMAC/examples/multiCellMuMimoScheduler/mMimoNetwork.cu`
  - 64T64R 的 `channel_config`、`SLS`、`antenna_panels` 接线
- `cuMAC/src/64T64R/multiCellMuUeSort.cu`
  - `SRS SNR` 门限驱动的 MU 可行性筛选
- `cuMAC/src/64T64R/multiCellMuUeGrp.cu`
  - 基于相关性的 UE 分组
- `cuMAC/src/64T64R/multiCellBeamform.cu`
  - 64T64R 的 `RZF` beamforming
- `cuMAC/src/api.h`
  - `precodingScheme`、`bfPowAllocScheme`、`prdMat_asim` 等外部接口定义
- `cuMAC/examples/tools/h5TvLoad.cpp`
  - HDF5 / TV 侧 precoding、beamforming 数据接入
- `cuPHY-CP/ru-emulator/ru_emulator/config_parser.cpp`
  - RU 侧动态 beamforming table 解析
