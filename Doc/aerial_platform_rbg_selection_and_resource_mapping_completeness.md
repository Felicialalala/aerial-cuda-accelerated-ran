# Aerial 平台 PRG(RBG) 选择、分配与资源映射完备度分析

## 1. 文档目标

本文从 `Aerial/cuMAC` 平台整体能力出发，分析其在 `PRG(RBG)` 频域资源管理上的完备程度，重点回答四个问题：

1. 平台当前如何定义和划分 `RBG/PRG`
2. 平台当前支持哪些 `RBG` 选择与分配方法
3. `RBG` 资源如何从调度解映射到 UE、PRB、干扰计算与 precoding/beamforming
4. 平台整体能力与“当前项目 Stage-B 实验配置”之间是什么关系

说明：

- 在 `Aerial/cuMAC` 代码里，主术语是 `PRG`；在调度语义上与常见 `RBG` 基本等价。
- 本文统一写作 `PRG(RBG)`，在涉及具体代码字段时保留原字段名。
- 本文强调“平台整体”，不是只站在当前 `Stage-B` 脚本或 `GNNRL` 接入点看问题。

## 2. 一页结论

先给结论：

1. 平台已经具备比较完整的 `PRG(RBG)` 级调度抽象：有统一的 `nPrbGrp`、`allocType`、`allocSol`、`postEqSinr`、`sinVal`、`prdMat` 等接口，足以支撑 `GPU` 调度、`CPU` reference、`per-PRG` SINR/precoding/beamforming 计算。
2. `4T4R` 路径是频域资源分配能力最完整的一条线：
   - `DL` 支持 `type-0` 非连续 bitmap 式分配
   - `DL/UL` 支持 `type-1` 连续区间分配
   - `PF` 与 `RR` 两类调度器都具备
3. `64T64R MU-MIMO` 路径的频域资源管理更偏“连续子带/子带组”风格：
   - 只支持 `type-1`
   - 不支持 `type-0 bitmap`
   - 更强调 `UE grouping + contiguous PRG block + beamforming`
4. 平台对 `PRG(RBG)` 的“调度抽象”是完整的，但对 `NR 标准化资源映射语义` 不是完全产品化的：
   - 当前没有看到 `cuMAC` 调度路径里显式的 `BWP -> RBG size` 自动推导
   - 没有看到显式的 `VRB/CRB/RIV/DCI frequencyDomainAssignment` 这一层
   - 现有 `type-0/type-1` 更像调度器内部资源表示，而不是完整的 38.214/38.331/FAPI 资源分配栈
5. 当前项目 `Stage-B` 只是平台能力的一个窄子集：
   - 固定 `4T4R`
   - 固定 `68 PRG × 4 PRB`
   - 固定 `type-0 bitmap`
   - 当前 native baseline 在该模式下本质上比较的是 `PRG bitmap allocation`，不是完整“UE selection + RBG selection”双阶段语义

如果只看“平台是否已经有 RBG 级资源管理骨架”，答案是“有，而且比较完整”。  
如果问“平台是否已经具备完整 NR 标准语义下的 RBG 划分、BWP 映射、VRB/CRB 映射、所有天线模式统一的一套产品化实现”，答案是“还没有，当前更像一套强调调度算法与 PHY 闭环的研究/工程平台抽象”。

## 3. 平台里的 PRG(RBG) 抽象是什么

### 3.1 统一资源粒度

平台把频域资源统一抽象成 `nPrbGrp` 个 `PRG`：

- `nPrbGrp`：每个 cell 当前可调度的 `PRG` 数
- `nPrbPerGrp`：每个 `PRG` 包含多少 `PRB`
- `W`：单个 `PRG` 的带宽，代码中定义为 `12 * SCS * (#PRB per PRG)`

这里有一个接口层与样例层的区别：

1. `cuMAC API` 主结构里直接携带的是 `nPrbGrp` 和 `W`
2. `nPrbPerGrp` 主要保存在 `example/network` 这一层
3. 当前样例是先给 `nPrbPerGrp`，再推导 `W`

也就是说，平台接口更关心“每个 PRG 的带宽是多少、总共有多少个 PRG”，而不是把“每个 PRG 包几块 PRB”作为所有接口都显式携带的一级字段。

对应实现：

- [`cuMAC/src/api.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/api.h)
- [`cuMAC/examples/parameters.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/parameters.h)
- [`cuMAC/examples/network.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/network.cu)

典型配置：

- [`cuMAC/examples/parameters.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/parameters.h) 里当前 `Stage-B` 固定为：
  - `nPrbsPerGrpConst = 4`
  - `nPrbGrpsConst = 68`
  - 即总调度栅格为 `68 × 4 = 272 PRB`

这也意味着在当前 `30 kHz + 100 MHz` 的 `273 PRB` 载波语境下，现有 `Stage-B` 样例并不是把 273 个 PRB 全部按组建模，而是使用了一个 `272 PRB` 的规则分组网格。

### 3.2 平台上限与抽象边界

在 API 常量里：

- [`cuMAC/src/api.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/api.h) 定义 `maxNumPrgPerCell_ = 273`

这说明平台内部数据结构可以覆盖“最多 273 个 PRG 单元”的情况。  
但这不代表平台自动按 `273 PRB -> 某种 3GPP RBG 规则` 来划分；它只说明平台的上层调度抽象可以容纳这一数量级的频域单元。

### 3.3 这里的 PRG(RBG) 不是完整 3GPP BWP/RIV/VRB 语义

从当前仓库 `cuMAC` 调度路径看，`PRG(RBG)` 的划分主要是“调度器内部频域网格抽象”，而不是完整的 `NR MAC/RRC/FAPI` 资源语义。

更准确地说：

1. `cuMAC` 明确使用 `nPrbGrp`、`nPrbPerGrp`、`allocType`、`allocSol`
2. 但没有看到 `cuMAC` 调度路径里显式的：
   - `BWP -> RBG size` 自动推导
   - `resourceAllocationType0/1` 对应 `DCI frequencyDomainAssignment`
   - `VRB -> PRB` 交织/非交织映射
   - `RIV` 编解码
3. `BWP` 相关逻辑在仓库里更多出现在 `cuPHY` 的 PDSCH/DMRS 发送接收实现，不在 `cuMAC` 这条调度主链里

因此：

- `Aerial/cuMAC` 的 `type-0/type-1` 应理解为“内部调度资源表示”
- 不应直接等同为“已经完整实现 3GPP 所有标准层资源映射语义”

## 4. 平台当前支持如何划分 PRG(RBG)

### 4.1 已支持：静态可配置分组

平台当前明确支持的是“静态分组”：

1. 先由配置给出 `nPrbPerGrp`
2. 再给出 `nPrbGrp`
3. 调度器、SINR 计算、layer selection、beamforming 都围绕这个固定网格运行

在 `4T4R` 示例里：

- [`cuMAC/examples/network.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/network.cu) 将 `nPrbGrp = nPrbGrpsConst`、`nPrbPerGrp = nPrbsPerGrpConst`
- [`cuMAC/examples/parameters.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/parameters.h) 进一步固定为 `68 × 4`

在 `64T64R` 示例里：

- [`cuMAC/examples/multiCellMuMimoScheduler/config.yaml`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/multiCellMuMimoScheduler/config.yaml) 直接给出：
  - `nPrbPerGrp: 4`
  - `nPrbGrp: 68`

### 4.2 未看到：按 BWP 大小自动推导 RBG size

从当前仓库实现看，没有看到 `cuMAC` 里按 `BWP size` 或 `3GPP table` 自动决定 `RBG size` 的逻辑。

这意味着平台当前更像：

- “用户/样例先把调度频域网格定义好”
- “调度器在这个网格上工作”

而不是：

- “平台先读 RRC/FAPI/BWP 配置，再自动导出资源块组大小”

### 4.3 未看到：尾部不等宽 RBG、混合粒度、动态重分组

当前没有看到以下能力：

1. 同一 cell 内不同 `RBG` 使用不同 `PRB` 宽度
2. 尾部 `RBG` 特殊大小处理
3. 同一场景中按 `BWP` 或 `slot` 动态切换 `RBG` 粒度
4. 同时存在“粗粒度 + 细粒度”两套频域资源网格

所以从“RBG 划分机制”角度看，平台当前是：

- `调度抽象完整`
- `标准语义和自动化分组机制不完整`

## 5. 平台当前支持哪些 PRG(RBG) 分配方法

## 5.1 4T4R SU-MIMO：支持最完整

`4T4R` 是当前平台里最完整的 `RBG/PRG` 调度实现。

相关模块：

- `PF UE selection`：
  - [`cuMAC/src/4T4R/multiCellUeSelection.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/multiCellUeSelection.cu)
  - [`cuMAC/src/4T4R/multiCellUeSelectionCpu.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/multiCellUeSelectionCpu.cpp)
- `RR UE selection`：
  - [`cuMAC/src/4T4R/roundRobinUeSel.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/roundRobinUeSel.cu)
  - [`cuMAC/src/4T4R/roundRobinUeSelCpu.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/roundRobinUeSelCpu.cpp)
- `PF scheduler`：
  - [`cuMAC/src/4T4R/multiCellScheduler.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/multiCellScheduler.cu)
  - [`cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/multiCellSchedulerCpu.cpp)
- `RR scheduler`：
  - [`cuMAC/src/4T4R/roundRobinScheduler.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/roundRobinScheduler.cu)
  - [`cuMAC/src/4T4R/roundRobinSchedulerCpu.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/roundRobinSchedulerCpu.cpp)

### 5.1.1 Type-0：非连续、bitmap 式 PRG 分配

`allocType = 0` 的语义是：

- 一个 `PRG` 可以独立分给某个 UE
- UE 在频域上不要求连续
- 输出形式本质上是“`PRG × cell -> UE` 的离散映射”

#### PF 在 type-0 下的行为

`PF` 型 `type-0` 调度器的核心思想是：

1. 对每个 `cell`
2. 对每个 `PRG`
3. 在候选 UE 中计算该 `PRG` 上的 PF metric
4. 选择该 `PRG` 的最优 UE

在 CPU 参考路径里，这一点非常直接：

- [`cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/multiCellSchedulerCpu.cpp) 的 `multiCellSchedulerCpu_noPrdMmse()` 和 `multiCellSchedulerCpu_svdMmse()` 都是“逐 `RBG` 选最优 UE”
- 结果写成 `allocSol[rbgIdx*totNumCell + cIdx] = ueIdx`

因此：

- `type-0 PF` 是真正的“按 PRG 独立选择 UE”
- 这条线在调度粒度上是最细的
- 它更适合研究 `bitmap/non-contiguous frequency scheduling`

#### RR 在 type-0 下的行为

`RR type-0` 并不是“每个 PRG 严格轮转一次 UE”，而是：

1. 找出该 cell 当前关联/已选中的 UE
2. 按 UE 数把 `nPrbGrp` 尽量平均切开
3. 余数从前往后补
4. 写回 `allocSol[prgIdx*stride + cell]`

也就是说：

- `RR type-0` 的输出仍然是 bitmap 形式
- 但算法本质是“均分频域资源块”
- 它不是基于信道质量的选择器

### 5.1.2 Type-1：连续区间式 PRG 分配

`allocType = 1` 的语义是：

- 每个 UE 拿到的是一个连续 `PRG` 区间 `[startPrg, endPrg)`
- 不支持一个 UE 在同一 TTI 拿多个离散频域块

#### PF 在 type-1 下的行为

`PF type-1` 的关键特点是：

1. 先计算所有 `UE × PRG` 的 PF metric
2. 再做排序
3. 用“sequential riding peaks”一类的逻辑，把高 metric 的 `PRG` 连续扩展成区间
4. 最终输出每个 UE 的连续频域段

这意味着：

- `type-1 PF` 不是简单的“先逐 PRG 选 winner，再后处理压连续”
- 它从设计上就带连续块约束
- 因而更接近传统产品调度器里“单 UE 一段连续资源”的语义

#### RR 在 type-1 下的行为

`RR type-1` 则比较直接：

1. 统计当前关联 UE 数
2. 把连续 `PRG` 轴按 UE 数尽量均分
3. 每个 UE 得到一段连续区间

所以它是：

- 最简单的连续块分配器
- 适合做 reference/baseline

### 5.1.3 4T4R 的支持边界

从平台能力角度看，`4T4R` 路径当前边界如下：

1. `DL`：
   - 支持 `type-0`
   - 支持 `type-1`
   - 支持 `PF`
   - 支持 `RR`
2. `UL`：
   - 只支持 `type-1`
   - `type-0` 明确不支持
3. `HARQ`：
   - API 与实现都表明 `HARQ` 只支持 `type-1`
   - `type-0 + HARQ` 不是当前平台主支持路径
4. 若细分 kernel 组合，还存在一些限制：
   - 某些 `row-major`/`column-major` 组合不支持
   - 某些 `lightweight kernel` 未实现
   - Aerial Sim 路径对 `allocType`/layout 也有限制

结论：

- `4T4R` 在“RBG 选择方法”上是平台里最成熟的一侧
- 但真正完整且稳定的主能力仍然更偏 `DL` 和 `type-1`

## 5.2 64T64R MU-MIMO：支持连续子带型分配，不支持 type-0 bitmap

`64T64R` 的设计重点不是“每个 PRG 独立选一个 UE”，而是：

1. 先做 `UE grouping`
2. 再做 `subband/PRG block` 级资源分配
3. 最后在每个 `PRG` 上做 `beamforming`

相关模块：

- [`cuMAC/src/64T64R/multiCellMuUeGrp.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/64T64R/multiCellMuUeGrp.cu)
- [`cuMAC/src/64T64R/muMimoSchedulerBaseCpu.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/64T64R/muMimoSchedulerBaseCpu.cpp)
- [`cuMAC/src/64T64R/multiCellBeamform.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/64T64R/multiCellBeamform.cu)

### 5.2.1 平台边界

这条线的边界非常明确：

1. 只支持 `type-1`
2. 不支持 `type-0`
3. baseline `CPU` 路径当前只支持 `DL`
4. 频域分配和 beamforming 强耦合

因此从“RBG 选择完备度”看：

- `64T64R` 的重点不是“非连续 bitmap 选择”
- 而是“连续子带块 + UEG + beamforming”

### 5.2.2 baseline CPU：RME 风格连续 PRG 块分配

在 [`cuMAC/src/64T64R/muMimoSchedulerBaseCpu.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/64T64R/muMimoSchedulerBaseCpu.cpp) 中，可以看到如下逻辑：

1. 对每个 UE、每个 PRG 计算 PF metric
2. 找到一个 anchor UE 和其最优 PRG
3. 向左、向右连续扩展，只要该 UE 在相邻 PRG 上仍具优势
4. 填补 gap
5. 再在该连续子带上迭代加入 MU 候选 UE

它本质上是：

- `per-PRG` 度量驱动
- 但输出保持为 `contiguous block`
- 更像“连续子带选择器”而不是“bitmap 选择器”

### 5.2.3 GPU MU grouping：semi-static / dynamic 两种模式

在 [`cuMAC/src/64T64R/multiCellMuUeGrp.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/64T64R/multiCellMuUeGrp.cu) 中，`DL` 路径支持两类频域分配模式：

#### 1. semi-static subband allocation

思路是：

1. 先把整个 `nPrbGrp` 频带切成 `nMaxUegPerCellDl` 个固定子带
2. 再把这些子带分给排名靠前的 `UEG`
3. 每个 `UEG` 内部所有 UE 共享同一连续频域块

这是一种“先切子带，再选组”的模式。

#### 2. dynamic subband allocation

思路是：

1. 先为重传 `UEG` 预留连续 PRG
2. 再做 top-K 新传 `UEG` 选择
3. 将剩余 `PRG` 预算按 `UEG` 大小与剩余量分配成连续块
4. 同一 `UEG` 内所有 UE 共享同一 `[start,end)` 区间

这是一种“先 group，再分连续 PRG budget”的模式。

### 5.2.4 64T64R 的结论

从平台完备度角度看，`64T64R` 在频域上已经具备：

1. `UEG` 级频域资源分配
2. `semi-static` 与 `dynamic` 两类连续子带分配模式
3. 与 `beamforming` 严格一致的 `per-PRG` 映射链路

但它不具备：

1. `type-0 bitmap`
2. `per-PRG` 非连续离散调度
3. 一套和 `4T4R type-0` 对称的 RBG 选择框架

所以它的频域资源能力是“深但窄”：

- 在 `连续子带 + MU grouping + beamforming` 上很强
- 在“自由形 bitmap RBG 选择”上不完整

## 6. 当前支持如何分配和映射 PRG(RBG) 资源

这一节回答最关键的问题：`RBG` 分配解在平台里到底是怎么落到 UE、PRB、干扰和 precoding 上的。

## 6.1 核心数据结构

最关键的四类字段是：

1. `setSchdUePerCellTTI`
2. `allocSol`
3. `postEqSinr / sinVal`
4. `prdMat / bfGainPrgCurrTx`

语义分别是：

### 1. `setSchdUePerCellTTI`

- 表示“本 TTI 每个 cell 被调度的 active UE ID 集合”
- `4T4R` 下大小是 `nCell * numUeSchdPerCellTTI`
- `64T64R` 下大小是 `nCell * numUeForGrpPerCell`

### 2. `allocSol`

这是平台里真正的频域资源解。

但它在不同模式下语义不同：

#### 4T4R + type-0

- `allocSol` 形状：`totNumCell * nPrbGrp`
- 每个元素表示：
  - 某个 `(cell, prg)` 被分给哪个“已选 UE 索引”
- `-1` 表示未分配

#### 4T4R + type-1

- `allocSol` 形状：`2 * nUe`
- 每两个元素表示一个已选 UE 的：
  - `startPrg`
  - `endPrgPlusOne`

#### 64T64R + type-1

- `allocSol` 形状：`2 * nActiveUe`
- 每两个元素表示一个 active UE 的：
  - `startPrg`
  - `endPrgPlusOne`

### 3. `postEqSinr / sinVal`

这两类数组都已经做到 `per-UE × per-PRG`：

1. `postEqSinr[u, prg, layer]`
2. `sinVal[u, prg, layer]`

这意味着平台已经具备：

- 以 `PRG(RBG)` 为粒度的质量感知输入
- 支撑 `RBG` 级调度决策所需的核心 PHY 观测

### 4. `prdMat / bfGainPrgCurrTx`

这两类数组说明平台并没有把 `RBG` 分配停留在 MAC 层抽象，而是已经映射到 PHY：

1. `prdMat[cell, prg, ant, layer]`
2. `bfGainPrgCurrTx[ue, prg]`

也就是说：

- `PRG(RBG)` 资源不是“逻辑占位符”
- 它直接决定了每个 `PRG` 上的 precoder/beamformer 计算与使用

## 6.2 Type-0 下的映射方式

### 6.2.1 资源解长什么样

在 `type-0` 下，平台使用“bitmap-like occupancy map”：

- `allocSol[prgIdx * stride + cell] = ueSelectedIdx`

这里有一个很重要的实现细节：

1. API 注释把 `4T4R type-0` 描述为 `nCell * nPrbGrp`
2. 但实际 `example/network` 侧分配和使用时，stride 采用的是 `totNumCell`
3. 原因是要把协调 cell 与干扰 cell 一起放到同一个 `PRG` 占用图里，便于同频干扰计算

对应实现：

- [`cuMAC/src/api.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/api.h)
- [`cuMAC/examples/network.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/network.cu)

### 6.2.2 `allocSol` 存的是谁

这点很容易混淆。

在 `4T4R type-0` 下，`allocSol` 存的不是直接的 active UE ID，而是“已选 UE 索引”。

要拿到真正 active UE ID，需要：

1. 先读 `allocSol[prg, cell] -> selectedUeIdx`
2. 再用 `setSchdUePerCellTTI[selectedUeIdx] -> activeUeId`

也就是说：

- `allocSol` 负责回答“这个 PRG 分给了哪一个已入调度集合的 UE slot”
- `setSchdUePerCellTTI` 负责把这个 slot 解释成真实 active UE

### 6.2.3 当前 Stage-B 为什么看起来像“直接存 active UE ID”

这是当前项目的一个特殊情况，不是平台通用语义。

在 [`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/multiCellSchedulerUeSelection/main.cpp) 里，当前 `type-0` baseline 会先调用 `populateType0AllUeSelection()`：

1. 直接把所有 active UE 填进 `setSchdUePerCellTTI`
2. 且顺序就是 `0, 1, 2, ...`

所以在这个特定实验里：

- `selectedUeIdx == activeUeId`

这会让人误以为：

- `type-0 allocSol` 直接存 active UE ID

但这只是当前 `Stage-B` 的特例，不是平台一般语义。

### 6.2.4 Type-0 如何参与干扰和 KPI 计算

在 `network.cu` 中，`type-0` 的 KPI/吞吐/干扰计算会：

1. 对某个 UE 遍历全部 `PRG`
2. 找到属于该 UE 的 `PRG`
3. 再在同一个 `PRG` 上扫描其他 cell 的 `allocSol`
4. 找到同频干扰 UE
5. 进一步计算 `SINR`、`TBS`、BLER、吞吐

因此 `type-0` 的映射天然支持：

1. `per-PRG` 同频干扰分析
2. 非连续频域占用
3. 逐 `PRG` 的资源利用率统计

这是它相对 `type-1` 最大的表达能力优势。

## 6.3 Type-1 下的映射方式

### 6.3.1 资源解长什么样

在 `type-1` 下，平台统一采用连续区间表示：

- `[startPrg, endPrgPlusOne)`

区别只在于索引域：

1. `4T4R`：
   - `allocSol` 索引域是“已选 UE 索引”
2. `64T64R`：
   - `allocSol` 索引域是“active UE ID”

### 6.3.2 为什么 type-1 还需要临时展开回 `PRG -> UE`

虽然 `type-1` 的主输出是区间，但后续很多 PHY/KPI 逻辑仍然要知道：

- 某个 `PRG` 上究竟是谁在发

所以在 `network.cu` 里常能看到这样的模式：

1. 先根据每个 UE 的 `[start,end)` 区间
2. 构造一个临时 `allocSol_rbg2Ue`
3. 再用它做每 `PRG` 的干扰、SINR、吞吐计算

也就是说：

- `type-1` 只是压缩表示
- 平台内部仍然会在需要时把它展开成 `PRG -> UE` 占用图

## 6.4 PRG(RBG) 到 PRB 的映射方式

平台当前的 `PRG -> PRB` 换算是直接按固定组宽完成的：

1. `startPrb = startPrg * nPrbPerGrp`
2. `endPrb = endPrg * nPrbPerGrp`

在代码里：

- [`cuMAC/examples/network.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/network.cu) 的 `convertPrbgAllocToPrbAlloc()` 就是做这个乘法

这说明当前平台默认假设：

1. 所有 `PRG` 宽度相同
2. `PRG` 在频域上线性顺序排列
3. 不存在额外交织或非线性 `VRB -> PRB` 映射

这再次说明它更像“调度器内部资源网格”，而不是完整 `NR` 资源映射器。

## 6.5 PRG(RBG) 与 precoding / beamforming 的映射

### 6.5.1 4T4R

在 `4T4R` 路径中：

1. `prdMat` 是 `per-cell × per-PRG`
2. `sinVal` / `postEqSinr` 也是 `per-UE × per-PRG`

因此 `RBG` 分配不仅决定“谁占这个频带”，还直接决定：

1. 这个 `PRG` 上用哪组 precoder
2. 这个 `PRG` 上对应 UE 的后均衡 `SINR`

### 6.5.2 64T64R

在 [`cuMAC/src/64T64R/multiCellBeamform.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/64T64R/multiCellBeamform.cu) 中，beamforming kernel 是按 `(cell, rbgIdx)` 发起的。

其基本流程是：

1. 取出当前 `cell`、当前 `rbgIdx`
2. 扫描 `setSchdUePerCellTTI`
3. 找出那些区间覆盖该 `rbgIdx` 的 UE
4. 组装 stacked channel
5. 计算该 `PRG` 上的 beamformer
6. 写回 `prdMat` 和 `bfGainPrgCurrTx`

这意味着：

- `64T64R` 的频域资源解与 beamforming 是逐 `PRG` 对齐的
- 所谓“连续子带分配”，最终仍落实为“每个 `PRG` 上谁参与 beamforming”

## 7. 平台完备程度评估

下面按几个维度给出判断。

## 7.1 从“调度抽象是否完整”看：较完整

这是平台最强的一面。

它已经具备：

1. 统一的 `PRG(RBG)` 资源抽象
2. `type-0` 与 `type-1` 两类资源表示
3. `UE selection`、`PRG allocation`、`layer selection`、`MCS selection` 的分层流水
4. `per-PRG` 的信道、SINR、precoding、beamforming 数据结构
5. `CPU` reference 与 `GPU` 主路径

如果目标是：

- 研究调度算法
- 做 `GPU` 版频域调度实验
- 做 `RBG` 级 `RL/GNN` 接管

那么这套抽象已经足够完整。

## 7.2 从“RBG 选择方法是否丰富”看：4T4R 较完整，64T64R 中等

### 4T4R

优点：

1. 有 `PF`
2. 有 `RR`
3. 有 `type-0` 非连续分配
4. 有 `type-1` 连续分配
5. `DL` 和 `UL` 都覆盖到一部分

不足：

1. `UL type-0` 不支持
2. `type-0 + HARQ` 不支持
3. 某些 kernel 组合仍有限制

结论：

- `4T4R` 的 `RBG` 选择完备度可以认为是“高于平台平均水平”

### 64T64R

优点：

1. 有 `UEG` 级频域分配
2. 有 `semi-static` 子带分配
3. 有 `dynamic` 子带分配
4. 与 `beamforming` 结合紧密

不足：

1. 没有 `type-0 bitmap`
2. 没有自由非连续 `RBG` 选择
3. baseline `UL` 不完整

结论：

- `64T64R` 的频域资源能力是“连续子带调度很完整，bitmap 调度不完整”

## 7.3 从“标准化资源映射语义是否完整”看：中低

如果用 3GPP/FAPI 产品化标准要求来衡量，当前平台仍有明显缺口：

1. 没有看到 `BWP -> RBG size` 自动推导
2. 没有看到显式 `VRB/CRB` 映射层
3. 没有看到 `RIV`/`DCI frequencyDomainAssignment` 这一层
4. `PRG -> PRB` 主要是线性乘法映射

因此：

- 这套平台已经是“调度算法平台”
- 但还不是“完整 NR 资源映射产品栈”

## 7.4 从“当前项目暴露出来的能力”看：只是平台子集

当前 `Stage-B` 实验配置暴露出来的能力比平台整体窄很多：

1. 固定 `4T4R`
2. 固定 `DL`
3. 固定 `68 PRG × 4 PRB`
4. 固定 `type-0 bitmap`
5. 当前 native baseline 下，`UE selection` 在 `type-0` 被简化为“所有 active UE 都进入 slot”

所以如果只看当前项目，很容易误判成：

1. 平台只有 `type-0`
2. 平台只会 bitmap 分配
3. 平台没有真正的 UE selection

这些结论都不成立。  
更准确的说法是：

- 当前项目只选中了平台能力树上的一个很窄的分支来做实验

## 8. 与当前 Stage-B / GNNRL 项目的关系

这一节专门澄清“平台整体”和“当前项目实现”的关系。

## 8.1 当前 Stage-B 实际使用的是哪一支

当前 `Stage-B` 基线实际使用的是：

1. `4T4R`
2. `DL`
3. `nPrbPerGrp = 4`
4. `nPrbGrp = 68`
5. `allocType = 0`

参考文档：

- [`Doc/current_stageB_effective_configuration.md`](/home/oai2/aerial-cuda-accelerated-ran/Doc/current_stageB_effective_configuration.md)

## 8.2 当前 baseline 比较的其实是什么

当前 `type-0` baseline 下：

1. `setSchdUePerCellTTI` 先被填成“所有 active UE”
2. 真正可比的差异主要落在 `PRG bitmap allocation`

因此当前 `RR vs PF` 在 `Stage-B` 中比较的本质是：

- “PRG bitmap 分配策略”

而不是：

- “完整 UE selection + RBG selection + layer/mcs 联合调度”

## 8.3 当前自定义/训练路径与平台整体能力的关系

当前仓库里的自定义/训练路径里：

1. `CustomUePrgScheduler` 已经兼容 `allocType 0/1`
2. 但 `GNNRL runtime` 当前主要只支持 `allocType = 0`
3. `online bridge` 当前也要求 `allocType = 0` 且 `nTotCell == nCell`

这再次说明：

- 项目接入点的限制，不等于平台调度框架本身的限制

## 9. 最终判断

如果把问题简化成一句话：

**Aerial 平台当前已经具备比较完整的 `PRG(RBG)` 级调度与 PHY 映射骨架，其中 `4T4R` 的 `RBG` 选择方法最完整，`64T64R` 则偏向连续子带和 UEG/beamforming；但在 `BWP/VRB/CRB/RIV/DCI` 这一层标准化资源映射语义上，平台还没有做成完整产品化栈。**

进一步拆开看：

1. “有没有 `RBG` 级调度能力？”
   - 有，而且不弱
2. “有没有 `type-0` 和 `type-1` 两类资源表示？”
   - `4T4R` 有，`64T64R` 只有 `type-1`
3. “有没有从 `RBG` 到 SINR、precoding、beamforming 的闭环映射？”
   - 有
4. “有没有完整 3GPP 资源分配语义栈？”
   - 还没有
5. “当前项目是否代表平台全部能力？”
   - 不是，只代表其中一个窄子集

## 10. 若后续要补齐平台完备度，优先建议

如果后续要把平台从“调度算法平台”继续推向“更完整的 NR 资源管理平台”，建议优先补这几项：

1. 增加 `BWP -> RBG size` 自动推导层  
   让 `nPrbGrp/nPrbPerGrp` 不再只靠样例手工给定。

2. 显式增加 `VRB/CRB/RIV/DCI frequencyDomainAssignment` 适配层  
   把调度器内部 `allocSol` 和标准化控制面语义分离开。

3. 统一 `allocSol` 的索引域解释  
   尤其是 `4T4R type-0/type-1` 与 `64T64R type-1` 之间，最好明确提供统一 helper，而不是让调用方自己猜“是 selected UE index 还是 active UE ID”。

4. 把 `type-0` 的可用 `prgMsk`、HARQ、重传保留机制补齐  
   这样 `bitmap` 路径才算真正产品化。

5. 为 `64T64R` 增加更自由的非连续频域资源表示  
   至少补一条与 `4T4R type-0` 对称的表达能力，方便更高自由度的 MU-MIMO 频域调度研究。
