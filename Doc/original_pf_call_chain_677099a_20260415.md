# `677099a` 原始 PF 实现与完整调用链路梳理

## 1. 结论先说

- `677099a` 里“原始 PF”并不在 `cuMAC/examples/customScheduler/*`。
- 原始 PF 的原生实现分成两段：
  - `PF UE selection`：`cuMAC/src/4T4R/multiCellUeSelection.cu` 与 `cuMAC/src/4T4R/multiCellUeSelectionCpu.cpp`
  - `PF PRG/PRB scheduler`：`cuMAC/src/4T4R/multiCellScheduler.cu` 与 `cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`
- 对 Stage-B 而言，只要 `--custom-ue-prg 0`，最终就会走这套 native PF；如果 `--custom-ue-prg 1`，则会改走 `CustomUePrgScheduler`，那已经不是“原始 PF”。
- `677099a` 默认编译参数下，原始 PF 的默认主路径是：
  - `DL`
  - `no precoding`
  - `allocType = 0`，也就是 `type-0 / non-consecutive PRG allocation`
  - GPU kernel 落到 `cumac::multiCellSchedulerKernel_noPrdMmseIrc`
  - CPU reference 落到 `cumac::multiCellSchedulerCpu::multiCellSchedulerCpu_noPrdMmse`

## 2. 原始 PF 的代码位置

### 2.1 最关键的实现文件

| 角色 | 关键函数/类 | 文件 | `677099a` 行号 |
|---|---|---|---|
| Stage-B 脚本入口 | `BIN=.../multiCellSchedulerUeSelection` | `cuMAC/scripts/run_stageB_main_experiment.sh` | 580-582 |
| 示例主入口 | `int main(int argc, char* argv[])` | `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp` | 1-980 |
| PF 状态初始化 | `void cumac::network::createAPI()` | `cuMAC/examples/network.cu` | 778-1002 |
| 每 TTI API 同步 | `void cumac::network::setupAPI(cudaStream_t strm)` | `cuMAC/examples/network.cu` | 1004-1019 |
| SINR 计算 | `class cumac::multiCellSinrCal` | `cuMAC/src/tools/multiCellSinrCal.cuh` | 64-123 |
| PF UE 选择 GPU | `class cumac::multiCellUeSelection` | `cuMAC/src/4T4R/multiCellUeSelection.cuh` | 49-100 |
| PF UE 选择 CPU | `class cumac::multiCellUeSelectionCpu` | `cuMAC/src/4T4R/multiCellUeSelectionCpu.h` | 53-89 |
| UE downselect 桥接 | `__global__ void cumac::ueDownSel4TrKernel(...)` | `cuMAC/examples/network.cu` | 3871-3978 |
| PF 调度 GPU | `class cumac::multiCellScheduler` | `cuMAC/src/4T4R/multiCellScheduler.cuh` | 87-229 |
| PF 调度 CPU | `class cumac::multiCellSchedulerCpu` | `cuMAC/src/4T4R/multiCellSchedulerCpu.h` | 55-124 |
| PF 平均速率回写 | `updateDataRate*` / `updateDataRateAllActiveUe*` | `cuMAC/examples/network.cu` | 2128-2739, 4346-4408 |

### 2.2 原始 PF 与 custom UE+PRG 的分界

`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp` 中已经把两条路径分得很清楚：

- `useCustomUePrg == 1` 时：
  - `cumac::CustomUePrgScheduler::run(...)`
  - 跳过 native `multiCellUeSelection` 与 `multiCellScheduler`
  - 见 `main.cpp` 537-547、589-635
- `useCustomUePrg == 0` 时：
  - 走 native PF UE selection + native PF scheduler
  - 见 `main.cpp` 548-609、637-655

所以，如果你要找“原始 PF”，要看的就是 `useCustomUePrg == 0` 这条分支。

## 3. `677099a` 默认配置下 PF 走哪条分支

### 3.1 编译期默认值

`cuMAC/examples/parameters.h` 中：

- `gpuAllocTypeConst = 0`，见 79 行
- `cpuAllocTypeConst = 0`，见 80 行
- `prdSchemeConst = 0`，见 81 行
- `initAvgRateConst = 1.0`，见 102 行
- `pfAvgRateUpdConst = 0.001`，见 103 行
- `betaCoeffConst = 1.0`，见 104 行

这意味着默认就是：

- type-0 PF 调度
- 不使用 SVD precoding
- PF 指数 `beta = 1`

### 3.2 Stage-B 脚本也会把分支钉死到 native type-0 PF

`cuMAC/scripts/run_stageB_main_experiment.sh` 在 `677099a` 中会显式设置：

- `gpuAllocTypeConst "0"`，见 572 行
- `cpuAllocTypeConst "0"`，见 573 行
- binary 指向 `multiCellSchedulerUeSelection`，见 580 行

所以从脚本入口看，Stage-B baseline/native PF 也确实是 type-0 原始 PF，而不是 type-1 分支。

## 4. PF 相关核心状态

下面这几个数据结构，是理解调用链最关键的部分。

### 4.1 API 结构体

定义见 `cuMAC/src/api.h`：

- `struct cumac::cumacCellGrpUeStatus`
  - `avgRates`，98-104 行
  - `avgRatesActUe`，105-112 行
  - `bufferSize`，150-156 行
  - `tbErrLastActUe`，157-167 行
  - `tbErrLast`，169-182 行
  - `newDataActUe`，186-194 行
- `struct cumac::cumacSchdSol`
  - `allocSol`，671-679 行
  - `pfMetricArr`，688-693 行
  - `pfIdArr`，694-698 行
  - `setSchdUePerCellTTI`，718-731 行
- `struct cumac::cumacCellGrpPrms`
  - `postEqSinr`，533-541 行
  - `wbSinr`，543-552 行

### 4.2 这几个状态分别干什么

- `avgRatesActUe`
  - 面向“所有 active UE”
  - UE selection 用它做 PF 分母
- `setSchdUePerCellTTI`
  - UE selection 的输出
  - 存的是 active UE ID
- `avgRates`
  - 面向“本 TTI 被挑出来的 selected UE 集合”
  - 是 scheduler 计算 PF metric 时真正使用的分母
  - 它不是直接初始化好的，而是经过 `ueDownSel4TrKernel()` 从 `avgRatesActUe` 投影出来
- `postEqSinr`
  - 每 active UE、每 PRG、每 layer 的后均衡 SINR
  - `wbSinr` 由它平均得到
- `allocSol`
  - scheduler 的输出
  - type-0 时维度是 `[nPrbGrp, totNumCell]`
  - type-1 时维度是 `[2 * nUe]`

### 4.3 PF 状态初始化在哪里

`cumac::network::createAPI()` 中完成了 PF 初始状态创建和赋值：

- 分配 `avgRates` / `avgRatesActUe`，见 `network.cu` 400-428 行
- 初始化两者为 `initAvgRate`，见 422-428 行、939-947 行
- 把 `betaCoeff` 写入 GPU/CPU `cellGrpPrms`，见 967、989 行

这一步很关键，因为后续所有 PF metric 都要依赖：

- `avgRatesActUe`
- `avgRates`
- `betaCoeff`

## 5. 完整 PF 调用链路

下面按 `677099a` 默认 native PF 路径，把整条链从脚本入口一直串到平均速率回写。

### 5.1 脚本/二进制入口

1. Stage-B 脚本 `cuMAC/scripts/run_stageB_main_experiment.sh`
2. 运行二进制 `build/.../multiCellSchedulerUeSelection`
3. 进入 `int main(int argc, char* argv[])`
   - 文件：`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
4. `main()` 中先调用：
   - `cumac::network::createAPI()`，见 319 行
5. 然后创建原始 PF 相关对象：
   - `cumac::multiCellSinrCal`，340 行
   - `cumac::multiCellUeSelection`，343 行
   - `cumac::multiCellScheduler`，346 行
   - CPU reference：
     - `cumac::multiCellUeSelectionCpu`，398 行
     - `cumac::multiCellSchedulerCpu`，402 行

### 5.2 每个 TTI 的总体顺序

`main.cpp` 437-785 行里的 TTI 主循环，native PF 路径顺序是：

1. `net->setupAPI(cuStrmMain)`，479 行
2. `mcSinrCalGpu->setup(...)`，489 行
3. `mcSinrCalGpu->run(...)`，495 行
4. `mcSinrCalGpu->setup_wbSinr(...)`，504 行
5. `mcSinrCalGpu->run(...)`，510 行
6. `net->cpySinrGpu2Cpu()`，515 行
7. `mcUeSelGpu->setup(...)`，550 行
8. `mcUeSelGpu->run(...)`，556 行
9. `mcUeSelCpu->setup(...)`，568 行
10. `mcUeSelCpu->run()`，571 行
11. `net->ueDownSelectGpu()`，577 行
12. `net->ueDownSelectCpu()`，579 行
13. `mcSchGpu->setup(...)`，598 行
14. `mcSchCpu->setup(...)`，609 行
15. `mcSchGpu->run(...)`，639 行
16. `mcSchCpu->run()`，653 行
17. `net->run(cuStrmMain)`，702 行
18. `net->updateDataRateUeSelCpu(t)`，731 行
19. `net->updateDataRateGpu(t)`，734 行
20. `net->updateDataRateAllActiveUeCpu(t)`，739 行
21. `net->updateDataRateAllActiveUeGpu(t)`，742 行

这 21 步组成了完整 PF feedback loop。

## 6. 详细链路拆解

### 6.1 先产出 `wbSinr`，因为 PF UE selection 要用它

#### 关键函数引用

- `void cumac::multiCellSinrCal::setup(cumacCellGrpPrms* cellGrpPrms, uint8_t in_columnMajor, cudaStream_t strm)`
  - 文件：`cuMAC/src/tools/multiCellSinrCal.cu`
  - 行号：891-933
- `void cumac::multiCellSinrCal::setup_wbSinr(cumacCellGrpPrms* cellGrpPrms, cudaStream_t strm)`
  - 文件：`cuMAC/src/tools/multiCellSinrCal.cu`
  - 行号：935-970
- `void cumac::multiCellSinrCal::run(cudaStream_t strm)`
  - 文件：`cuMAC/src/tools/multiCellSinrCal.cu`
  - 行号：972-980
- `static __global__ void cumac::multiCellSinrCalKernel_noPrdMmseIrc_cm(mcSinrCalDynDescr_t* pDynDescr)`
  - 文件：`cuMAC/src/tools/multiCellSinrCal.cu`
  - 行号：49 起
- `static __global__ void cumac::multiCellWideBandSinrCalKernel(mcSinrCalDynDescr_t* pDynDescr)`
  - 文件：`cuMAC/src/tools/multiCellSinrCal.cu`
  - 行号：820-832

#### 默认分支怎么选

`cumac::multiCellSinrCal::kernelSelect()` 会根据：

- `DL`
- `precodingScheme`
- `columnMajor`

选择具体 kernel。默认配置是：

- `DL = 1`
- `precodingScheme = 0`
- `columnMajor = 1`

因此会落到：

- `multiCellSinrCalKernel_noPrdMmseIrc_cm`
  - 见 `multiCellSinrCal.cu` 834-849 行的选择逻辑

#### 这一步输出什么

- 第一次 `setup()+run()` 输出 `postEqSinr`
- 第二次 `setup_wbSinr()+run()` 调用 `multiCellWideBandSinrCalKernel`
  - 对每个 active UE、每个 layer，把所有 PRG 的 `postEqSinr` 求平均
  - 写入 `wbSinr`
  - 核心代码见 824-829 行

随后 `net->cpySinrGpu2Cpu()` 把 GPU 上的：

- `postEqSinr`
- `wbSinr`

拷到 CPU 侧，供 CPU reference PF 使用。

### 6.2 PF UE selection：按 `wbSinr / avgRatesActUe` 选 active UE

#### 关键函数引用

- GPU setup：
  - `void cumac::multiCellUeSelection::setup(cumacCellGrpUeStatus* cellGrpUeStatus, cumacSchdSol* schdSol, cumacCellGrpPrms* cellGrpPrms, cudaStream_t strm)`
  - 文件：`cuMAC/src/4T4R/multiCellUeSelection.cu`
  - 行号：260-306
- GPU run：
  - `void cumac::multiCellUeSelection::run(cudaStream_t strm)`
  - 文件：`cuMAC/src/4T4R/multiCellUeSelection.cu`
  - 行号：308-339
- GPU kernel：
  - `static __global__ void cumac::multiCellUeSelKernel(mcUeSelDynDescr_t* pDynDescr)`
  - 文件：`cuMAC/src/4T4R/multiCellUeSelection.cu`
  - 行号：102-161
- CPU setup：
  - `void cumac::multiCellUeSelectionCpu::setup(cumacCellGrpUeStatus* cellGrpUeStatus, cumacSchdSol* schdSol, cumacCellGrpPrms* cellGrpPrms)`
  - 文件：`cuMAC/src/4T4R/multiCellUeSelectionCpu.cpp`
  - 行号：152-187
- CPU run：
  - `void cumac::multiCellUeSelectionCpu::run()`
  - 文件：`cuMAC/src/4T4R/multiCellUeSelectionCpu.cpp`
  - 行号：189-196
- CPU 核心实现：
  - `void cumac::multiCellUeSelectionCpu::multiCellUeSelCpu()`
  - 文件：`cuMAC/src/4T4R/multiCellUeSelectionCpu.cpp`
  - 行号：32-86

#### PF UE metric 的原始公式

CPU 与 GPU 两边的公式是一致的：

```text
dataRate = sum_j W * log2(1 + wbSinr[uIdx, j])
pfMetric = pow(dataRate, betaCoeff) / avgRatesActUe[uIdx]
```

直接代码锚点：

- CPU：`multiCellUeSelectionCpu.cpp` 47-53 行
- GPU：`multiCellUeSelection.cu` 130-136 行

#### 这一步还有两个额外条件

- 如果 `bufferSize[uIdx] == 0`，直接跳过，不参与 PF 排序
  - CPU：41-43 行
  - GPU：125-128 行
- 如果是 HARQ 重传，即 `newDataActUe[uIdx] != 1`
  - metric 直接置为 `std::numeric_limits<float>::max()`
  - 等价于强制优先调度重传
  - CPU：56-63 行
  - GPU：137-140 行

#### 这一步输出什么

输出写到：

- `cumac::cumacSchdSol::setSchdUePerCellTTI`

也就是：

- 每小区本 TTI 先选出若干个 active UE
- 后续 scheduler 只在这个“已选中 UE 集合”上继续做 PRG 级 PF 分配

### 6.3 `ueDownSel4TrKernel()`：把 active UE 状态投影到 selected UE 视图

这一步非常重要，也是很多人第一次看代码时最容易漏掉的桥接层。

#### 关键函数引用

- `__global__ void cumac::ueDownSel4TrKernel(...)`
  - 文件：`cuMAC/examples/network.cu`
  - 行号：3871-3978
- `void cumac::network::ueDownSelectGpu()`
  - 文件：`cuMAC/examples/network.cu`
  - 行号：4070-4132
- `void cumac::network::ueDownSelectCpu()`
  - 文件：`cuMAC/examples/network.cu`
  - 行号：4019-4068

#### 这一步做了什么

`ueDownSel4TrKernel()` 用 `setSchdUePerCellTTI` 中的 active UE ID，把“active UE 域”的数据拷到“selected UE 域”：

- `avgRatesActUe[globalUeId] -> avgRates[uIdx]`
  - 3932-3935 行
- `tbErrLastActUe[globalUeId] -> tbErrLast[uIdx]`
  - 3932-3935 行
- `estH_fr_actUe -> estH_fr`
  - 3937-3952 行
- `prdMat_actUe -> prdMat`
  - 3954-3960 行
- `detMat_actUe -> detMat`
  - 3962-3968 行
- `sinVal_actUe -> sinVal`
  - 3970-3975 行

同时还会生成 selected UE 视图下的 `cellAssocArr`：

- 3908-3923 行

#### 这一步为什么必不可少

因为：

- UE selection 的输入是 active UE 域：`avgRatesActUe`, `wbSinr`
- scheduler 的输入是 selected UE 域：`avgRates`, `estH_fr`, `cellAssoc`

也就是说，原始 PF 实现并不是同一组索引直接从头跑到尾，而是中间经过了一次“active UE -> selected UE”的索引投影。

### 6.4 PF scheduler 默认主路径：type-0 / no-precoding / DL

#### 关键函数引用

- GPU kernel 选择：
  - `void cumac::multiCellScheduler::kernelSelect()`
  - 文件：`cuMAC/src/4T4R/multiCellScheduler.cu`
  - 行号：5261-5397
- GPU setup：
  - `void cumac::multiCellScheduler::setup(cumacCellGrpUeStatus* cellGrpUeStatus, cumacSchdSol* schdSol, cumacCellGrpPrms* cellGrpPrms, cumacSimParam* simParam, uint8_t in_columnMajor, uint8_t in_halfPrecision, uint8_t in_lightWeight, float in_percSmNumThrdBlk, cudaStream_t strm)`
  - 文件：`cuMAC/src/4T4R/multiCellScheduler.cu`
  - 行号：5400-5499
- GPU run：
  - `void cumac::multiCellScheduler::run(cudaStream_t strm)`
  - 文件：`cuMAC/src/4T4R/multiCellScheduler.cu`
  - 行号：5739-5762
- GPU 默认 kernel：
  - `static __global__ __launch_bounds__ (1024, MinBlkPerSM_) void cumac::multiCellSchedulerKernel_noPrdMmseIrc(mcDynDescr_t* pDynDescr)`
  - 文件：`cuMAC/src/4T4R/multiCellScheduler.cu`
  - 行号：693-988
- CPU setup：
  - `void cumac::multiCellSchedulerCpu::setup(cumacCellGrpUeStatus* cellGrpUeStatus, cumacSchdSol* schdSol, cumacCellGrpPrms* cellGrpPrms, cumacSimParam* simParam, uint8_t in_columnMajor)`
  - 文件：`cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`
  - 行号：837-881
- CPU run：
  - `void cumac::multiCellSchedulerCpu::run()`
  - 文件：`cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`
  - 行号：1011-1075
- CPU 默认实现：
  - `void cumac::multiCellSchedulerCpu::multiCellSchedulerCpu_noPrdMmse()`
  - 文件：`cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`
  - 行号：41-130

#### 默认 kernel 是怎么选中的

`kernelSelect()` 的选择逻辑是：

- `DL == 1`
- `precodingScheme == 0`
- `allocType == 0`
- `columnMajor == 1`
- `lightWeight == 0`

于是最终落到：

- `multiCellSchedulerKernel_noPrdMmseIrc`

直接代码锚点：

- `multiCellScheduler.cu` 5288-5315 行

#### 这个 kernel 里 PF metric 怎么算

默认 type-0 PF scheduler 的核心逻辑在 `multiCellSchedulerKernel_noPrdMmseIrc()`：

1. 对每个 `(cell, rbg)`，枚举该 cell 下的 candidate selected UE
2. 为每个 UE 计算干扰协方差矩阵 `C`
3. 计算 `C^-1`
4. 计算 `H^H C^-1 H + I`
5. 取逆后从对角元素恢复每层 `sinrTemp`
6. 累加该 UE 在当前 PRG 上的速率
7. 再做 PF 归一化
8. 当前 `(cell, rbg)` 选取 metric 最大的 UE

代码锚点：

- 干扰协方差矩阵 `C`：782-809 行
- `C^-1`：812-846 行
- `H^H C^-1`：854-868 行
- `H^H C^-1 H + I`：877-897 行
- 最终逆矩阵：906-940 行
- 每 layer `sinrTemp` 与 `postEqSinr`：949-953 行
- PF metric：
  - `pfMetric = pow(sum_rate, betaCoeff) / avgRates[uIdx]`
  - 957-959 行
- 最终取最大 UE 并写 `allocSol`：972-985 行

#### 这里的速率和 SINR 细节

内核里先算：

```text
sinrTemp = 1 / diag((H^H C^-1 H + I)^-1)
```

随后：

- `postEqSinr = sinrTemp - 1`
- `rate += W * log2(sinrTemp)`

见 949-953 行。

也就是说这里的 `sinrTemp` 其实是 `1 + SINR`，因此后面速率公式没有再写 `1 + ...`。

### 6.5 CPU 参考实现与 GPU kernel 是一一对应的

`multiCellSchedulerCpu_noPrdMmse()` 与 GPU 默认 kernel 做的是同一件事：

- 计算干扰协方差矩阵
- 计算 MMSE-IRC 等效 SINR
- 按当前 selected UE 的 `avgRates[ueIdx]` 做 PF 归一化
- 对每个 `(cell, rbg)` 选 metric 最大的 UE

对应代码锚点：

- 干扰协方差矩阵：67-88 行
- 逆矩阵：88-102 行
- 速率与 `postEqSinr`：104-115 行
- PF metric：115-118 行
- 写 `allocSol`：121 行

### 6.6 type-1 PF 是原始 PF 的“可选分支”，不是默认分支

虽然 `677099a` 默认不走 type-1，但原始 PF 的 type-1 实现也在同一套 native 代码里。

#### type-1 关键函数引用

- GPU no-precoding column-major：
  - `static __global__ __launch_bounds__ (1024, MinBlkPerSM_) void cumac::multiCellSchedulerKernel_type1_NoPrdMmseIrc_cm(mcDynDescr_t* pDynDescr)`
  - 文件：`cuMAC/src/4T4R/multiCellScheduler.cu`
  - 行号：1274-1916
- GPU no-precoding row-major：
  - `static __global__ __launch_bounds__ (1024, MinBlkPerSM_) void cumac::multiCellSchedulerKernel_type1_NoPrdMmseIrc_rm(mcDynDescr_t* pDynDescr)`
  - 文件：`cuMAC/src/4T4R/multiCellScheduler.cu`
  - 行号：2417 起
- GPU SVD：
  - `static __global__ __launch_bounds__ (1024, MinBlkPerSM_) void cumac::multiCellSchedulerKernel_type1_svdPrdMmseIrc_cm(mcDynDescr_t* pDynDescr)`
  - 文件：`cuMAC/src/4T4R/multiCellScheduler.cu`
  - 行号：1919-2415
- CPU no-precoding column-major：
  - `void cumac::multiCellSchedulerCpu::multiCellSchedulerCpu_type1_NoPrdMmse_cm()`
  - 文件：`cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`
  - 行号：402-561
- CPU no-precoding row-major：
  - `void cumac::multiCellSchedulerCpu::multiCellSchedulerCpu_type1_NoPrdMmse_rm()`
  - 文件：`cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`
  - 行号：563-722
- CPU SVD：
  - `void cumac::multiCellSchedulerCpu::multiCellSchedulerCpu_type1_svdPrdMmse_cm()`
  - 文件：`cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`
  - 行号：235-400

#### type-1 PF 的核心区别

type-1 不是给每个 PRG 独立选 UE，而是：

- 先对每个 `(UE, PRG)` 计算 PF metric
- 再把离散的 per-PRG metric 合并成“连续 PRG 区间”
- 最后写入 type-1 格式的 `allocSol[2*uIdx], allocSol[2*uIdx+1]`

#### CPU type-1 的实现方式

CPU 版本非常直观：

- 为每个 `(ueIdx, rbgIdx)` 算 `pfMetric`
  - 497 行或 658 行附近
- 把所有 metric 放进 `std::vector<pfMetric>`
- 全局排序
  - 502-506 行、663-667 行
- 顺序扫一遍，尽量把同一个 UE 的带宽区间向左右扩展成连续段
  - 514-552 行、675-713 行

#### GPU type-1 的实现方式

GPU 版本会把 `(UE, PRG)` metric 先写到：

- `pfMetricArr`
- `pfIdArr`

然后再做连续带宽分配：

- `multiCellSchedulerKernel_type1_NoPrdMmseIrc_cm()` 使用 `Parallel Riding Peaks`
  - 1532-1911 行
- `multiCellSchedulerKernel_type1_svdPrdMmseIrc_cm()` 先全局 bitonic sort，再做 peak/gap fill
  - 2202-2411 行

所以 `pfMetricArr/pfIdArr` 是 type-1 才真正有意义的中间结果，`api.h` 688-698 行的注释也明确说明了这一点。

### 6.7 `network::run()` 只是把 GPU 结果搬回 host，不负责 PF 决策

关键函数：

- `void cumac::network::run(cudaStream_t strm)`
  - 文件：`cuMAC/examples/network.cu`
  - 行号：1179-1187

它做的是：

- `allocSol` 从 GPU 拷回 host
- `layerSelSol` 从 GPU 拷回 host
- `mcsSelSol` 从 GPU 拷回 host
- 如有需要再拷 `cellAssoc`

这一步不是 PF 算法本身，但它是 GPU PF 与后续速率更新之间的桥梁。

### 6.8 PF feedback 的闭环：平均速率回写

这是原始 PF 的最后一环，也是下一 TTI 会再次用到的反馈状态。

#### 关键函数引用

- `void cumac::network::updateDataRateUeSelCpu(int slotIdx)`
  - 文件：`cuMAC/examples/network.cu`
  - 行号：2128-2329
- `void cumac::network::updateDataRateGpu(int slotIdx)`
  - 文件：`cuMAC/examples/network.cu`
  - 行号：2530-2740
- `void cumac::network::updateDataRateAllActiveUeCpu(const int slotIdx)`
  - 文件：`cuMAC/examples/network.cu`
  - 行号：4380-4408
- `void cumac::network::updateDataRateAllActiveUeGpu(const int slotIdx)`
  - 文件：`cuMAC/examples/network.cu`
  - 行号：4346-4378

#### 第一步：更新 selected UE 域的 `avgRates`

`updateDataRateUeSelCpu()` / `updateDataRateGpu()` 会重新根据本 TTI 的调度结果求每个 selected UE 的瞬时吞吐 `perUeThr[uIdx]`，然后做指数滑动平均：

```text
avgRates[uIdx] = (1 - pfAvgRateUpd) * avgRates[uIdx] + pfAvgRateUpd * perUeThr[uIdx]
```

代码锚点：

- CPU：2313 行
- GPU：2724 行

这里更新的是 selected UE 域的分母，直接影响“下一轮 scheduler 的 PF metric”。

#### 第二步：把 selected UE 域再回写到 active UE 域

`updateDataRateAllActiveUeCpu()` / `updateDataRateAllActiveUeGpu()` 会遍历所有 active UE：

- 如果该 UE 在本 TTI 出现在 `setSchdUePerCellTTI`
  - 用对应 selected UE 的 `avgRates` 覆盖 `avgRatesActUe`
- 否则只做衰减：

```text
avgRatesActUe[uIdx] = (1 - pfAvgRateUpd) * avgRatesActUe[uIdx]
```

代码锚点：

- CPU：4385-4395 行
- GPU：4352-4363 行

这一步更新的是“下一轮 UE selection 的 PF 分母”。

#### 这就是完整闭环

所以 native PF 的反馈闭环其实是：

1. `avgRatesActUe` 参与 UE selection
2. `ueDownSel4TrKernel()` 把它投影成 `avgRates`
3. `avgRates` 参与 PRG scheduler
4. 调度结束后重新更新 `avgRates`
5. 再把 `avgRates` 回写/衰减成新的 `avgRatesActUe`

下一 TTI 再重复上述过程。

## 7. 一张图看懂默认 native PF 主链路

```text
run_stageB_main_experiment.sh (--custom-ue-prg 0)
  -> multiCellSchedulerUeSelection/main.cpp::main()
  -> cumac::network::createAPI()
  -> TTI loop
     -> cumac::network::setupAPI()
     -> cumac::multiCellSinrCal::{setup, run}
     -> cumac::multiCellSinrCal::{setup_wbSinr, run}
     -> cumac::network::cpySinrGpu2Cpu()
     -> cumac::multiCellUeSelection::{setup, run}
     -> cumac::multiCellUeSelectionCpu::{setup, run}
     -> cumac::network::{ueDownSelectGpu, ueDownSelectCpu}
     -> cumac::multiCellScheduler::{setup, run}
     -> cumac::multiCellSchedulerCpu::{setup, run}
     -> cumac::network::run()
     -> cumac::network::{updateDataRateUeSelCpu, updateDataRateGpu}
     -> cumac::network::{updateDataRateAllActiveUeCpu, updateDataRateAllActiveUeGpu}
```

## 8. 最值得记住的三个“关键理解点”

### 8.1 原始 PF 不是一个函数，而是三段式

原始 PF 真正是三段一起构成的：

- `wbSinr -> PF UE selection`
- `selected UE -> PF scheduler`
- `instantaneous throughput -> avgRates / avgRatesActUe 回写`

只看 `multiCellScheduler*.cu/cpp` 还不够，必须把 `multiCellUeSelection*` 和 `network::updateDataRate*` 一起看，才是完整 PF。

### 8.2 `avgRatesActUe` 和 `avgRates` 不是一回事

- `avgRatesActUe`：active UE 域，供 UE 选择使用
- `avgRates`：selected UE 域，供 PRG scheduler 使用

这两个数组之间靠 `ueDownSel4TrKernel()` 和 `updateDataRateAllActiveUe*()` 来回投影。

### 8.3 `677099a` 默认 baseline PF 是 type-0，不是 type-1

如果你后面要把这份梳理和 PFQ、GNNRL、自定义 UE+PRG 做对比，默认应该先对齐到：

- `multiCellUeSelKernel`
- `multiCellSchedulerKernel_noPrdMmseIrc`
- `updateDataRateUeSelCpu / updateDataRateGpu`
- `updateDataRateAllActiveUeCpu / updateDataRateAllActiveUeGpu`

这才是 `677099a` 的“原始 PF 默认链路”。

## 9. 关键函数引用清单

下面给一份更适合检索代码时直接使用的“完整函数引用”清单。

| 完整函数引用 | 文件 | `677099a` 行号 | 作用 |
|---|---|---:|---|
| `int main(int argc, char* argv[])` | `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp` | 1-980 | 原始 PF 示例/Stage-B 二进制入口 |
| `void cumac::network::createAPI()` | `cuMAC/examples/network.cu` | 778-1002 | 创建 PF 相关 API 缓冲区并初始化平均速率 |
| `void cumac::network::setupAPI(cudaStream_t strm)` | `cuMAC/examples/network.cu` | 1004-1019 | 每 TTI 同步 active UE PF 状态到 GPU |
| `void cumac::multiCellSinrCal::setup(cumacCellGrpPrms* cellGrpPrms, uint8_t in_columnMajor, cudaStream_t strm)` | `cuMAC/src/tools/multiCellSinrCal.cu` | 891-933 | 配置 subband SINR kernel |
| `void cumac::multiCellSinrCal::setup_wbSinr(cumacCellGrpPrms* cellGrpPrms, cudaStream_t strm)` | `cuMAC/src/tools/multiCellSinrCal.cu` | 935-970 | 配置 wideband SINR kernel |
| `void cumac::multiCellSinrCal::run(cudaStream_t strm)` | `cuMAC/src/tools/multiCellSinrCal.cu` | 972-980 | 执行 SINR kernel |
| `static __global__ void cumac::multiCellSinrCalKernel_noPrdMmseIrc_cm(mcSinrCalDynDescr_t* pDynDescr)` | `cuMAC/src/tools/multiCellSinrCal.cu` | 49-280 | 默认 subband SINR 计算 |
| `static __global__ void cumac::multiCellWideBandSinrCalKernel(mcSinrCalDynDescr_t* pDynDescr)` | `cuMAC/src/tools/multiCellSinrCal.cu` | 820-832 | `postEqSinr -> wbSinr` |
| `void cumac::multiCellUeSelection::setup(cumacCellGrpUeStatus* cellGrpUeStatus, cumacSchdSol* schdSol, cumacCellGrpPrms* cellGrpPrms, cudaStream_t strm)` | `cuMAC/src/4T4R/multiCellUeSelection.cu` | 260-306 | 配置 GPU PF UE selection |
| `void cumac::multiCellUeSelection::run(cudaStream_t strm)` | `cuMAC/src/4T4R/multiCellUeSelection.cu` | 308-339 | 执行 GPU PF UE selection |
| `static __global__ void cumac::multiCellUeSelKernel(mcUeSelDynDescr_t* pDynDescr)` | `cuMAC/src/4T4R/multiCellUeSelection.cu` | 102-161 | 默认 GPU PF UE selection kernel |
| `void cumac::multiCellUeSelectionCpu::setup(cumacCellGrpUeStatus* cellGrpUeStatus, cumacSchdSol* schdSol, cumacCellGrpPrms* cellGrpPrms)` | `cuMAC/src/4T4R/multiCellUeSelectionCpu.cpp` | 152-187 | 配置 CPU PF UE selection |
| `void cumac::multiCellUeSelectionCpu::run()` | `cuMAC/src/4T4R/multiCellUeSelectionCpu.cpp` | 189-196 | 执行 CPU PF UE selection |
| `void cumac::multiCellUeSelectionCpu::multiCellUeSelCpu()` | `cuMAC/src/4T4R/multiCellUeSelectionCpu.cpp` | 32-86 | CPU PF UE selection 核心逻辑 |
| `__global__ void cumac::ueDownSel4TrKernel(...)` | `cuMAC/examples/network.cu` | 3871-3978 | active UE 视图转 selected UE 视图 |
| `void cumac::network::ueDownSelectCpu()` | `cuMAC/examples/network.cu` | 4019-4068 | CPU scheduler 前的数据重排 |
| `void cumac::network::ueDownSelectGpu()` | `cuMAC/examples/network.cu` | 4070-4132 | GPU scheduler 前的数据重排 |
| `void cumac::multiCellScheduler::kernelSelect()` | `cuMAC/src/4T4R/multiCellScheduler.cu` | 5261-5397 | 选择 PF scheduler kernel |
| `void cumac::multiCellScheduler::setup(cumacCellGrpUeStatus* cellGrpUeStatus, cumacSchdSol* schdSol, cumacCellGrpPrms* cellGrpPrms, cumacSimParam* simParam, uint8_t in_columnMajor, uint8_t in_halfPrecision, uint8_t in_lightWeight, float in_percSmNumThrdBlk, cudaStream_t strm)` | `cuMAC/src/4T4R/multiCellScheduler.cu` | 5400-5499 | 配置 GPU PF scheduler |
| `void cumac::multiCellScheduler::run(cudaStream_t strm)` | `cuMAC/src/4T4R/multiCellScheduler.cu` | 5739-5762 | 执行 GPU PF scheduler |
| `static __global__ __launch_bounds__ (1024, MinBlkPerSM_) void cumac::multiCellSchedulerKernel_noPrdMmseIrc(mcDynDescr_t* pDynDescr)` | `cuMAC/src/4T4R/multiCellScheduler.cu` | 693-988 | 默认 GPU type-0 PF scheduler |
| `static __global__ __launch_bounds__ (1024, MinBlkPerSM_) void cumac::multiCellSchedulerKernel_type1_NoPrdMmseIrc_cm(mcDynDescr_t* pDynDescr)` | `cuMAC/src/4T4R/multiCellScheduler.cu` | 1274-1916 | GPU type-1 no-precoding PF scheduler |
| `static __global__ __launch_bounds__ (1024, MinBlkPerSM_) void cumac::multiCellSchedulerKernel_type1_svdPrdMmseIrc_cm(mcDynDescr_t* pDynDescr)` | `cuMAC/src/4T4R/multiCellScheduler.cu` | 1919-2415 | GPU type-1 SVD PF scheduler |
| `void cumac::multiCellSchedulerCpu::setup(cumacCellGrpUeStatus* cellGrpUeStatus, cumacSchdSol* schdSol, cumacCellGrpPrms* cellGrpPrms, cumacSimParam* simParam, uint8_t in_columnMajor)` | `cuMAC/src/4T4R/multiCellSchedulerCpu.cpp` | 837-881 | 配置 CPU PF scheduler |
| `void cumac::multiCellSchedulerCpu::run()` | `cuMAC/src/4T4R/multiCellSchedulerCpu.cpp` | 1011-1075 | 执行 CPU PF scheduler |
| `void cumac::multiCellSchedulerCpu::multiCellSchedulerCpu_noPrdMmse()` | `cuMAC/src/4T4R/multiCellSchedulerCpu.cpp` | 41-130 | CPU 默认 type-0 PF scheduler |
| `void cumac::multiCellSchedulerCpu::multiCellSchedulerCpu_type1_NoPrdMmse_cm()` | `cuMAC/src/4T4R/multiCellSchedulerCpu.cpp` | 402-561 | CPU type-1 no-precoding PF scheduler |
| `void cumac::network::run(cudaStream_t strm)` | `cuMAC/examples/network.cu` | 1179-1187 | 把 GPU 调度结果拷回 host |
| `void cumac::network::updateDataRateUeSelCpu(int slotIdx)` | `cuMAC/examples/network.cu` | 2128-2329 | 更新 CPU selected UE 域平均速率 |
| `void cumac::network::updateDataRateGpu(int slotIdx)` | `cuMAC/examples/network.cu` | 2530-2740 | 更新 GPU selected UE 域平均速率 |
| `void cumac::network::updateDataRateAllActiveUeGpu(const int slotIdx)` | `cuMAC/examples/network.cu` | 4346-4378 | GPU selected UE 域回写到 active UE 域 |
| `void cumac::network::updateDataRateAllActiveUeCpu(const int slotIdx)` | `cuMAC/examples/network.cu` | 4380-4408 | CPU selected UE 域回写到 active UE 域 |

