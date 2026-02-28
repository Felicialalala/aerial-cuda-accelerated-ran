# 簇级智能调度分步实施详情（先接管 UE Selection + PRG/RBG）

## 1. 文档目标

本方案基于当前仓库可运行环境（`multiCellSchedulerUeSelection` + Stage-B 脚本）设计一条低风险落地路线：

1. 第一步只接管 `UE selection + PRG(RBG) allocation`。
2. `layerSel/mcsSel` 暂时沿用现有模块，确保链路先稳定可跑。
3. 最终演进到时域、频域、空域、功率域联合调度。

说明：当前 4T4R 路径代码里主称呼为 `PRG`，与常见 `RBG` 在本项目语境等价。本文统一写作 `PRG(RBG)`。

## 2. 当前实验环境基线（冻结点）

1. 入口程序：`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
2. Stage-B 脚本：`cuMAC/scripts/run_stageB_main_experiment.sh`
3. 当前调度流水（每 TTI）：
   1. CSI/SINR 更新
   2. UE selection
   3. UE downselection
   4. PRG allocation
   5. layer selection
   6. MCS selection
   7. `net->run()` 下发与吞吐统计
4. 目标配置：7 cell，4T4R，30 kHz，100 MHz，`nPrbGrp=68`，CDL-C（可扩展）。

## 3. 总体架构设计

## 3.1 新增模块边界

新增一个可替换模块（建议名：`CustomUePrgScheduler`），只负责输出两类结果：

1. `setSchdUePerCellTTI`（UE 选择）
2. `allocSol`（PRG 分配）

保持不变：

1. `multiCellLayerSel`（空域层选择，短期保留）
2. `mcsSelectionLUT`（MCS，短期保留）
3. `multiCellSinrCal`、信道、traffic、KPI 汇总链路

## 3.2 运行模式

通过开关切换：

1. `native`：全原生流程
2. `custom_ue_prg`：UE+PRG 使用自定义模块，layer/mcs 仍原生

推荐在 `multiCellSchedulerUeSelection` 增加 CLI 选项：

1. `--custom-ue-prg 0|1`
2. `--custom-policy <name>`
3. `--custom-config <path>`

## 3.3 数据契约（只接管两个输出）

必须严格写入 API 缓冲：

1. `schdSol->setSchdUePerCellTTI`
2. `schdSol->allocSol`

不改写（继续由原模块给出）：

1. `schdSol->layerSelSol`
2. `schdSol->mcsSelSol`

## 4. 分步路线图（四域支持）

## 阶段 S0：基线冻结与回归集

目标：

1. 固定当前 Stage-B 结果作为基线。
2. 固定 3 组种子与 2 组信道配置用于回归。

输出：

1. `native` 模式基线 KPI（`inst_mean`, `per_ue_p5`, `drop_rate`, `queue_delay`）。
2. 失败诊断与关键日志模板。

## 阶段 S1：时域 + 频域（先实现）

目标：

1. 接管 `UE selection`（时域用户调度入口）。
2. 接管 `PRG(RBG) allocation`（频域资源分配入口）。
3. layer/mcs 继续原生。

核心原则：

1. 先做“可运行正确”，再做“策略更优”。
2. 首版策略可用规则法，不强制上学习模型。

建议首版策略：

1. UE 选择：按 PF 权重 + 队列长度加权排序（每 cell 选 `numUeSchdPerCellTTI`）。
2. PRG 分配：基于每 UE 宽带 SINR 或估算速率的贪心分配，附带最小连续块约束（匹配 type-1）。

验收：

1. 全流程无崩溃，400 TTI 可稳定跑完。
2. `allocSol` 合法性 100%（无越界、无非法区间）。
3. 对比 native：吞吐损失不超过 10%（初版门槛），并可复现。

## 阶段 S2：空域支持（第二步）

目标：

1. 在不替换 `layerSel/mcs` 的前提下，空域“可感知”纳入状态与目标。
2. 提供未来替换 layerSel 的接口位置与数据结构。

实现要点：

1. 在 S1 的状态向量加入 `RI/PMI/奇异值/SINR分层统计`。
2. 增加“空域一致性约束”，避免给 layerSel 不可行的候选 UE/PRG 组合。
3. 保持 `multiCellLayerSel` 原生运行，作为空域保底。

验收：

1. 维持 S1 稳定性。
2. 在高干扰 case 中，`per_ue_p5` 不劣于 S1。

## 阶段 S3：功率域支持（第三步）

目标：

1. 先实现“功率可控输入”，再做“功率可学习决策”。
2. 对 4T4R 与 64T64R 分开推进。

实现要点：

1. 4T4R：先做功率状态驱动与软约束（发射功率预算、边缘UE权重），不直接改底层复杂功控内核。
2. 64T64R：利用现有 `bfPowAllocScheme` 支持范围（0/1）与 beamforming 链路逐步接入。
3. SRS 调度链路可作为 UL 功率控制参考路径。

验收：

1. 功率相关参数变更可观测影响 KPI。
2. 不出现明显稳定性回退。

## 5. 阶段 S1 详细实施计划（本次执行重点）

## 5.1 代码改造清单

1. 新增：`cuMAC/examples/customScheduler/CustomUePrgScheduler.h`
2. 新增：`cuMAC/examples/customScheduler/CustomUePrgScheduler.cpp`
3. 修改：`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
4. 可选：`cuMAC/examples/CMakeLists.txt`（注册新源文件）
5. 可选：`cuMAC/scripts/run_stageB_main_experiment.sh` 增加开关透传参数

## 5.2 主循环替换点（每 TTI）

1. 在 UE selection 阶段：
   1. `native`：调用 `mcUeSelGpu->setup/run`
   2. `custom_ue_prg`：调用 `CustomUePrgScheduler::selectUe(...)` 写入 `setSchdUePerCellTTI`
2. 之后保持 `net->ueDownSelectGpu()` 不变
3. 在 PRG allocation 阶段：
   1. `native`：调用 `mcSchGpu->setup/run`
   2. `custom_ue_prg`：调用 `CustomUePrgScheduler::allocPrg(...)` 写入 `allocSol`
4. layer/mcs 继续调用原有 `mcLayerSelGpu` 与 `mcsSelGpu`

## 5.3 输入状态（首版最小集合）

1. 每 UE：`avgRatesActUe`、`bufferSize`、`wbSinr`
2. 每 cell：可用 PRG 数、负载（平均 buffer）
3. 基础配置：`nCell`、`nUe`、`nActiveUe`、`nPrbGrp`、`numUeSchdPerCellTTI`

## 5.4 输出合法性检查（必须实现）

1. `setSchdUePerCellTTI`：
   1. UE ID 在 `[0, nActiveUe-1]`
   2. 每 cell 不重复
2. `allocSol`（type-1）：
   1. 每 UE 要么 `[-1,-1]`，要么满足 `0 <= start < end <= nPrbGrp`
   2. 同 cell 内 PRG 不冲突
3. 任一检查失败：
   1. 回退 native 结果
   2. 打印 fail reason 计数

## 5.5 日志与KPI增强

新增统计：

1. `custom_ue_changed_ratio`
2. `custom_prg_used_ratio`
3. `fallback_count`
4. `edge_ue_share`（边缘UE被调度占比）

保留现有：

1. throughput 三类统计
2. traffic KPI（offered/served/drop/queue）

## 5.6 里程碑

1. M1：编译通过 + 模式切换可运行
2. M2：400 TTI 稳定跑通，零非法解
3. M3：给出 3C/7C 对比表（native vs custom_ue_prg）

## 6. 与四域调度支持的对应关系

1. 时域：S1 通过 `UE selection` 完成首个可控入口
2. 频域：S1 通过 `PRG(RBG) allocation` 完成首个可控入口
3. 空域：S2 先支持“空域感知+约束”，再考虑替换 `layerSel`
4. 功率域：S3 先接“功率状态与参数”，再逐步扩展到可学习功率决策

## 7. 风险与规避

1. 风险：自定义结果破坏后续 layer/mcs 假设  
   对策：先做严格合法性检查，失败自动回退 native。
2. 风险：首版策略吞吐明显下降  
   对策：设置回退阈值与灰度开关（按场景开关）。
3. 风险：调试成本高  
   对策：增加 `run_key.log` 的 custom 专项关键行。

## 8. 本阶段完成标准（可进入编码）

1. 文档评审通过。
2. 明确输入输出与回退机制。
3. 确定先实现规则版 `CustomUePrgScheduler`，不直接引入 RL/GNN。
4. 完成 S1 后再进入 S2/S3。

