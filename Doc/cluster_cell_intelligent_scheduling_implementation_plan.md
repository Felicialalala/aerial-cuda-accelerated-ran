# 簇级同频小区智能调度实施计划（基于当前仓库能力）

## 1. 目标与约束

- 目标：在当前 Aerial SDK 代码基线上，完成“3 小区可跑通 -> 7 小区主实验 -> 64T64R 扩展”的簇级同频智能调度研究链路。
- 关键制式对齐：TDD + 30 kHz SCS + 100 MHz + 273 PRB。
- 工程原则：优先复用现有 `cuMAC`/`chanModels`/`phase4_test_scripts` 接口，不重造调度主干。

## 2. 现有能力盘点（与计划直接相关）

### 2.1 多小区调度与 MU-MIMO

- 4T4R 多小区调度入口：`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
  - 已支持 DL/UL、Rayleigh/TDL/CDL 切换（`-f 0/1/2/3/4`）。
- 64T64R 多小区 MU-MIMO 入口：`cuMAC/examples/multiCellMuMimoScheduler/main.cpp`
  - 配置文件：`cuMAC/examples/multiCellMuMimoScheduler/config.yaml`
  - 已包含 `nCell`、`nActiveUePerCell`、`nPrbGrp`、`scs`、`nBsAnt` 等关键参数。
- 核心库模块已就位：
  - `cuMAC/src/4T4R/*`
  - `cuMAC/src/64T64R/*`
  - `cuMAC/src/tools/multiCellSinrCal.*`

### 2.2 信道模型与可控干扰

- 3GPP 38.901 信道模型：`testBenches/chanModels/src/*`
- 系统级统计信道配置模板：`testBenches/chanModels/config/statistic_channel_config.yaml`
  - 已具备 `sc_spacing_hz=30000`、`n_prb=273`、`bandwidth_hz=100e6` 字段。
- 示例入口：
  - `testBenches/chanModels/examples/sls_chan/sls_chan_ex.cpp`
  - `testBenches/chanModels/examples/cdl_chan/cdl_chan_ex.cpp`

### 2.3 E2E 原型与多小区自动化

- phase4 自动化（Test MAC + RU emulator + cuPHY controller）：
  - `testBenches/phase4_test_scripts/README.md`
  - `testBenches/phase4_test_scripts/test_config.sh`
- 已支持 `--num-cells`、pattern 化运行、多实例 ML2 等。

## 3. 总体实施路径（建议 3 阶段 + 2 扩展）

## 阶段 A：MVP 跑通（3 小区，4T4R，Rayleigh）

### A1. 场景固化

- 小区：3（中心 + 2 邻区）
- UE：8 UE/小区（中心 2、中间 4、边缘 2）
- 无线参数：30 kHz，100 MHz，273 PRB，TDD
- 信道：先 `Rayleigh`，低速（0-1 m/s）

### A2. 代码接入策略

- 复用 `multiCellSchedulerUeSelection`，先不改核心算子。
- 通过配置/宏参数将 `numCellConst`、`numUePerCellConst`、`numActiveUePerCellConst` 收敛到 MVP 规模（见 `cuMAC/examples/parameters.h`）。
- 先在入口层插入“观测与动作 Hook”，不破坏原调度流水：
  - 观测：每 cell/UE 的 CQI/RI/PMI、buffer、历史速率、SINR 统计。
  - 动作：PRG 配置偏置、功率缩放因子、UE/层选择策略系数。

### A3. 产出

- 可重复运行脚本（固定 seed、固定拓扑）。
- 三组对比：
  - baseline（原始调度）
  - 干扰增强（减 ISD 或边缘 UE 朝向邻区）
  - 协同策略开启（动作生效）
- KPI 至少包括：sum throughput、5%ile UE 吞吐、Jain 公平性、边缘 UE 平均 SINR。

## 阶段 B：主实验（7 小区，4T4R，CDL-C/D）

### B1. 拓扑升级

- 7 小区六边形：1+6。
- ISD 初值建议：400 m（可扫 300/400/500）。
- UE：8 或 12 UE/小区（建议先 8，稳定后扩到 12）。

### B2. 信道升级

- 从 `Rayleigh` 切到 `CDL`（`delay_profile` 与 `delay_spread` 分层）。
- 速度场景分层：0/3/30 km/h。
- 仍保持 30 kHz + 100 MHz + 273 PRB。

### B3. 智能调度策略最小闭环

- 先用“规则+学习混合”方案：
  - 规则层：硬约束（最小保障、功率上限、PRG 冲突规避）。
  - 学习层：在可行动作空间内做簇级优化。
- 建议动作空间分解，降低一次性复杂度：
  - 第 1 版：仅 PRG 协调。
  - 第 2 版：PRG + 功率。
  - 第 3 版：PRG + 功率 + 层/波束（4T4R 下先做层选择）。

## 阶段 C：扩展到 64T64R（MU-MIMO）

### C1. 迁移入口

- 切换到 `multiCellMuMimoScheduler`。
- 使用现有 `config.yaml` 作为主配置，增加实验配置分层（3C/7C、UE 数、相关阈值 `chanCorrThr`）。

### C2. 关注点

- 先锁定 `nPrbPerGrp=4`、`nPrbGrp=68` 的现有分组体系，避免一次改 PRB 粒度。
- 在 MU 分组前后分别记录指标：
  - 组内相关性
  - 组间干扰
  - 波束增益与边缘损失

## 阶段 D：E2E 原型化（可选但建议）

- 将阶段 B/C 已验证策略接入 `phase4_test_scripts` 链路，形成可演示闭环：
  - `run1_RU.sh`
  - `run2_cuPHYcontroller.sh`
  - `run3_testMAC.sh`
- 重点不是 full stack 新开发，而是验证策略在 E2E 配置下的稳定性与可观测性。

## 阶段 E：合同规模与回归体系

- 目标规模：每站 8 UE，7 小区常态运行，必要时扩 12 UE。
- 建立 nightly 回归：固定 3C/7C 两组 smoke case + 1 组压力 case。

## 4. 具体工程改造清单（按代码路径）

### 4.1 新增目录与文件建议

建议新增：

- `Doc/`（当前文档目录）
- `cuMAC/examples/clusterScheduler/`（智能调度实验入口，后续实现）
- `cuMAC/examples/clusterScheduler/config/`（3C/7C/64TR 配置）
- `cuMAC/examples/clusterScheduler/scripts/`（运行与汇总脚本）

### 4.2 观测与动作接口（建议）

依托 `cuMAC/src/api.h` 中 `cumacCellGrpUeStatus`/`cumacCellGrpPrms`/`cumacSchdSol`：

- 观测（State）
  - per-cell：负载、可用 PRG、邻区干扰统计
  - per-UE：CQI/RI/PMI、buffer、历史吞吐、post-eq SINR
- 动作（Action）
  - PRG 资源偏置（cell 间冲突规避）
  - UE 优先级权重（边缘 UE 拉升）
  - 功率缩放（按 PRG 或按 cell）
  - 64TR 阶段加入 MU 分组阈值动态化（如 `chanCorrThr`）
- 奖励（Reward）
  - `R = alpha*sumRate + beta*edgeRate5 - gamma*interfPenalty - delta*fairnessLoss`

### 4.3 干扰可控机制

- 利用 `multiCellSinrCal` 输出 `postEqSinr/wbSinr`，作为干扰响应观测。
- 在拓扑与 UE 摆位层面固定随机种子，保证“动作变化 -> KPI 变化”可复现。

## 5. 配置模板（可直接落地）

## 5.1 MVP（3C, 4T4R）

- `nCell=3`
- `nActiveUePerCell=8`
- `scs=30000`
- `bandwidth=100e6`
- `n_prb=273`
- `fading_type=0`（先 Rayleigh）
- `velocity=[0,0,0]`

## 5.2 主场景（7C, 4T4R）

- `nCell=7`
- `nActiveUePerCell=8/12`
- 信道改为 `CDL`，delay spread 分层（100ns/300ns/1us）

## 5.3 进阶（7C, 64T64R）

- `nBsAnt=64`
- `nUeAnt=4`
- `numUeSchdPerCellTTI` 从 6/8 起步，不直接拉满
- 保持 `nPrbPerGrp=4, nPrbGrp=68`

## 6. 里程碑与验收标准

## M1（1-2 周）

- 3C Rayleigh 跑通，输出基础 KPI 与日志。
- 验收：同样 seed 下结果可复现；动作开启后 KPI 方向正确（至少一项边缘指标改善）。

## M2（2-4 周）

- 7C CDL 主场景完成，产出主结果表与消融实验（PRG-only / PRG+Power / +Layer）。
- 验收：形成可用于论文/汇报的稳定图表。

## M3（4-6 周）

- 64T64R 扩展完成，给出可运行 case 与性能-收益权衡。
- 验收：至少 1 组 64TR 场景稳定运行并输出簇级收益结论。

## 7. 实验矩阵建议（最小充分）

- 拓扑：3C, 7C
- 天线：4T4R, 64T64R
- 信道：Rayleigh, CDL
- UE/小区：8, 12
- 速度：0, 3, 30 km/h
- ISD：300, 400, 500 m

输出维度：

- 吞吐：cell sum / UE 5%ile / cell-edge mean
- 稳定性：多 seed 方差
- 复杂度：单 TTI 调度时延、GPU 占用

## 8. 可执行命令参考（当前仓库）

- 构建示例：
  - `cmake --build <build_dir> --target multiCellSchedulerUeSelection`
  - `cmake --build <build_dir> --target multiCellMuMimoScheduler`
- 4T4R 多小区入口：
  - `<build_dir>/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection -d 1 -f 0`
- 64TR 多小区入口：
  - `<build_dir>/cuMAC/examples/multiCellMuMimoScheduler/multiCellMuMimoScheduler -c cuMAC/examples/multiCellMuMimoScheduler/config.yaml`
- phase4 参数化：
  - `testBenches/phase4_test_scripts/test_config.sh <pattern> --num-cells=7`

## 9. 风险与对策

- 风险 1：信道随机性掩盖策略收益。
  - 对策：先静态低速 + 固定 seed，逐步加复杂度。
- 风险 2：动作空间过大导致训练/调参失控。
  - 对策：分阶段动作解耦（PRG -> PRG+Power -> +空域）。
- 风险 3：从 testbench 到 E2E 偏差较大。
  - 对策：阶段 D 早介入最小 E2E case，不等算法“完美”再接。

## 10. 本计划与现有代码的对应关系（摘要）

- 调度核心：`cuMAC/src/4T4R`, `cuMAC/src/64T64R`
- 调度入口：`cuMAC/examples/multiCellSchedulerUeSelection`, `cuMAC/examples/multiCellMuMimoScheduler`
- 干扰/SINR：`cuMAC/src/tools/multiCellSinrCal.*`
- 信道模型：`testBenches/chanModels/*`
- E2E 自动化：`testBenches/phase4_test_scripts/*`

---

如果下一步需要，我可以继续在 `Doc/` 下补两份配套文档：

- `参数清单与默认值（3C/7C/64TR）`
- `实验结果记录模板（KPI表 + 图表命名规范 + 回归检查项）`
