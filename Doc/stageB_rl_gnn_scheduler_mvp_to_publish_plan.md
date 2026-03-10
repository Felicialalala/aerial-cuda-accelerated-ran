# Stage-B 强化学习 + GNN 智能调度设计（MVP→可发表版）

## 1. 目标与结论（先给结论）

基于当前仓库的 Stage-B 实现，已经具备 RL+GNN 调度所需的大部分底层输入能力，尤其是：

- UE 级队列/历史速率/ACK-BLER 相关状态；
- UE×RBG 级信道质量（`postEqSinr`）与 CFR（`estH_fr*`）；
- 调度动作结果（`setSchdUePerCellTTI` + `allocSol`）；
- 每 TTI 的 SINR 更新链路与调度执行闭环。

当前主要缺口有两个：

1. 没有原生导出严格的 HOL delay（现有是 `queue_delay_est_ms` 估计值）；
2. 缺少面向 RL 训练的 per-TTI transition 持久化（现有 Stage-B 汇总脚本主要看终态 KPI）。

因此可行路径是：先按 MVP 跑通（不依赖 HOL、TE、IB），再增量加入 TE 剪枝与信息瓶颈正则。

---

## 2. Stage-B 闭环映射（输入→模型→输出→反馈）

- A. TTI 观测采集：来自 `cumacCellGrpUeStatus/cumacCellGrpPrms/cumacSchdSol` 与运行时统计。
- B. 构图：以 cell 为节点、邻区干扰耦合为边（先静态 Top-K，再加动态权重）。
- C. Cell-GNN 编码：得到每个 cell 的干扰上下文 `c_i^t`。
- D. RBG 动作头：每个 `cell×RBG` 输出 `UE/NO_TX` 多分类（含 action mask）。
- E. 下发动作：转换为 Type-0/Type-1 资源分配并进入现有调度执行链路。
- F. 反馈收集：ACK/NACK、吞吐、BLER、队列变化、SINR 等。
- G. PPO 更新：先标准 PPO，后续加入 KL 型信息瓶颈与 TE 离线剪枝。

---

## 3. 字段级输入清单（字段 -> 含义 -> 单位 -> 所属层 -> 消费模块）

下面按“节点/边/UE-RBG/反馈”分组，给出当前项目可直接使用或可低成本构造的字段。

### 3.1 节点（Cell）输入

| 字段（源码或衍生） | 含义 | 单位 | 所属层 | 被哪个模块消费 | 当前状态 |
|---|---|---|---|---|---|
| `sum(bufferSize for UE in cell)` | 小区总排队字节 | Bytes | MAC/队列 | `CellGraphBuilder`（节点负载） | 可直接构造 |
| `active_ue_count`（由 `cellAssocActUe`+`bufferSize`） | 活跃 UE 数 | 个 | MAC | `CellGraphBuilder` | 可直接构造 |
| `mean/p95(wbSinr)` | 小区链路质量统计 | dB 或线性值 | PHY | `CellGraphBuilder` | 可直接构造 |
| `ack_rate`（由 `tbErrLastActUe`） | 上一窗口 ACK 比率 | 比例 | PHY/MAC | `CellGraphBuilder`、`RewardBuilder` | 可直接构造 |
| `bler`（由 `tbErrLastActUe` 或 `tbErrLast`） | BLER 估计 | 比例 | PHY/MAC | `CellGraphBuilder`、`RewardBuilder` | 可直接构造 |
| `rbg_utilization`（由 `allocSol`） | 上 TTI RBG 占用率 | 比例 | MAC | `CellGraphBuilder`、`EdgeFeatureBuilder` | 可直接构造 |
| `avg_mcs`（由 `mcsSelSol` 聚合） | 小区平均 MCS | index | PHY/MAC | `CellGraphBuilder` | 可直接构造 |
| `pf_history`（`avgRatesActUe`聚合） | PF 历史吞吐上下文 | bps | MAC | `CellGraphBuilder` | 可直接构造 |
| `p95_hol_delay` | 严格 HOL 时延 | ms | L2/队列 | `CellGraphBuilder`、`RewardBuilder` | 缺失（需新增） |

### 3.2 边（j→i）输入

| 字段（源码或衍生） | 含义 | 单位 | 所属层 | 被哪个模块消费 | 当前状态 |
|---|---|---|---|---|---|
| `g_{j->i}`（由 `estH_fr*`/`postEqSinr`/拓扑统计衍生） | 邻区到本区干扰耦合强度 | 归一化无量纲 | PHY | `EdgeFeatureBuilder`、`CellGraphEncoder` | 可构造（建议离线静态先行） |
| `neighbor_load`（邻区 buffer/active UE） | 邻区负载活跃度 | Bytes/个 | MAC | `EdgeFeatureBuilder` | 可直接构造 |
| `neighbor_rbg_util`（由邻区 `allocSol`） | 邻区发射活跃度代理 | 比例 | MAC | `EdgeFeatureBuilder` | 可直接构造 |
| `rbg_overlap_ratio`（本区与邻区 bitmap 重叠） | 频域重叠比例 | 比例 | MAC | `EdgeFeatureBuilder`、`CellGraphEncoder` | 可构造（MVP 可选） |
| `te_score`（离线计算） | 传输熵边权 | nat/bit | 统计层 | `EdgePruner` | 后续增强 |

### 3.3 UE/RBG 输入（动作头核心）

| 字段（源码或衍生） | 含义 | 单位 | 所属层 | 被哪个模块消费 | 当前状态 |
|---|---|---|---|---|---|
| `bufferSize[u]` | UE 队列字节 | Bytes | MAC/队列 | `RbgPolicyHead`、`ActionMaskBuilder` | 可直接使用 |
| `avgRatesActUe[u]` | UE 历史平均吞吐（PF） | bps | MAC | `RbgPolicyHead`、`RewardBuilder` | 可直接使用 |
| `tbErrLastActUe[u]` | UE 上次 TB 成败（ACK/NACK） | {-1,0,1} | PHY/MAC | `RbgPolicyHead`、`RewardBuilder` | 可直接使用 |
| `newDataActUe[u]` | 新传/重传标记 | {0,1} | MAC/HARQ | `ActionMaskBuilder` | 可直接使用 |
| `cqiActUe[u]` | UE 宽带 CQI | index | PHY | `RbgPolicyHead` | 可直接使用 |
| `riActUe[u]` | UE RI | layers | PHY | `RbgPolicyHead` | 可直接使用 |
| `wbSinr[u,*]` | UE 宽带 SINR | dB/linear | PHY | `RbgPolicyHead` | 可直接使用 |
| `postEqSinr[u,m,*]` | UE×RBG×层 SINR | linear | PHY | `RbgPolicyHead`（UE×RBG 质量） | 可直接使用 |
| `estH_fr*_actUe` | UE×RBG CFR 原始复信道 | complex | PHY | `FeatureExtractor`（可衍生 CQI/SINR/耦合） | 可直接使用（需导出/内存访问） |
| `rbg_index` | RBG 序号嵌入 | index | MAC | `RbgPolicyHead` | 可直接构造 |
| `bwp_edge_flag` | RBG 频域边缘标记 | bool | MAC/PHY | `RbgPolicyHead` | 可构造（MVP 可选） |
| `hol_delay[u]` | UE 严格 HOL delay | ms | L2/队列 | `RbgPolicyHead`、`RewardBuilder` | 缺失（需新增） |

### 3.4 动作与反馈（训练信号）

| 字段（源码或衍生） | 含义 | 单位 | 所属层 | 被哪个模块消费 | 当前状态 |
|---|---|---|---|---|---|
| `setSchdUePerCellTTI` | 每 cell 当 TTI 调度 UE 集 | UE ID | MAC | `ActionAdapter`、`ReplayWriter` | 可直接使用 |
| `allocSol` | PRG/RBG 分配解 | 索引区间或bitmap | MAC | `ActionAdapter`、`ReplayWriter` | 可直接使用 |
| `mcsSelSol` | UE MCS 动作 | index | PHY/MAC | `ActionAdapter`、`ReplayWriter` | 可直接使用 |
| `layerSelSol` | UE 层数动作 | layers | PHY | `ActionAdapter`、`ReplayWriter` | 可直接使用 |
| `predBler`（`getPredictedBlerCpu`） | 预测 BLER | 比例 | PHY | `RewardBuilder` | 可直接使用 |
| `tbErrLast/tbErrLastActUe` | 实际 TB 成败 | 比例/标记 | PHY/MAC | `RewardBuilder` | 可直接使用 |
| `sumInsThrRecords*` | TTI 系统瞬时吞吐 | bps | PHY/MAC | `RewardBuilder`、评测 | 可直接使用 |
| `avgRatesActUe` 更新 | UE 长期吞吐更新 | bps | MAC | `RewardBuilder`（公平项） | 可直接使用 |
| `queue_delay_est_ms` | 队列时延估计值 | ms | 统计层 | `RewardBuilder`（过渡方案） | 可直接使用（非严格 HOL） |

---

## 4. “MVP→可发表版”输入侧满足度评估

### 4.1 MVP（先跑通）必需输入与判定

| MVP 必需项 | 当前是否满足 | 说明 |
|---|---|---|
| Cell 负载（buffer/active UE） | 满足 | 可由 `bufferSize` + `cellAssocActUe` 聚合 |
| Cell 链路质量（SINR/ACK/BLER） | 满足 | `wbSinr` + `tbErrLastActUe` 可得 |
| Edge 耦合强度（静态） | 部分满足 | 可由 CFR/拓扑离线构造，建议先固定 Top-K 邻居 |
| Edge 动态活跃度 | 满足 | `allocSol` 可得到邻区发射活跃度 |
| UE 队列/历史吞吐/CQI | 满足 | `bufferSize/avgRatesActUe/cqiActUe` |
| UE×RBG 质量 | 满足 | `postEqSinr`（或由 `estH_fr` 聚合） |
| 动作可执行（UE/NO_TX + mask） | 满足 | `allocSol/setSchdUePerCellTTI` 与 buffer/HARQ 可做 mask |
| 奖励（吞吐+BLER+时延） | 部分满足 | 吞吐/BLER可直接取，时延先用 `queue_delay_est_ms` 替代 HOL |

### 4.2 可发表版新增要求与判定

| 可发表版增强项 | 当前是否满足 | 建议 |
|---|---|---|
| 严格 HOL（平均/P95/P99） | 不满足 | 在 traffic 队列中增加包级时间戳并导出 HOL |
| 跨场景泛化评测输入（拓扑/负载/信道 shift） | 满足 | 已有拓扑 seed、CDL profile、负载参数可扫场景 |
| TE/DI 边剪枝特征 | 部分满足 | 先离线从 replay 计算 TE，再回写 edge mask |
| 信息瓶颈（KL on message） | 满足（模型侧） | 不依赖新字段，只需在 GNN 消息层加分布参数与 KL |
| 复杂度指标（推理时延/FLOPs/稀疏度） | 部分满足 | 增加推理计时与图统计日志 |

---

## 5. 分阶段实施路线（兼容 MVP→可发表版）

## 阶段 M0：数据接口最小补齐（先于训练）

- 新增 `RLReplayWriter`（建议 JSONL/Parquet）按 TTI 写入：
  - `state_cell`, `state_edge`, `state_ue_rbg`, `action`, `reward_terms`, `next_state`。
- 复用现有内存字段，不改 cuMAC 核心算子，只在 `main.cpp` TTI 循环加 hook。
- 保持与现有 `run_stageB_main_experiment.sh` 共存，新增开关如 `--rl-trace 1`。

## 阶段 M1：MVP 训练闭环（可跑通）

- 模型：`1~2 层 GAT/GCN + per-RBG MLP 分类头`；
- 动作：`UE/NO_TX`，mask 至少包含 `buffer=0` 和不可调度 UE；
- 奖励：`r = α*sum_throughput - γ*avg_queue_delay_est - δ*bler`；
- 边特征：先“静态耦合 + 动态活跃度”，不引入 TE/IB；
- 目标：稳定优于 PF/RR baseline 的至少一个关键指标（如 edge throughput 或 P95 delay proxy）。

## 阶段 M2：可发表版增强-1（TE 剪枝）

- 从 M1 replay 离线计算 TE/DI，得到边重要度；
- 仅保留 Top-K 或 TE 超阈值边，形成稀疏图；
- 报告稀疏度-性能曲线与推理时延变化。

## 阶段 M3：可发表版增强-2（信息瓶颈）

- 在 `m_{j->i}` 消息上输出高斯参数并采样；
- 在 PPO 目标中加入 `-β * KL(q(m|x,e) || N(0,I))`；
- 报告 Reward-Information frontier（性能 vs KL 预算）。

---

## 6. 对当前脚本/产物的具体说明（避免误用）

- `cuMAC/scripts/summarize_stageB_matrix.py` 是场景级 KPI 汇总器，输入为 `kpi_summary.json`；它不提供 per-TTI RL 训练样本。
- Stage-B 默认产物侧重最终统计，且在 `compact-output=1` 下会清理部分日志。
- 若要做 RL 训练，请依赖运行时内存字段 + 新增 replay 导出，而不是只依赖 `stageB_kpi_matrix.csv`。

---

## 7. 最小验收标准（建议）

- MVP 验收：
  - 能持续产出 `(s_t, a_t, r_t, s_{t+1})`；
  - PPO 训练可收敛；
  - 至少 1 个关键指标不劣于 PF baseline。

- 可发表版验收：
  - 补齐 HOL（至少平均 + P95）；
  - 完成 TE 剪枝与信息瓶颈消融；
  - 完成 topology/load/channel shift 泛化实验并给出 generalization gap。

