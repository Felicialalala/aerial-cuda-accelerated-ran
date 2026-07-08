# Stage-B HGraph 下一步优化方案

## 1. 文档定位

本文记录当前 HGraph 智能调度算法的真实实现口径，并给出下一轮优化方向。

当前要解决的问题是：

- HGraph action-mode 已经能在 `3cell + 36UE + RBG16 + TTL200ms + SVD` 场景下达到接近或略优于 PFQ/RRQ 的 aggregate served throughput。
- 但策略表现为低 PRG 利用率、低调度频率、单次集中服务。
- 结果是 packet delay、packet effective service rate、UE 服务节奏仍落后 PFQ/RRQ。

本文不替代 [`current_stageB_effective_configuration.md`](./current_stageB_effective_configuration.md)，而是补充 HGraph 算法结构和下一步训练/部署优化设计。

### 1.1 源码对应关系

当前 HGraph 主链路可按以下源码核对：

| 模块 | 文件 | 说明 |
|---|---|---|
| online bridge / observation | `training/stageb_hgraph/bridge_env.py`、`training/stageb_hgraph/online_protocol.py` | 从仿真侧接收 cell/UE/PRG/post-eq SINR/mask/reward terms |
| snapshot 对齐 | `training/stageb_hgraph/feature_snapshot.py`、`training/stageb_hgraph/types.py` | 将 observation 对齐为固定 `C/U/P/S` 图输入 |
| 图边构造 | `training/stageb_hgraph/edge_generator.py` | 构造 `E_CC/E_CU/E_CP/E_UP/E_PP` 和 UE-PRG edge attrs |
| 图操作/动作合法化 | `training/stageb_hgraph/graph_ops.py` | candidate packing、Type-0 target/action decode、mask 约束 |
| 模型 | `training/stageb_hgraph/model.py` | `StageBHGraphPolicy`、entity encoders、`HeteroMessageLayer`、action heads |
| PPO 训练 | `training/stageb_hgraph/online_ppo.py` | rollout、reward shaping、slot repair、checkpoint 选择 |
| reward | `training/stageb_hgraph/reward_utils.py` | `stageb_v1` 和 `stageb_blankaware` 训练 reward |
| online eval | `training/stageb_hgraph/eval_checkpoint_online.py` | checkpoint 多 seed/单 seed 在线评测 |
| ONNX 导出 | `training/stageb_hgraph/export_onnx.py` | 固定形状 action-mode/logits-mode 导出 |

## 2. 当前现象和诊断

### 2.1 代表性结果

当前 HGraph action-mode seed42 smoke：

| 指标 | HGraph action s42 | PFQ s42 | RRQ s42 |
|---|---:|---:|---:|
| `traffic.served_mbps_est` | 1736.670 | 1734.346 | 1725.653 |
| `traffic.goodput_mbps` | 1555.805 | 1548.486 | 1548.063 |
| `global_kpi.global_tb_bler` | 0.105984 | 0.108009 | 0.108990 |
| `global_kpi.prg_utilization_ratio` | 0.603208 | 0.906142 | 0.866892 |
| `global_kpi.scheduled_ratio_mean` | 0.226817 | - | - |
| `global_kpi.scheduled_ratio_p5` | 0.080250 | - | - |
| `traffic.packet_delay_mean_ms` | 2.353 | 0.692 | 0.685 |
| `traffic.packet_delay_p95_ms` | 6.500 | 2.000 | 1.500 |
| `traffic.packet_effective_service_rate_mbps` | 10.198 | - | - |

当前 HGraph v33 3-seed Python online eval：

| 指标 | HGraph v33 mean | PFQ mean |
|---|---:|---:|
| `traffic.served_mbps_est` | 1730.930 | 1729.518 |
| `traffic.goodput_mbps` | 1537.405 | 1541.352 |
| `global_kpi.prg_utilization_ratio` | 0.704002 | 0.913458 |
| `global_kpi.scheduled_ratio_mean` | 0.458035 | 0.737530 |
| `global_kpi.scheduled_ratio_p5` | 0.264396 | 0.548063 |
| `traffic.packet_delay_mean_ms` | 1.492 | 1.355 |
| `traffic.packet_effective_service_rate_mbps` | 16.343 | 21.737 |

结论：

- 吞吐已经接近 offered load 上限，单纯追 `served_mbps_est` 的空间有限。
- goodput 还有来自 BLER/重传浪费的空间。
- packet delay 和 packet effective service rate 的主要缺口来自服务不够连续，而不是缺少瞬时峰值传输能力。

### 2.2 主要成因

当前策略更像“高效率 burst 服务”：

1. reward 中有 `active_prg_goodput_efficiency = spectral_eff / prg_utilization`，会奖励单位 PRG 效率。
2. reward 中有 harmful packing penalty，会惩罚高 PRG utilization 与高 reuse 下的 `throughput - goodput` 浪费。
3. 时延/服务连续性相关权重当前默认较弱或为 0：
   - `BACKLOG_DEBT_WEIGHT=0`
   - `PACKET_RATE_WEIGHT=0`
   - `PACKET_COMPLETION_WEIGHT=0`
   - `AGED_BACKLOG_WEIGHT=0`
4. action-mode 部署采用 deterministic argmax，容易把 slot 和 PRG 聚到 logits 最高的一小批 UE 上。
5. decoder 的 demand-cap 会把超出 UE 需求估计的 PRG drop 或 fallback；如果策略没有主动覆盖更多 backlog UE，就会留下较多 final blank。

因此当前模型已经学会“少用 PRG 也把系统吞吐做满”，但还没学会“以低 delay 的节奏持续服务所有有需求 UE”。

## 3. 当前算法架构

### 3.1 运行场景

当前代表性部署目标固定为：

- `C = 3` coordinated cells
- `U = 36` active/global UE
- `P = 17` local PRG/RBG per cell
- `C * P = 51` PRG tokens
- `S = 36` Type-0 scheduling slots
- `RBG16`，即每个 PRG/RBG 对应 `16 PRB`
- `4T4R + SVD precoding`
- `packet_size_bytes = 3000`
- `traffic_arrival_rate = 1.0 pkt/TTI`
- `packet_ttl_ms = 200`
- `slotDuration = 0.5 ms`

当前 HGraph 部署 ABI 固定为 `3cell/36UE/17PRG`。如果要泛化到其他小区数、UE 数或 PRG 数，需要重新定义 ONNX ABI 或做动态形状 runtime。

### 3.2 图节点

当前图是 sparse entity graph，节点类型包括：

| 节点类型 | 数量 | 粒度 | 含义 |
|---|---:|---|---|
| `Cell` | `C` | cell / TTI | 协同小区级状态 |
| `UE` | `U` | global UE / TTI | active UE 状态，UE 通过 `action_mask_cell_ue` 关联到 serving cell |
| `PRG` | `C * P` | `(cell, local_prg)` / TTI | 每个小区每个本地 PRG 的频域资源 token |

PRG token 的图内 flatten 顺序是：

```text
prg_token = cell_id * n_prg + local_prg_id
```

C++ Type-0 native `action_prg_alloc` 使用 native order：

```text
native_linear = local_prg_id * n_cell + owner_cell
```

因此 Python/ONNX/C++ 间需要显式转换 PRG 顺序。

### 3.3 图边

当前边集合：

| 边 | 方向 | 粒度 | 当前构造方式 | 主要作用 |
|---|---|---|---|---|
| `E_CC` | `Cell -> Cell` | cell-pair | 来自 `obs_edge_index`，缺省为 complete directed graph 去掉 self-loop | 小区间干扰/拓扑上下文传播 |
| `E_CU` | `Cell -> UE` | legal cell-UE pair | 来自 `action_mask_cell_ue` | UE 归属和 cell context 注入 |
| `E_CP` | `Cell -> PRG` | owned PRG | 静态 `cell_id -> (cell_id, prg_id)` | 小区状态注入每个 PRG token |
| `E_UP` | `UE -> PRG` | candidate UE-PRG pair | 优先用 `post_eq_sinr` 构造 per-PRG 候选，否则用 shared pool | PRG 频域质量、队列紧迫度、服务 debt 注入 |
| `E_PP` | `PRG -> PRG` | same local PRG across neighboring cells | 对每条 `Cell -> Cell` 边复制到同 local PRG | 同频 PRG reuse/interference 上下文 |

`E_UP` 是当前最关键的动态边：

- online bridge 导出 `post_eq_sinr` 时，候选按每个 `local_prg` 上各 UE 的 post-eq SINR 排序。
- `candidate_mode=all_demand_ues` 时，每个 PRG 会连接该 cell 下所有有 demand 的 legal UE。
- `candidate_mode=topk_diversity` 时，先取 SINR top-k，再用 urgency/debt 做 diversity extra。
- `edge_attr` 当前为 5 维：
  - `quality_db`
  - `quality_gap_db = best_quality_db - quality_db`
  - `urgency`
  - `debt`
  - `prg_risk`

其中：

```text
urgency = 0.45 * log1p(buffer_bytes)
        + 0.30 * normalized_hol_delay
        + 0.25 * ttl_urgency

debt = 0.60 * recent_goodput_deficit_norm
     + 0.40 * (1 - recent_scheduled_ratio)
```

v34 当前 `prg_risk` 口径已经对齐 PRG 节点第 6/7 维：

```text
prg_risk = 0.5 * clamp(same_prg_conflict_ratio, 0, 1)
         + 0.5 * clamp(ici_proxy, 0, 1)
```

它作为 `E_UP.edge_attr[4]` 输入消息传递，并在 ONNX action-mode 导出路径中保持同一公式。第 4/5 维 `neighbor_max_top1_sinr_db / neighbor_mean_top1_sinr_db` 仍保留为 PRG 节点特征，但不再直接相加作为 UE-PRG 边风险。

注意：当前 `post_eq_sinr` shape 是 `[U, P]` 或 `[U, P, L]`，即按 UE 和本地 PRG 对齐；cell 维度由 `action_mask_cell_ue[cell, ue]` 决定。

### 3.4 输入特征

当前 snapshot 来自 online bridge 的 observation dict：

- `obs_cell_features`
- `obs_ue_features`
- `obs_prg_features`
- `post_eq_sinr`，可选但 action-mode 推荐开启
- `obs_edge_index`
- `obs_edge_attr`
- `action_mask_ue`
- `action_mask_cell_ue`
- `action_mask_prg_cell`
- `teacher_action_ue_select`，训练可选
- `teacher_action_prg_alloc`，训练可选

#### Cell Features

shape：`cell_features[C,5]`

| idx | 字段 | 粒度 | 说明 |
|---:|---|---|---|
| 0 | `cell_load_bytes` | cell / TTI | 当前 cell 下队列负载 |
| 1 | `active_ue_count` | cell / TTI | 当前 active UE 数 |
| 2 | `mean_wb_sinr_lin` | cell / TTI | cell 内 UE wideband SINR 均值，线性域 |
| 3 | `mean_avg_rate_mbps` | cell / TTI | cell 内长期平均速率均值 |
| 4 | `tb_err_rate` | cell / TTI | transport block error 统计/EMA 输入 |

#### UE Features

shape：`ue_features[U,12]`

| idx | 字段 | 粒度 | 说明 |
|---:|---|---|---|
| 0 | `buffer_bytes` | UE / TTI | MAC queue 当前 backlog |
| 1 | `avg_rate_mbps` | UE / TTI | scheduler 长期平均速率状态 |
| 2 | `wb_sinr_lin` | UE / TTI | wideband SINR，线性域 |
| 3 | `cqi` | UE / TTI | 当前 CQI |
| 4 | `ri` | UE / TTI | rank indicator / layer 相关信息 |
| 5 | `tb_err_last` | UE / TTI | 上一次 TB error 指示/统计 |
| 6 | `new_data_flag` | UE / TTI | 本 TTI 是否有新数据需求 |
| 7 | `stale_slots` | UE / TTI | 距上次服务的 slot/TTI 累计 |
| 8 | `hol_delay_ms` | UE / TTI | head-of-line delay |
| 9 | `ttl_slack_ms` | UE / TTI | 到 TTL 截止的剩余 slack |
| 10 | `recent_scheduled_ratio` | UE / recent window | 近期被调度比例 |
| 11 | `recent_goodput_deficit_norm` | UE / recent window | 近期 goodput deficit |

这些字段已经包含时延和服务节奏信息。当前问题不是“状态缺时延”，而是 reward/action/decode 对这些字段的使用强度不足。

#### PRG Features

shape：`prg_features[C*P,8]`，来自 `obs_prg_features[C,P,8]` 展平。

| idx | 字段 | 粒度 | 说明 |
|---:|---|---|---|
| 0 | `top1_sinr_db` | cell-PRG / TTI | 本 PRG 上最佳候选 UE 的 SINR |
| 1 | `top2_gap_db` | cell-PRG / TTI | top1/top2 SINR gap |
| 2 | `prev_prg_assigned` | cell-PRG / previous action | 上一动作该 PRG 是否被分配 |
| 3 | `prev_prg_reuse_ratio` | cell-PRG / previous action | 上一动作同 PRG reuse 情况 |
| 4 | `neighbor_max_top1_sinr_db` | cell-PRG / TTI | 邻居小区同 PRG top1 SINR 最大值 |
| 5 | `neighbor_mean_top1_sinr_db` | cell-PRG / TTI | 邻居小区同 PRG top1 SINR 均值 |
| 6 | `same_prg_conflict_ratio` | cell-PRG / TTI | 同 PRG 冲突 proxy |
| 7 | `ici_proxy` | cell-PRG / TTI | inter-cell interference proxy |

`same_prg_conflict_ratio` 和 `ici_proxy` 在 C++ online observation extras 中生成，源码入口是 `cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h`：

1. 对每个 `(cell, prg)`，在该 cell 关联 UE 中找当前 PRG 上 `postEqSinr` 最高的 UE，得到 `top1_sinr_db`；若没有 `postEqSinr`，退化使用 UE 的 `wbSinr`。
2. 对同一个 `local_prg`，遍历其他 cell 的 `top1_sinr_db`。若邻区同 PRG 的 `top1_sinr_db >= 本 cell top1_sinr_db - 3 dB`，则计为一次同频竞争。
3. 归一化得到：

```text
same_prg_conflict_ratio = conflict_count / (n_cell - 1)
```

4. 对本 cell 当前 PRG 的 top1 UE，比较其 wideband SINR 与当前 PRG post-eq SINR 的退化：

```text
sinr_drop_db = max(0, top1_winner_wb_sinr_db - top1_sinr_db)
prg_sinr_drop_norm = min(1, sinr_drop_db / 12 dB)
```

5. 合成轻量 ICI 风险 proxy：

```text
ici_proxy = 0.5 * same_prg_conflict_ratio
          + 0.5 * prg_sinr_drop_norm
```

两者最终都会被 clamp 到 `[0, 1]`。因此它们是“同 PRG 竞争强度 + 本 PRG 相对 wideband 退化”的启发式 proxy，不是 `multiCellSinrCal` 内部精确 ICI 协方差或真实跨小区链路增益。

#### Masks And Metadata

| 字段 | shape | 粒度 | 用途 |
|---|---:|---|---|
| `action_mask_ue` | `[U]` | UE | UE 是否可调度 |
| `action_mask_cell_ue` | `[C,U]` | cell-UE | UE 是否属于/可由该 cell 调度 |
| `action_mask_prg_cell` | `[C,P]` | cell-PRG | PRG 是否可用 |
| `prg_owner_cell` | `[C*P]` | PRG token | token 所属 cell |
| `prg_local_index` | `[C*P]` | PRG token | token 对应本地 PRG id |
| `cell_adjacency` | `[E,2]` | cell-pair | 小区干扰/邻接关系 |

### 3.5 编码器和消息传递

当前模型是 `StageBHGraphPolicy`：

1. `CellTemplateEncoder` 编码 `cell_features`
2. `UeTemplateEncoder` 编码 `ue_features`
3. `PrgTemplateEncoder` 编码 `prg_features`
4. 每类节点经过 residual block
5. 经过若干层 `HeteroMessageLayer`

`HeteroMessageLayer` 的关系聚合：

- cell 收 `E_CC`
- UE 收 `E_CU`
- PRG 收三类聚合：
  - `E_CP` from Cell
  - `E_UP` from UE
  - `E_PP` from PRG

当前聚合是 relation-specific MLP message + scatter mean，再用 residual update。它不显式建模 sequence/history，只依赖 observation 中已有的 recent/stale/history 特征。

### 3.6 动作空间

当前动作头是 joint Type-0 两阶段动作：

1. `slot -> UE`
   - logits shape：`ue_logits[S, U+1]`
   - 前 `U` 类是具体 UE
   - 最后一类是 no-UE/null
   - 通过 `action_mask_cell_ue` 和 Type-0 slot layout 限制每个 slot 只能选择同 cell legal UE
2. `PRG -> slot`
   - logits shape：`prg_logits[C, P, S+1]`
   - 前 `S` 类是调度 slot
   - 最后一类是 no-PRG/null
   - 通过 selected slot mask、cell 对齐、PRG mask 限制每个 PRG 只能指向同 cell 已选中的 slot

部署 action-mode ONNX 输出：

- `action_ue_select [1,36]`
- `action_prg_alloc [1,51]`

ONNX 内部使用 deterministic argmax：

```text
ue_class = argmax(ue_logits)
prg_class = argmax(masked_prg_logits)
```

这也是当前服务节奏偏 burst 的一个原因：argmax 会放大 logits 排名差异，使 PRG/slot 更容易集中到少数 UE。

### 3.7 Decode / Finalize

当前 Python evaluator 和 C++ runtime 都保留最终合法化逻辑，核心约束包括：

- Type-0 slot / cell 合法性
- PRG owner cell 对齐
- invalid slot/empty slot 置 blank
- demand-active UE 判断
- per-UE PRG demand-cap
- over-cap PRG fallback 到同 cell 其他有 headroom 的 slot/UE

当前 demand-cap 估算：

```text
est_bytes_per_prg = clamp(3000 * log2(1 + max_post_eq_sinr) / 2, 1200, 12000)
demand_bytes = buffer_bytes + 3000 slack
cap = ceil(demand_bytes / est_bytes_per_prg)
```

这会避免过度给单个 UE 堆 PRG，但如果上游 action 没有覆盖足够多有 backlog 的 UE，最终就会产生较多 blank。

### 3.8 Reward

当前存在两类主 reward：

#### `stageb_v1`

```text
reward =
  goodput_weight * goodput_mbps
- tb_err_weight * tb_err_rate
- expiry_weight * expiry_drop_rate
- conflict_weight * conflict_mean
- ici_weight * ici_mean
- urgent_backlog_weight * urgent_backlog_penalty
```

#### `stageb_blankaware`

```text
reward =
  0.05 * goodput_mbps
+ 0.25 * active_prg_goodput_efficiency
- 2.0 * tb_err_rate
- 4.0 * expiry_drop_rate
- 10.0 * harmful_packing_penalty
+ 0.25 * fairness_jain
- backlog_debt_weight * backlog_debt_penalty
+ packet_rate_weight * packet_rate_bonus
+ packet_completion_weight * packet_completion_bonus
- aged_backlog_weight * aged_backlog_penalty
```

当前关键问题：

- `active_prg_goodput_efficiency` 鼓励低 PRG utilization 下的高效率。
- `harmful_packing_penalty` 抑制高利用率高复用的冒进动作。
- `backlog_debt_weight`、`packet_rate_weight`、`packet_completion_weight`、`aged_backlog_weight` 默认是 0，导致 packet delay 和服务连续性的优化压力不足。

## 4. 下一步优化目标

下一轮不是简单“把 PRG 填满”，而是要学习：

> 在不显著提高 BLER/重传浪费的前提下，提高服务连续性和有效 PRG 填充率。

建议目标分层：

| 层级 | 目标 | 约束 |
|---|---|---|
| G0 | 保持 `served_mbps_est` 不低于 PFQ/RRQ 当前水平 | 不牺牲 aggregate served throughput |
| G1 | `packet_delay_mean_ms` 接近 PFQ，`packet_delay_p95_ms` 不恶化 | 优先服务节奏 |
| G2 | `packet_effective_service_rate_mbps` 提升 | 减少 packet system time |
| G3 | `global_tb_bler` 不高于 PFQ/RRQ 明显水平 | 避免用高 BLER 换填充 |
| G4 | `prg_utilization_ratio` 提升到 `0.75-0.85` 区间 | 不必追 PFQ 的 `0.91+`，先找 Pareto 点 |
| G5 | `scheduled_ratio_p5`、`scheduled_ratio_jain` 提升 | 避免边缘 UE 长时间不服务 |

## 5. 优化方向

### 5.1 Reward 从 throughput-only 改为 latency-aware Pareto

第一优先级是启用已经预留但默认关闭的时延/服务连续性项：

- `--backlog-debt-weight`
- `--packet-rate-weight`
- `--packet-completion-weight`
- `--aged-backlog-weight`

建议从温和权重开始：

```bash
--reward-mode stageb_blankaware \
--backlog-debt-weight 0.03 \
--packet-rate-weight 0.20 \
--packet-completion-weight 0.05 \
--aged-backlog-weight 0.02
```

含义：

- `backlog_debt_weight`：惩罚实际未服务 backlog，避免“有包但本 TTI 不管”。
- `packet_rate_weight`：奖励 packet effective service rate，直接压缩 packet system time。
- `packet_completion_weight`：奖励每 TTI packet completion 数，避免只攒大 burst。
- `aged_backlog_weight`：惩罚 recent100 debt + no-service streak，针对长期未服务 UE。

第二阶段可做权重 sweep：

| 实验 | backlog debt | packet rate | packet completion | aged backlog | 预期 |
|---|---:|---:|---:|---:|---|
| R0 | 0 | 0 | 0 | 0 | 复现当前 |
| R1 | 0.03 | 0.10 | 0.03 | 0.01 | 温和降时延 |
| R2 | 0.05 | 0.20 | 0.05 | 0.02 | 平衡版本 |
| R3 | 0.08 | 0.30 | 0.08 | 0.04 | 强时延版本，观察 goodput 是否掉 |

筛选规则：

1. `traffic.goodput_mbps` 不低于 PFQ 均值太多。
2. `traffic.packet_delay_mean_ms` 和 `packet_effective_service_rate_mbps` 明显改善。
3. `global_tb_bler` 不明显上升。
4. `prg_utilization_ratio` 不盲目超过 PFQ，而是观察 `goodput / utilization` 是否仍合理。

### 5.2 增加 coverage-aware slot repair

当前已有 `_repair_ue_action_for_coverage`，但默认 `slot_repair_max_per_cell=0`，等于关闭。

建议开启小幅 repair：

```bash
--slot-repair-max-per-cell 2 \
--slot-repair-min-backlog-bytes 1
```

repair 逻辑会在每个 cell 内优先替换 invalid/duplicate slot，候选 UE 按：

1. backlog
2. recent deficit
3. stale slots
4. HOL delay
5. TTL urgency

排序。它适合作为“最小服务覆盖”补丁，不会直接强制填满所有 PRG。

后续可扩展为更明确的 coverage 约束：

- 每 cell 每 TTI 至少覆盖 `min(K, demand_ue_count)` 个 demand UE。
- 对 `recent_scheduled_ratio` 最低或 `no_service_streak` 最高的 UE 做 hard coverage。
- 对 high SINR UE 和 high debt UE 分别保留 slot，避免全被瞬时高质量 UE 吸走。

### 5.3 调整 PRG fill 与 blank policy

当前低 utilization 来自两部分：

- policy 直接选择 no-PRG/null
- decode demand-cap 后 final blank 增加

建议做三个轻量改动方向：

1. 训练期加入 fill regularizer：

```text
reward += fill_weight * clamp(target_fill - final_blank_ratio, ...)
```

或更稳妥地加入 under-fill penalty：

```text
if prg_utilization < util_floor and tb_err_rate < bler_guard:
    penalty = util_floor - prg_utilization
```

推荐从 `util_floor=0.75` 开始，不直接追 `0.90+`。

2. decode fallback 增强：

- over-cap PRG 先 fallback 到同 cell 高 backlog/high HOL UE。
- 若所有 selected slots 都无 headroom，允许补一个 slot repair UE 再接收 PRG。
- fallback priority 增加 `hol_delay_ms`、`stale_slots`、`recent_goodput_deficit_norm`，不只看 headroom/backlog。

3. action-mode ONNX 加 runtime knobs：

- `min_prg_ratio`
- `max_final_blank_ratio`
- `max_prg_share_per_ue`

这些 knobs 适合先做部署侧 ablation，验证“填充率拉高是否真的降低 delay 且不伤 BLER”。

### 5.4 改善两阶段动作头的集中性

当前 `slot -> UE` 先 argmax，再让 `PRG -> slot` 依赖这个 hard slot assignment。改进方向：

1. 训练期提高 UE coverage 多样性：

- 增强 teacher auxiliary，延长 `teacher_aux_decay_iters`。
- 使用 PFQ passthrough 作为更强 warm start，让策略先学到服务节奏，再 PPO 微调。

2. action head 加分布约束：

- 对同 cell slot 的 UE 分布加入 duplicate/集中惩罚。
- 对 per-UE PRG share 加软约束。
- 对低 `recent_scheduled_ratio` UE 加 logit bias。

3. PRG head 增加 direct UE candidate branch：

保留 `PRG -> slot` 以兼容 Type-0，但训练时增加辅助 head：

```text
PRG -> candidate UE
```

辅助 loss 用来约束 PRG 不只追随 slot argmax，改善 slot bottleneck。

### 5.5 Candidate 生成从 pure SINR 排序变成 quality + urgency

`E_UP` 当前 post-eq 模式下：

- `all_demand_ues`：所有 demand UE 都连边，但排序仍按 SINR。
- `topk_diversity`：top-k 以 quality 为 core，extra 才混 urgency/debt。

下一步可做：

```text
candidate_score =
  0.55 * normalized_post_eq_quality
+ 0.25 * urgency
+ 0.20 * debt
- 0.10 * recent_tb_err
```

用于：

- top-k candidate selection
- edge priority
- fallback priority

目标是让高 HOL / high debt UE 更早进入 PRG 决策核心，而不是只作为边属性等待模型自己学。

v34 已在 `training/stageb_hgraph/edge_generator.py` 中落地 candidate 排序改造：

```text
candidate_score =
  0.55 * normalized_post_eq_quality
+ 0.25 * normalized_urgency
+ 0.20 * normalized_debt
- 0.10 * normalized_tb_err_last
```

该分数现在用于：

- `candidate_mode=all_demand_ues` 下的 E_UP 边插入顺序；由于 candidate packing 会按边顺序截断到 `max_prg_candidates`，这会影响实际进入 action head 的候选集合。
- `candidate_mode=topk_diversity` 下的 core top-k 和 diversity extra 选择。

无 `post_eq_sinr` 时的 shared-pool ranking 也加入了 `tb_err_last` 抑制项。

### 5.6 训练和评测协议

建议下一轮训练不只看固定 seed42：

1. 先固定 seed42 做快速 sweep：
   - 100-150 iterations
   - 观察 `rollout_packet_effective_service_rate_mbps_mean`
   - 观察 `rollout_no_service_streak_p90_mean`
   - 观察 `rollout_recent100_service_rate_p10_mean`
2. 选 2-3 个候选后跑 `s41/s42/s43`。
3. 最后再跑 `s41-s50`。

推荐主评估指标：

| 指标 | 目标 |
|---|---|
| `traffic.goodput_mbps` | 不明显低于 PFQ |
| `traffic.packet_delay_mean_ms` | 尽量接近 PFQ |
| `traffic.packet_delay_p95_ms` | 不出现长尾恶化 |
| `traffic.packet_effective_service_rate_mbps` | 高于当前 HGraph |
| `global_kpi.prg_utilization_ratio` | 从 `0.60/0.70` 提升到 `0.75-0.85` |
| `global_kpi.scheduled_ratio_p5` | 明显提升 |
| `global_kpi.scheduled_ratio_jain` | 明显提升 |
| `global_kpi.global_tb_bler` | 不明显高于当前 |

## 6. 推荐实验命令

### 6.1 Baseline 复现

```bash
CUDA_VISIBLE_DEVICES=1 ./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --build-method skip \
  --tti 4000 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 42 \
  --topology-seed-mode fixed \
  --baseline-scheduler pfq \
  --online-persistent 1 \
  --episode-horizon 1024 \
  --rollout-steps 1024 \
  --iterations 120 \
  --ue-prg-topk 6 \
  --hidden-dim 192 \
  --message-layers 3 \
  --dropout 0.0 \
  --device cuda \
  --lr 1.5e-4 \
  --actor-lr 1.0e-4 \
  --critic-lr 3.0e-4 \
  --reward-mode stageb_blankaware \
  --best-metric goodput \
  --sim-env CUDA_VISIBLE_DEVICES=0 \
  --out-dir training/stageb_hgraph/runs/online_ppo_seed42_latency_r0
```

### 6.2 Latency-aware R2 候选

```bash
CUDA_VISIBLE_DEVICES=1 ./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --build-method skip \
  --tti 4000 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 42 \
  --topology-seed-mode fixed \
  --baseline-scheduler pfq \
  --online-persistent 1 \
  --episode-horizon 1024 \
  --rollout-steps 1024 \
  --iterations 180 \
  --ue-prg-topk 6 \
  --hidden-dim 192 \
  --message-layers 3 \
  --dropout 0.0 \
  --device cuda \
  --lr 1.5e-4 \
  --actor-lr 1.0e-4 \
  --critic-lr 3.0e-4 \
  --reward-mode stageb_blankaware \
  --backlog-debt-weight 0.05 \
  --packet-rate-weight 0.20 \
  --packet-completion-weight 0.05 \
  --aged-backlog-weight 0.02 \
  --slot-repair-max-per-cell 2 \
  --slot-repair-min-backlog-bytes 1 \
  --teacher-aux-coef 0.20 \
  --teacher-aux-decay-iters 120 \
  --best-metric goodput \
  --sim-env CUDA_VISIBLE_DEVICES=0 \
  --out-dir training/stageb_hgraph/runs/online_ppo_seed42_latency_r2
```

### 6.3 三 seed 评估

```bash
./cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh \
  --build-method skip \
  --seed-list 41,42,43 \
  --checkpoint-path training/stageb_hgraph/runs/online_ppo_seed42_latency_r2/ppo_actor_absolute_best.pt \
  --model-label hgraph_latency_r2 \
  --baseline-mean-root output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s50_ue36_ttl200_rbg16_svd_20260506_091443 \
  --decode-mode argmax \
  --eval-episode-horizon 4000 \
  --eval-device cuda \
  --eval-print-every-steps 500 \
  --sim-env CUDA_VISIBLE_DEVICES=0 \
  --eval-env CUDA_VISIBLE_DEVICES=1 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --exec-mode both \
  --precoding svd \
  --baseline-scheduler pfq \
  --compact-output 1 \
  --compact-tti-log 1 \
  --progress-tti 0 \
  --kpi-tti-log 100 \
  --compare-tti 0 \
  --tti 4000 \
  --compare-output-dir output/stageB_hgraph_latency_r2_s41_s42_s43
```

## 7. 建议落地顺序

1. **文档和指标口径固定**
   - 本文作为 HGraph 优化主文档。
   - 每次训练必须记录 reward weights、slot repair、candidate mode、decode mode。
2. **先做 reward weight sweep**
   - R0/R1/R2/R3 固定 seed42。
   - 只要 goodput 不明显掉，优先选择 packet delay 和 `packet_effective_service_rate_mbps` 改善最大的版本。
3. **开启 slot repair**
   - 从 `slot_repair_max_per_cell=2` 开始。
   - 观察 scheduled_ratio_p5 是否上升。
4. **做 fill floor / fallback ablation**
   - 目标 PRG utilization 先到 `0.75-0.85`。
   - 防止 BLER 因填充过猛上升。
5. **三 seed -> 十 seed**
   - 三 seed 通过后再跑 s41-s50。
6. **再导出 action-mode ONNX**
   - 只把 Pareto 明显更好的 checkpoint 导出到 `training/stageb_hgraph/exports/`。

## 8. 预期结果

理想下一版不是所有指标同时大幅上升，而是形成更好的 Pareto 点：

- `served_mbps_est` 维持接近满 offered load。
- `goodput_mbps` 不低于当前 HGraph，尽量接近或超过 PFQ。
- `prg_utilization_ratio` 从 `0.60/0.70` 提到 `0.75-0.85`。
- `scheduled_ratio_p5` 和 `scheduled_ratio_jain` 明显改善。
- `packet_delay_mean_ms`、`packet_delay_p95_ms` 明显下降。
- `packet_effective_service_rate_mbps` 明显上升。

如果出现以下情况，应回退：

- PRG utilization 上升但 BLER 明显变差。
- packet delay 下降但 goodput 明显低于 PFQ。
- scheduled_ratio 更公平但 p5 goodput 下降，说明服务了更多 UE 但资源质量不够。
- packet completion 上升但 packet effective service rate 不升，说明只是小包完成多，系统时间没有真正下降。

## 9. 当前判断

当前 HGraph 的优势在于已经能用较少资源获得接近 PFQ/RRQ 的 aggregate rate，说明图特征、post-eq SINR、PRG risk 和 Type-0 action path 是有效的。

下一步最有价值的优化不是继续增加模型规模，而是：

1. 把已有的时延/服务连续性信号真正接入 reward。
2. 让 slot repair 和 fallback 保障 minimum service coverage。
3. 在 `0.75-0.85` PRG utilization 区间寻找 BLER、goodput、delay 的 Pareto 最优点。

## 10. v34 本轮改造记录

### 10.1 代码改造

本轮先做训练前的输入语义校准和 candidate 改造：

| 项目 | 文件 | 变更 |
|---|---|---|
| normalized PRG risk | `training/stageb_hgraph/edge_generator.py` | `E_UP.edge_attr[4]` 从 `neighbor_max + neighbor_mean` 改为 `0.5 * same_prg_conflict_ratio + 0.5 * ici_proxy` |
| ONNX risk 对齐 | `training/stageb_hgraph/export_onnx.py` | 固定形状导出 wrapper 使用同一 `prg_risk` 公式 |
| candidate 排序 | `training/stageb_hgraph/edge_generator.py` | post-eq candidate 从纯质量排序改为 `quality + urgency + debt - tb_err` |
| sanity 日志修正 | `training/stageb_hgraph/sanity_logger.py`、`training/stageb_hgraph/online_imitation.py` | 修正 8 维 PRG 特征索引：`prev=2`、`neighbor_max=4`、`conflict=6`、`ici=7` |

### 10.2 已完成验证

- `git diff --check`：通过。
- `.venv/bin/python -m compileall training/stageb_hgraph`：通过。
- `edge_generator` 轻量 smoke：通过，`E_UP` 风险特征保持在 `[0, 1]`。
- `sanity_logger` 轻量 smoke：通过，8 维 PRG 特征索引修正后不再访问越界字段。

### 10.3 本轮训练配置

本轮训练必须在 Aerial 容器内运行，否则宿主环境可能缺少 HDF5、TensorRT、fmtlog 等 simulator 动态库：

```bash
./cuPHY-CP/container/attach_aerial.sh
source .venv/bin/activate
```

本轮 smoke 训练采用 latency-aware R2 方向，但缩短 rollout 和 iteration，用于验证闭环：

```bash
CUDA_VISIBLE_DEVICES=1 ./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --build-method skip \
  --restore-params 1 \
  --tti 1600 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 42 \
  --topology-seed-mode fixed \
  --baseline-scheduler pfq \
  --online-persistent 1 \
  --episode-horizon 256 \
  --rollout-steps 256 \
  --iterations 8 \
  --ue-prg-topk 6 \
  --ue-prg-diversity-extra 2 \
  --candidate-mode topk_diversity \
  --hidden-dim 192 \
  --message-layers 3 \
  --dropout 0.0 \
  --device cuda \
  --lr 1.5e-4 \
  --actor-lr 1.0e-4 \
  --critic-lr 3.0e-4 \
  --reward-mode stageb_blankaware \
  --backlog-debt-weight 0.05 \
  --packet-rate-weight 0.20 \
  --packet-completion-weight 0.05 \
  --aged-backlog-weight 0.02 \
  --slot-repair-max-per-cell 2 \
  --slot-repair-min-backlog-bytes 1 \
  --teacher-aux-coef 0.20 \
  --teacher-aux-decay-iters 8 \
  --best-metric goodput \
  --sim-env CUDA_VISIBLE_DEVICES=0 \
  --out-dir training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_smoke
```

若 smoke 正常，再扩展到 `iterations=180`、`episode-horizon=1024`、`rollout-steps=1024`，并沿用相同 reward/candidate/slot-repair 配置。

### 10.4 Smoke 训练结果

2026-05-21 已在 `c_aerial_oai2` 容器中完成 v34 smoke 训练。

输出目录：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_smoke
```

关键产物：

- `online_ppo_metrics.csv`
- `online_ppo_summary.json`
- `ppo_actor_absolute_best.pt`
- `ppo_actor_best.pt`
- `ppo_actor_last.pt`

8-iteration smoke 指标：

| iter | goodput Mbps | TB err | fill ratio | final blank | candidate coverage | unserved demand UE | no-service p90 | recent100 service p10 | packet effective Mbps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1318.895 | 0.2283 | 0.6580 | 0.3420 | 0.9987 | 3.512 | 0.689 | 0.756 | 31.366 |
| 2 | 1552.624 | 0.1033 | 0.7361 | 0.2639 | 0.9965 | 4.422 | 0.825 | 0.716 | 23.483 |
| 5 | 1589.937 | 0.1049 | 0.7639 | 0.2361 | 0.9958 | 5.004 | 0.941 | 0.681 | 23.019 |
| 8 | 1556.425 | 0.1015 | 0.7721 | 0.2279 | 0.9973 | 3.305 | 0.599 | 0.762 | 25.122 |

`online_ppo_summary.json` 记录：

- `completed_iterations = 8`
- `absolute_best_iter = 5`
- `absolute_best_score = 1589.937`
- `threshold_best_iter = 5`
- `threshold_best_score = 1589.937`

初步判断：

- 训练闭环已经恢复，online bridge、post-eq 输入、candidate graph、slot repair、reward shaping 都能跑通。
- `candidate_coverage_ratio` 稳定在约 `0.996-0.999`，说明 candidate 改造没有削弱 demand UE 覆盖。
- `fill_ratio` 从 iter1 的 `0.658` 提升到 iter8 的 `0.772`，已进入目标 `0.75-0.85` 区间。
- `goodput_mbps` 在 iter2 后稳定回到 `1550+`，best iter5 达到 `1589.937`。
- 末轮 `recent100_service_rate_p10` 回升到 `0.762`，`no_service_streak_p90` 降到 `0.599`，服务连续性指标方向较好。

下一步建议以 `ppo_actor_absolute_best.pt` 为 warm start，启动长训版本：

```bash
--init-checkpoint training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_smoke/ppo_actor_absolute_best.pt \
--iterations 180 \
--episode-horizon 1024 \
--rollout-steps 1024 \
--teacher-aux-decay-iters 120
```

### 10.5 正式长训启动记录

2026-05-21 按 smoke 的 `ppo_actor_absolute_best.pt` 继续启动 180-iteration 长训，目标是观察 v34 candidate/risk/reward 配置是否能在更长 rollout 上收敛到稳定的 goodput、PRG utilization 和服务连续性折中点。

运行前置条件仍然是进入 Aerial 容器并激活仓库内虚拟环境：

```bash
./cuPHY-CP/container/attach_aerial.sh
source .venv/bin/activate
```

正式长训命令：

```bash
CUDA_VISIBLE_DEVICES=1 ./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --build-method skip \
  --restore-params 1 \
  --tti 1600 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 42 \
  --topology-seed-mode fixed \
  --baseline-scheduler pfq \
  --online-persistent 1 \
  --episode-horizon 1024 \
  --rollout-steps 1024 \
  --iterations 180 \
  --ue-prg-topk 6 \
  --ue-prg-diversity-extra 2 \
  --candidate-mode topk_diversity \
  --hidden-dim 192 \
  --message-layers 3 \
  --dropout 0.0 \
  --device cuda \
  --lr 1.5e-4 \
  --actor-lr 1.0e-4 \
  --critic-lr 3.0e-4 \
  --reward-mode stageb_blankaware \
  --backlog-debt-weight 0.05 \
  --packet-rate-weight 0.20 \
  --packet-completion-weight 0.05 \
  --aged-backlog-weight 0.02 \
  --slot-repair-max-per-cell 2 \
  --slot-repair-min-backlog-bytes 1 \
  --teacher-aux-coef 0.20 \
  --teacher-aux-decay-iters 120 \
  --best-metric goodput \
  --init-checkpoint training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_smoke/ppo_actor_absolute_best.pt \
  --sim-env CUDA_VISIBLE_DEVICES=0 \
  --out-dir training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_180_from_smoke
```

训练过程重点观察：

- `rollout_goodput_mbps_mean` 是否能稳定超过 smoke best 附近的 `1589 Mbps`。
- `rollout_prg_utilization_ratio_mean` 是否维持在 `0.75-0.85` 区间，避免继续向高 BLER packing 漂移。
- `rollout_tb_err_rate_mean` 是否保持在约 `0.10` 附近或下降。
- `rollout_no_service_streak_p90_mean`、`rollout_recent100_service_rate_p10_mean` 是否相对 smoke 末轮继续改善。
- `rollout_packet_effective_service_rate_mbps_mean` 是否随长训稳定上升，而不是只靠集中服务提升 aggregate goodput。

### 10.6 长训中期观察与 challenger 设计

长训进行到 iter50 后，主线出现了比较明确的平台期：

- 按 `best_metric=goodput` 选择的当前 best 出现在 iter7，`goodput_mbps=1575.737`，但 `unserved_demand_ue=9.576`、`recent100_service_rate_p10=0.539`，不是理想的时延/连续服务解。
- 最近 10 个已完成 iteration 的均值约为：`goodput=1557 Mbps`、`TB err=0.102`、`fill=0.761`、`unserved_demand_ue=4.45`、`recent100_service_rate_p10=0.709`。
- 结论是：当前 v34 R2 reward 能把策略从集中服务拉回较好的服务覆盖，但单纯按 goodput 保存 best checkpoint 会偏向早期集中服务高吞吐点；后续需要把服务覆盖约束显式接入 reward 和 checkpoint 选择。

本轮已加入一个默认关闭的 challenger reward 扩展，不改变历史实验复现：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--unserved-ue-weight` | `0.0` | 对 `actual_unserved_demand_ue_count / total_ue_count` 加惩罚，直接压未服务 demand UE 数 |
| `--service-gap-weight` | `0.0` | 对 `max(0, service_rate_target - recent100_service_rate_p10)` 加惩罚 |
| `--service-rate-target` | `0.0` | 服务连续性 p10 目标值，challenger 初始设为 `0.72` |

同时加入 `--best-metric service_guarded_goodput`，用于保存服务守门后的 best checkpoint：

```text
score = goodput_mbps
        - best_service_gap_penalty_mbps * max(0, service_rate_target - recent100_service_rate_p10)
        - best_unserved_penalty_mbps * max(0, unserved_demand_ue - best_unserved_ue_target)
```

建议在当前 180-iteration 主线 early-stop 或完成后，启动 challenger 对照：

```bash
CUDA_VISIBLE_DEVICES=1 ./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --build-method skip \
  --restore-params 1 \
  --tti 1600 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 42 \
  --topology-seed-mode fixed \
  --baseline-scheduler pfq \
  --online-persistent 1 \
  --episode-horizon 1024 \
  --rollout-steps 1024 \
  --iterations 80 \
  --ue-prg-topk 6 \
  --ue-prg-diversity-extra 2 \
  --candidate-mode topk_diversity \
  --hidden-dim 192 \
  --message-layers 3 \
  --dropout 0.0 \
  --device cuda \
  --lr 1.5e-4 \
  --actor-lr 1.0e-4 \
  --critic-lr 3.0e-4 \
  --reward-mode stageb_blankaware \
  --backlog-debt-weight 0.08 \
  --packet-rate-weight 0.20 \
  --packet-completion-weight 0.05 \
  --aged-backlog-weight 0.03 \
  --unserved-ue-weight 2.0 \
  --service-gap-weight 4.0 \
  --service-rate-target 0.72 \
  --slot-repair-max-per-cell 3 \
  --slot-repair-min-backlog-bytes 1 \
  --teacher-aux-coef 0.20 \
  --teacher-aux-decay-iters 80 \
  --best-metric service_guarded_goodput \
  --best-service-gap-penalty-mbps 100.0 \
  --best-unserved-ue-target 4.3 \
  --best-unserved-penalty-mbps 3.0 \
  --init-checkpoint training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_smoke/ppo_actor_absolute_best.pt \
  --sim-env CUDA_VISIBLE_DEVICES=0 \
  --out-dir training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_challenger_service_guard
```

challenger 的成功标准不是单点 goodput 最高，而是同时满足：

- `rollout_goodput_mbps_mean` 稳定不低于主线平台期约 `1555-1565 Mbps`。
- `rollout_recent100_service_rate_p10_mean >= 0.72` 或明显高于主线最近窗口。
- `rollout_unserved_demand_ue_mean <= 4.3`，并避免回到 iter7 那种 `>9` 的集中服务点。
- `rollout_tb_err_rate_mean` 继续保持在约 `0.10` 附近。

### 10.7 主线与 challenger 对照结果

2026-05-21 实际训练中，主线长训在 iter60 后人工停止。原因是 iter50 后已经进入平台期，并在 teacher auxiliary 继续衰减后再次向低服务覆盖漂移；继续跑到 180 iter 的边际价值低于启动服务守门 challenger。

主线输出目录：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_180_from_smoke
```

challenger 输出目录：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_challenger_service_guard
```

关键结果：

| run | 选择点 | iter | score metric | goodput Mbps | TB err | fill | unserved UE | service p10 | packet effective Mbps |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 主线 | pure goodput best | 53 | goodput | 1576.868 | 0.0985 | 0.7582 | 5.048 | 0.621 | 24.642 |
| 主线 | service-guarded 后处理 best | 36 | guarded score | 1568.846 | 0.1020 | 0.7595 | 4.058 | 0.716 | 25.187 |
| 主线 | last completed | 60 | goodput | 1559.960 | 0.1023 | 0.7522 | 5.511 | 0.657 | 24.656 |
| challenger | service-guarded best | 9 | service_guarded_goodput | 1578.172 | 0.1007 | 0.7828 | 2.021 | 0.842 | 25.585 |
| challenger | last completed | 30 | service_guarded_goodput | 1551.813 | 0.1008 | 0.7795 | 2.028 | 0.824 | 25.674 |

最近窗口均值：

| run | window | goodput Mbps | unserved UE | service p10 | packet effective Mbps |
|---|---:|---:|---:|---:|---:|
| 主线 | last 10 | 1564.121 | 5.278 | 0.641 | 24.842 |
| challenger | last 10 | 1555.298 | 2.405 | 0.807 | 25.796 |
| challenger | last 5 | 1559.679 | 2.100 | 0.824 | 25.808 |

结论：

- `service_guarded_goodput` challenger 成功找到了比主线更好的 Pareto 点：best goodput 略高于主线 pure goodput best，同时 `unserved UE` 从 `5.05` 降到 `2.02`，`service p10` 从 `0.621` 提升到 `0.842`。
- challenger 的后期均值 goodput 约 `1555-1560 Mbps`，略低于主线最后窗口，但服务连续性和 packet effective service rate 更好。
- `slot_repair_max_per_cell=3` 与 service gap penalty 的组合能够强力保障覆盖，但也可能偏保守；下一档建议做一个 lighter guard：`slot_repair_max_per_cell=2`、`service_gap_weight=3.0`、`unserved_ue_weight=1.5`，看能否保留 `service p10 >= 0.78` 的同时把尾段 goodput 稳定推回 `1565+ Mbps`。

当前最推荐用于后续评估/导出的 checkpoint：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_challenger_service_guard/ppo_actor_absolute_best.pt
```

### 10.8 推荐 checkpoint 评估与导出记录

2026-05-22 对上一节推荐 checkpoint 做了部署式 argmax 在线评估和 action-mode ONNX 导出：

```text
checkpoint=training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_challenger_service_guard/ppo_actor_absolute_best.pt
checkpoint_iter=9
eval_seeds=41,42,43
scenario=RAYLEIGH / 3cell / 36 UE / TTL=200 ms / RBG=16 / SVD
decode_mode=argmax
eval_horizon=4000 TTI
eval_stats_warmup=1000 TTI
```

评估输出目录：

```text
output/stageB_hgraph_v34_challenger_service_guard_s41_s42_s43_ue36_ttl200_rbg16_svd_eval
```

主要落盘文件：

```text
RAYLEIGH/hgraph_v34_challenger_service_guard_kpi_mean.csv
RAYLEIGH/hgraph_v34_challenger_service_guard_kpi_mean.json
RAYLEIGH/hgraph_v34_challenger_service_guard_kpi_mean.txt
RAYLEIGH/hgraph_v34_challenger_service_guard_vs_rrq_pfq_selected_compare.csv
RAYLEIGH/hgraph_v34_challenger_service_guard_vs_rrq_pfq_selected_compare.txt
per_seed_eval/s41/eval_summary.json
per_seed_eval/s42/eval_summary.json
per_seed_eval/s43/eval_summary.json
```

三 seed 模型自身均值：

| metric | mean | std | 说明 |
|---|---:|---:|---|
| `traffic.served_mbps_est` | 1720.766 Mbps | 3.881 | PHY served throughput estimate |
| `traffic.goodput_mbps` | 1518.062 Mbps | 8.328 | 成功解码 goodput，含完整 4000 TTI |
| `eval.stats_mean_goodput_mbps` | 1539.374 Mbps | 9.819 | 去掉前 1000 TTI warmup 后的 evaluator goodput |
| `global_kpi.global_tb_bler` | 12.686% | 0.500% | 全局 TB BLER |
| `global_kpi.prg_utilization_ratio` | 75.006% | 4.118% | 完整 run PRG 利用率 |
| `eval.stats_mean_prg_utilization_ratio` | 75.331% | 4.565% | warmup 后 PRG 利用率 |
| `eval.stats_mean_uniq_ue` | 12.908 | 0.154 | warmup 后每 TTI unique scheduled UE |
| `traffic.packet_delay_mean_ms` | 2.741 ms | 0.368 | packet tracker mean delay |
| `traffic.packet_delay_p90_ms` | 5.833 ms | 0.624 | packet tracker p90 delay |
| `traffic.packet_delay_p95_ms` | 7.000 ms | 0.707 | packet tracker p95 delay |
| `global_kpi.scheduled_ratio_mean` | 35.809% | 0.399% | UE 平均被调度 TTI 比例 |
| `global_kpi.scheduled_ratio_p5` | 14.623% | 1.226% | UE 调度频率低分位 |

与 2026-05-01 的 RRQ/PFQ 三 seed baseline 做选择性对照：

| metric | RRQ | PFQ | HGraph v34 guard | 结论 |
|---|---:|---:|---:|---|
| `traffic.goodput_mbps` | 1526.624 | 1541.352 | 1518.062 | HGraph 比 PFQ 低 23.290 Mbps，比 RRQ 低 8.563 Mbps |
| `traffic.served_mbps_est` | 1703.827 | 1729.518 | 1720.766 | HGraph 高于 RRQ 16.940 Mbps，低于 PFQ 8.752 Mbps |
| `global_kpi.global_tb_bler` | 11.012% | 11.087% | 12.686% | HGraph BLER 仍偏高 |
| `global_kpi.prg_utilization_ratio` | 86.592% | 91.346% | 75.006% | HGraph PRG 利用率仍是主要缺口 |
| `traffic.packet_delay_mean_ms` | 9.015 | 1.355 | 2.741 | HGraph 明显好于 RRQ，但仍高于 PFQ 1.386 ms |
| `traffic.packet_delay_p95_ms` | 90.833 | 4.667 | 7.000 | HGraph 尾时延大幅好于 RRQ，但仍高于 PFQ 2.333 ms |
| `global_kpi.ue_goodput_p5_mbps` | 37.359 | 39.680 | 39.379 | HGraph 接近 PFQ，明显好于 RRQ |
| `global_kpi.ue_goodput_jain` | 0.9961 | 0.9987 | 0.9981 | HGraph 公平性接近 PFQ |

解释：

- 这个 checkpoint 相比早期主线，服务覆盖和 packet delay 已经明显改善；`eval.stats_mean_uniq_ue≈12.9` 表明每个 TTI 仍能覆盖较多 UE。
- 与 PFQ 的剩余差距主要来自 PRG 利用率和 BLER：HGraph 的服务守门和 repair 机制让调度更保守，PRG 使用率只有约 `75%`，同时 BLER 比 PFQ 高约 `1.6` 个百分点，所以 aggregate goodput 未超过 PFQ。
- 从 packet delay 看，HGraph 已经不是 RRQ 那种长尾积压形态；但与 PFQ 相比仍有 `mean/p95 delay` 缺口，说明下一步应在保持服务连续性的同时提高可安全复用 PRG 的比例。

ONNX 导出产物：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_challenger_service_guard/export/ppo_actor_absolute_best_action_posteq.onnx
training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_challenger_service_guard/export/ppo_actor_absolute_best_action_posteq.onnx.metadata.json
```

导出配置：

```text
output_mode=action
use_post_eq_input=true
opset=17
decode_path=masked_argmax_to_action_ue_select_and_pre_cap_action_prg_alloc
input_shapes:
  obs_cell_features: [1, 3, 5]
  obs_ue_features: [1, 36, 12]
  obs_prg_features: [1, 3, 17, 8]
  obs_edge_index: [1, 6, 2]
  obs_edge_attr: [1, 6, 2]
  obs_post_eq_sinr: [1, 36, 17]
  action_mask_ue: [1, 36]
  action_mask_cell_ue: [1, 3, 36]
output_shapes:
  action_ue_select: [1, 36]
  action_prg_alloc: [1, 51]
```

导出校验：

```text
onnx_checker=ok
onnxruntime=ok
max_abs_diff.action_ue_select=0.0
max_abs_diff.action_prg_alloc=0.0
```

当前结论：

- 该 checkpoint 可作为 v34 service-guarded 分支的可复现实验产物和 ONNX 集成候选。
- 若目标是直接压倒 PFQ，它还不是最终最优点；下一轮应优先做 lighter guard 或利用率恢复实验，例如 `slot_repair_max_per_cell=2`、较低 `service_gap_weight`、以及对高置信 PRG reuse 的 reward/decoder 放松。

### 10.9 lighter guard + utilization floor 实验设计

2026-05-22 启动下一轮第一阶段优化，目标是在保持服务连续性的前提下恢复 PRG 利用率，先判断是否能直接超过 PFQ。

本阶段先不改 action head，只增加一个默认关闭的 BLER-gated utilization floor reward：

```text
util_floor_gap = max(0, util_floor_target - prg_utilization_ratio)

if tb_err_rate <= util_floor_tb_err_guard:
    util_floor_gate = 1
elif util_floor_tb_err_softness > 0:
    util_floor_gate = clamp((guard + softness - tb_err_rate) / softness, 0, 1)
else:
    util_floor_gate = 0

reward -= util_floor_weight * util_floor_gate * util_floor_gap
```

新增参数：

| 参数 | 默认值 | 本轮建议值 | 作用 |
|---|---:|---:|---|
| `--util-floor-weight` | `0.0` | `8.0` | 低利用率惩罚权重，默认关闭 |
| `--util-floor-target` | `0.0` | `0.84` | 期望 PRG utilization floor |
| `--util-floor-tb-err-guard` | `1.0` | `0.115` | TB err 不高于该值时完整施加 floor |
| `--util-floor-tb-err-softness` | `0.0` | `0.020` | TB err 在 `0.115-0.135` 间线性衰减 |

lighter guard 配置相对 challenger 做三处放松：

| 项 | challenger | lighter guard |
|---|---:|---:|
| `slot_repair_max_per_cell` | `3` | `2` |
| `unserved_ue_weight` | `2.0` | `1.5` |
| `service_gap_weight` | `4.0` | `3.0` |
| `service_rate_target` | `0.72` | `0.70` |

预期判据：

- 若训练/评估能把 `PRG utilization` 从约 `75%` 拉到 `82-86%`，且 `TB BLER` 不明显高于 `12%`，则有机会直接超过 PFQ。
- 若 `goodput` 仍低于 PFQ，同时 utilization 已明显恢复，则问题更可能在 PRG 级选择质量/slot action 结构，应进入 `PRG -> candidate UE` action head 改造。
- 若 utilization 仍拉不动，则优先检查 decoder demand-cap/fallback 和 reward 中 `active_prg_goodput_efficiency` 对低利用率的隐性奖励。

### 10.10 lighter guard + utilization floor 结果

2026-05-22 已完成第一阶段 v35 challenger：

```text
run_dir=training/stageb_hgraph/runs/online_ppo_seed42_v35_lighter_guard_util_floor_from_challenger
init_checkpoint=training/stageb_hgraph/runs/online_ppo_seed42_v34_latency_r2_challenger_service_guard/ppo_actor_absolute_best.pt
action_head_mode=slot
completed_iterations=10
absolute_best_iter=8
absolute_best_score=1572.0198
checkpoint=ppo_actor_absolute_best.pt
eval_dir=output/stageB_hgraph_v35_lighter_guard_util_floor_s41_s42_s43_ue36_ttl200_rbg16_svd_eval
```

关键训练观察：

| iter | goodput Mbps | TB err | fill | blank | post-decode drop | capDrop | unserved UE | service p10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1499.951 | 13.141% | 0.7366 | 0.1989 | 0.0645 | 3.29 | 3.95 | 0.751 |
| 5 | 1575.901 | 9.917% | 0.7486 | 0.0492 | 0.2021 | 10.31 | 5.77 | 0.654 |
| 8 | 1573.041 | 9.970% | 0.7650 | 0.1364 | 0.0987 | 5.03 | 4.51 | 0.706 |
| 10 | 1558.635 | 10.004% | 0.7641 | 0.1762 | 0.0597 | 3.04 | 4.00 | 0.730 |

v35 的 3-seed 评估均值：

| metric | PFQ baseline | v35 lighter guard | 结论 |
|---|---:|---:|---|
| `traffic.goodput_mbps` | 1541.352 Mbps | 1537.356 Mbps | 仍低 3.996 Mbps |
| `traffic.served_mbps_est` | 1729.518 Mbps | 1725.780 Mbps | 仍低 3.738 Mbps |
| `global_kpi.global_tb_bler` | 11.087% | 11.555% | BLER 仍高 0.468 pct |
| `global_kpi.prg_utilization_ratio` | 91.346% | 73.854% | PRG 利用率仍低 17.492 pct |
| `traffic.packet_delay_mean_ms` | 1.355 ms | 1.833 ms | 时延仍高 0.478 ms |
| `traffic.packet_delay_p95_ms` | 4.667 ms | 5.667 ms | 尾时延仍高 1.000 ms |
| `global_kpi.ue_goodput_p5_mbps` | 39.680 Mbps | 39.630 Mbps | cell-edge goodput 已接近 |
| `global_kpi.ue_goodput_jain` | 0.9987 | 0.9985 | 公平性接近 |

结论：

- v35 明显改善 v34 guard 的 aggregate goodput 和 delay，但没有越过 PFQ。
- utilization floor 没有把有效 PRG fill 推到目标区间；训练中经常把显式 blank 转成 post-decode drop / capDrop，而不是稳定的有效填充。
- 这说明主要瓶颈已经不是 reward 里少一个“用 PRG”的信号，而是 `slot -> UE` 先验和 `PRG -> slot` 依赖链限制了 PRG 级选择空间。按本节判据，进入 `PRG -> candidate UE` action head 改造。

### 10.11 PRG -> candidate UE action head 改造记录

2026-05-22 已落地第一版 candidate action head，保持旧路径默认兼容：

| 模块 | 改造 |
|---|---|
| `training/stageb_hgraph/model.py` | 新增 `HGraphModelConfig.action_head_mode`，默认 `slot`；`candidate` 模式下新增 candidate pair scorer |
| `training/stageb_hgraph/online_ppo.py` | PPO sample/log-prob/entropy/teacher-aux 按 `candidate_prg_logits` 自动切换 |
| `training/stageb_hgraph/eval_checkpoint_online.py` | 在线 checkpoint eval 支持 candidate logits + candidate decoder |
| `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh` | 新增 `--action-head-mode slot|candidate` |

新路径的动作结构：

```text
slot branch:
  slot -> UE logits: [S, U+1]

candidate PRG branch:
  pack E_UP by PRG token -> candidate UE ids: [C*P, K]
  candidate logits: [C*P, K+1]
  last class = blank/no-PRG
  decode_candidate_actions_to_type0() maps PRG->UE choices back to:
    action_ue_select [S]
    action_prg_alloc [C*P native allocSol order]
```

candidate scorer 输入：

```text
score_ctx(cell, prg, candidate_ue) =
  concat(
    prg_ctx[cell, prg],          # PRG token embedding after message passing
    ue_ctx[candidate_ue],        # UE embedding after message passing
    edge_attr_proj(E_UP attr)    # [post_eq_db, quality_gap, urgency, debt, prg_risk]
  )
```

当前实现保持两个约束：

- `slot -> UE` branch 仍存在，用于给 Type-0 slot 表提供初始 UE 集合，也给 candidate decoder 提供可替换/补充的 slot 空间。
- candidate decoder 可以给 PRG 选中的新 UE 分配同 cell 空闲 slot 或替换未承载 PRG 的 slot，因此 PRG 决策不再完全受最初 hard slot argmax 限制。

兼容性：

- `action_head_mode=slot` 时不创建 candidate 参数，旧 checkpoint / ONNX slot 路径保持兼容。
- `action_head_mode=candidate` 时可从 v35/v34 checkpoint 兼容加载 encoder、message layers、slot branch 和 legacy PRG branch；candidate scorer 新参数随机初始化，需要 imitation 或 teacher auxiliary 稳定启动。
- 当前 Python online eval 已支持 candidate checkpoint；ONNX action-mode 仍是 legacy slot PRG decode，candidate checkpoint 若要部署到 C++ 还需要后续导出路径同步。

smoke 结果：

```text
out_dir=/tmp/stageb_hgraph_candidate_head_smoke
iterations=1
rollout_steps=32
init_checkpoint=v35 lighter guard absolute best
action_head_mode=candidate
result=online bridge, candidate decode, PPO update all passed
```

smoke 首轮是随机 candidate scorer，`fill=0.4737`、`blank=0.2770`、`post_decode_drop=0.2494`，因此正式 challenger 需要先用 PFQ teacher 做 imitation warm-start，再进入 PPO。

### 10.12 candidate action head 训练与评估结果

2026-05-22 已完成 v36 candidate head 主实验：

```text
run_dir=training/stageb_hgraph/runs/online_ppo_seed42_v36_candidate_head_from_v35_lighter_guard
init_checkpoint=training/stageb_hgraph/runs/online_ppo_seed42_v35_lighter_guard_util_floor_from_challenger/ppo_actor_absolute_best.pt
action_head_mode=candidate
imitation_episodes=1
teacher_mode=pfq_passthrough
completed_iterations=10
absolute_best_iter=5
absolute_best_score=1567.5263
checkpoint=ppo_actor_absolute_best.pt
eval_dir=output/stageB_hgraph_v36_candidate_head_s41_s42_s43_ue36_ttl200_rbg16_svd_eval
```

v36 训练中的关键变化：

| iter | goodput Mbps | TB err | fill | blank | post-decode drop | capDrop | unique UE | service p10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1490.104 | 12.890% | 0.7198 | 0.2422 | 0.0380 | 0.00 | 23.55 | 0.851 |
| 5 | 1567.526 | 9.585% | 0.7280 | 0.2174 | 0.0546 | 0.00 | 23.72 | 0.785 |
| 8 | 1566.812 | 9.657% | 0.7435 | 0.2005 | 0.0560 | 0.00 | 23.98 | 0.798 |
| 10 | 1559.211 | 9.766% | 0.7460 | 0.1918 | 0.0622 | 0.00 | 23.91 | 0.830 |

v36 的 3-seed 评估均值：

| metric | PFQ baseline | v36 candidate | 结论 |
|---|---:|---:|---|
| `traffic.goodput_mbps` | 1541.352 Mbps | 1522.414 Mbps | 低 18.938 Mbps |
| `traffic.served_mbps_est` | 1729.518 Mbps | 1718.783 Mbps | 低 10.735 Mbps |
| `global_kpi.global_tb_bler` | 11.087% | 11.559% | 高 0.472 pct |
| `global_kpi.prg_utilization_ratio` | 91.346% | 75.132% | 低 16.214 pct |
| `eval.stats_mean_prg_utilization_ratio` | n/a | 75.230% | warmup 后仍未拉高 |
| `eval.stats_mean_uniq_ue` | n/a | 22.848 | 调度覆盖明显高于 slot head |
| `eval.stats_mean_cap_drop_count` | n/a | 0.000 | demand-cap/drop 链路已消除 |
| `traffic.packet_delay_mean_ms` | 1.355 ms | 5.543 ms | 时延退化 |
| `traffic.packet_delay_p95_ms` | 4.667 ms | 21.667 ms | 尾时延退化 |

per-seed warmup 视角：

| seed | goodput Mbps | TB err | PRG util | blank | unique UE | expiry drop |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 1538.326 | 11.440% | 80.465% | 12.724% | 24.260 | 0.000% |
| 42 | 1548.397 | 10.564% | 73.784% | 21.214% | 22.559 | 0.051% |
| 43 | 1527.613 | 11.132% | 71.442% | 19.460% | 21.726 | 0.284% |

结论：

- candidate action head 解决了旧 slot head 的一个明确结构问题：`capDrop=0`、`fallback=0`，说明 PRG->UE 决策不再被 slot hard selection 和 demand-cap fallback 大量吞掉。
- 但 candidate head 没有把 PRG utilization 推到 PFQ 区间。策略仍偏向 blank，尤其 seed 42/43 的 `blank≈19-21%`，导致有效 fill 仍停在 `71-74%`。
- packet delay 退化说明它虽然每 TTI 调度更多 UE，但对部分积压 UE 的持续服务不足；低利用率加长了队列驻留时间。

### 10.13 candidate blank-bias 挑战

2026-05-22 为验证 candidate blank 倾向是否是主因，增加 `candidate_blank_logit_bias` 参数，并从 v36 absolute best 继续做短挑战：

```text
run_dir=training/stageb_hgraph/runs/online_ppo_seed42_v37_candidate_blankbias_m06_from_v36
init_checkpoint=training/stageb_hgraph/runs/online_ppo_seed42_v36_candidate_head_from_v35_lighter_guard/ppo_actor_absolute_best.pt
action_head_mode=candidate
candidate_blank_logit_bias=-0.6
imitation_episodes=0
teacher_aux_coef=0.08
teacher_aux_decay_iters=40
completed_iterations=6
absolute_best_iter=5
absolute_best_score=1564.0827
checkpoint=ppo_actor_absolute_best.pt
eval_dir=output/stageB_hgraph_v37_candidate_blankbias_m06_s41_s42_s43_ue36_ttl200_rbg16_svd_eval
```

v37 训练观察：

| iter | goodput Mbps | TB err | fill | blank | post-decode drop | capDrop | unique UE | service p10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1498.072 | 12.860% | 0.7484 | 0.1720 | 0.0796 | 0.00 | 23.74 | 0.906 |
| 3 | 1541.418 | 9.861% | 0.7504 | 0.1436 | 0.1060 | 0.00 | 23.68 | 0.871 |
| 5 | 1564.083 | 9.554% | 0.7306 | 0.2241 | 0.0453 | 0.00 | 23.80 | 0.808 |
| 6 | 1563.119 | 9.736% | 0.7294 | 0.2067 | 0.0638 | 0.00 | 23.98 | 0.813 |

v37 的 3-seed 评估均值：

| metric | PFQ baseline | v37 blank-bias | 结论 |
|---|---:|---:|---|
| `traffic.goodput_mbps` | 1541.352 Mbps | 1472.304 Mbps | 低 69.048 Mbps |
| `traffic.served_mbps_est` | 1729.518 Mbps | 1660.517 Mbps | 低 69.001 Mbps |
| `global_kpi.global_tb_bler` | 11.087% | 11.295% | 接近 PFQ，但仍略高 |
| `global_kpi.prg_utilization_ratio` | 91.346% | 70.515% | 明显退化 |
| `eval.stats_mean_goodput_mbps` | n/a | 1492.426 Mbps | warmup 后也未恢复 |
| `eval.stats_mean_prg_utilization_ratio` | n/a | 70.824% | seed 42/43 拖累明显 |
| `traffic.expiry_drop_rate` | n/a | 2.823% | 出现明显 TTL drop |
| `traffic.packet_delay_mean_ms` | 1.355 ms | 11.437 ms | 时延显著退化 |
| `traffic.packet_delay_p95_ms` | 4.667 ms | 88.408 ms | 尾时延不可接受 |

per-seed warmup 视角：

| seed | goodput Mbps | TB err | PRG util | blank | unique UE | expiry drop |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 1540.512 | 11.264% | 80.367% | 10.073% | 24.413 | 0.000% |
| 42 | 1537.560 | 10.347% | 68.780% | 21.658% | 21.827 | 0.260% |
| 43 | 1399.207 | 10.183% | 63.325% | 22.897% | 19.426 | 6.708% |

结论：

- `candidate_blank_logit_bias=-0.6` 不是稳定收益。它能在 seed 41 上把 blank 降到约 `10%`，但 seed 42/43 仍保持高 blank，并造成严重低利用率。
- BLER 下降没有转化成 throughput。v37 进入了“更保守、更低错包、但更少服务”的区域，seed 43 直接出现 TTL 过期和长尾时延。
- 当前瓶颈不是简单 blank bias，而是 candidate scorer 对拓扑/干扰风险的校准不稳：在部分拓扑上它宁可 blank，也不愿把中等置信 PRG 分给 backlog UE。

下一轮建议：

1. 先回到 v36 checkpoint 作为 candidate 主线，而不是 v37。
2. teacher 需要从 `pfq_passthrough` 改成 PRG 级 soft target：对每个 PRG candidate UE 记录 PFQ 选择、后验成功、post-eq SINR margin、TTL/backlog，训练 scorer 直接学“该 PRG 给谁”，而不是只用最终 Type-0 动作反推。
3. decoder 需要增加 safe-fill fallback：当 candidate 选择 blank 且存在 `TB err guard` 内的高 backlog UE 时，按 `post_eq_db - risk` 排序填充部分 blank PRG，用 evaluator 统计 guarded fill 是否提高 goodput。
4. reward 需要把 utilization floor 从全局平均改成 per-cell/per-risk bucket floor，避免 seed 41 过用、seed 42/43 欠用的平均化失败。
5. ONNX action export 仍需同步 candidate decode；当前 Python online eval 已支持 candidate checkpoint，但 C++ 部署路径仍是 legacy slot action decode。

### 10.14 v38 设计：PRG soft teacher + guarded safe-fill + candidate deployment

2026-05-22 根据 v36/v37 结果继续从 v36 主线推进，不沿 v37 blank-bias 分支。

本轮目标不是单纯“填满更多 PRG”。PFQ 的优势来自感知 PF 与需求 cap 的组合，盲目提高 PRG utilization 可能只会提高 BLER、重复服务或无效发包。因此 v38 的优化目标定义为：

```text
primary:   higher goodput
secondary: higher packet effective service rate / average served speed
guard:     lower TB err, lower TTL expiry, lower backlog tail, lower packet delay
non-goal:  forcing PRG utilization upward as an isolated objective
```

#### 10.14.1 PRG 级 teacher soft target

v36 之前的 candidate teacher 是 hard target：

```text
PFQ Type-0 action -> candidate class id
loss = cross_entropy(candidate_logits, hard_class)
```

这个监督过硬，容易把 PFQ 的单个离散选择当作唯一正确答案，也容易把 PFQ blank 误学成“blank 是安全默认”。v38 改成 PRG 级 soft target：

```text
target_distribution(PRG, candidate UE / blank)
  = softmax(
      quality / success proxy
      + urgency / backlog / TTL
      + PFQ selected UE bonus
      - risk / gap / recent TB err
      + guarded blank prior
    )
```

soft target 输入来自每个 `E_UP` candidate edge：

| feature | 含义 | 用途 |
|---|---|---|
| `post_eq_db` | candidate UE 在该 PRG 的 post-eq SINR dB | 高速率/成功概率代理 |
| `quality_gap` | 与同 PRG 最佳候选的 dB 差距 | 防止质量太差的填充 |
| `urgency` | backlog、HOL delay、TTL slack 组合 | 降 packet delay |
| `debt` | recent scheduling / goodput deficit | 防止长期欠服务 |
| `prg_risk` | `same_prg_conflict_ratio` 与 `ici_proxy` | 控制干扰和 BLER |
| `ue.tb_err_last` | UE 最近 TB err | 避免持续给高错包 UE |
| `PFQ PRG choice` | PFQ 在该 PRG 上选择的 UE 或 blank | teacher prior，不是唯一标签 |

blank 的 soft target 逻辑：

- 如果没有候选同时满足质量、risk、backlog、TB err guard，则 blank 获得较高 prior。
- 如果存在安全候选，则 blank 会被压低，但不会被完全禁止。
- 这样避免 v37 的问题：只靠 blank bias 会在部分拓扑上进入“低错包、低服务、低 goodput”的保守区域。

新增训练参数：

| 参数 | 默认值 | v38 建议 |
|---|---:|---:|
| `--teacher-soft-target` | `1` | `1` |
| `--teacher-soft-temperature` | `0.85` | `0.85` |
| `--teacher-soft-hit-bonus` | `2.0` | `1.6-2.0` |
| `--teacher-soft-blank-bonus` | `0.35` | `0.20-0.35` |
| `--teacher-soft-blank-safe-penalty` | `1.25` | `1.0-1.25` |
| `--teacher-soft-blank-no-safe-bonus` | `1.0` | `1.0` |
| `--teacher-aux-min-coef` | `0.0` | `0.03-0.05` |

`teacher-aux-min-coef` 用于避免 teacher auxiliary 过早衰减到 0。v36/v37 的 best iter 很早，一个原因是 PPO 很快漂离 teacher prior；保留一个小的 soft teacher 下限，可以让策略先探索一段时间再收敛，而不是前几轮偶然高点就成为 best。

#### 10.14.2 guarded safe-fill fallback

v38 增加 candidate decode 后处理：

```text
if policy chooses blank on a PRG:
    find best candidate UE only if all guards pass:
      backlog >= min_backlog
      post_eq_db >= min_quality_db
      quality_gap <= max_quality_gap
      prg_risk <= max_prg_risk
      ue_tb_err_last <= max_ue_tb_err
    fill at most safe_fill_max_per_cell blank PRGs per cell
else:
    keep policy choice
```

默认关闭，v38 训练建议小步开启：

| 参数 | 默认值 | v38 建议 |
|---|---:|---:|
| `--safe-fill-max-per-cell` | `0` | `2` |
| `--safe-fill-min-quality-db` | `-2.0` | `-1.0~-2.0` |
| `--safe-fill-max-quality-gap-db` | `8.0` | `7.0-8.0` |
| `--safe-fill-max-prg-risk` | `0.70` | `0.65-0.70` |
| `--safe-fill-min-backlog-bytes` | `1.0` | `1.0` |
| `--safe-fill-max-ue-tb-err` | `0.45` | `0.35-0.45` |

这不是 utilization floor。safe-fill 只处理“policy blank 但存在高置信 backlog candidate”的 PRG，目的是减少无谓等待和 TTL/packet delay，不直接奖励填满。

新增诊断：

```text
rollout_safe_fill_count_mean
eval.mean_safe_fill_count
eval.stats_mean_safe_fill_count
```

#### 10.14.3 best metric 改造

新增 `latency_guarded_goodput`，避免只按单轮 goodput 保存早期偶然 checkpoint：

```text
score =
    goodput_mbps
  + packet_rate_bonus
  - service_gap_penalty
  - unserved_ue_penalty
  - tb_err_penalty
  - expiry_penalty
  - backlog_p90_penalty
```

它仍以 goodput 为主，但会显式惩罚“高 goodput 偶然点但 packet service rate / backlog / expiry 不好”的策略。v38 训练时配合更长 teacher 辅助下限，期望 best iter 不再集中出现在前几轮。

#### 10.14.4 candidate 部署路径

`training/stageb_hgraph/export_onnx.py` 已同步 candidate action-mode：

```text
action_head_mode=slot:
  legacy masked PRG->slot decode

action_head_mode=candidate:
  fixed-shape PRG candidate pack
  candidate pair scorer
  candidate argmax -> Type-0 action_ue_select/action_prg_alloc
```

当前部署路径说明：

- Python online eval 使用完整训练 decode，包括 guarded safe-fill。
- ONNX action export 已不再把 candidate checkpoint 错导成 legacy slot PRG decode。
- ONNX candidate pack 为固定 3cell/36UE/17PRG action-mode 路径，按 post-eq/urgency/debt/risk 选 top candidates；后续如果 C++ runtime 要完全复刻 Python safe-fill，还应把 safe-fill 参数固化进 export 或放在 C++ finalization 中实现。

### 10.15 v38 实施与评估记录

2026-05-22 已完成 v38 第一版实现和训练挑战。

代码改造范围：

- `training/stageb_hgraph/graph_ops.py`：增加 PRG 级 soft teacher target、guarded safe-fill fallback，并在 candidate decode 输出 `safe_fill_count`。
- `training/stageb_hgraph/online_ppo.py`：candidate teacher auxiliary 支持 soft target CE；teacher aux 支持 `teacher_aux_min_coef`；新增 `latency_guarded_goodput` checkpoint metric。
- `training/stageb_hgraph/eval_checkpoint_online.py`：evaluator 支持 checkpoint 参数继承的 guarded safe-fill，并输出 `mean_safe_fill_count`。
- `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh` 和 `cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh`：透传 soft teacher、safe-fill、latency-guarded best metric 参数。
- `training/stageb_hgraph/export_onnx.py`：action-mode candidate checkpoint 可导出为 `candidate argmax -> Type-0 action_ue_select/action_prg_alloc`。
- `cuMAC/scripts/aggregate_model_kpi_compare.py`：聚合新增 `safe_fill_count` 指标，并兼容当前 `rrq_vs_pfq_compare_mean.csv` baseline 文件名。

#### 10.15.1 训练记录

初始化 checkpoint：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v36_candidate_head_from_v35_lighter_guard/ppo_actor_absolute_best.pt
```

v38 运行目录：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v38_soft_teacher_safe_fill_from_v36
```

本次原计划 60 iter，实际训练到 iter 15 后停止，用已有 checkpoint 评估。停止原因不是程序错误，而是观察到 iter 9-15 已进入吞吐/时延权衡平台期，继续盲跑收益不高。

`ppo_actor_absolute_best.pt` 最终来自 iter 12，而不是前 1-2 轮。iter 9 有最高 rollout goodput，但 iter 12 的 backlog/service 指标更好，因此被 `latency_guarded_goodput` 选中。

| iter | goodput Mbps | TB err | policy blank | drop | fill | safeFill | unserved UE | srv100 p10 | backlog p90 KB | pkt service Mbps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1499.845 | 12.765% | 13.264% | 13.172% | 73.677% | 2.163 | 0.564 | 0.926 | 13.5 | 25.690 |
| 2 | 1566.882 | 9.833% | 9.582% | 13.879% | 76.570% | 1.765 | 0.576 | 0.922 | 13.5 | 24.879 |
| 5 | 1567.860 | 9.830% | 7.755% | 15.127% | 77.133% | 1.286 | 0.683 | 0.909 | 13.8 | 24.565 |
| 9 | 1575.841 | 9.838% | 7.440% | 14.881% | 77.703% | 1.251 | 0.691 | 0.906 | 14.5 | 23.851 |
| 12 | 1565.559 | 9.971% | 4.004% | 17.898% | 78.106% | 0.674 | 0.557 | 0.922 | 12.6 | 25.862 |
| 15 | 1558.437 | 9.815% | 9.576% | 14.447% | 75.990% | 1.451 | 0.848 | 0.895 | 15.4 | 22.886 |

观察：

- v38 不再是 iter 1 最好，策略在训练一段时间后能达到更高 rollout goodput。
- safe-fill 在训练 rollout 中从约 `2.16` 降到 `0.67-1.45`，说明 fallback 不是主要吞吐来源；candidate head 本体在接管选择。
- iter 9 证明模型存在超过 PFQ goodput 的局部动作空间，但它会伴随 backlog/service p10 波动。
- iter 12 的特征是更低 backlog 和更好的 packet service，因此更适合作为 latency-guarded checkpoint。

#### 10.15.2 3-seed 部署评估

评估使用 seeds `41,42,43`，每 seed `4000` evaluator step，warmup `1000` step。

输出目录：

```text
output/stageB_hgraph_v38_soft_teacher_safe_fill_s41_s42_s43_ue36_ttl200_rbg16_svd_eval
```

对比 CSV：

```text
output/stageB_hgraph_v38_soft_teacher_safe_fill_s41_s42_s43_ue36_ttl200_rbg16_svd_eval/RAYLEIGH/rr_vs_pfq_vs_hgraph_v38_soft_teacher_safe_fill_compare_mean.csv
```

核心 mean 对比：

| metric | PFQ | v36 candidate | v38 soft+safe | v38 vs PFQ | v38 vs v36 |
|---|---:|---:|---:|---:|---:|
| `traffic.goodput_mbps` | 1541.352 | 1522.414 | 1530.315 | -11.037 | +7.901 |
| `traffic.served_mbps_est` | 1729.518 | 1718.783 | 1726.980 | -2.538 | +8.196 |
| `global_kpi.global_tb_bler` | 11.087% | 11.559% | 11.608% | +0.521 pp | +0.049 pp |
| `global_kpi.prg_utilization_ratio` | 91.346% | 75.132% | 78.220% | -13.125 pp | +3.088 pp |
| `traffic.expiry_drop_rate` | 0.000% | 0.199% | 0.000% | tie | -0.199 pp |
| `traffic.packet_delay_mean_ms` | 1.355 | 5.543 | 2.133 | +0.779 | -3.409 |
| `traffic.packet_delay_p90_ms` | 2.333 | 4.833 | 3.000 | +0.667 | -1.833 |
| `traffic.packet_delay_p95_ms` | 4.667 | 21.667 | 9.000 | +4.333 | -12.667 |
| `traffic.packet_effective_service_rate_mbps` | 21.737 | 5.686 | 14.994 | -6.744 | +9.308 |
| `global_kpi.scheduled_ratio_mean` | 73.753% | 63.400% | 64.639% | -9.114 pp | +1.239 pp |

eval warmup 后 HGraph 指标：

| metric | v36 | v38 | delta |
|---|---:|---:|---:|
| `eval.stats_mean_goodput_mbps` | 1538.112 | 1547.302 | +9.190 |
| `eval.stats_mean_tb_err_rate` | 11.045% | 11.041% | -0.004 pp |
| `eval.stats_mean_fill_ratio` | 75.230% | 78.341% | +3.111 pp |
| `eval.stats_mean_final_blank_count` | 12.633 | 11.046 | -1.587 |
| `eval.stats_mean_uniq_ue` | 22.848 | 23.293 | +0.445 |
| `eval.stats_mean_safe_fill_count` | n/a | 0.0002 | n/a |

per-seed eval summary：

| seed | mean goodput | stats goodput | mean TB err | stats TB err | fill | safeFill | stats final blank | stats uniq UE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 41 | 1527.000 | 1543.139 | 12.424% | 11.492% | 81.649% | 0.0000 | 9.432 | 24.958 |
| 42 | 1530.290 | 1550.573 | 11.612% | 10.672% | 78.380% | 0.0015 | 10.871 | 22.855 |
| 43 | 1533.653 | 1548.195 | 11.651% | 10.958% | 74.632% | 0.0010 | 12.834 | 22.067 |

结论：

- v38 明确优于 v36：goodput、served Mbps、packet delay、tail delay、packet service rate、expired packet 都有改善。
- v38 还没有整体压过 PFQ：全程 goodput 低 `11.0 Mbps`，packet mean/p95 delay 仍高于 PFQ。
- warmup 后 `eval.stats_mean_goodput_mbps=1547.3 Mbps` 已超过 PFQ 全程 mean `1541.35 Mbps`，但这不是严格同口径胜利；它说明冷启动和部署稳定性是主要问题之一。
- 部署 argmax 下 `policy blank` 基本为 0，`safeFill` 也几乎为 0。safe-fill 目前更像训练期护栏，最终 checkpoint 自身已经学成了强非 blank 倾向。
- PRG utilization 只有 `78.2%`，仍低于 PFQ 的 `91.3%`。这不代表下一步要粗暴追 utilization；要让更多 PRG 被“高成功率、高服务价值”的 UE 使用。

#### 10.15.3 ONNX candidate action export

导出检查命令：

```text
.venv/bin/python training/stageb_hgraph/export_onnx.py \
  --checkpoint training/stageb_hgraph/runs/online_ppo_seed42_v38_soft_teacher_safe_fill_from_v36/ppo_actor_absolute_best.pt \
  --out /tmp/hgraph_v38_candidate_action.onnx \
  --output-mode action \
  --use-post-eq-input \
  --check \
  --device cpu
```

结果：

```text
onnx_checker: ok
onnxruntime: ok
max_abs_diff.action_ue_select = 0.0
max_abs_diff.action_prg_alloc = 0.0
```

candidate checkpoint 的 action-mode 部署路径已经可导出、可 onnxruntime 对齐。导出日志仍有 TracerWarning，主要来自固定 shape 和 slot-layout 中的 Python shape 分支；当前 3cell/36UE/17PRG 固定部署可用，但泛化到其他拓扑时应重新导出或进一步去掉 shape 常量。

#### 10.15.4 下一步 v39 调整建议

v38 的主要限制不是“PRG 用得还不够多”，而是 candidate argmax 对 blank 的抑制过强，同时对 BLER/等待债务的稳定绑定不够。建议 v39 继续从 v38 absolute best 或 v36 主线重启，做以下调整：

1. 不再使用本次训练里的 `--best-tb-err-target 0.38`。该值过松，基本不会惩罚 10%-12% BLER 区间。下一轮建议 `--best-tb-err-target 0.108-0.115`，`--best-tb-err-penalty-mbps 500`。
2. soft teacher 降低“存在 safe candidate 就压 blank”的强度：`teacher_soft_blank_safe_penalty` 从 `1.35` 降到 `0.8-1.0`，同时把 `teacher_soft_hit_bonus` 从 `1.15` 降到 `0.8-1.0`，让 PFQ prior 不再把所有 PRG 都推向非 blank。
3. reward/best metric 加强 packet service 与 backlog：`packet_rate_weight 0.30-0.35`，`aged_backlog_weight 0.04-0.05`，`unserved_ue_weight 1.5`，避免 iter 13/15 这类 high backlog 策略。
4. 尝试小的正 `candidate_blank_logit_bias`，例如 `+0.10~+0.20`，让 argmax 恢复“风险高则 blank”的自由度；guarded safe-fill 再负责把明显安全的 blank 补回来。不要使用 v37 那种负 blank bias。
5. 如果仍不能过 PFQ，再做结构改造：把 candidate scorer 输出拆成 `success_score` 与 `service_value_score` 两路，最终 logit 使用 `service_value * success_gate`，避免一个标量同时承担“速率高”和“低错包”的冲突目标。

### 10.16 v39 执行计划：blank freedom + tighter BLER guard

2026-05-22 启动 v39，目标是在 v38 的基础上减少“argmax 几乎不 blank”的副作用。v39 不追求单独提高 PRG utilization，而是让策略重新拥有“风险高则 blank、确定安全才服务”的自由度。

初始化 checkpoint：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v38_soft_teacher_safe_fill_from_v36/ppo_actor_absolute_best.pt
```

计划输出目录：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v39_blank_free_risk_guard_from_v38
```

关键配置调整：

| 项 | v38 | v39 计划 | 目的 |
|---|---:|---:|---|
| `candidate_blank_logit_bias` | `0.0` | `+0.15` | 恢复风险高 PRG blank 的自由度 |
| `best_tb_err_target` | `0.38` | `0.110` | checkpoint 选择显式惩罚 BLER 超过 PFQ 附近水平 |
| `best_tb_err_penalty_mbps` | `180` | `500` | 避免高 goodput 但高 BLER 被保存 |
| `teacher_soft_hit_bonus` | `1.15` | `0.95` | 降低 PFQ hard prior 的压迫 |
| `teacher_soft_blank_safe_penalty` | `1.35` | `0.90` | 不再强迫存在 safe candidate 时一定非 blank |
| `teacher_soft_blank_no_safe_bonus` | `0.70` | `0.90` | 高风险 PRG 更自然 blank |
| `packet_rate_weight` | `0.25` | `0.35` | 提升 packet service rate |
| `aged_backlog_weight` | `0.035` | `0.05` | 压低等待债务和尾时延 |
| `unserved_ue_weight` | `1.2` | `1.5` | 抑制高 backlog / high unserved 策略 |

预期诊断：

- 如果 v39 生效，rollout 中 `policy blank` 应高于 v38 部署的近零水平，但 `safeFill` 仍应保持小量。
- `TB err` 应接近或低于 `0.110-0.115`，而不是用高 BLER 换吞吐。
- `packet_effective_service_rate_mbps`、`srv100_p10`、`backlog_p90` 应比 v38 iter 13/15 更稳定。

### 10.17 v39 实施与评估记录

2026-05-22 已完成 v39 与一个小调整分支 v39b。

#### 10.17.1 v39 训练结果

v39 运行目录：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v39_blank_free_risk_guard_from_v38
```

v39 从 v38 absolute best 继续，使用 `candidate_blank_logit_bias=+0.15`、`best_tb_err_target=0.110`、更低 soft teacher blank 压制和更高 packet/backlog 权重。训练到 iter 10 后停止，原因是后期进入高 blank / 高 backlog 区间。

v39 absolute best 来自 iter 6：

| metric | value |
|---|---:|
| `rollout_goodput_mbps_mean` | 1568.006 |
| `rollout_tb_err_rate_mean` | 10.069% |
| `rollout_blank_ratio_mean` | 5.850% |
| `rollout_action_fill_ratio_mean` | 77.771% |
| `rollout_safe_fill_count_mean` | 0.866 |
| `rollout_unserved_demand_ue_mean` | 0.548 |
| `rollout_recent100_service_rate_p10_mean` | 0.925 |
| `rollout_backlog_p90_bytes_mean` | 13.2 KB |
| `rollout_packet_effective_service_rate_mbps_mean` | 25.454 |

训练观察：

- `+0.15` blank bias 在 rollout 中确实恢复了 blank 自由度，早期 blank 约 `10%-13%`。
- iter 6 是一个健康点：goodput 仍高，BLER 低于 11%，service p10 和 backlog 均比 v38 后期好。
- iter 7-10 逐渐变保守，blank 上升到 `12%-14%`，unserved/backlog 增大，reward 下降。因此没有继续跑满 60 iter。

#### 10.17.2 v39b 小调整

v39b 运行目录：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v39b_blank_bias008_service_guard_from_v39
```

v39b 从 v39 absolute best 继续，将 `candidate_blank_logit_bias` 从 `+0.15` 降到 `+0.08`，并进一步提高 backlog/service 权重。训练到 iter 8 后停止。

v39b absolute best 来自 iter 5：

| metric | value |
|---|---:|
| `rollout_goodput_mbps_mean` | 1553.588 |
| `rollout_tb_err_rate_mean` | 9.965% |
| `rollout_blank_ratio_mean` | 5.825% |
| `rollout_action_fill_ratio_mean` | 77.407% |
| `rollout_safe_fill_count_mean` | 0.779 |
| `rollout_recent100_service_rate_p10_mean` | 0.930 |
| `rollout_backlog_p90_bytes_mean` | 12.2 KB |
| `rollout_packet_effective_service_rate_mbps_mean` | 25.923 |

v39b 也出现过高吞吐点：

| iter | goodput Mbps | TB err | blank | fill | safeFill | srv100 p10 | backlog p90 KB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1568.557 | 9.91% | 8.22% | 77.74% | 1.18 | 0.901 | 15.3 |
| 6 | 1573.912 | 9.68% | 9.68% | 76.79% | 1.28 | 0.901 | 14.8 |
| 7 | 1576.076 | 9.90% | 10.02% | 76.50% | 1.30 | 0.904 | 14.9 |

但 latency-guarded best 选择了 iter 5，因为 iter 3/6/7 的 backlog 和 service p10 仍不够稳。v39b 说明降低 bias 可以重新找到高吞吐点，但当前 best metric 与训练目标会把 checkpoint 选向更保守的低 backlog 区域。

#### 10.17.3 v39 3-seed 评估

评估 checkpoint：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v39_blank_free_risk_guard_from_v38/ppo_actor_absolute_best.pt
```

评估输出：

```text
output/stageB_hgraph_v39_blank_free_risk_guard_s41_s42_s43_ue36_ttl200_rbg16_svd_eval
```

对比 CSV：

```text
output/stageB_hgraph_v39_blank_free_risk_guard_s41_s42_s43_ue36_ttl200_rbg16_svd_eval/RAYLEIGH/rr_vs_pfq_vs_hgraph_v39_blank_free_risk_guard_compare_mean.csv
```

核心 mean 对比：

| metric | PFQ | v38 | v39 | v39-v38 | v39-PFQ |
|---|---:|---:|---:|---:|---:|
| `traffic.goodput_mbps` | 1541.352 | 1530.315 | 1525.522 | -4.793 | -15.830 |
| `traffic.served_mbps_est` | 1729.518 | 1726.980 | 1726.477 | -0.503 | -3.041 |
| `global_kpi.global_tb_bler` | 11.087% | 11.608% | 11.840% | +0.232 pp | +0.753 pp |
| `global_kpi.prg_utilization_ratio` | 91.346% | 78.220% | 79.113% | +0.893 pp | -12.235 pp |
| `traffic.packet_delay_mean_ms` | 1.355 | 2.133 | 2.350 | +0.217 | +0.995 |
| `traffic.packet_delay_p90_ms` | 2.333 | 3.000 | 3.833 | +0.833 | +1.500 |
| `traffic.packet_delay_p95_ms` | 4.667 | 9.000 | 9.833 | +0.833 | +5.167 |
| `traffic.packet_effective_service_rate_mbps` | 21.737 | 14.994 | 13.534 | -1.460 | -8.203 |
| `eval.stats_mean_goodput_mbps` | n/a | 1547.302 | 1541.384 | -5.919 | n/a |
| `eval.stats_mean_tb_err_rate` | n/a | 11.041% | 11.318% | +0.277 pp | n/a |

per-seed eval summary：

| seed | mean goodput | stats goodput | mean TB err | stats TB err | fill | safeFill | stats final blank | stats uniq UE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 41 | 1522.761 | 1541.323 | 12.349% | 11.458% | 82.910% | 0.0000 | 8.341 | 25.437 |
| 42 | 1522.474 | 1536.862 | 11.965% | 11.140% | 78.322% | 0.0035 | 11.058 | 22.799 |
| 43 | 1531.330 | 1545.965 | 12.094% | 11.354% | 76.107% | 0.0003 | 12.028 | 22.474 |

结论：

- v39 没有替代 v38。虽然训练 rollout 中 v39/v39b 有漂亮的局部点，但 3-seed deterministic argmax 部署下 v39 goodput、packet delay、packet service rate 均弱于 v38。
- 正 `candidate_blank_logit_bias` 在 stochastic rollout 中能增加 blank，但 deterministic eval argmax 下 `policy blank` 仍接近 0。这说明单纯调 logit bias 不能真正恢复部署时“风险高则 blank”的能力。
- v39 的部署形态仍是强非 blank candidate argmax，safe-fill 基本不参与；因此 fallback 不是瓶颈，真正瓶颈是 candidate head 的 deterministic decision boundary。
- 下一步不建议继续只调 scalar bias / reward 权重。应进入结构改造：将 candidate scorer 拆成 `success_score` 与 `service_value_score`，并显式引入 blank/success gate，让部署 argmax 也能在低成功率 PRG 上选择 blank，而不是只在训练采样里偶尔 blank。

### 10.18 v40 结构改造：success/service 双分支 candidate head

2026-05-22 进入 candidate action head 结构改造。目标不是单纯提高 PRG 填充率，而是让策略在每个 PRG 上同时表达两个量：

- `success_score`: 该 PRG->candidate UE 分配在当前 post-eq quality、相对 gap、ICI risk、UE 近期 TB error 下是否可能成功。
- `service_value_score`: 若成功，服务该 UE 对 goodput、packet service、backlog/debt 的价值。

新的 candidate 主 logits 采用：

```text
candidate_logit = service_value_score + log_sigmoid(success_score * success_scale)
blank_logit = blank_value + blank_bias + blank_success_scale * clamp(-max_success_score * success_scale, -8, 8)
```

含义：

- 成功率低的 candidate 会被显式下压，即使 service value 很高也不会无条件抢占。
- 当一个 PRG 上所有 candidate 的 `success_score` 都低时，`blank_logit` 会得到正向 failure margin，deterministic argmax 部署也可以选择 blank。这里不能只用 `log_sigmoid(-max_success)`，因为它最多接近 0，只能“少惩罚 blank”，不能把旧 checkpoint 中偏低的 blank head 真正抬起来。
- 如果存在高成功率 candidate，blank gate 自动减弱，让策略仍可以选择高价值服务，不靠固定负 bias 强行填 PRG。

实现记录：

| 文件 | 改动 |
|---|---|
| `training/stageb_hgraph/model.py` | candidate scorer 拆为 `candidate_pair_score`/service value 与 `candidate_success_score`；新增 `candidate_success_logit_scale`、`candidate_blank_success_scale`；旧 checkpoint 缺省关闭 success gate，避免随机新 head 改变 v38/v39 行为；从旧 checkpoint 续训时 success head 采用零权重、`+1.5` bias 的近中性初始化，避免随机分支破坏 v38 起点 |
| `training/stageb_hgraph/graph_ops.py` | 新增 `candidate_prg_success_targets`，用 `quality_db - 0.35*quality_gap - 7.5*prg_risk - 6.0*tb_err_last` 构造 PRG-candidate 软成功监督 |
| `training/stageb_hgraph/online_ppo.py` | 新增 `candidate_success_aux_coef`，PPO update 中对 `candidate_success_logits` 加 BCE auxiliary loss；summary/metrics 记录 success aux |
| `training/stageb_hgraph/export_onnx.py` | action-mode ONNX candidate argmax 同步 success/service/blank gate 公式，确保部署路径与训练 forward 一致 |
| `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh` | 暴露 `--candidate-success-logit-scale`、`--candidate-blank-success-scale`、`--candidate-success-aux-coef` |

v40 初始建议从 v38 absolute best 继续，而不是 v39/v39b：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v38_soft_teacher_safe_fill_from_v36/ppo_actor_absolute_best.pt
```

建议首轮参数：

| 参数 | 建议值 | 说明 |
|---|---:|---|
| `candidate_success_logit_scale` | `1.2` | 让 success branch 参与 candidate-vs-blank 决策 |
| `candidate_blank_success_scale` | `1.5` | 让低成功率 PRG 的 blank 在 argmax 下可见 |
| `candidate_success_aux_coef` | `0.04-0.05` | 校准 success score，但不压过 PPO reward |
| `candidate_blank_logit_bias` | `0.0` 或 `+0.03` | 不再主要依赖固定 blank bias |
| `teacher_aux_coef` | `0.06-0.08` | 保留 soft teacher 稳定性，但降低 PFQ hard prior |
| `safe_fill_max_per_cell` | `2` | 仅补明显安全的 blank，不把目标变成填满 PRG |

验证重点：

- rollout 与 deterministic eval 的 `blank/fill/safeFill` 是否一致，不再出现 rollout 有 blank、argmax 几乎无 blank 的错位。
- `success_aux_loss` 是否平稳下降；若下降但 goodput 走低，说明 success gate 太强，应先降 `blank_success_scale` 或 `candidate_success_aux_coef`。
- checkpoint best iter 是否不再集中在极早期；若 v40 仍早期最好，说明 reward/teacher 仍在把策略推离高 goodput 区，应进一步降低 teacher aux 或放宽 latency guard。

### 10.19 v40 实施、训练与部署校准记录

2026-05-22 已完成 v40 candidate scorer 结构改造与三组短训练/快速评估。

#### 10.19.1 代码改造落点

已落地：

- `training/stageb_hgraph/model.py`：candidate head 拆为 `candidate_success_score` 与 `candidate_pair_score`/service value；主 candidate logit 使用 `service + log_sigmoid(success)`；blank logit 使用 failure margin gate；success head 对旧 checkpoint 采用零权重、`+1.5` bias 的中性初始化。
- `training/stageb_hgraph/graph_ops.py`：新增 `candidate_prg_success_targets`，按 PRG-candidate 的质量、gap、PRG risk、UE TB err 构造 soft success target。
- `training/stageb_hgraph/online_ppo.py`：新增 `candidate_success_aux_coef`，PPO update 中记录 `candidate_success_aux_loss`。
- `training/stageb_hgraph/export_onnx.py`：ONNX action-mode candidate argmax 同步 success/service/blank gate。
- `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh`：暴露 success gate 与 success aux 参数。

静态验证：

```text
compileall training/stageb_hgraph: pass
bash -n run_stageB_hgraph_online_ppo.sh: pass
git diff --check: pass
```

#### 10.19.2 v40 原始 gate：`log_sigmoid(-max_success)` 不够

运行目录：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v40_success_service_gate_from_v38
```

关键参数：

```text
candidate_success_logit_scale=1.0
candidate_blank_success_scale=1.0
candidate_success_aux_coef=0.04
blank gate = log_sigmoid(-max_success)
```

12 iter 后 best 为 iter 10：

| metric | value |
|---|---:|
| `rollout_goodput_mbps_mean` | 1579.948 |
| `rollout_tb_err_rate_mean` | 9.447% |
| `rollout_blank_ratio_mean` | 12.770% |
| `rollout_action_fill_ratio_mean` | 75.623% |
| `rollout_safe_fill_count_mean` | 2.248 |
| `rollout_recent100_service_rate_p10_mean` | 0.866 |
| `candidate_success_aux_loss` | 0.334 |

快速 seed41 deterministic eval 暴露问题：

| metric | value |
|---|---:|
| `stats_mean_goodput_mbps` | 1554.951 |
| `stats_mean_tb_err_rate` | 11.231% |
| `stats_mean_blank_ratio` | 0.000 |
| `stats_mean_final_blank_count` | 8.572 |

结论：`log_sigmoid(-max_success)` 只会让 blank gate 接近 0，不能给旧 blank head 正向 margin；训练采样里有 blank，部署 argmax 仍几乎不 blank。

#### 10.19.3 v40b failure margin：能 blank，但过强

将 blank gate 改为：

```text
blank_margin = clamp(-max_success * success_scale, -8, 8)
blank_logit = blank_value + blank_bias + blank_success_scale * blank_margin
```

运行目录：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v40b_failure_margin_gate_from_v38
```

best 为 iter 3：

| metric | value |
|---|---:|
| `rollout_goodput_mbps_mean` | 1563.224 |
| `rollout_tb_err_rate_mean` | 9.284% |
| `rollout_blank_ratio_mean` | 13.221% |
| `rollout_action_fill_ratio_mean` | 73.824% |
| `rollout_recent100_service_rate_p10_mean` | 0.909 |
| `rollout_backlog_p90_bytes_mean` | 17.5 KB |

快速 seed41 deterministic eval：

| metric | value |
|---|---:|
| `stats_mean_goodput_mbps` | 1164.503 |
| `stats_mean_tb_err_rate` | 9.986% |
| `stats_mean_blank_ratio` | 31.453% |
| `stats_mean_final_blank_count` | 23.406 |

结论：failure margin 可以让部署 argmax 真的 blank，但 `candidate_blank_success_scale=1.0` 过强，牺牲大量 fill 和 goodput。

#### 10.19.4 v40c scale=0.4：训练强，但部署仍不 blank

运行目录：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v40c_failure_margin_scale04_from_v38
```

关键参数：

```text
candidate_success_logit_scale=1.0
candidate_blank_success_scale=0.4
candidate_success_aux_coef=0.05
```

8 iter 完成，best 为 iter 7：

| metric | value |
|---|---:|
| `rollout_goodput_mbps_mean` | 1601.508 |
| `rollout_tb_err_rate_mean` | 9.938% |
| `rollout_blank_ratio_mean` | 7.059% |
| `rollout_action_fill_ratio_mean` | 78.417% |
| `rollout_safe_fill_count_mean` | 1.120 |
| `rollout_recent100_service_rate_p10_mean` | 0.894 |
| `rollout_backlog_p90_bytes_mean` | 18.5 KB |
| `candidate_success_aux_loss` | 0.332 |

快速 seed41 deterministic eval：

| checkpoint | `stats_mean_goodput_mbps` | `stats_mean_tb_err_rate` | `stats_mean_blank_ratio` | `stats_mean_final_blank_count` |
|---|---:|---:|---:|---:|
| scale 0.4 trained | 1548.558 | 11.536% | 0.000% | 9.411 |
| scale 0.6 posthoc | 1538.909 | 11.258% | 0.000% | 9.510 |
| scale 0.8 posthoc | 1552.473 | 11.431% | 0.000% | 9.080 |
| scale 1.0 posthoc | 1541.976 | 11.442% | 0.000% | 9.346 |

结论：

- v40c 在训练 rollout 中达到了目前最好的 seed42 局部 goodput，但部署 argmax 仍不主动 blank。
- 单纯后验调 `candidate_blank_success_scale` 对 v40c 权重无效，说明 success logits 在该 checkpoint 中整体仍偏高；需要在训练期校准 success score 的绝对尺度。
- v40b 与 v40c 共同说明：结构方向成立，但现在缺少“blank target / success threshold”的训练约束，导致部署 blank 决策呈现过陡阈值。

#### 10.19.5 当前结论与下一步

当前不建议把 v40/v40b/v40c 替代 v38 作为主线部署 checkpoint。推荐保留：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v40c_failure_margin_scale04_from_v38/ppo_actor_absolute_best.pt
```

作为下一轮结构校准的起点或对照，而不是发布 checkpoint。

下一步建议：

1. 增加 PRG 级 explicit blank target：当一个 PRG 的所有 candidate success target 都低于阈值，例如 `max_success_target < 0.45`，给 blank 一个 margin target，而不是只靠 BCE success aux 间接塑形。
2. 增加 success logit calibration 监控：记录 `max_success_logit` 的均值、p10/p50/p90，以及 blank PRG 和 filled PRG 的分布差异。
3. 在 PPO update 中加入 argmax shadow diagnostics：每个 minibatch 额外计算 deterministic candidate choice 的 blank/fill/drop，不参与环境交互也能提前发现“sample 有 blank、argmax 无 blank”。
4. 用 `blank_success_scale=0.6-0.8` 训练，而不是只后验改配置，并配合 explicit blank target，避免 v40b 的高 blank 和 v40c 的零 argmax blank 两个极端。

### 10.20 v41 计划：explicit blank target 与 success calibration

2026-05-25 根据 v40c 的结论进入 v41。目标是让 `success_score` 的绝对 logit 尺度在训练期就对齐部署 argmax，而不是训练后再调 `candidate_blank_success_scale`。

新增训练项：

| loss / metric | 参数 | 作用 |
|---|---|---|
| candidate BCE success aux | `candidate_success_aux_coef` | 对每个 PRG-candidate 的 `success_logit` 拟合 soft physical-success target |
| max-success logit calibration | `candidate_success_max_calib_coef` | 将每个 PRG 的 `max_success_logit` 对齐到 `logit(max_success_target)`，解决 success logit 整体偏高/偏低 |
| explicit blank target margin | `candidate_blank_target_coef` | 当 `max_success_target < threshold` 时要求 `blank_logit > best_candidate_logit + margin`；反之要求 best candidate 至少高出 blank 一个 margin |
| argmax shadow blank | `candidate_shadow_blank_ratio` | PPO minibatch 内的 deterministic argmax blank 诊断，不参与环境交互 |

blank/fill target 定义：

```text
max_success_target = max_k success_target[prg, k]
blank_target = max_success_target < candidate_blank_target_threshold
decision_logit = blank_logit - max_candidate_logit

loss = softplus(margin - signed_target * decision_logit)
signed_target = +1 if blank_target else -1
```

为避免阈值附近噪声，`abs(max_success_target - threshold) < candidate_blank_target_deadband` 的 PRG 不参与 blank margin loss。

新增 CLI 参数：

```text
--candidate-success-max-calib-coef
--candidate-success-target-eps
--candidate-blank-target-coef
--candidate-blank-target-threshold
--candidate-blank-target-margin
--candidate-blank-target-deadband
```

建议首轮从 v40c best 继续，而不是从 v38 重新开始：

```text
init_checkpoint=training/stageb_hgraph/runs/online_ppo_seed42_v40c_failure_margin_scale04_from_v38/ppo_actor_absolute_best.pt
candidate_blank_success_scale=0.7
candidate_success_aux_coef=0.04
candidate_success_max_calib_coef=0.08
candidate_blank_target_coef=0.06
candidate_blank_target_threshold=0.45
candidate_blank_target_margin=0.45
candidate_blank_target_deadband=0.05
```

观察标准：

- `candidate_success_max_logit_mean` 应逐步接近 `logit(clamp(max_success_target, eps, 1-eps))` 的量级；例如 `eps=0.05` 时上界是 `logit(0.95)=2.944`，不能直接拿未 clamp 的 `candidate_success_max_target_mean` 做 logit 解释。
- `candidate_shadow_blank_ratio` 应在训练 update 内维持非零但不过大，期望先落在 `3%-12%`。
- rollout blank/fill 和 quick deterministic eval blank 不应再分裂成“rollout 有 blank、argmax 为 0”。
- 若 `candidate_shadow_blank_ratio` 上升但 goodput/packet service 明显下降，先降 `candidate_blank_target_coef` 或提高 threshold deadband，而不是关闭 success calibration。

#### 10.20.1 v41 实现与 smoke

已在 `training/stageb_hgraph/online_ppo.py` 中落地：

- `candidate_success_max_calib_loss`：对每个 PRG 的 `max_success_logit` 做 SmoothL1 校准，target 为 `logit(clamp(max_success_target, eps, 1-eps))`。
- `candidate_blank_target_loss`：在 PPO minibatch 内显式约束 `blank_logit - max_candidate_logit` 的 margin。
- `candidate_shadow_blank_ratio`：不参与环境交互，仅诊断 deterministic argmax 在 update batch 内是否会主动 blank。

wrapper `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh` 已透传新增参数。静态验证通过：

```text
compileall training/stageb_hgraph
bash -n cuMAC/scripts/run_stageB_hgraph_online_ppo.sh cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh
git diff --check
```

smoke 目录：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v41_blank_calib_smoke
```

1 iter / 48 steps 成功完成，update 指标：

| metric | value |
|---|---:|
| `candidate_success_aux_loss` | 0.4101 |
| `candidate_success_max_calib_loss` | 0.5061 |
| `candidate_blank_target_loss` | 0.0948 |
| `candidate_shadow_blank_ratio` | 2.78% |

#### 10.20.2 v41 短训结果

运行目录：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v41_explicit_blank_success_calib_from_v40c
```

起点：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v40c_failure_margin_scale04_from_v38/ppo_actor_absolute_best.pt
```

关键参数：

```text
candidate_blank_success_scale=0.7
candidate_success_aux_coef=0.04
candidate_success_max_calib_coef=0.08
candidate_success_target_eps=0.05
candidate_blank_target_coef=0.06
candidate_blank_target_threshold=0.45
candidate_blank_target_margin=0.45
candidate_blank_target_deadband=0.05
```

8 iter 完成，`latency_guarded_goodput` best 为 iter 3：

| iter | goodput Mbps | TB err | blank | fill | safeFill | `shadowBlank` | blank target ratio | max success logit | max success target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1401.145 | 18.17% | 2.50% | 73.47% | 0.33 | 0.41% | 0.49% | 3.281 | 0.988 |
| 2 | 1550.566 | 9.76% | 3.52% | 78.09% | 0.49 | 0.11% | 0.18% | 2.894 | 0.995 |
| 3 | 1576.178 | 9.64% | 3.26% | 77.40% | 0.34 | 0.26% | 0.14% | 2.961 | 0.997 |
| 8 | 1565.376 | 10.20% | 1.68% | 79.15% | 0.20 | 0.12% | 0.07% | 2.934 | 0.997 |

解释：

- `candidate_success_max_logit_mean` 很快收敛到约 `2.94`，这正是 `eps=0.05` clamp 后的上界 `logit(0.95)`；success logit 的绝对尺度校准项是有效的。
- `candidate_blank_target_ratio` 只有 `0.07%-0.49%`，说明用 `max_success_target < 0.45` 定义 explicit blank 太保守。当前 success proxy 下，几乎每个 PRG 都至少有一个 candidate 的 soft success target 接近 1。
- 因此 `candidate_shadow_blank_ratio` 仍只有 `0.1%-0.4%`，训练期 deterministic argmax 没有形成稳定 blank 自由度。

#### 10.20.3 v41 seed41 deterministic quick eval

评估目录：

```text
output/stageB_hgraph_v41_explicit_blank_calib_quick_s41_eval
```

checkpoint：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v41_explicit_blank_success_calib_from_v40c/ppo_actor_absolute_best.pt
```

快速 seed41 / 1200 TTI / argmax / warmup 200 结果：

| metric | value |
|---|---:|
| `eval.stats_mean_goodput_mbps` | 1555.763 |
| `eval.stats_mean_tb_err_rate` | 10.952% |
| `eval.stats_mean_prg_utilization_ratio` | 81.686% |
| `eval.stats_mean_blank_ratio` | 0.000% |
| `eval.stats_mean_final_blank_count` | 9.340 |

与 PFQ mean 的聚合对比，v41 在 served throughput 上略高于 PFQ，但 goodput 和 packet delay 仍落后：

| metric | PFQ mean | v41 seed41 quick | delta |
|---|---:|---:|---:|
| `traffic.served_mbps_est` | 1729.518 | 1733.788 | +4.270 |
| `traffic.goodput_mbps` | 1541.352 | 1503.581 | -37.771 |
| `global_kpi.global_tb_bler` | 11.087% | 12.780% | +1.693 pp |
| `global_kpi.prg_utilization_ratio` | 91.346% | 80.554% | -10.792 pp |
| `traffic.packet_delay_mean_ms` | 1.355 | 2.341 | +0.987 |
| `traffic.packet_effective_service_rate_mbps` | 21.737 | 10.252 | -11.486 |

结论：

- v41 完成了 success logit 的训练期绝对尺度校准，但未解决部署 argmax 主动 blank。
- 失败点不是 loss 没接上，而是 explicit blank target 的正样本定义太少；`max_success_target < 0.45` 在当前 success proxy 下几乎不会触发。
- 下一步应把 blank target 从“绝对 success 低于 0.45”改为更贴近部署风险/收益的目标，例如：
  1. 使用更高阈值或分位阈值，例如先试 `candidate_blank_target_threshold=0.995`、`candidate_blank_target_deadband=0.0-0.002`，观察 `blank_target_ratio` 是否进入 `3%-12%`。
  2. 或基于 safe-guard/teacher-soft blank probability 定义 blank target：当没有 candidate 满足 quality gap、PRG risk、UE TB err、backlog guard 时要求 blank；当存在高 service-value safe candidate 时要求 fill。
  3. 增加 `candidate_success_max_target` 分位数和 `blank_logit - max_candidate_logit` 分位数日志，避免只看均值误判。

当前不建议用 v41 替代 v38/v40c 作为主线 checkpoint；它的价值是确认了 success calibration 生效，并定位了 blank target source 的问题。

### 10.21 v41b：提高 blank target 阈值并做正样本加权

2026-05-25 继续 v41b，目标是验证 explicit blank target 能否真正解决“训练采样/部署 argmax 不对齐”。本轮新增诊断和训练参数：

- `candidate_success_max_target_p10/p50/p90`：观察 success target 的分布，而不是只看均值。
- `candidate_blank_decision_logit_p10/p50/p90`：其中 `blank_decision_logit = blank_logit - best_candidate_logit`，大于 0 表示部署 argmax 会选 blank。
- `candidate_blank_target_pos_weight` / `candidate_blank_target_fill_weight`：对 explicit blank margin loss 中的 blank 正样本和 fill 样本分开加权。
- `candidate_blank_target_weighted_ratio`：记录加权后 blank 样本在 margin loss 中的有效占比。

实现位置：

```text
training/stageb_hgraph/online_ppo.py
cuMAC/scripts/run_stageB_hgraph_online_ppo.sh
```

静态验证通过：

```text
compileall training/stageb_hgraph
bash -n cuMAC/scripts/run_stageB_hgraph_online_ppo.sh cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh
git diff --check
```

#### 10.21.1 阈值 probe

从 v41 best 继续：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v41_explicit_blank_success_calib_from_v40c/ppo_actor_absolute_best.pt
```

| run | key params | 训练现象 | 判断 |
|---|---|---|---|
| `v41b_threshold995_smoke` | threshold=0.995, no weighting | cold smoke `blank_target_ratio=38.8%`, `shadowBlank=3.0%` | 冷启动过强，但证明阈值能产生 blank 样本 |
| `v41b_threshold95_long_from_v41` | threshold=0.95, no weighting | iter2 `blank_target_ratio=1.0%`, `shadowBlank=0.4%` | 长窗口过弱 |
| `v41b_threshold99_probe_from_v41` | threshold=0.99, no weighting | iter2 `blank_target_ratio=2.7%`, `shadowBlank=0.4%` | 仍过弱 |
| `v41b_threshold995_long_from_v41` | threshold=0.995, no weighting | iter3 train goodput 1592.7 Mbps, TB err 9.81%, but `shadowBlank=0.18%` | KPI 看起来好，但部署 argmax 仍基本不 blank |

unweighted threshold995 的 seed41 argmax quick eval：

| metric | value |
|---|---:|
| `eval.stats_mean_goodput_mbps` | 1527.488 |
| `eval.stats_mean_tb_err_rate` | 12.262% |
| `eval.stats_mean_blank_ratio` | 0.043% |
| `eval.stats_mean_prg_utilization_ratio` | 82.929% |

结论：单纯提高 `candidate_blank_target_threshold` 不能解决部署对齐；blank target 样本虽存在，但 margin loss 被大量 fill target 和 teacher aux 淹没。

#### 10.21.2 加权 blank target

强权重 probe：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v41b_weighted_blank_probe_from_v41
```

参数：

```text
threshold=0.995
candidate_blank_target_coef=0.08
candidate_blank_target_margin=0.45
candidate_blank_target_pos_weight=3.0
candidate_blank_target_fill_weight=0.5
teacher_aux_coef=0.04
```

训练端第 4 iter：

| metric | value |
|---|---:|
| `rollout_goodput_mbps_mean` | 1563.266 |
| `rollout_tb_err_rate_mean` | 9.829% |
| `rollout_blank_ratio_mean` | 8.534% |
| `rollout_recent100_service_rate_p10_mean` | 0.914 |
| `candidate_blank_target_ratio` | 7.31% |
| `candidate_blank_target_weighted_ratio` | 31.70% |
| `candidate_shadow_blank_ratio` | 10.39% |
| `candidate_blank_decision_logit_p90` | -0.054 |

这说明加权后训练端 deterministic argmax 终于被推到 blank 边界附近。但部署 seed41 quick eval 显示闭环放大过强：

| metric | value |
|---|---:|
| `eval.stats_mean_goodput_mbps` | 964.934 |
| `eval.stats_mean_tb_err_rate` | 10.143% |
| `eval.stats_mean_blank_ratio` | 50.059% |
| `eval.stats_mean_prg_utilization_ratio` | 39.594% |

从该 probe 继续长训会进一步过 blank，已中止：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v41b_weighted_blank_long_from_probe
```

第 2 iter 已出现：

| metric | value |
|---|---:|
| `rollout_goodput_mbps_mean` | 1516.342 |
| `rollout_blank_ratio_mean` | 23.113% |
| `rollout_recent100_service_rate_p10_mean` | 0.703 |
| `candidate_shadow_blank_ratio` | 19.21% |

轻权重 probe：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v41b_weighted_blank_lite_probe_from_v41
```

参数：

```text
threshold=0.995
candidate_blank_target_coef=0.06
candidate_blank_target_margin=0.35
candidate_blank_target_pos_weight=2.0
candidate_blank_target_fill_weight=0.7
teacher_aux_coef=0.04
```

训练端第 3 iter：

| metric | value |
|---|---:|
| `rollout_goodput_mbps_mean` | 1574.211 |
| `rollout_tb_err_rate_mean` | 9.862% |
| `rollout_blank_ratio_mean` | 6.431% |
| `candidate_blank_target_weighted_ratio` | 18.73% |
| `candidate_shadow_blank_ratio` | 0.94% |

部署 seed41 quick eval：

| metric | value |
|---|---:|
| `eval.stats_mean_goodput_mbps` | 1389.712 |
| `eval.stats_mean_tb_err_rate` | 10.634% |
| `eval.stats_mean_blank_ratio` | 13.029% |
| `eval.stats_mean_prg_utilization_ratio` | 68.114% |

#### 10.21.3 v41b 结论

v41b 回答了两个问题：

1. 可以通过 explicit blank target 正样本加权，让部署 argmax 真正选择 blank；v41/v41b unweighted 的“不 blank”问题不是结构不可达。
2. 但当前 blank target 一旦进入部署闭环，会被放大成 PRG 利用率下降和服务节奏下降，goodput 与 packet service 远低于 PFQ。

因此 v41b 没有解决最终 KPI 问题，不建议替代 v38/v40c。下一步不应继续用全局 blank target 权重硬调，而应改成更局部的部署约束：

- 只对 `blank_target=True` 的 PRG 加 blank margin，不再让多数 fill target 通过同一个 loss 反向推 blank；fill 仍交给 teacher/service value。
- blank target 应绑定 service guard：若 UE backlog/debt 高或 recent service 低，即使 success target 稍低也不应 blank。
- 部署解码侧需要 lightweight safe-fill 或 min-service repair，与训练端 blank target 同步，防止少量 blank 决策在闭环里造成服务债。
- 下一轮建议命名为 v42：`positive-only blank margin + service-aware blank mask + deployment min-service repair`。

### 10.22 v42：positive-only blank margin + service-aware blank mask

2026-05-25 进入 v42。目标不是继续提高 blank 数量，也不是单纯提高 PRG utilization，而是修正 v41b 暴露出的训练/部署闭环放大：

- v41b 强加权能让部署 argmax blank，但 blank target 是全局的，会把低成功率 PRG 直接推向 blank。
- 一旦部署 blank 增多，若没有同步保护 backlog/debt/HOL 高的 UE，就会降低服务频率，导致 packet delay 和 effective service rate 退化。
- 因此 v42 只在“低成功率且没有可服务高压力候选”的 PRG 上推 blank；存在可接受成功率的服务候选时，把该 PRG 从 blank target 中剔除。

本轮代码改造：

| 文件 | 改动 |
|---|---|
| `training/stageb_hgraph/graph_ops.py` | 新增 `candidate_prg_service_protection_mask`，按 success target、safe-fill 质量/risk guard、backlog、recent service、HOL/debt 判断某 PRG 是否应免受 blank target |
| `training/stageb_hgraph/online_ppo.py` | `candidate_blank_target` 支持 positive-only；新增 `raw/block/fill` 诊断，记录原始 blank target、被 service guard 拦截比例、局部 fill anchor 比例 |
| `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh` | 透传 v42 参数，便于训练脚本复现实验 |

新增参数：

| 参数 | 作用 |
|---|---|
| `--candidate-blank-target-positive-only` | `1` 时只对正 blank target 加 margin，fill 不再通过同一个 loss 被强推 |
| `--candidate-blank-target-service-guard` | `1` 时启用 service-aware mask |
| `--candidate-blank-target-service-min-success` | 候选 success target 至少达到该值才可阻断 blank target |
| `--candidate-blank-target-service-min-debt` | recent-goodput deficit 超过该值视为有服务压力 |
| `--candidate-blank-target-service-min-hol-ms` | HOL delay 超过该值视为有服务压力 |
| `--candidate-blank-target-service-max-scheduled-ratio` | recent scheduled ratio 低于该值视为有服务压力 |
| `--candidate-blank-target-service-fill-weight` | 对高成功率/有服务压力 PRG 加很轻的 candidate-over-blank anchor，防止 blank head 全局漂移 |
| `--candidate-blank-target-service-fill-margin` | 局部 fill anchor 的 candidate-over-blank margin |
| `--candidate-blank-target-service-fill-min-success` | success target 高于该值时即使不是 service-protected，也可作为 fill anchor |

v42 首轮建议从 v41 best 继续，而不是从 v41b 加权 checkpoint 继续，避免继承过 blank 的闭环状态：

```text
init_checkpoint=training/stageb_hgraph/runs/online_ppo_seed42_v41_explicit_blank_success_calib_from_v40c/ppo_actor_absolute_best.pt
```

首轮 probe 参数：

```text
candidate_blank_target_threshold=0.995
candidate_blank_target_coef=0.055
candidate_blank_target_margin=0.35
candidate_blank_target_pos_weight=2.0
candidate_blank_target_fill_weight=0.0
candidate_blank_target_positive_only=1
candidate_blank_target_service_guard=1
candidate_blank_target_service_min_success=0.70
candidate_blank_target_service_min_debt=0.25
candidate_blank_target_service_min_hol_ms=1.0
candidate_blank_target_service_max_scheduled_ratio=0.65
safe_fill_max_per_cell=2
safe_fill_min_quality_db=-0.5
safe_fill_max_quality_gap_db=6.5
safe_fill_max_prg_risk=0.62
safe_fill_max_ue_tb_err=0.32
```

观察标准：

- `candidate_blank_target_raw_ratio` 可以较高，但 `candidate_blank_target_service_block_ratio` 应能拦住一部分 service-sensitive PRG。
- `candidate_shadow_blank_ratio` 不应像 v41b 强加权一样快速超过 `10%-20%`。
- `rollout_recent100_service_rate_p10_mean`、`rollout_backlog_p90_bytes_mean` 不应随 blank 增加而恶化。
- seed41 deterministic quick eval 中，若 `stats_mean_blank_ratio` 小幅恢复但 goodput、packet service 不崩，才进入更长训练。

#### 10.22.1 首轮 probe 结果与修正

第一版 v42 probe：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v42_positive_service_guard_probe_from_v41
```

参数核心是 `threshold=0.995, coef=0.055, margin=0.35, pos_weight=2.0, fill_weight=0.0, positive_only=1, service_guard=1`。

结果：第 1 iter `raw/block=100.0%/6.5%`，说明 blank target 仍覆盖几乎所有 PRG；第 2 iter `shadowBlank=31.2%`，第 3 iter rollout blank 约 `33.8%`，`srv100_p10` 降到 `0.494`，`backlog_p90` 升到 `329 KB`。该 probe 已中止。

第二版收窄 probe：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v42_narrow_positive_service_guard_probe_from_v41
```

参数改为 `threshold=0.990, coef=0.04, margin=0.25, pos_weight=1.5`，并放宽 service guard。第 1 iter `raw/block=7.5%/1.0%`、`shadowBlank=1.1%`，看起来合理；但第 2 iter 即使 raw target 只剩 `1.8%`，`shadowBlank` 仍跳到 `28.4%`，第 3 iter rollout blank 约 `33.9%`。这说明 positive-only blank margin 会通过共享 blank head 产生全局上浮，不能只靠降低 target ratio 修复。

因此 v42 增加局部 `service_fill_anchor`：

- positive blank target 仍只作用在低成功率 PRG；
- 对 `service_protected` 或 `max_success_target >= service_fill_min_success` 的 PRG，用小权重约束 `candidate_logit > blank_logit`；
- 这不是 v41b 的全局 fill target，而是只锚住高成功率/高服务压力 PRG 的局部边界，目标是防止少量 positive blank 样本把整个 blank head 推穿。

下一版 probe 使用：

```text
threshold=0.990
candidate_blank_target_coef=0.03
candidate_blank_target_margin=0.20
candidate_blank_target_pos_weight=1.0
candidate_blank_target_service_fill_weight=0.10
candidate_blank_target_service_fill_margin=0.10
candidate_blank_target_service_fill_min_success=0.995
```

#### 10.22.2 service fill anchor 与部署 blank budget

第三版 probe：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v42_service_fill_anchor_probe_from_v41
```

该版本把 `service_fill_anchor` 设为较强的局部锚：`coef=0.03, margin=0.20, service_fill_weight=0.10, service_fill_min_success=0.995`。训练端 4 iter 没有再出现 blank 崩塌，best iter 3：

| metric | value |
|---|---:|
| `rollout_goodput_mbps_mean` | 1560.5 |
| `rollout_tb_err_rate_mean` | 9.91% |
| `rollout_blank_ratio_mean` | 7.06% |
| `rollout_action_fill_ratio_mean` | 76.01% |
| `candidate_shadow_blank_ratio` | 0.7% |

但 seed41 deterministic quick eval 显示部署 argmax 几乎不 blank：

| metric | value |
|---|---:|
| `eval.stats_mean_goodput_mbps` | 1535.712 |
| `eval.stats_mean_tb_err_rate` | 11.224% |
| `eval.stats_mean_blank_ratio` | 0.029% |
| `eval.stats_mean_fill_ratio` | 81.288% |
| `traffic.goodput_mbps` | 1480.494 |
| `traffic.packet_delay_mean_ms` | 2.596 |

结论：强 `service_fill_anchor` 能压住 blank head 漂移，但退回到“几乎不 blank”的部署形态，且 goodput/packet delay 仍弱于 PFQ。

第四版 lite anchor：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v42_lite_service_fill_anchor_probe_from_v41
```

参数降为 `coef=0.025, margin=0.18, service_fill_weight=0.04, service_fill_min_success=0.9995`。训练端 best iter 4：

| metric | value |
|---|---:|
| `rollout_goodput_mbps_mean` | 1582.2 |
| `rollout_tb_err_rate_mean` | 9.90% |
| `rollout_blank_ratio_mean` | 6.05% |
| `rollout_action_fill_ratio_mean` | 77.68% |
| `candidate_shadow_blank_ratio` | 10.8% |

但部署侧出现 v41b 类似放大：seed41 quick eval `stats_mean_blank_ratio=53.33%`，`stats_mean_fill_ratio=36.79%`，`stats_mean_goodput_mbps=945.926`。把 `safe_fill_max_per_cell` 从 2 提到 8 仍只有 `stats_mean_fill_ratio=39.57%`；完全放宽 safe-fill 虽可把 fill 拉回 `81.25%`，但 `stats_mean_tb_err_rate=11.55%`、全程 `traffic.goodput_mbps=1463.802`，仍明显落后 PFQ。

这说明：

- train-side `shadowBlank` 只有约 10% 时，deployment argmax 仍可放大到 50% blank；
- 保守 safe-fill 无法补回足够 PRG；
- 极宽松 safe-fill 能补利用率，但会选择质量不足的候选，BLER/goodput 下降。

因此补充部署侧 `candidate_blank_budget`，用于诊断和后续可控部署：

| 新参数 | 作用 |
|---|---|
| `--candidate-blank-budget-max-per-cell` | `-1` 关闭；`0` 表示每 cell 不保留 policy blank；正数表示每 cell 只保留 blank-minus-best-candidate logit 最高的若干 PRG |
| `--candidate-blank-budget-min-decision-logit` | 只有超过该 blank decision logit 的 PRG 才能消耗 blank budget |

seed41 quick eval 结果：

| checkpoint / decode | goodput warmup | TB err warmup | fill | final blank count | 备注 |
|---|---:|---:|---:|---:|---|
| v42 lite 原部署 | 945.926 | 10.881% | 36.79% | 32.24 | blank 崩塌 |
| v42 lite + budget=2/cell | 1502.060 | 11.882% | 79.61% | 10.40 | blank 被压住，但 BLER 偏高 |
| v42 lite + budget=0/cell | 1535.818 | 11.506% | 81.50% | 9.43 | 接近强 anchor，但仍不压 PFQ |

关键观察：`blank_ratio` 已降到 0，但 `final_blank_count` 仍约 9-10 个 PRG/TTI，说明剩余缺口不是 policy blank，而是 candidate->Type0 decode 后因为 demand cap/slot assignment 形成的空洞。

#### 10.22.3 v42b：blank target 退场后的 candidate ranking probe

为了确认是否能靠 PPO 奖励继续修正 candidate 排序，做短训 v42b：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v42b_budget0_rank_probe_from_v42_lite
```

核心设置：

```text
init_checkpoint=training/stageb_hgraph/runs/online_ppo_seed42_v42_lite_service_fill_anchor_probe_from_v41/ppo_actor_absolute_best.pt
candidate_blank_target_coef=0.0
candidate_blank_budget_max_per_cell=0
candidate_blank_success_scale=0.3
safe_fill_max_per_cell=0
teacher_aux_coef=0.04 -> 0.02
```

训练 seed42 10 iter 结果：

| iter | goodput | TB err | fill | post-decode drop | blankBudget |
|---:|---:|---:|---:|---:|---:|
| 1 | 1389.849 | 18.115% | 73.05% | 26.70% | 20.38 |
| 2 | 1563.847 | 10.010% | 79.78% | 20.22% | 21.29 |
| 6 best | 1573.427 | 10.324% | 80.09% | 19.91% | 18.39 |
| 10 | 1553.671 | 10.178% | 79.36% | 20.64% | 12.70 |

seed41 quick eval：

| metric | v42 lite + budget=0 | v42b best |
|---|---:|---:|
| `eval.stats_mean_goodput_mbps` | 1535.818 | 1532.307 |
| `eval.stats_mean_tb_err_rate` | 11.506% | 10.997% |
| `eval.stats_mean_fill_ratio` | 81.50% | 80.67% |
| `eval.stats_mean_final_blank_count` | 9.43 | 9.86 |
| `traffic.packet_delay_mean_ms` | 2.309 | 1.896 |
| `traffic.goodput_mbps` | 1468.492 | 1467.183 |

v42b 降低了 TB err 和 packet delay，但没有提升 goodput，也没有解决解码后空洞。它说明只靠“budget=0 + PPO ranking 微调”不能突破 PFQ。

#### 10.22.4 candidate decode demand cap slack 诊断

v42b 后，`policy blank=0` 但 final blank 仍约 9-10 PRG/TTI，因此将 candidate decode 中的 `_estimate_ue_prg_demand_cap(..., slack_bytes=3000)` 参数化：

| 新参数 | 作用 |
|---|---|
| `--candidate-decode-demand-slack-bytes` | candidate->Type0 解码时，每个 UE 需求 cap 的额外 backlog slack；默认 `3000`，用于防止过度重复服务同一 UE |

seed41 quick eval 结果：

| decode slack | goodput warmup | TB err warmup | fill | final blank count | 全程 goodput | packet delay mean |
|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 1532.307 | 10.997% | 80.67% | 9.86 | 1467.183 | 1.896 ms |
| 6000 | 1517.124 | 11.612% | 86.18% | 7.05 | 1457.967 | 4.090 ms |
| 9000 | 1501.527 | 11.738% | 91.72% | 4.22 | 1430.899 | 4.696 ms |

结论：放松 demand cap 能直接提高利用率，但会显著恶化 BLER 和 packet delay；当前 candidate scorer 没有足够能力在更高 fill 下选择“安全且有服务价值”的 UE/PRG 组合。PFQ 能保持 `~91%` PRG utilization 和 `~11.1%` BLER，而 v42b 在相同 utilization 附近 BLER/packet service 都明显变差。

#### 10.22.5 v42 当前结论

v42 的主要价值是把问题定位清楚：

1. `explicit blank target` 可以让部署 argmax blank，但很容易被闭环放大；
2. `service_fill_anchor` 可以防止 blank 放大，但会回到几乎不 blank；
3. `candidate_blank_budget` 能控制部署 blank，但 goodput 仍不超 PFQ；
4. `demand_cap_slack` 证明低 utilization 是一部分表象，强行提高 utilization 会暴露 candidate ranking/BLER 风险问题。

因此 v42 不建议作为最终 checkpoint。下一步应从“blank/填充约束”转向 candidate scorer 的排序目标：

- 增加 PRG-level pairwise ranking loss：对同一 PRG 内候选，显式要求高 success、高 service value、低 recent TB err 的 UE 排在前面。
- 增加 decode-aware auxiliary：训练时统计 active PRG 在 candidate->Type0 后未分配的原因，把 `post_decode_drop` 纳入 loss/诊断，而不是只看 policy blank。
- 将 success score 的校准从单点 max 扩展到 per-candidate calibrated probability，并在部署 argmax 使用 `service_value + alpha * log(success_prob)` 这类乘性风险模型。
- 保留 `candidate_blank_budget` 和 `candidate_decode_demand_slack_bytes` 作为部署/评估开关，但不要把更高 fill 当作目标本身；目标仍是 goodput、平均速率和 packet delay。

### 10.23 v43：PRG 内 pairwise ranking + per-candidate success calibration + post-decode drop 诊断

2026-05-25 进入 v43。目标是落实 v42 结论：不再继续单独推 blank 或填满 PRG，而是让 candidate scorer 在同一个 PRG 内学会更可靠地区分“安全且有服务价值”的候选，并把 candidate->Type0 decode 后的空洞来源记录清楚。

#### 10.23.1 代码改造

本轮涉及以下文件：

| 文件 | 改动 |
|---|---|
| `training/stageb_hgraph/graph_ops.py` | 新增 `candidate_prg_pairwise_rank_targets(...)`；扩展 `DecodedType0Action`，记录 post-decode drop 原因 |
| `training/stageb_hgraph/online_ppo.py` | 新增 per-candidate success logit calibration、PRG 内 pairwise ranking loss、rollout drop reason 日志 |
| `training/stageb_hgraph/eval_checkpoint_online.py` | eval summary 增加 post-decode reason count，支持部署侧定位空洞来源 |
| `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh` | 透传 v43 训练参数 |

新增 post-decode reason 口径：

| 字段 | 含义 |
|---|---|
| `post_decode_policy_blank_count` | policy/blank-budget 后仍选择 blank 的 PRG 数 |
| `post_decode_no_valid_candidate_count` | PRG active 但没有合法 candidate 的数量 |
| `post_decode_demand_cap_count` | candidate 有效，但因 UE demand cap / headroom 不足被裁掉的数量 |
| `post_decode_no_slot_count` | candidate 有效，但无法找到/创建合法 Type-0 slot 的数量 |
| `post_decode_other_drop_count` | 其他未分类 active drop |

v43 的 pairwise ranking target 在每个 PRG 内构造，不跨 PRG 比较。target 由 success、service value 和 risk 三部分组成：

```text
rank_utility =
  success_weight * success_target
+ service_weight * service_value
- risk_weight    * risk_value
```

其中：

- `success_target` 复用 candidate success target 口径，用 post-eq SINR、quality gap、PRG risk、UE recent TB err 等估计候选成功概率。
- `service_value` 使用 candidate edge 中已有的 urgency/debt，并补充 UE backlog、HOL delay、recent scheduled ratio、recent goodput deficit。
- `risk_value` 使用 PRG risk 和 UE `tb_err_last`。
- 最后对每个 PRG 内 valid candidates 做 masked z-score，使 pairwise margin 只表达同 PRG 内相对顺序。

训练 loss：

```text
for candidate pair (i, j) in same PRG:
    if target_i - target_j >= min_gap:
        loss += softplus(margin - (logit_i - logit_j)) * clipped(target_i - target_j)
```

首轮参数：

```text
candidate_success_aux_coef=0.03
candidate_success_calib_coef=0.03
candidate_success_max_calib_coef=0.05
candidate_pairwise_rank_coef=0.03
candidate_pairwise_rank_margin=0.25
candidate_pairwise_rank_min_gap=0.10
candidate_pairwise_rank_success_weight=0.70
candidate_pairwise_rank_service_weight=0.30
candidate_pairwise_rank_risk_weight=0.15
candidate_blank_budget_max_per_cell=0
candidate_decode_demand_slack_bytes=3000
safe_fill_max_per_cell=0
```

#### 10.23.2 v43 smoke

smoke run：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v43_pairwise_calib_smoke
```

从 v42b checkpoint 继续，1 iter / 80 rollout steps。结果：

| metric | value |
|---|---:|
| `rollout_goodput_mbps_mean` | 938.706 |
| `rollout_tb_err_rate_mean` | 44.19% |
| `candidate_success_prob_mae` | 0.105 |
| `candidate_success_calib_loss` | 0.369 |
| `candidate_pairwise_rank_loss` | 0.377 |
| `post_decode_demand_cap_count_mean` | 21.88 |
| `post_decode_no_slot_count_mean` | 0.00 |
| `post_decode_other_drop_count_mean` | 0.00 |

smoke 是冷启动短窗口，不看 KPI，只确认新增 loss 和 reason diagnostics 能跑通。关键发现是 drop 原因几乎全是 `demand_cap`，不是 slot 分配失败。

#### 10.23.3 v43 10-iter probe

主 probe：

```text
training/stageb_hgraph/runs/online_ppo_seed42_v43_pairwise_calib_probe_from_v42b
init_checkpoint=training/stageb_hgraph/runs/online_ppo_seed42_v42b_budget0_rank_probe_from_v42_lite/ppo_actor_absolute_best.pt
```

训练 seed42 10 iter 结果：

| iter | goodput | TB err | fill | demand-cap drop | success MAE | rank loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1388.214 | 17.74% | 72.59% | 13.85 | 0.094 | 0.374 |
| 2 | 1549.448 | 9.99% | 78.83% | 10.80 | 0.084 | 0.319 |
| 3 best-score | 1576.654 | 9.89% | 78.76% | 10.83 | 0.072 | 0.324 |
| 6 | 1574.527 | 10.24% | 79.67% | 10.37 | 0.071 | 0.269 |
| 8 | 1569.769 | 10.50% | 80.32% | 10.04 | 0.069 | 0.239 |
| 10 last | 1558.821 | 9.99% | 78.75% | 10.84 | 0.067 | 0.240 |

训练内结论：

- per-candidate success calibration 有效，`succMAE` 从 `0.094` 降到 `0.067`。
- pairwise rank loss 从 `0.374` 降到约 `0.24`，说明候选排序辅助信号稳定。
- `post_decode_no_slot_count=0`、`post_decode_other_drop_count=0`，final blank 的主要来源明确是 demand cap/headroom 裁剪。
- `best_score` 仍在第 3 iter，说明现有 checkpoint 选择指标仍偏向短期 reward/goodput；但校准指标在更晚 iter 继续变好。

#### 10.23.4 seed41 quick eval

使用 `ppo_actor_absolute_best.pt` 和 `ppo_actor_last.pt` 都做了 seed41 quick eval。

| checkpoint | full goodput | full TB err | full fill | warmup goodput | warmup TB err | warmup fill | warmup demand-cap drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| v42 lite budget=0 | 1468.492 | 14.573% | 80.04% | 1535.818 | 11.506% | 81.50% | - |
| v42b best | 1467.183 | 14.268% | 79.24% | 1532.307 | 10.997% | 80.67% | - |
| v43 absolute best iter3 | 1483.190 | 14.518% | 79.33% | 1549.729 | 11.355% | 80.86% | 9.76 |
| v43 last iter10 | 1487.354 | 14.470% | 79.94% | 1551.905 | 11.068% | 81.32% | 9.53 |

相对 PFQ 3-seed mean：

| metric | PFQ | v43 last seed41 | delta |
|---|---:|---:|---:|
| `traffic.served_mbps_est` | 1729.518 | 1726.669 | -2.849 |
| `traffic.goodput_mbps` | 1541.352 | 1487.354 | -53.998 |
| `global_kpi.global_tb_bler` | 11.087% | 13.203% | +2.116% |
| `global_kpi.prg_utilization_ratio` | 91.346% | 79.939% | -11.407% |
| `traffic.packet_delay_mean_ms` | 1.355 | 2.360 | +1.005 |
| `traffic.packet_delay_p95_ms` | 4.667 | 8.500 | +3.833 |

`last` checkpoint 虽然不是训练 `absolute_best`，但部署 quick eval 更好：warmup goodput `1551.905 Mbps`，TB err `11.068%`，fill `81.32%`。这说明 v43 的校准和 pairwise ranking 在继续训练后有实际部署收益，当前 best checkpoint selection 需要纳入更贴近部署的选择口径。

#### 10.23.5 v43 结论

v43 有三个明确收获：

1. per-candidate success calibration 和 PRG 内 pairwise ranking 能稳定学习，且 `last` checkpoint 部署优于 `absolute_best`。
2. post-decode drop 原因已定位：在 `candidate_blank_budget=0` 后，policy blank 基本不是主因；`demand_cap/headroom` 裁剪贡献了 warmup 约 `9.5-9.8` 个 PRG/TTI 的 final blank。
3. v43 把 warmup goodput 从 v42b 的 `1532.3 Mbps` 提升到 `1551.9 Mbps`，但全程 goodput 仍低 PFQ，主要因为全程 BLER 约 `13.2%`，高于 PFQ 的 `11.1%`。

因此 v43 不应回退到继续“强行提高 fill”。下一步更合理的方向是：

- checkpoint 选择：增加 `last/EMA` 或基于部署 quick-eval proxy 的选择，不只用训练 rollout score。
- 风险模型：把部署 argmax 的 candidate score 改为更强的乘性风险形式，例如 `service_value + alpha * log(success_prob)`，并约束 success calibration 的温度/尺度。
- decode-aware 训练：对 `demand_cap` drop 反向约束候选/slot 组合，例如 penalize 同 UE over-headroom 的 PRG 排序，或在 pairwise target 中加入 per-UE remaining headroom/已选 slot load。
- action head：考虑增加 PRG->candidate UE 的 headroom-aware auxiliary，直接学习“当前 PRG 给这个 UE 是否会在 Type-0 finalize 中被保留”，而不是只学习物理成功概率。
