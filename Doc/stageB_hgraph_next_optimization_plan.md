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

当前 `prg_risk` 实现口径需要特别注意：`training/stageb_hgraph/edge_generator.py` 里使用的是 PRG 特征第 4/5 维之和，即 `neighbor_max_top1_sinr_db + neighbor_mean_top1_sinr_db`。PRG 节点自身仍保留第 6/7 维 `same_prg_conflict_ratio` 和 `ici_proxy`，但它们不是当前 `E_UP.edge_attr[4]` 的直接来源。

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
