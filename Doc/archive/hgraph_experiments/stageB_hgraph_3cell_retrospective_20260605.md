# Stage-B 3-Cell HGraph Retrospective

本文是 3 小区 HGraph Stage-B 的复盘文档。它保留算法结构、v34-v50 关键实验和机制教训，用于支持新的 7 小区设计；它不再是当前下一步执行计划。

当前 7 小区执行主线见：

- [`stageB_7cell_tar_gnn_pfq_design.md`](../../stageB_7cell_tar_gnn_pfq_design.md)

v34-v43 的完整逐轮日志已归档到：

- [`stageB_hgraph_v34_v43_full_experiment_log_20260525.md`](./stageB_hgraph_v34_v43_full_experiment_log_20260525.md)

本文只保留仍有决策价值的内容：算法架构、关键实验结果、为什么多轮优化没有明显压过 PFQ，以及哪些机制应迁移到 7 小区 TAR-GNN-PFQ。

## 1. 3-Cell 复盘结论

3-cell HGraph 已经具备可运行的 online PPO + candidate action-mode 部署链路，但 v34-v50 多轮优化没有稳定压过 PFQ。

核心结论：

- HGraph 能接近 PFQ 的 served throughput，但全程 goodput 仍经常低于 PFQ。
- 主要差距不是“不会填 PRG”这么简单，而是高 fill 下 candidate 风险排序不够准，导致 BLER 或 packet delay 变差。
- v42/v43 已确认：`candidate_blank_budget=0` 后，policy blank 不是主要瓶颈；final blank 主要来自 candidate->Type0 decode 后的 `demand_cap/headroom` 裁剪。
- PFQ 机制简单，但它的 inductive bias 很强：按 PF 指标、可服务需求、信道质量持续调度，天然贴合当前仿真 KPI。HGraph 学到的策略反而容易在 reward、argmax、decode 约束之间错位。
- 对 7-cell 新主线最有价值的迁移经验是：不要继续调 blank 或强行提高 PRG utilization，而要做 PFQ anchor、decode/headroom-aware candidate ranking、expected-goodput scorer 和更贴近部署的 checkpoint 选择。

## 2. 当前算法架构

### 2.1 场景和固定 ABI

代表性场景：

| 项 | 当前口径 |
|---|---|
| cell | `3` |
| UE | `36` |
| PRG/RBG | `17` per cell, `51` total |
| Type-0 slots | `36` |
| bandwidth grouping | `RBG16` |
| channel | Rayleigh |
| precoding | `4T4R + SVD` |
| traffic | `3000 bytes/pkt`, `1 pkt/TTI`, `TTL=200ms` |

当前 action-mode ONNX ABI 固定为 `3cell/36UE/17PRG`。换 cell/UE/PRG 数量需要重新处理导出和 runtime 形状。

### 2.2 图结构

节点：

| 节点 | 数量 | 粒度 |
|---|---:|---|
| `Cell` | `C` | cell / TTI |
| `UE` | `U` | global UE / TTI |
| `PRG` | `C * P` | `(cell, local_prg)` / TTI |

边：

| 边 | 方向 | 作用 |
|---|---|---|
| `E_CC` | Cell -> Cell | 小区间上下文和干扰关系 |
| `E_CU` | Cell -> UE | UE 归属和 cell context 注入 |
| `E_CP` | Cell -> PRG | cell 状态注入 PRG token |
| `E_UP` | UE -> PRG | candidate UE-PRG 质量、服务压力、风险 |
| `E_PP` | PRG -> PRG | 同 local PRG 跨小区 reuse/interference 上下文 |

PRG token flatten：

```text
prg_token = cell_id * n_prg + local_prg_id
```

C++ Type-0 native order：

```text
native_linear = local_prg_id * n_cell + owner_cell
```

因此 Python/ONNX/C++ 间必须做 PRG 顺序转换。

### 2.3 输入特征

Cell feature `shape=[C,5]`：

| idx | 字段 |
|---:|---|
| 0 | `cell_load_bytes` |
| 1 | `active_ue_count` |
| 2 | `mean_wb_sinr_lin` |
| 3 | `mean_avg_rate_mbps` |
| 4 | `tb_err_rate` |

UE feature `shape=[U,12]`：

| idx | 字段 |
|---:|---|
| 0 | `buffer_bytes` |
| 1 | `avg_rate_mbps` |
| 2 | `wb_sinr_lin` |
| 3 | `cqi` |
| 4 | `ri` |
| 5 | `tb_err_last` |
| 6 | `new_data_flag` |
| 7 | `stale_slots` |
| 8 | `hol_delay_ms` |
| 9 | `ttl_slack_ms` |
| 10 | `recent_scheduled_ratio` |
| 11 | `recent_goodput_deficit_norm` |

PRG feature `shape=[C*P,8]`：

| idx | 字段 |
|---:|---|
| 0 | `top1_sinr_db` |
| 1 | `top2_gap_db` |
| 2 | `prev_prg_assigned` |
| 3 | `prev_prg_reuse_ratio` |
| 4 | `neighbor_max_top1_sinr_db` |
| 5 | `neighbor_mean_top1_sinr_db` |
| 6 | `same_prg_conflict_ratio` |
| 7 | `ici_proxy` |

`same_prg_conflict_ratio` 和 `ici_proxy` 是 C++ observation extras 中生成的轻量 proxy：

```text
same_prg_conflict_ratio = conflict_count / (n_cell - 1)
ici_proxy = 0.5 * same_prg_conflict_ratio + 0.5 * prg_sinr_drop_norm
```

它们不是真实跨小区信道协方差，只是“邻区同 PRG 竞争 + 当前 PRG 相对 wideband 退化”的启发式风险特征。

### 2.4 动作头和解码

历史两阶段 Type-0 动作：

1. `slot -> UE`，shape `S x (U+1)`。
2. `PRG -> slot`，shape `C x P x (S+1)`。

v36 后 candidate action head：

1. 仍先选 `slot -> UE`，保证 Type-0 slot layout。
2. 每个 PRG 在 packed candidates 中选 candidate UE 或 blank。
3. candidate->Type0 decode 将 PRG 分配回 slot，并执行合法化。

decode 约束：

- cell / slot / PRG mask 合法性
- selected UE 必须有 demand
- per-UE demand cap / headroom
- over-cap fallback
- optional safe-fill
- optional blank budget

当前关键诊断：

```text
final blank = policy blank + no valid candidate + demand-cap drop + no-slot drop + other drop
```

v43 结果显示，在 `candidate_blank_budget=0` 后，warmup final blank 约 `9.5-9.8 PRG/TTI`，几乎全部来自 `demand_cap/headroom` 裁剪。

## 3. v34-v43 优化操作总览

| 版本 | 主要动作 | 预期 | 实际结果 | 关键教训 |
|---|---|---|---|---|
| v34 | PRG risk 对齐、candidate 排序加入 urgency/debt、latency reward、slot repair | 提升服务连续性 | service guard 改善 delay，但 PRG utilization / goodput 仍落后 PFQ | reward 可以改变服务节奏，但 checkpoint 容易选到早期高吞吐点 |
| v35 | lighter guard + utilization floor | 在不升 BLER 下提高 fill | 比 v34 好，但仍没过 PFQ | 问题不是单纯缺少 utilization 奖励 |
| v36 | 进入 PRG->candidate UE action head | 解除 PRG->slot bottleneck | 结构更灵活，但 BLER、delay 和 fill 仍不稳定 | candidate 排序质量成为主瓶颈 |
| v37 | blank bias 实验 | 恢复风险高则 blank | 更保守、BLER 降但服务不足，TTL/delay 恶化 | 单独调 blank 会进入低服务区域 |
| v38 | PRG soft teacher + guarded safe-fill | 稳定 candidate 部署并改善 delay | 明显优于 v36，3-seed goodput 仍低 PFQ 约 11 Mbps | 有效但未突破；仍受 candidate 风险校准限制 |
| v39 | tighter TB target、降低 teacher blank 压制、增加 packet/backlog 权重、小正 blank bias | 让策略更自由地 blank | 训练有局部好点，部署弱于 v38 | 训练采样和 deterministic argmax 仍错位 |
| v40 | success/service 双分支 candidate scorer + blank/success gate | 显式分离成功率和服务价值 | 结构方向成立，但 blank gate 尺度非常敏感 | success logit 绝对尺度不校准，部署 argmax 不可靠 |
| v41 | explicit blank target / success calibration | 对齐 success logits 和部署 argmax | success max logit 可校准，但 blank target 太弱 | 成功率校准有效，blank 监督来源不够 |
| v41b | 提高 blank threshold、正样本加权 | 让部署真的 blank | 能 blank，但强加权导致 blank 放大、goodput 崩 | 全局 blank target 会破坏服务节奏 |
| v42 | positive-only blank margin、service-aware mask、blank budget、demand slack 诊断 | 避免 blank 放大并定位 final blank | budget 可压住 policy blank；放松 demand cap 提高 fill 但 BLER/delay 恶化 | 瓶颈转为 candidate risk ranking + decode-aware 分配 |
| v43 | PRG 内 pairwise ranking、per-candidate success calibration、post-decode drop reason | 提高 candidate 排序和部署诊断 | warmup goodput 提升到 `1551.9 Mbps`，但全程 goodput 仍低 PFQ 约 `54 Mbps` | ranking/calibration 有收益，但仍缺 headroom/decode-aware 信号 |

## 4. 代表性结果

### 4.1 v38 3-seed mean

| 指标 | PFQ | HGraph v36 | HGraph v38 |
|---|---:|---:|---:|
| `traffic.goodput_mbps` | 1541.352 | 1522.414 | 1530.315 |
| `traffic.served_mbps_est` | 1729.518 | 1718.783 | 1726.980 |
| `global_kpi.global_tb_bler` | 11.087% | 11.559% | 11.608% |
| `global_kpi.prg_utilization_ratio` | 91.346% | 75.132% | 78.220% |
| `traffic.packet_delay_mean_ms` | 1.355 | 5.543 | 2.133 |
| `traffic.packet_delay_p95_ms` | 4.667 | 21.667 | 9.000 |
| `traffic.packet_effective_service_rate_mbps` | 21.737 | 5.686 | 14.994 |

v38 的意义：它证明 soft teacher + safe-fill + candidate head 可以显著改善 v36，但还没有稳定超过 PFQ。

### 4.2 v42/v43 seed41 quick eval

| checkpoint | full goodput | full TB err | full fill | warmup goodput | warmup TB err | warmup fill | warmup demand-cap drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| v42 lite budget=0 | 1468.492 | 14.573% | 80.04% | 1535.818 | 11.506% | 81.50% | - |
| v42b best | 1467.183 | 14.268% | 79.24% | 1532.307 | 10.997% | 80.67% | - |
| v43 absolute best iter3 | 1483.190 | 14.518% | 79.33% | 1549.729 | 11.355% | 80.86% | 9.76 |
| v43 last iter10 | 1487.354 | 14.470% | 79.94% | 1551.905 | 11.068% | 81.32% | 9.53 |

v43 的意义：

- success calibration 和 pairwise ranking 确实让 warmup 部署指标改善。
- `ppo_actor_last.pt` 比训练选出的 `absolute_best` 更好，说明当前 best checkpoint 选择口径不贴近部署。
- 全程 goodput 仍弱，主要是冷启动和全程 BLER 偏高。

### 4.3 v43 last 相对 PFQ

| 指标 | PFQ | v43 last seed41 | 差值 |
|---|---:|---:|---:|
| `traffic.served_mbps_est` | 1729.518 | 1726.669 | -2.849 |
| `traffic.goodput_mbps` | 1541.352 | 1487.354 | -53.998 |
| `global_kpi.global_tb_bler` | 11.087% | 13.203% | +2.116% |
| `global_kpi.prg_utilization_ratio` | 91.346% | 79.939% | -11.407% |
| `traffic.packet_delay_mean_ms` | 1.355 | 2.360 | +1.005 |
| `traffic.packet_delay_p95_ms` | 4.667 | 8.500 | +3.833 |

served throughput 已经非常接近 PFQ，但 goodput 被 BLER 拉开。这说明继续追 served throughput 意义不大，必须降低错误传输和无效 PRG 分配。

## 5. 为什么没有达到预期提升

### 5.1 优化目标与真实部署指标错位

训练中经常出现早期 best，后续校准指标更好但训练 score 反而下降。v43 里 `ppo_actor_last.pt` 部署优于 `ppo_actor_absolute_best.pt` 就是典型例子。

原因：

- PPO rollout reward、best metric、最终 C++/Python deterministic argmax eval 不是同一个目标。
- 训练采样能探索 blank/fill 分布，但部署 argmax 会放大 logit 边界。
- reward 里 goodput、packet service、BLER、utilization、backlog 互相拉扯，单个 scalar score 很难稳定选中 Pareto 最优点。

结果是：训练中看起来的高点，不一定是部署中最好的 checkpoint。

### 5.2 PFQ 虽简单，但偏置高度匹配当前问题

PFQ 的强点不是模型复杂，而是目标对齐：

- 它持续服务有需求 UE，不容易形成长时间未服务。
- PF 指标天然兼顾瞬时信道和长期公平。
- 它不会学习到奇怪的 blank/free-fill 边界。
- 它和 MAC demand / Type-0 调度约束在机制上更一致。

HGraph 虽然表达能力更强，但必须自己从 reward 中学出这些机制。当前 reward 和 auxiliary loss 给出的信号还不够直接。

### 5.3 Candidate success/risk 模型仍太粗

当前 success target 主要来自 post-eq SINR、quality gap、PRG risk proxy、UE TB error 等启发式特征。

问题：

- `same_prg_conflict_ratio` 和 `ici_proxy` 只是轻量 proxy，不是真实跨小区干扰矩阵。
- post-eq SINR 能反映局部质量，但不充分表达最终 TB decode 的复杂性。
- candidate 的成功率和“本 TTI 同 UE 已分配多少 PRG / 同 PRG 邻区怎么复用 / Type-0 slot load”有关，而当前 scorer 还没有完整地把这些 decode 后状态纳入排序。

因此 HGraph 可以接近 PFQ 的 served throughput，但高风险 PRG/UE 组合会带来更高 BLER，goodput 被吃掉。

### 5.4 Decode 后约束没有进入 action head 的核心决策

v42/v43 证明 final blank 主要来自 `demand_cap/headroom` 裁剪。这说明模型上游经常选中“看起来有服务价值，但 finalize 后会被裁掉”的 PRG/UE 组合。

当前 candidate head 更像在回答：

```text
这个 PRG 给哪个 UE 物理上比较好？
```

但实际部署需要回答：

```text
这个 PRG 给哪个 UE，经过 Type-0 slot、UE headroom、demand cap、fallback 后仍会被保留，并贡献 goodput？
```

少了后者，模型会反复制造 decode 后空洞。强行放松 demand cap 虽能提高 fill，但会导致 BLER 和 delay 恶化，说明被裁掉的 PRG 并不都是“可安全补上的好 PRG”。

### 5.5 Blank 不是主瓶颈，过度调 blank 反而伤害策略

v37-v42 已经把 blank 方向试清楚：

- 负 blank bias / 强 fill 会让策略冒进或低质量服务。
- 正 blank bias / 强 blank target 会让策略过度保守。
- service fill anchor 能防止 blank 崩，但会回到几乎不 blank。
- blank budget 能控制部署 blank，但不能解决 decode 后 demand-cap 空洞。

因此 blank 只是症状开关，不是根因。

### 5.6 Action structure 仍受 Type-0 slot 粒度限制

即使 candidate head 让 PRG 可以直接挑 UE，最终仍要落回 Type-0 slot：

- slot 先选 UE，会限制 PRG 可落到的 UE 集合。
- PRG 选择和 slot/headroom 并不是完全联合优化。
- decoder 再做 cap/fallback，会产生训练前向和部署结果的不一致。

这也是“模型有 candidate 选择能力，但最终仍留下 9-10 个 final blank PRG/TTI”的重要原因。

### 5.7 单 seed / quick eval 容易误导

多轮实验中经常出现 seed42 rollout 或 seed41 warmup 的漂亮局部点，但 3-seed 或全程指标不稳定。

原因：

- 冷启动、拓扑 seed、traffic backlog 初始状态对 BLER/delay 影响很大。
- warmup 后指标能反映稳态能力，但报告中的 full KPI 仍会被前段拉低。
- 当前 checkpoint selection 没有把多 seed / warmup eval 融入训练闭环。

因此不能再用单个训练 rollout 的 best score 判断是否超过 PFQ。

### 5.8 RRQ 与 PFQ 的 backlog 感知差异

当前 RRQ/PFQ 都已经不是完全不看队列的传统 RR/PF。它们共同点是：`bufferSize == 0` 的 UE 不应被当成有效新传需求，Type-0 下也会用 demand cap 避免明显超发。但两者的 backlog 感知深度不同。

RRQ 的 backlog 感知是**门控/上限型**：

- 入口上，当前 Type-0 baseline 会先把所有 active UE 放进 Type-0 slot，RRQ scheduler 再按 cell 收集关联 UE。
- 若 `rrQueueDemandAwareCap=1`，每个 UE 的最多 PRG 数约为：

```text
cap_prg = ceil((buffer_bytes + slack_bytes) / rr_estimated_bytes_per_prg)
```

- PRG 分配仍按 round-robin cursor 扫 UE。backlog 只决定“这个 UE 还能不能继续拿 PRG”，不决定“谁更应该先拿”。
- 因此 RRQ 对大 backlog UE 的响应是滞后的：只有当其他 UE 达到 cap 或无需求后，大 backlog UE 才会拿到额外 PRG。
- RRQ 不看每个 PRG 上的瞬时信道差异，也不看长期 avgRate；它的主要 inductive bias 是服务节奏均匀和实现稳定。

PFQ 的 backlog 感知是**排序/上限混合型**：

- 基础分数仍是 PF：

```text
base_pf = rate^beta / avgRate
```

- backlog 会直接进入候选排序，形成 queue-aware PF 权重：

```text
queue_weight = 1 + coeff * log2(1 + buffer_bytes / scale_bytes)
score = base_pf * queue_weight
```

- Type-0 逐 PRG 分配时，还会用每个 candidate 的 `predBytes` 和当前 TTI 已分配字节做 demand-aware cap：

```text
remaining = buffer_bytes + slack_bytes - assigned_bytes_this_tti
if remaining <= 0:
    adjusted_score = 0
elif predBytes > remaining:
    adjusted_score = score * remaining / predBytes
```

- 同一 UE 在同一 TTI 内已经拿到多个 PRG 后，还会做 intra-TTI decay：

```text
adjusted_score /= 1 + decay_coeff * assigned_prg_count_this_tti
```

所以 PFQ 同时回答三个问题：这个 UE backlog 是否值得服务、这个 PRG 上它的 PF/信道是否好、当前 TTI 是否已经给够。相比 RRQ，PFQ 更像是“带 headroom 的逐 PRG 贪心排序”。

两者流程可概括为：

```text
RRQ Type-0:
1. 每个 cell 收集关联 active UE。
2. 对每个 UE 计算 cap_prg。
3. 按 PRG 顺序扫描；每个 PRG 给下一个未到 cap 的 UE。
4. 若所有 UE 都到 cap，则后续 PRG 保持空。

PFQ Type-0:
1. GPU 先计算每个 (cell, PRG, UE) 的 PF metric 和 predBytes。
2. CPU/GPU host postprocess 按 cell、PRG 顺序重选最终 UE。
3. 对每个候选应用 buffer weight、remaining-demand scaling、intra-TTI decay。
4. 当前 PRG 给 adjusted_score 最高的 UE，并更新该 UE 的 assigned bytes / PRG count。
5. 若所有候选 adjusted_score 为 0，则该 PRG 为空。
```

下面换成容器内真实 Stage-B replay trace，而不是 toy topology 或离线近似。运行口径：

- 进入容器：`./cuPHY-CP/container/attach_aerial.sh`，再 `source .venv/bin/activate`。
- 共同参数：`--topology-scenario 3cell --total-ue-count 36 --prbs-per-group 16 --packet-size-bytes 3000 --traffic-arrival-rate 1.0 --packet-ttl-ms 200 --precoding svd --topology-seed 42 --custom-ue-prg 0 --exec-mode both --replay-dump 1`。
- 分别运行 `--baseline-scheduler rrq` 和 `--baseline-scheduler pfq`。
- Replay 维度：`n_cell=3`、`n_active_ue=36`、`n_prg=17`，所以每 TTI 有 `51` 个 Type-0 PRG opportunity。
- 运行命令请求了 `--tti 10`，但当前 skip-built binary 仍按编译态配置写出 `2000` 条 replay record；下面只解码 `tti=0..9`。
- `PFQ` 这次 full-run harness 因 CPU/GPU per-UE CDF gap 被标成 `WARN`，sum-throughput check 通过；这里使用 replay 中记录的 native action 做分配形状分析。

表中 `--` 表示 blank；每个 allocation 串按 `PRG0 ... PRG16` 排列，数字是 global UE id。`q_p90/q_max/activeQ` 来自该 TTI 开始时的 `obs_ue_features[:, buffer_bytes]`，短窗口只用于观察机制，不作为稳态 KPI。

2000 TTI 的整体结果先给一个参照：

| baseline | PRG util | served Mbps | goodput Mbps | pkt delay mean | pkt delay p95 | residual buffer ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RRQ | 0.883961 | 1728.766 | 1548.282 | 1.248 ms | 5.500 ms | 0.000501 |
| PFQ | 0.914029 | 1722.931 | 1528.953 | 0.737 ms | 2.000 ms | 0.000476 |

RRQ 前 10 TTI 的摘要：

| TTI | alloc/51 | blank | unique UE | top repeat | activeQ | q_p90 bytes | q_max bytes |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 34 | 17 | 23 | UE00x3, UE33x3, UE01x2 | 23 | 6000 | 9000 |
| 1 | 36 | 15 | 22 | UE21x4, UE09x3, UE26x3 | 22 | 7500 | 12000 |
| 2 | 38 | 13 | 22 | UE15x4, UE19x4, UE10x3 | 22 | 9000 | 12000 |
| 3 | 35 | 16 | 24 | UE03x3, UE11x3, UE04x2 | 24 | 6000 | 9000 |
| 4 | 42 | 9 | 25 | UE16x3, UE19x3, UE23x3 | 25 | 9000 | 9000 |
| 5 | 44 | 7 | 27 | UE31x4, UE10x3, UE00x2 | 27 | 6000 | 12000 |
| 6 | 35 | 16 | 22 | UE12x4, UE32x3, UE35x3 | 22 | 6000 | 12000 |
| 7 | 38 | 13 | 25 | UE12x4, UE27x3, UE32x3 | 25 | 6000 | 12000 |
| 8 | 42 | 9 | 24 | UE18x4, UE02x3, UE09x3 | 24 | 7500 | 12000 |
| 9 | 40 | 11 | 26 | UE17x4, UE29x4, UE07x3 | 26 | 6000 | 12000 |

RRQ 前 10 TTI 的具体分配：

| TTI | cell0 PRG0-16 | cell1 PRG0-16 | cell2 PRG0-16 |
| ---: | --- | --- | --- |
| 0 | `00 04 08 10 11 20 27 33 00 27 33 00 33 -- -- -- --` | `12 13 15 16 21 22 23 29 34 12 13 34 -- -- -- -- --` | `01 05 24 25 31 32 01 05 24 -- -- -- -- -- -- -- --` |
| 1 | `02 04 06 07 09 10 11 27 33 35 09 33 35 09 -- -- --` | `14 15 16 21 23 28 29 14 15 21 21 21 -- -- -- -- --` | `05 26 30 31 32 26 31 32 26 31 -- -- -- -- -- -- --` |
| 2 | `03 04 06 08 09 10 11 27 33 35 10 27 33 35 10 -- --` | `12 15 16 19 23 29 34 15 16 19 15 16 19 15 19 -- --` | `01 17 25 26 31 01 26 26 -- -- -- -- -- -- -- -- --` |
| 3 | `03 04 06 07 09 10 11 20 33 35 03 04 10 11 33 03 11` | `12 14 15 16 18 19 21 22 29 34 22 -- -- -- -- -- --` | `17 24 31 32 17 24 31 -- -- -- -- -- -- -- -- -- --` |
| 4 | `00 02 03 04 06 07 08 09 10 11 20 33 00 02 03 07 11` | `13 14 16 19 23 28 29 34 16 19 23 29 34 16 19 23 29` | `05 25 30 31 32 05 31 32 -- -- -- -- -- -- -- -- --` |
| 5 | `00 02 04 06 07 08 09 10 11 20 27 35 00 10 20 35 10` | `12 14 15 16 18 19 21 22 28 29 14 15 16 18 21 22 29` | `01 17 25 31 32 25 31 32 31 31 -- -- -- -- -- -- --` |
| 6 | `00 02 03 04 06 10 11 20 35 06 11 35 35 -- -- -- --` | `12 13 15 16 19 29 12 19 29 12 12 -- -- -- -- -- --` | `01 17 24 25 30 31 32 30 31 32 32 -- -- -- -- -- --` |
| 7 | `02 03 04 07 09 11 27 33 35 02 03 27 35 27 -- -- --` | `12 13 14 15 16 18 19 21 22 28 34 12 14 22 28 12 12` | `17 26 30 31 32 32 32 -- -- -- -- -- -- -- -- -- --` |
| 8 | `00 02 04 09 10 11 20 27 33 00 02 09 20 02 09 -- --` | `12 14 16 18 21 23 28 29 34 18 21 23 29 34 18 18 --` | `01 17 25 30 31 32 01 25 30 31 30 -- -- -- -- -- --` |
| 9 | `02 03 04 06 07 10 11 20 27 33 03 04 07 07 -- -- --` | `13 14 15 16 18 22 28 29 34 14 16 18 29 34 29 29 --` | `01 17 24 25 30 31 32 17 17 17 -- -- -- -- -- -- --` |

PFQ 前 10 TTI 的摘要：

| TTI | alloc/51 | blank | unique UE | top repeat | activeQ | q_p90 bytes | q_max bytes |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 46 | 5 | 20 | UE05x5, UE28x5, UE02x4 | 20 | 9000 | 15000 |
| 1 | 44 | 7 | 21 | UE17x5, UE07x4, UE28x4 | 22 | 7500 | 9000 |
| 2 | 47 | 4 | 23 | UE01x4, UE20x4, UE05x3 | 23 | 6000 | 12000 |
| 3 | 45 | 6 | 19 | UE05x5, UE28x5, UE34x5 | 19 | 6000 | 9000 |
| 4 | 42 | 9 | 18 | UE28x6, UE07x5, UE30x4 | 19 | 9000 | 15000 |
| 5 | 48 | 3 | 19 | UE28x6, UE05x5, UE20x4 | 21 | 7500 | 15000 |
| 6 | 42 | 9 | 19 | UE20x6, UE34x6, UE05x5 | 23 | 9000 | 12000 |
| 7 | 51 | 0 | 24 | UE02x7, UE29x5, UE01x4 | 28 | 6000 | 12000 |
| 8 | 40 | 11 | 23 | UE29x4, UE05x3, UE20x3 | 25 | 9000 | 15000 |
| 9 | 51 | 0 | 22 | UE28x6, UE35x6, UE01x4 | 23 | 7500 | 14040 |

PFQ 前 10 TTI 的具体分配：

| TTI | cell0 PRG0-16 | cell1 PRG0-16 | cell2 PRG0-16 |
| ---: | --- | --- | --- |
| 0 | `10 03 11 03 02 27 20 20 02 03 07 07 20 27 02 07 02` | `15 22 19 21 18 16 12 23 16 28 23 28 16 28 28 21 28` | `25 17 05 17 01 01 05 05 01 05 17 05 -- -- -- -- --` |
| 1 | `06 11 04 10 11 09 04 27 35 35 27 07 07 07 35 07 35` | `18 12 22 23 16 28 12 29 23 29 28 29 16 28 29 28 --` | `26 30 32 25 24 26 17 17 17 17 17 -- -- -- -- -- --` |
| 2 | `06 10 00 11 08 08 20 20 20 08 20 11 03 07 03 07 07` | `18 22 21 15 23 16 13 18 16 23 34 29 34 29 34 29 21` | `32 26 24 30 32 01 30 05 01 05 05 01 01 -- -- -- --` |
| 3 | `33 00 06 09 33 27 20 02 20 27 33 02 02 09 20 02 07` | `15 21 16 34 16 21 28 34 34 28 34 34 28 28 28 -- --` | `26 24 25 31 05 01 05 01 05 05 01 05 01 -- -- -- --` |
| 4 | `06 00 04 09 04 20 20 20 09 03 03 03 07 07 07 07 07` | `22 15 19 23 23 22 34 34 23 28 28 34 34 28 28 28 28` | `24 30 25 31 26 30 30 30 -- -- -- -- -- -- -- -- --` |
| 5 | `06 09 04 33 09 08 20 27 20 08 09 20 27 20 06 04 33` | `21 14 18 16 16 28 28 28 28 28 28 12 13 12 13 13 12` | `30 31 05 30 05 05 05 05 01 01 17 01 17 17 -- -- --` |
| 6 | `06 04 11 33 09 27 20 20 04 27 20 20 20 09 06 20 33` | `19 15 14 18 22 23 34 34 29 28 14 34 29 34 28 34 34` | `30 31 05 30 05 05 05 05 -- -- -- -- -- -- -- -- --` |
| 7 | `10 00 09 06 09 02 02 02 02 03 02 20 02 07 03 20 02` | `21 18 19 22 23 16 23 34 29 21 34 29 29 29 34 29 16` | `32 30 25 26 31 30 05 05 05 05 32 01 01 01 01 17 17` |
| 8 | `04 11 33 27 11 27 04 33 27 00 08 06 08 00 20 20 20` | `19 18 22 23 29 29 23 29 29 15 16 14 28 13 28 12 14` | `25 32 05 05 05 26 -- -- -- -- -- -- -- -- -- -- --` |
| 9 | `04 33 27 27 04 35 33 35 35 35 35 35 09 00 03 20 20` | `23 23 29 23 29 29 29 16 16 28 15 28 16 28 28 28 28` | `24 25 31 32 30 30 26 05 05 05 17 01 01 17 01 01 17` |

从这 10 个 TTI 能看到更贴近真实 Stage-B 的分配特性：

- RRQ 的分配串有明显的 association-order 扫描痕迹：每个 cell 先按固定 UE 顺序给有 backlog 且未到 cap 的 UE 发一轮，后面的 PRG 才给仍有 cap 的 UE；如果都到 cap，就出现 `--`。因此 RRQ 前 10 TTI 只用了 `34..44/51` 个 PRG，blank 常见。
- RRQ 的 backlog 感知主要体现在 `cap_prg`：backlog 大的 UE 允许被重复分配，但不会因为 backlog 大而提前插队，也不会因为某个 PRG 上信道更好而被优先挑中。
- PFQ 的分配串更不规则，同一 cell 内相邻 PRG 会跳到不同 UE，这是逐 PRG 比较 `PF score * queue_weight` 的结果；例如 PFQ 的 UE28、UE20、UE05、UE02 等会在某些 TTI 拿到 `5..7` 个 PRG。
- PFQ 的 unique UE 数通常少于 RRQ，但 PRG 填充更积极，前 10 TTI 有两轮填满 `51/51`，2000 TTI 的 PRG utilization 也更高。它牺牲了一部分“轮询外观上的均匀”，换来更强的 backlog/信道/公平性联合排序和更低的 packet delay p95。
- 因此，两者都“看 backlog”，但 RRQ 是把 backlog 当作能不能继续服务的硬门槛；PFQ 是把 backlog 同时放进排序、需求上限和同 TTI 内抑制里。前者的分配形状更像稳定轮询，后者更像带 headroom 的逐 PRG 贪心调度。

## 6. 已经不建议继续投入的方向

以下方向已经基本验证过，继续小幅调参收益很有限：

- 单独提高 PRG utilization。
- 单独调 `candidate_blank_logit_bias`。
- 全局 explicit blank target 正样本加权。
- 简单放松 `candidate_decode_demand_slack_bytes`。
- 只靠 safe-fill 补 PRG。
- 只用训练 rollout goodput 保存 best checkpoint。

这些方向要么转成过度 blank，要么提高 BLER，要么改善局部 warmup 但全程不稳。

## 7. 下一步若继续突破 PFQ，应做什么

### 7.1 Decode-aware candidate target

把 target 从“物理成功概率”扩展到“decode 后是否保留并贡献 goodput”：

```text
target = kept_after_decode * decode_success * delivered_bytes
```

可先做辅助监督，不直接替代 PPO：

- 对被 `demand_cap` 裁掉的 candidate 降权。
- 对同 UE 已接近 headroom 的后续 PRG 加 penalty。
- 对能覆盖 high backlog / high HOL 且 headroom 充足的 candidate 加权。

### 7.2 Headroom-aware action scoring

candidate scorer 输入或 target 应显式包含：

- 当前 UE estimated PRG cap。
- 已被同 TTI 其他 PRG 选中的数量。
- selected slot load。
- per-cell active demand UE 数。
- UE backlog 是否足够支撑更多 PRG。

这会让模型从“选好 UE”变成“选会被 finalize 保留的 UE”。

### 7.3 Checkpoint 选择改造

保留训练 best，但新增部署选择口径：

- `last` / EMA checkpoint。
- 固定短 eval seed 的 warmup goodput + TB err + demand-cap drop proxy。
- checkpoint score 中加入 `success_prob_mae`、`post_decode_demand_cap_count`、`stats_mean_tb_err_rate`。

v43 已经证明 `last` 可能比训练 `absolute_best` 更适合部署。

### 7.4 风险模型改为乘性 goodput 口径

部署 argmax 不应只看 service value，也不应只看 success logit。候选分数应更接近：

```text
expected_goodput = service_bytes_or_rate * calibrated_success_prob - risk_penalty
```

可在 scorer 或 auxiliary loss 中实现：

```text
score = service_value + alpha * log(success_prob) - beta * headroom_risk
```

重点是让低成功率候选即使 backlog 很高也不会被误选。

### 7.5 更强 teacher 或 imitation source

如果目标是压过 PFQ，仅靠 PPO 从当前 reward 学出 PFQ 的调度规律成本较高。可以考虑：

- 用 PFQ decision trace 做更强 imitation warm start。
- teacher target 不只给单一 action，而给 PRG 内 candidate distribution。
- 将 PFQ 的可解释项拆成 auxiliary target：PF metric、headroom、expected delivered bytes。

这不是回到 PFQ，而是把 PFQ 的强 inductive bias 显式注入网络，再让网络学习超越它的干扰/多小区场景。

## 8. 当前推荐保留文档

当前主线只需要优先阅读：

- [`README.md`](../../README.md)：项目入口和当前状态。
- [`current_stageB_effective_configuration.md`](../../current_stageB_effective_configuration.md)：当前有效配置。
- 本文：HGraph 算法结构、v34-v43 复盘和下一步判断。

历史细节已归档：

- [`stageB_hgraph_v34_v43_full_experiment_log_20260525.md`](./stageB_hgraph_v34_v43_full_experiment_log_20260525.md)
- [`../hgraph_design/`](../hgraph_design/)
- [`../platform/`](../platform/)

## 9. 2026-05-25 追加验证：active power 与部署侧 headroom ranking

基于 v43 `ppo_actor_last.pt` 做 quick eval，配置为 seed41/42/43、`1000 TTI`、`200 warmup`、3cell/36UE/RBG16/SVD/TTL200。当前代码默认 `candidate_headroom_rank_mode=stable`，新增实验均为显式开关。

| 方案 | 3-seed warmup goodput | TB err | fill | demand-cap drop | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| stable current | `1551.05` | `11.01%` | `79.03%` | `10.70` | 当前对齐基线 |
| active-RBG power, max boost 3 dB | `1553.66` | `11.05%` | `78.31%` | `11.06` | 均值小幅 +2.61 Mbps，但 seed 方向不一致 |

active-RBG power 的 per-seed goodput delta 为：

```text
seed41 +15.36 Mbps
seed42  -1.40 Mbps
seed43  -6.14 Mbps
```

因此它是一个值得保留的功率模型开关，但还不能作为“稳定压过 PFQ”的证据。后续若继续该方向，必须同时补 PFQ/RRQ under active-power 的公平基线。

同轮还测了若干部署侧 headroom ranking quick 实验：

| 方案 | seed41 warmup goodput | TB err | fill | demand-cap drop | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| stable current | `1546.56` | `11.06%` | `82.65%` | `8.85` | seed41 当前基线 |
| headroom logit ranking | `1547.04` | `11.18%` | `81.70%` | `9.34` | 仅 +0.49 Mbps，噪声级 |
| headroom phy ranking | `1546.17` | `11.11%` | `82.93%` | `8.71` | 基本持平 |
| headroom utility ranking | `1532.74` | `11.20%` | `81.17%` | `9.60` | 变差 |
| slack 6000 | `1530.73` | `11.45%` | `85.39%` | `7.45` | 多填但 goodput 下降 |

判断：

- 简单放宽 headroom 或手写部署 ranking 不能稳定提升，且容易把 fill 换成更高 BLER。
- `logit/phy` ranking 可作为 gated debug 开关保留，但不建议作为下一轮主线。
- 真正需要的是训练期 target/scorer 对齐：让模型直接学习 `kept_after_decode * success * delivered_bytes`，而不是部署阶段临时重排。

## 10. 2026-05-25 追加验证：direct target/scorer 学习，active power 关闭

本轮按要求关闭 active power（训练和 eval 命令均未设置 `CUMAC_ACTIVE_RBG_POWER`），从 v43 checkpoint 继续验证“target/scorer 直接学习”。

### 10.1 已实现机制

- `candidate_prg_decode_goodput_targets`：rollout 后读取 actual UE diagnostics，只监督本轮 sampled main candidate；若该 candidate 被 Type-0/headroom decode 裁掉或 PHY 未产生 goodput，则 target 为 0；若保留并产生 goodput，则 target 约为 `per_prg_goodput_bytes / target_bytes`。
- `candidate_decode_goodput_loss`：直接作用在部署 scorer 的 `candidate_prg_logits[:, :, :-1]`，而不是只校准 success proxy。
- `candidate_argmax_decode_loss`：训练时对当前 scorer 的 argmax 做 shadow Type-0 decode，直接惩罚“argmax 选中但会被 demand/headroom 裁掉”的 candidate。

### 10.2 快速训练结果

| run | init | 训练设置 | rollout 结论 |
| --- | --- | --- | --- |
| v44 decode-goodput | v43 best | `decode_goodput_coef=0.05`, pairwise 关闭 | sampled rollout 可学，`decode_gp` loss `0.852 -> 0.795`，goodput 到 `1572.5 Mbps`，post-decode drop 到 `0.211` |
| v45b argmax-shadow | v43 last | `argmax_decode_coef=0.10`, `decode_goodput_coef=0.02`, pairwise `0.015` | sampled rollout 保持 `1565.9 Mbps`，但 `argKeep` 只从 `0.132 -> 0.146`，argmax 保留率改善很慢 |

### 10.3 deterministic argmax eval 结果

同一口径：seed42、`1000 TTI`、`200 warmup`、argmax decode、3cell/36UE/RBG16/SVD/TTL200。

| checkpoint | full goodput | warmup goodput | warmup fill | warmup final blank | warmup demand-cap drop | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v43 last control | `250.55` | `283.68` | `25.15%` | `38.17` | `36.88` | deterministic argmax 已严重塌缩 |
| v44 decode-goodput best | `255.93` | `289.34` | `25.67%` | `37.91` | `36.31` | 比 v43 略好，但仍不可用 |
| v45b argmax-shadow best | `258.82` | `295.75` | `25.92%` | `37.78` | `36.95` | 有小幅改善，但没有解决 argmax/headroom 塌缩 |

判断：

- sampled PPO rollout 与 deterministic argmax eval 已经明显分裂；sampled rollout 约 `1560 Mbps`，argmax eval 仅约 `290 Mbps warmup`。
- 单独的 per-candidate BCE 无法表达“同一 UE 已被多少 PRG 选中”的全局 headroom 状态；argmax 会把大量 PRG 压到同一批 UE，然后被 demand-cap 裁掉。
- v45b 的 shadow argmax loss 方向正确但信号太弱/太局部，不能靠继续小步训练解决。下一步若继续，应把 scorer 从 per-PRG 独立打分改成 sequential/top-k-with-budget 或显式加入 per-UE headroom state。

## 11. 当前工作判断

HGraph 目前不是“还差几个参数”的状态，而是进入了机制瓶颈：

```text
PFQ 的简单机制 = 与当前 MAC/Type-0/demand/headroom 约束天然对齐
HGraph 当前机制 = 表达能力强，但 reward、argmax 和 decode 约束仍未完全对齐
```

如果目标是近期拿到稳定超过 PFQ 的结果，优先级应是：

1. decode-aware / headroom-aware candidate ranking。
2. 部署口径 checkpoint selection。
3. calibrated expected-goodput scorer。
4. PFQ-style teacher distribution。

如果不做这些机制层改造，继续围绕 blank、utilization floor、safe-fill 做局部调参，大概率只会在“低利用率低 BLER”和“高利用率高 BLER”之间来回摆动，难以稳定压过 PFQ。

## 12. 2026-05-26 追加验证：sequential budgeted candidate decoder

本轮按要求继续关闭 active power，并改 scorer/decoder 部署机制，让 candidate ranking 显式维护“同一 UE 已经占用多少 PRG / 还剩多少 decode headroom”。

### 12.1 已实现机制

- 新增 `select_budgeted_candidate_actions`：按每个 PRG 的最佳 candidate logit 从高到低顺序处理，维护 `ue_prg_used`，并用 `_estimate_ue_prg_demand_cap` 得到每个 UE 的当前 PRG headroom。
- 每个候选分数按同 UE 已用 PRG 做递减：`score -= usage_penalty * used_prg`。
- `hard_cap=1` 时，UE headroom 用尽后该 UE 后续 candidate 被 hard mask；该 PRG 若无可选 candidate 才 blank。
- `hard_cap=0` 时允许超 headroom，但对超 cap candidate 加 `overcap_penalty`。
- budgeted selector 当前作为 deployment/scorer ranking 机制插入，默认关闭；训练、评估 wrapper 均新增 `--candidate-budgeted-*` 参数。
- 评估 summary 新增 `budgeted_changed/no_headroom/active_ue` 指标，便于判断 decoder 是否真的在约束同 UE 过量分配。

关键修正：

- 初版 budgeted selector 允许 blank logit 参与竞争，导致 hard-cap 虽然把 demand-cap drop 压到 0，但 fill 掉到约 `19%`。
- 已改为 candidate-forcing 语义：只要存在有效且未超 headroom 的 candidate，就只在 candidate 中选择；只有没有可选 candidate 时才 blank。这与之前 `candidate_blank_budget=0` 的部署口径一致。

### 12.2 seed42 快速 A/B

口径：seed42、`1000 TTI`、`200 warmup`、argmax decode、3cell/36UE/RBG16/SVD/TTL200，active power 未开启。

| 方案 | checkpoint | slack | hard cap | warmup goodput | TB err | fill | reuse | demand-cap drop | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v43 control | v43 last | `3000` | off | `283.68` | `14.84%` | `25.15%` | `42.42%` | `36.88` | argmax 大量过量服务同 UE，被 Type-0 裁掉 |
| v45b control | v45b best | `3000` | off | `295.75` | `14.33%` | `25.92%` | `43.76%` | `36.95` | direct target 有小幅提升，但仍未解决 headroom |
| hard-cap | v43 last | `3000` | on | `299.44` | `14.37%` | `25.22%` | `16.57%` | `0.00` | 成功消除过量服务，goodput 高于 v45b |
| soft-cap | v43 last | `3000` | off | `291.24` | `14.26%` | `25.12%` | `28.67%` | `37.00` | 超 cap 重新出现，收益不足 |
| hard-cap + slack6000 | v43 last | `6000` | on | `303.27` | `14.24%` | `29.07%` | `18.28%` | `0.00` | 当前 seed42 最好点 |
| hard-cap + slack9000 | v43 last | `9000` | on | `296.75` | `15.91%` | `33.12%` | `20.56%` | `0.00` | 过宽 headroom 提高 fill，但 BLER/goodput 变差 |
| hard-cap + slack6000 | v45b best | `6000` | on | `294.46` | `14.60%` | `28.84%` | `18.98%` | `0.00` | v45b scorer 不如 v43 pairwise/calib scorer 适配该 decoder |

判断：

- `hard_cap=1` 是必要条件；soft-cap 会把 Type-0 demand-cap drop 带回来。
- slack 从 `3000` 增到 `6000` 能改善填充和 goodput；继续到 `9000` 会把更多 PRG 分出去，但成功率下降，goodput 回落。
- 当前最有希望的组合是 `v43 last + budgeted greedy + hard_cap=1 + decode slack 6000`，而不是 v45b。

### 12.3 3-seed 顺序快验

同一配置 `v43 last + hard-cap + slack6000`，seed41/42/43，`parallel-seeds=1` 顺序运行。并行三 seed 曾出现 seed42 数字偏移，因此当前以顺序结果为准。

| seed | warmup goodput | TB err | fill | reuse | final blank | demand-cap drop | no-headroom | active UE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | `291.71` | `14.81%` | `25.70%` | `16.74%` | `37.89` | `0.00` | `35.47` | `8.15` |
| 42 | `294.46` | `14.73%` | `28.25%` | `17.50%` | `36.59` | `0.00` | `34.64` | `8.57` |
| 43 | `292.16` | `15.65%` | `27.37%` | `17.71%` | `37.04` | `0.00` | `34.45` | `8.44` |
| mean | `292.78` | `15.06%` | `27.11%` | `17.32%` | `37.18` | `0.00` | `34.85` | `8.39` |

注意：

- 这组 3-seed 结果是新 decoder 的绝对稳定性验证，不是严格 A/B；历史 control 目前只有 seed42 quick eval 可直接对齐。
- 顺序三 seed 均将 demand-cap drop 压到 `0`，说明机制确实解决了 deterministic argmax 下“同 UE 被重复过量分配”的主要 decode mismatch。
- 与 seed42 单独 run 的 `303.27 Mbps` 相比，三 seed均值约 `292.78 Mbps`，说明短评估仍有一定随机/拓扑波动；后续应使用相同顺序口径补 v43/v45b/PFQ control。

### 12.4 下一步建议

短期推荐默认验证配置：

```text
--decode-mode argmax
--candidate-budgeted-select-mode greedy
--candidate-budgeted-hard-cap 1
--candidate-budgeted-usage-penalty 0.35
--candidate-budgeted-overcap-penalty 8.0
--candidate-decode-demand-slack-bytes 6000
```

后续最值得做的训练方向：

1. 用上述 budgeted decoder 作为 rollout/deployment 口径继续从 v43 last 训练，而不是从 v45b best 继续。
2. 将 direct target 改为对 budgeted-selected candidate 学习：target 由 `kept_after_budgeted_decode * PHY success * delivered_bytes` 定义。
3. 给 scorer 加“剩余 headroom / 已服务 PRG”形式的 sequential feature 或 recurrent/top-k decode head；否则 per-PRG 独立 logits 只能在部署时被外部 decoder 修正。
4. 补同口径顺序 3-seed control：v43 control、v45b control、PFQ/RR baseline，再决定是否进入长训。

## 13. 2026-05-26 追加训练：v46 budgeted rollout from v43 last

本轮按要求从 v43 `ppo_actor_last.pt` 继续训练，active power 仍关闭。目标是把上一节最有效的 deployment 机制放进 rollout：

```text
init_checkpoint=training/stageb_hgraph/runs/online_ppo_seed42_v43_pairwise_calib_probe_from_v42b/ppo_actor_last.pt
run=training/stageb_hgraph/runs/online_ppo_seed42_v46_budgeted_sample_from_v43last
iterations=12
rollout_steps=400
candidate_budgeted_select_mode=sample
candidate_budgeted_hard_cap=1
candidate_decode_demand_slack_bytes=6000
candidate_decode_goodput_coef=0.02
candidate_success_aux/calib/max_calib=0.03/0.03/0.05
candidate_pairwise_rank_coef=0.03
lr=5e-5
active power=off
```

训练时用 `sample` 而不是 `greedy`，是为了保留 PPO 探索；deployment eval 仍使用 `greedy`。需要注意：当前 PPO logp 仍按原始 per-PRG categorical 估计，budgeted selector 是后处理约束层，因此这轮是“机制 probe”，不是严格的 logp-correct sequential policy gradient。

### 13.1 训练内 rollout

| iter | goodput | TB err | fill | demand-cap drop | no-headroom | success MAE | 备注 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `1385.98` | `18.14%` | `78.15%` | `0.00` | `11.06` | `0.072` | 新 rollout 分布冷启动，BLER 高 |
| 2 best | `1583.14` | `9.91%` | `83.48%` | `0.00` | `8.43` | `0.068` | 训练内最好 |
| 7 | `1577.22` | `9.44%` | `83.17%` | `0.00` | `8.59` | `0.053` | 另一个高点 |
| 10 | `1555.93` | `9.66%` | `81.91%` | `0.00` | `9.18` | `0.053` | 中后段回落 |
| 12 last | `1551.05` | `9.55%` | `83.24%` | `0.00` | `8.55` | `0.052` | scorer 校准继续改善，但 goodput 未同步上升 |

判断：

- budgeted rollout 的核心约束有效：`post_decode_demand_cap_count` 全程为 `0`。
- 训练内能达到 `1580 Mbps` 级别，说明机制不是不可学。
- 后续没有单调收敛，`best_iter=2`；success MAE 下降但 goodput 不跟随，说明 loss/selection/checkpoint 口径仍未完全对齐。

### 13.2 Greedy deployment eval

口径与 12.2 相同：seed42、`1000 TTI`、`200 warmup`、argmax/greedy budgeted decoder、hard-cap、slack6000。

| checkpoint | warmup goodput | TB err | fill | reuse | demand-cap drop | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v43 last + budgeted hard/slack6000 | `303.27` | `14.24%` | `29.07%` | `18.28%` | `0.00` | 当前 deployment baseline |
| v46 best iter2 | `297.31` | `14.37%` | `28.88%` | `18.42%` | `0.00` | 未超过 v43 |
| v46 last iter12 | `293.07` | `14.13%` | `28.47%` | `16.69%` | `0.00` | 更保守，goodput 更低 |

结论：

- 本轮训练未收敛到比 v43 last + budgeted decoder 更满意的 deployment 状态。
- 训练内 rollout goodput 与 greedy deployment goodput 仍有明显错位；直接拉长训练可能有帮助，但不是最高优先级。
- 下一步应先修机制：让 PPO 的 logp/entropy 按 budgeted sequential distribution 计算，或加入 budgeted-greedy shadow/deployment selection loss，再做长训。

### 13.3 关于 warmup 和短训是否科学

`1000 TTI + 200 warmup` 的用途是 quick A/B 和机制筛选，不应作为最终“压过 PFQ”的证据。它的优点是能快速暴露明显问题，例如：

- deterministic argmax 是否被 demand-cap 大量裁掉；
- hard-cap 是否真的把同 UE 过量服务压到 0；
- slack 过宽是否把 BLER 推高；
- checkpoint 之间是否存在明显 deployment 差异。

但它的局限也明确：

- 200 warmup 只能去掉最早期队列/EMA 冷启动，无法覆盖长时间 backlog、packet delay、重传状态的尾部行为。
- 单 seed seed42 容易受 topology/channel/在线 bridge 状态波动影响。
- PPO 12 iter / 400 steps 只能证明“短训趋势”，不能证明“充分训练后的上限”。

更科学的最终口径应是：

```text
training: 至少 50-100 iter 或按 eval plateau/early-stop 选择 checkpoint
eval: 顺序 3-5 seed，episode_horizon >= 4000，warmup >= 1000 或统计后 75%
baseline: 同 seed、同 power、同 traffic 的 PFQ/RR/HGraph paired compare
selection: 用 deployment eval score 选 checkpoint，而不是 rollout reward 单指标
```

因此，本轮结论不是“长训一定没用”，而是：

```text
当前 sample-budgeted PPO 口径可以改善 rollout 约束，但不能稳定转化为更优 greedy deployment。
在投入长训前，优先补 logp-correct sequential budgeted policy 或 budgeted-greedy deployment loss。
```

## 14. 2026-05-26 实现：logp-correct sequential budgeted PPO

为后续长训，已将 budgeted sequential selector 的 PPO 口径补齐。此前 v46 的问题是：

```text
rollout action = sequential budgeted selector 后处理后的 PRG action
old/new logp = 原始 per-PRG independent categorical 的 logp
```

这会让 PPO ratio、entropy、KL 都不对应真实执行策略。当前已改为：

```text
rollout old_logp = sequential budgeted distribution 下被采样 action 的 logp
update new_logp = 当前网络下，对同一 action 重放 sequential budgeted distribution 得到的 logp
entropy = sequential budgeted distribution 的候选熵，而不是原始独立 PRG 熵
```

### 14.1 代码实现

核心实现位于 `training/stageb_hgraph/online_ppo.py`：

- `_budgeted_candidate_logp_entropy`
  - 复刻 `select_budgeted_candidate_actions` 的 PRG 访问顺序。
  - 每一步维护 `ue_prg_used`。
  - 用 `candidate_decode_demand_slack_bytes` 估计 UE headroom。
  - `hard_cap=1` 时，只在 headroom 未耗尽的 candidate 上计算 categorical logp；无 candidate 时 blank 为强制动作，logp=0、entropy=0。
  - `hard_cap=0` 时，对超 cap candidate 加 overcap penalty。
- `_budgeted_candidate_logp_entropy_batch`
  - PPO update minibatch 中逐 graph 重算 sequential logp/entropy。
- `_sample_joint_action`
  - 当 `candidate_budgeted_select_mode != off` 时，`old_logp` 改用 sequential budgeted logp。
- `_action_logp_entropy_batch`
  - 当 `candidate_budgeted_select_mode != off` 时，`new_logp` 改用 sequential budgeted logp。
- `_ppo_update`
  - 新增 budgeted 参数透传，保证 update 与 rollout 使用同一 selector 口径。

默认 `candidate_budgeted_select_mode=off`，旧训练路径不变。

### 14.2 Smoke 验证

已跑 1 iter / 80 TTI smoke：

```text
run=training/stageb_hgraph/runs/online_ppo_seed42_v47_budgeted_logp_smoke
init=v43 last
candidate_budgeted_select_mode=sample
candidate_budgeted_hard_cap=1
candidate_decode_demand_slack_bytes=6000
```

结果：

| metric | value |
| --- | ---: |
| rollout goodput | `986.27 Mbps` |
| TB err | `42.70%` |
| fill | `64.73%` |
| demand-cap drop | `0.00` |
| budgeted no-headroom | `17.4` |
| entropy | `0.529` |
| approx KL | `0.131` |

这个 smoke 是冷启动短窗口，只用于确认新 logp/entropy 路径能完整跑通。`entropy` 低于旧独立 PRG categorical 是预期现象，因为每个 PRG 的可采样集合被 headroom state 收窄。`approx KL` 在 80 TTI 小样本下偏高，长训时建议继续保守设置 LR/KL。

### 14.3 后续长训建议

建议下一轮长训从 v43 last 开始，而不是从 v46：

```text
init_checkpoint=training/stageb_hgraph/runs/online_ppo_seed42_v43_pairwise_calib_probe_from_v42b/ppo_actor_last.pt
candidate_budgeted_select_mode=sample
candidate_budgeted_hard_cap=1
candidate_decode_demand_slack_bytes=6000
lr=2e-5 或 5e-5
clip_eps=0.1
target_kl=0.02-0.03
iterations >= 50
rollout_steps >= 400，稳定后扩到 1024
deployment eval 使用 candidate_budgeted_select_mode=greedy
```

更新：v48/v49 长训后，`2e-5` 对 logp-correct sequential policy 仍偏大，实际稳定配置应以 `lr=5e-6`、`clip_eps=0.05`、`target_kl=0.01`、`update_epochs=1` 作为起点。

长训判断标准不应只看 rollout best，而应每隔若干 iter 做 deployment eval：

```text
seed42 quick eval: 1000 TTI / 200 warmup
候选 checkpoint: best, threshold_best, last
通过后再做顺序 3-5 seed、4000+ TTI 长评估
```

## 15. 2026-05-26 长训验证：v48/v49 logp-correct budgeted PPO

本轮目标是回答：修正 budgeted sequential selector 的 PPO logp/entropy 后，长训是否能把训练收益转化为更好的 greedy deployment。active power 仍关闭，初始化均从 v43 last：

```text
init_checkpoint=training/stageb_hgraph/runs/online_ppo_seed42_v43_pairwise_calib_probe_from_v42b/ppo_actor_last.pt
deployment eval=candidate_budgeted_select_mode=greedy, hard_cap=1, slack6000
```

### 15.1 v48：原学习率过大

v48 使用 `lr=2e-5`、`clip_eps=0.10`、`target_kl=0.02`，3 iter 后主动停止：

| iter | rollout goodput | TB err | fill | demand-cap drop | approx KL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `1399.39 Mbps` | `17.94%` | `77.19%` | `0.00` | `0.1044` |
| 2 | `1565.27 Mbps` | `10.01%` | `82.33%` | `0.00` | `0.0966` |
| 3 | `1569.61 Mbps` | `9.56%` | `82.46%` | `0.00` | `0.1193` |

结论：logp-correct 后，KL 对学习率更敏感。v48 的 rollout goodput 不差，但 policy update 太大，不适合作为长训主线。

### 15.2 v49：低学习率 50 iter 长训

v49 改为更保守的 PPO 步长：

```text
run=training/stageb_hgraph/runs/online_ppo_seed42_v49_budgeted_logp_long_lr5e6_from_v43last
lr=5e-6
clip_eps=0.05
target_kl=0.01
update_epochs=1
minibatch_size=128
rollout_steps=400
iterations=50
candidate_budgeted_select_mode=sample
candidate_budgeted_hard_cap=1
candidate_decode_demand_slack_bytes=6000
```

训练结果：

| checkpoint / window | rollout goodput | TB err | fill | demand-cap drop | approx KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| best, iter5 | `1587.98 Mbps` | `10.40%` | `85.30%` | `0.00` | `0.0123` |
| last, iter50 | `1549.37 Mbps` | `10.06%` | `84.18%` | `0.00` | `0.0028` |
| all-iter mean | `1554.05 Mbps` | - | - | `0.00 sum` | `0.0235 max` |
| tail10 mean | `1552.37 Mbps` | - | - | `0.00 sum` | - |

结论：

- `lr=5e-6` 解决了 v48 的 KL 失控问题，50 iter 内 `demand-cap drop` 全程为 `0`。
- 训练没有随时间继续爬升；best 出现在 iter5，后半段围绕 `1540-1570 Mbps` 平台波动。
- 这说明“训练时长太短导致完全看不出效果”不是本轮主要问题。短训确实不足以证明最终效果，但 50 iter 已显示当前 reward/scorer 仍未把 rollout 优势转化成更强 greedy deployment。

### 15.3 Deployment eval

口径与第 12/13 节一致：seed42、`1000 TTI`、`200 warmup`、3cell/36UE/RBG16/SVD/TTL200、budgeted greedy hard-cap、slack6000、`ue_prg_topk=6`、`ue_prg_diversity_extra=2`。

| model | warmup goodput | TB err | fill | blank | demand-cap drop | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v43 last + budgeted hard/slack6000 | `303.27 Mbps` | `14.24%` | `29.07%` | `70.93%` | `0.00` | 当前 seed42 deployment baseline |
| v46 best | `297.31 Mbps` | `14.37%` | `28.88%` | `71.13%` | `0.00` | 旧 sample-budgeted，logp 不严格 |
| v46 last | `293.07 Mbps` | `14.13%` | `28.47%` | `71.53%` | `0.00` | 后段更保守 |
| v49 best | `295.89 Mbps` | `14.55%` | `28.60%` | `71.40%` | `0.00` | logp-correct，但未超过 v43/v46 best |
| v49 last | `292.69 Mbps` | `14.20%` | `28.00%` | `72.00%` | `0.00` | 后半段漂移更明显 |

### 15.4 当前判断

本轮最重要的正结果是机制层面的：budgeted sequential PPO 的 logp/entropy 现在是正确口径，且低学习率长训稳定。负结果也明确：仅把 rollout/deployment 对齐为 budgeted selector，不足以学到超过 v43 scorer + budgeted greedy 的部署策略。

下一步更 promising 的方向不是继续无脑加长 v49，而是把优化目标进一步贴近 deployment：

1. 对 greedy budgeted-selected candidate 加 shadow/deployment loss，让 scorer 直接学习“部署时会被选中且 decode 成功”的排序。
2. 用 deployment eval 或固定 rollout shadow score 做 checkpoint selection，避免 early rollout best 与 greedy deployment 脱节。
3. 若继续 PPO，降低 teacher_aux 衰减后的漂移：保留较小的 pairwise/calib anchor，或对 v43 scorer 做 KL/behavior regularization。
4. 4000+ TTI、顺序 3-5 seed 只应在 seed42 quick eval 超过 v43 baseline 后再跑；当前 v49 不满足这个门槛。
