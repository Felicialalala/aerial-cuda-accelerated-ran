# Stage-B GNNRL 当前实现参考（offline + online + deployment）

## 1. 文档定位

- 本文档描述当前仓库里已经落地、并且仍与代码一致的 GNN+RL 实现。
- 它已经吸收此前拆分的 offline、replay、RL I/O、online protocol 和 bridge 设计说明。
- 当前基线场景请先看 [`Doc/current_stageB_effective_configuration.md`](/home/oai2/aerial-cuda-accelerated-ran/Doc/current_stageB_effective_configuration.md)。

## 2. 当前代码地图

### 2.1 仿真与数据导出

- Stage-B 主入口：
  - [`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/multiCellSchedulerUeSelection/main.cpp)
- replay 导出：
  - [`cuMAC/examples/rlReplay/ReplayWriter.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/rlReplay/ReplayWriter.cpp)
- online bridge：
  - [`cuMAC/examples/onlineTrainBridge/OnlineProtocol.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/onlineTrainBridge/OnlineProtocol.h)
  - [`cuMAC/examples/onlineTrainBridge/OnlineBridgeServer.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/onlineTrainBridge/OnlineBridgeServer.cpp)
  - [`cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp)

### 2.2 训练与部署

- dataset：
  - [`training/gnnrl/dataset.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/dataset.py)
- BC：
  - [`training/gnnrl/bc_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/bc_train.py)
- offline PPO：
  - [`training/gnnrl/ppo_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/ppo_train.py)
- online PPO：
  - [`training/gnnrl/ppo_online_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/ppo_online_train.py)
- online client：
  - [`training/gnnrl/aerial_env.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/aerial_env.py)
  - [`training/gnnrl/online_protocol.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/online_protocol.py)
- 模型：
  - [`training/gnnrl/model.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/model.py)
- 导出 ONNX：
  - [`training/gnnrl/export_onnx.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/export_onnx.py)
- 候选 checkpoint 主实验评测：
  - [`training/gnnrl/eval_candidate_checkpoints.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/eval_candidate_checkpoints.py)
- C++ 推理落地：
  - [`cuMAC/examples/customScheduler/GnnRlPolicyRuntime.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/customScheduler/GnnRlPolicyRuntime.cpp)
  - [`cuMAC/examples/customScheduler/CustomUePrgScheduler.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/customScheduler/CustomUePrgScheduler.cpp)

## 3. 当前训练约束

当前训练/桥接/部署要求满足以下约束：

- `alloc_type = 0`
- `n_tot_cell == n_cell`

这意味着当前实现并不只锁死在 `7-cell` 或固定 `68 PRG`。
当前 `joint` 主线会从 online bridge 的 `EnvDims` 动态读取：

- `n_cell`
- `n_active_ue`
- `n_sched_ue`
- `n_prg`
- `cell/ue/edge/prg feature dims`

例如最近的固定部署主线配置 `3cell + total_ue_count=36 + prbs_per_group=16` 对应：

- `n_cell = 3`
- `n_active_ue = 36`
- `n_sched_ue = 36`
- `n_edges = 6`
- `n_prg = 17`
- `action_alloc_len = n_cell * n_prg = 51`

补充说明：

- offline dataset 只支持 `alloc_type=0`
- online PPO 只支持 `alloc_type=0`
- `gnnrl_model` C++ runtime 目前也只支持 `allocType=0`
- online bridge 当前要求 `n_tot_cell == n_cell`
- online bridge 当前不能和 `exec-mode=gpu` 组合
- 虽然当前 `joint` 训练/部署不再硬编码 `n_active_ue % n_cell == 0` 或 `n_prg == 68`，但大多数现有实验脚本仍然使用“每小区 UE 数相同、PRG 数由 `prbs_per_group` 固定推导”的规则化场景

## 4. 当前状态、动作、奖励与 KPI 口径

### 4.1 Actor 真正接收的观测

当前 `joint` 策略网络前向一共接收 7 个输入张量：

- `obs_cell_features: float32[n_cell, 5]`
- `obs_ue_features: float32[n_active_ue, 12]`
- `obs_prg_features: float32[n_cell, n_prg, 8]`
- `obs_edge_index: int16[n_edges, 2]`
- `obs_edge_attr: float32[n_edges, 2]`
- `action_mask_ue: bool[n_active_ue]`
- `action_mask_cell_ue: bool[n_cell, n_active_ue]`

其中：

- `obs_*_features` 是数值特征
- `obs_edge_index` 是固定的小区图连边拓扑，不是可学习的数值特征
- `action_mask_cell_ue` 是“cell-UE 合法关联”结构输入，用于：
  - 在 UE 塔里把小区上下文聚合到 UE
  - 在 UE head 上屏蔽非法 `cell -> UE` 类别

语义如下。

`obs_cell_features[..., 5]`

1. `cell_load_bytes`
2. `active_ue_count`
3. `mean_wb_sinr_lin`
4. `mean_avg_rate_mbps`
5. `tb_err_rate`

`obs_ue_features[..., 12]`

1. `buffer_bytes`
2. `avg_rate_mbps`
3. `wb_sinr_lin`
4. `cqi`
5. `ri`
6. `tbErrLastActUe`
7. `newDataActUe`
8. `staleSlots`
9. `hol_delay_ms`
10. `ttl_slack_ms`
11. `recent_scheduled_ratio`
12. `recent_goodput_deficit_norm`

其中后 4 维来自 [`OnlineObservationExtrasBuilder.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h)：

- `hol_delay_ms`
  - 当前 HOL packet 的年龄，单位毫秒
  - 来自 traffic queue tracker 的 `hol_age_tti * slotDurationMs`
- `ttl_slack_ms`
  - 当前 HOL packet 离 TTL 过期还剩多少毫秒
  - 当 TTL 关闭或不可用时记为 `-1`
- `recent_scheduled_ratio`
  - 该 UE 近期是否被调度的 EMA
  - 当前实现 `alpha = 0.05`
- `recent_goodput_deficit_norm`
  - 相对本小区近期平均 goodput 的欠账程度
  - 口径是 `max(0, cell_mean_recent_goodput - ue_recent_goodput) / cell_mean_recent_goodput`

`obs_prg_features[..., 8]`

1. `top1SinrDb`
2. `top2GapDb`
3. `prev_prg_assigned`
4. `prev_prg_reuse_ratio`
5. `neighborMaxTop1SinrDb`
6. `neighborMeanTop1SinrDb`
7. `samePrgConflictRatio`
8. `iciProxy`

其中：

- `top1_sinr_db`
  - 该 `cell × PRG` 上，本小区所有关联 UE 中最好的 subband SINR
- `top2_gap_db`
  - 第 1 名和第 2 名 subband SINR 的差值
- `prev_prg_assigned`
  - 上一个 TTI 该 `cell × PRG` 是否实际被分配
- `prev_prg_reuse_ratio`
  - 上一个 TTI 同一 `PRG` 在多少个小区同时被使用，按 `reuse_count / n_cell` 归一化
- `neighborMaxTop1SinrDb`
  - 其他小区在同一 `PRG` 上最强候选 UE 的最大 subband SINR
- `neighborMeanTop1SinrDb`
  - 其他小区在同一 `PRG` 上最强候选 UE 的平均 subband SINR
- `samePrgConflictRatio`
  - 与本小区 top-1 质量接近的邻小区比例，用于表达“同一 PRG 当前 TTI 的跨小区竞争强度”
- `iciProxy`
  - 一个轻量干扰 proxy：将“本 PRG 相对 winner UE 宽带 SINR 的退化”与“same-PRG 冲突比例”合成到 `[0,1]`

`obs_edge_attr[..., 2]`

1. `src_load_ratio`
2. `load_diff_ratio`

`obs_edge_index[..., 2]`

- 当前是小区之间的完全有向图，不含 self-loop
- 边数 `n_edges = n_cell * (n_cell - 1)`

### 4.2 训练和在线交互额外使用的字段

这些字段不是前向里的“数值特征”，但当前训练/在线交互仍然依赖它们：

- `action_mask_ue: bool[n_active_ue]`
- `action_mask_prg_cell: bool[n_cell, n_prg]`
- `reward_scalar`
- `reward_terms[12]`
- `done`

另外，离线 replay 仍会持久化：

- `next_cell_features`
- `next_ue_features`
- `next_edge_attr`

但当前 replay v2 还没有持久化 `obs_prg_features / next_prg_features`，因此它和当前 online bridge 主线观测并不是完全等价的。

### 4.3 动作语义

当前动作分两部分：

- `action_ue_select: int32[n_sched_ue]`
  - 每个调度 slot 选哪个 active UE
  - `-1` 表示该 slot 空置
- `action_prg_alloc: int16[action_alloc_len]`
  - Type-0 下展平顺序是 `[prgIdx * nCell + cIdx]`
  - 值不是 UE id，而是“该 PRG 分给哪个调度 slot”
  - `-1` 表示该 `cell×PRG` 不发射

这点非常重要：

- baseline 的最终比较对象是 `allocSol`
- 不是 logits 本身
- 也不是 Python 侧原始分类标签本身

### 4.4 奖励与在线诊断指标

当前 online bridge 协议已经升级到 `version=5`，`reward_terms` 扩展为 12 个标量：

1. `served_throughput_mbps`
2. `total_pending_buffer_mb`
3. `tb_err_rate`
4. `fairness_jain`
5. `goodput_mbps`
6. `sched_wb_sinr_db`
7. `prg_utilization_ratio`
8. `goodput_spectral_efficiency_bpshz`
9. `prg_reuse_ratio`
10. `expired_bytes`
11. `expired_packets`
12. `expiry_drop_rate`

口径说明：

- `served_throughput_mbps` 来自当前 TTI 的 `servedBytesThisTti`
- `goodput_mbps` 只统计当前 TTI 成功传输的 bytes，即 `tbErr == 0` 的有效交付
- `total_pending_buffer_mb = (mac_buffer_bytes + flow_queued_bytes) / 1e6`
- `fairness_jain` 基于当前 TTI 的 per-UE `goodputBytesThisTti` 计算
- `sched_wb_sinr_db` 是“本 TTI 实际被调度 UE”的平均 wideband SINR
- `prg_utilization_ratio` 是 `allocated PRG / total PRG capacity`
- `goodput_spectral_efficiency_bpshz` 是 `goodput bitrate / (n_prg * W)`
- `prg_reuse_ratio` 描述同一个 PRG 被多 cell 同时使用的强度，可作为同频复用/同频干扰代理量
- `expired_bytes / expired_packets / expiry_drop_rate` 描述 TTL 过期导致的 freshness 损失

当前 reward 支持 6 种模式：

- `goodput_only`
  - 默认模式
  - `reward_scalar = 0.05 * goodput_mbps`
- `goodput_soft_queue`
  - 在 `goodput` 目标上叠加有界 backlog 惩罚
  - `reward_scalar = 0.05 * goodput_mbps - 0.5 * log1p(total_pending_buffer_mb)`
- `goodput_reliability`
  - 保持 `goodput` 为主目标，只做轻量可靠性 shaping
  - `reward_scalar = 0.05 * goodput_mbps - 2.0 * tb_err_rate - 4.0 * expiry_drop_rate + 0.25 * fairness_jain`
- `goodput_reliability_reuseaware`
  - 在 `goodput_reliability` 基础上增加 `tbErrRate * prgReuseRatio` 惩罚
  - 当前实现是：
    `reward_scalar = 0.05 * goodput_mbps - 2.0 * tb_err_rate - 4.0 * expiry_drop_rate - 8.0 * (tb_err_rate * prg_reuse_ratio) + 0.25 * fairness_jain`
- `goodput_reliability_blankaware`
  - 在 `goodput_reliability` 基础上增加“活跃 PRG goodput 效率”奖励和“高利用/高复用下无效打包”惩罚
  - 当前实现是：
    `reward_scalar = 0.05 * goodput_mbps + 0.25 * active_prg_goodput_efficiency - 2.0 * tb_err_rate - 4.0 * expiry_drop_rate - 10.0 * harmful_packing_penalty + 0.25 * fairness_jain`
- `legacy`
  - 恢复旧版 `throughput - backlog - queue_delay - bler + fairness` 奖励
  - `reward_scalar = 0.05 * served_throughput_mbps - 0.05 * total_pending_buffer_mb - 0.002 * queue_delay_est_ms - 1.5 * tb_err_rate + 0.5 * fairness_jain`

保留 `legacy` 的原因只是做回溯对比；当前主线更常用的是：

- 不考虑 TTL / 干扰时：`goodput_only`
- TTL 场景下的基础稳健版：`goodput_reliability`
- 想探索“减少坏复用 / 留空部分 PRG”时：`goodput_reliability_blankaware`

原因：

- persistent online 模式下，backlog 会跨 TTI 持续积累
- 旧 reward 容易被 backlog 项主导，出现“吞吐没变差，但 reward 一路下坠”的假不稳定
- `goodput_only` 能把 PPO 主目标重新对齐到真实成功交付
- backlog / 干扰 / 频谱效率等量更适合作为单独 KPI 观测，而不是默认直接混进标量 reward

从最近几轮实现开始，观测补进了更直接的 TTL / 干扰相关状态：

- UE 侧新增 4 维：`HOL delay`、`TTL slack`、`recent scheduled ratio`、`recent goodput deficit`
- PRG 侧当前共 8 维，其中最近新增 4 维：`neighborMaxTop1SinrDb`、`neighborMeanTop1SinrDb`、`samePrgConflictRatio`、`iciProxy`
- `joint` 模型的 PRG head 不再只看 `cell context + PRG position`，而是还融合了由 UE head 推断得到的 soft slot-to-UE context

这样做的目的不是“堆更多维度”，而是解决此前两个根本盲点：

- 模型不知道哪些 packet 已经接近 TTL 截止
- 模型也看不到真正随 PRG 变化的子带质量 / 同频复用状态
- 模型此前也很难知道“邻小区是否同样强烈想抢这个 PRG”

### 4.5 C++ 部署后的真正执行输出

ONNX / TensorRT 推理侧输出的是两个 logits 张量，但 C++ 端还会 decode 并做保护：

- 同 cell 合法性检查
- UE 不重复占同 cell 多 slot
- `NO_UE / NO_PRG` 抑制
- 最小调度比例
- 最小 PRG 分配比例
- 单 slot 最大 PRG 占比

当前常用保护环境变量：

- `CUMAC_GNNRL_MODEL_NO_UE_BIAS`
- `CUMAC_GNNRL_MODEL_MIN_SCHED_RATIO`
- `CUMAC_GNNRL_MODEL_NO_PRG_BIAS`
- `CUMAC_GNNRL_MODEL_MIN_PRG_RATIO`
- `CUMAC_GNNRL_MODEL_MAX_PRG_SHARE_PER_UE`

当前默认值里最重要的变化是：

- `CUMAC_GNNRL_MODEL_MIN_PRG_RATIO` 默认已经从 `1.0` 改为 `0.0`
- 也就是 decode 不再强制“每个 cell 的合法 PRG 都必须分出去”
- 模型现在可以显式选择 `NO_PRG`，从而让部分 PRG 空置来换取更低的同频干扰
- `CUMAC_GNNRL_MODEL_MAX_PRG_SHARE_PER_UE` 如果不显式设置，`prg_only_type0` 下会启用 runtime auto guardrail，避免 greedy argmax decode 把一个 cell 的大部分 PRG 长期压到单个 slot 上
- Python 训练侧现在也会把 `prg_only_type0` 的空 slot 从 PRG 动作空间里屏蔽掉，避免“训练时合法、C++ 执行时被丢弃”的 dead-slot mismatch
- `prg_only_type0` 仍然保留一个结构性限制：每个 cell 在单个 step 里只暴露 `n_sched_ue / n_cell` 个 slot，若某 cell 的关联 UE 数超过这个上限，超出的 UE 在该 step 不会进入 PRG 动作空间

### 4.6 `ue_kpi.csv` / `kpi_summary.json` 当前最值得看的字段

当前做策略比较时，建议优先看：

- `global_kpi.cluster_sum_throughput_mbps`
- `global_kpi.ue_throughput_jain`
- `global_kpi.ue_throughput_p5_mbps`
- `global_kpi.residual_buffer_ratio`
- `global_kpi.served_buffer_ratio`
- `global_kpi.non_residual_buffer_ratio`
- `global_kpi.packet_delay_p95_ms`
- `global_kpi.queue_delay_p95_ms`
- `global_kpi.scheduled_ratio_p5`

关于时延，当前有两套口径：

- `queue_delay_est_ms`
  - backlog-based end-of-run 估计
  - 不是严格 packet timestamp 平均
- `packet_delay_*`
  - 来自 traffic queue tracker 的包级时延统计
  - 更接近当前业务流模型下的真实服务时延

## 5. Replay v2 数据契约

当前主线 replay 版本是 `version=2`。

每条 transition 顺序如下：

1. `tti` (`int32`)
2. `done` (`uint8`) + padding
3. `reward_scalar` (`float32`)
4. `reward_terms` (`float32[4]`)
5. `obs_cell_features`
6. `obs_ue_features`
7. `obs_edge_index`
8. `obs_edge_attr`
9. `action_ue_select`
10. `action_prg_alloc`
11. `action_mask_ue`
12. `action_mask_cell_ue`
13. `action_mask_prg_cell`
14. `next_cell_features`
15. `next_ue_features`
16. `next_edge_attr`

当前推荐只使用 replay v2。

原因：

- 旧版曾缺少 `action_mask_cell_ue`
- 现在 `dataset.py` 仍兼容旧数据，但那只是回退逻辑，不再是主线推荐
- replay v2 仍未持久化 `obs_prg_features`
  - offline loader 会把 PRG 分支视为可选输入
  - 这意味着当前 online `joint` 主线观测比 replay v2 更丰富

快速检查：

```bash
python3 cuMAC/scripts/inspect_rl_replay.py --replay-dir <replay_dir>
```

## 6. Online bridge 协议

### 6.1 传输层

- `AF_UNIX` stream socket
- little-endian
- 基本消息：`reset / step / close`

### 6.2 Header

```c
struct MsgHeader {
  uint32_t magic;       // 0x524C4E4F ("ONLR")
  uint16_t version;     // 5
  uint16_t type;        // MsgType
  uint32_t payloadBytes;
};
```

### 6.3 ResetReq

```c
struct ResetReqPayload {
  int32_t seed;
  int32_t episodeHorizon;
  uint32_t flags;
};
```

### 6.4 StepReq

```c
struct StepReqHeader {
  int32_t stepIdx;
  uint32_t ueActionLen;
  uint32_t prgActionLen;
};
```

后续紧跟：

- `int32[ueActionLen] action_ue_select`
- `int16[prgActionLen] action_prg_alloc`

### 6.5 ResetRsp / StepRsp

当前回包里固定包含：

- `tti`
- `done`
- `rewardScalar`
- `rewardTerms[12]`
- `EnvDims`
- `obs_cell_features`
- `obs_ue_features`
- `obs_prg_features`
- `obs_edge_index`
- `obs_edge_attr`
- `action_mask_ue`
- `action_mask_cell_ue`
- `action_mask_prg_cell`

对应 Python 解析：

- [`training/gnnrl/online_protocol.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/online_protocol.py)
- [`training/gnnrl/aerial_env.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/aerial_env.py)

## 7. 当前推荐工作流

## 7.1 前置条件

推荐优先使用 [`cuMAC/scripts/run_stageB_online_train.sh`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/run_stageB_online_train.sh)。

原因：

- 它会复用 Stage-B 主脚本的编译参数注入逻辑
- 现在已经支持 `--topology-scenario 7cell|3cell`
- 可以避免 online 训练和 baseline/回放采集之间的配置漂移

另外，开始任何 Python 训练前都要先确认当前环境可导入：

- `torch`
- `numpy`
- `matplotlib`（仅画图可选）

如果要直接用 `ppo_online_train.py` 拉起 binary，而不是走 launcher，需要先保证：

- `parameters.h` 已经是当前 Stage-B 固定配置
- binary 已经在这个配置下重新编译

否则 online 训练虽然能启动，但场景可能和当前基线不一致。

## 7.2 采集 replay

```bash
./cuMAC/scripts/run_stageB_main_experiment.sh \
  --build-method cmake \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --topology-scenario 3cell \
  --tti 2000 \
  --custom-ue-prg 0 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 0.8 \
  --baseline-scheduler pfq \
  --topology-seed 42 \
  --replay-dump 1 \
  --compact-output 0 \
  --tag replay_seed42_3cell_pfq
```

## 7.3 离线 BC / offline PPO / 导出

```bash
python3 training/gnnrl/bc_train.py \
  --replay-dir output/stageB_main_experiment_replay_seed42_3cell_pfq_*/RAYLEIGH/replay \
  --out-dir training/gnnrl/checkpoints/m1_bc_seed42
```

```bash
python3 training/gnnrl/ppo_train.py \
  --replay-dir output/stageB_main_experiment_replay_seed42_3cell_pfq_*/RAYLEIGH/replay \
  --init-policy-checkpoint training/gnnrl/checkpoints/m1_bc_seed42/checkpoint_best.pt \
  --out-dir training/gnnrl/checkpoints/m2_ppo_seed42
```

```bash
python3 training/gnnrl/export_onnx.py \
  --checkpoint training/gnnrl/checkpoints/m2_ppo_seed42/ppo_actor_best.pt \
  --out training/gnnrl/checkpoints/m2_ppo_seed42/model.onnx \
  --opset 18
```

## 7.4 Online bridge 冒烟

binary 端：

```bash
SOCK=/tmp/stageb_online_smoke.sock
CUMAC_ONLINE_BRIDGE=1 \
CUMAC_ONLINE_SOCKET="${SOCK}" \
CUMAC_ONLINE_PERSISTENT=1 \
CUMAC_TRAFFIC_ARRIVAL_RATE=0.2 \
CUMAC_TOPOLOGY_SEED=42 \
CUMAC_EXEC_MODE=both \
build.$(uname -m)/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection \
  -d 1 -b 0 -x 0 -f 0 -g 100 -r 3000
```

上面命令只是 binary 本体写法；实际推荐仍由 `ppo_online_train.py` 自动设置：

- `CUMAC_ONLINE_BRIDGE=1`
- `CUMAC_ONLINE_SOCKET=<socket>`
- `CUMAC_ONLINE_PERSISTENT=1`

因此更常用的还是直接跑训练脚本。

## 7.5 直接在线 PPO

```bash
./cuMAC/scripts/run_stageB_online_train.sh \
  --build-method skip \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --topology-scenario 3cell \
  --tti 4000 \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 0.8 \
  --packet-ttl-ms 200 \
  --topology-seed 42 \
  --topology-seed-mode fixed \
  --baseline-scheduler pfq \
  --reward-mode goodput_reliability_blankaware \
  --action-mode joint \
  --online-persistent 1 \
  --episode-boundary-mode trainer \
  --episode-horizon 1024 \
  --rollout-steps 1024 \
  --iters 500 \
  --ppo-epochs 6 \
  --minibatch-size 256 \
  --entropy-coef 0.003 \
  --actor-lr 2e-4 \
  --critic-lr 1e-4 \
  --target-kl 0.02 \
  --candidate-save-every-iters 20 \
  --candidate-save-start-iter 200 \
  --auto-main-eval 1 \
  --eval-build-method skip \
  --eval-decode-mode sample \
  --eval-sample-seeds 41,42,43 \
  --eval-candidate-limit 16 \
  --plot-after-train 1 \
  --out-dir training/gnnrl/checkpoints/m3_online_seed42
```

这里要注意：

- online bridge 训练必须用 `exec-mode=both`
- 如果你当前对齐的是 `3cell + pfq + seed=42 + traffic_arrival_rate=0.8` 基线，就把这些参数原样带进 launcher
- 若直接调用 `ppo_online_train.py`，需要自行保证 compile-time 参数与运行期环境变量完全一致
- 这套命令是“固定 `seed=42` 主实验目标”的主推荐配置；若目标仍是固定 `seed=42` 部署，对齐度和稳定性都会优于 `sequential` / `list_cycle`
- `--topology-seed-mode sequential` 现在仍可做顺序多 seed 在线训练；trainer 会在每个 episode 重启 simulator，并把 `topology-seed + episode_idx - 1` 写到 `CUMAC_TOPOLOGY_SEED`
- `--seed-list 42,43,44,...` 配合 `--topology-seed-mode list_cycle` 仍可做显式列表循环
- `--curve-every-episodes 500` 会每 500 个完成 episode 刷新最新曲线并额外保存一次快照
- 当 `--online-persistent 1` 且启用 `--topology-seed-mode != fixed` 或 `--curve-every-episodes` 时，trainer 会按 `--episode-horizon` 主动切 episode，不需要改 online bridge 协议
- 对 `fixed` topology seed，trainer 现在支持 soft episode rollover：到 `episode_horizon` 只切 episode 统计和 GAE bootstrap，不会因为 horizon 直接重启 simulator；只有 simulator 自己跑到 ring horizon / env done 时才会重连
- 这里的 `list_cycle` 只是“按列表循环切 seed”，和 native RR baseline 调度器不是一回事
- 最近几轮最重要的经验是：不要省略上面这组 tuned PPO 超参数。若退回脚本默认值（如 `minibatch-size=128`、`entropy-coef=0.01`、`actor-lr=1e-4`、`critic-lr=3e-4`、`target-kl=0.05`），训练通常会更随机，也更难达到前面 `v11/v15` 那类高 goodput 水平。

## 7.6 2026-04-07 近期训练进展与经验

### 已落地的代码进展

最近几轮已经把下面几项接进当前主线：

1. 观测增强
   - UE 侧增加 `HOL delay / TTL slack / recent scheduled ratio / recent goodput deficit`
   - PRG 侧从 `4` 维扩展到 `8` 维，新增
     - `neighborMaxTop1SinrDb`
     - `neighborMeanTop1SinrDb`
     - `samePrgConflictRatio`
     - `iciProxy`
2. reward 扩展
   - 增加 `goodput_reliability_reuseaware`
   - 增加 `goodput_reliability_blankaware`
3. checkpoint 保存修正
   - `ppo_actor_best.pt` 现在保存“生成该 rollout 的 pre-update actor”
   - 不再被 PPO 更新后的权重顶替
4. 部署对齐工作流
   - 增加周期性候选 checkpoint 保存
   - 增加自动导出 ONNX + 主实验评测选 best
   - 可将 deployment-best 提升为 `ppo_actor_best_eval.pt / model_best_eval.onnx`
5. 训练日志与曲线
   - simulator 原始日志默认写入文件
   - trainer 终端只保留精简 iter 摘要
   - 曲线图优先保留 `goodput / BLER / expiry / SINR / reuse / util / entropy / KL`

### 最近几轮最重要的实验结论

1. `fixed seed` 仍然是固定部署目标下最稳的主训练模式。
   - 对固定 `topology-seed=42` 的主实验，`fixed` 明显比 `sequential` / `list_cycle` 更容易稳定爬升到高 goodput。
2. `v11` 证明了“训练有效且部署可以超过基线”。
   - 在固定 seed、tuned PPO 参数下，导出 ONNX 后的实测 `goodput` 已经可以超过 RR / PFQ。
3. `reuseaware` / `blankaware` reward 说明“reward shaping 有帮助，但不够”。
   - `reuseaware` 没有真正稳定学出“少复用/留空更优”的部署策略。
   - `blankaware` 加上 `prg8` 特征后，训练里开始出现一些“较低 reuse / util 但 goodput 仍高”的窗口，但单靠 rollout `goodput` 选 best 仍可能挑中部署更差的 checkpoint。
4. rollout best 与 deployment best 之间存在真实落差。
   - `v15 blankaware_prg8` 的训练曲线很强，但导出后主实验明显掉队，说明“训练最好”不等于“部署最好”。
   - 因此现在推荐把 deployment 评测纳入选模流程，而不是只看 `ppo_actor_best.pt`。
5. tuned PPO 超参数非常关键。
   - 已验证更稳的组合是：
     - `minibatch-size=256`
     - `entropy-coef=0.003`
     - `actor-lr=2e-4`
     - `critic-lr=1e-4`
     - `target-kl=0.02`
   - 近期一次 `v16` 训练效果明显偏弱，主要原因之一就是误用了脚本默认值，而不是沿用这套 tuned 参数。

### 当前更可靠的实践建议

1. 主训练先做固定 seed + tuned PPO 参数。
2. 每隔 `20` iter 保存候选 checkpoint。
3. 训练结束后用导出的 ONNX 跑主实验选 deployment-best。
4. 若要研究泛化，再在第二阶段尝试 `list_cycle`，而不是用它替代主训练 run。

## 8. 当前限制与不应误解的点

1. 当前 online PPO 虽然输出 UE 选择和 PRG bitmap 两部分动作，但 native Type-0 baseline 自身是“全部 active UE 固定进入 slot，再比较 PRG bitmap 分配”的语义。
2. 若要严格对齐 baseline 的“每个基站关联 UE 都是候选 UE”动作空间，应优先使用 `joint` 模式。
   该模式会让每个 cell 的 scheduler slot 从本 cell 全部关联 UE 中选择，不再把候选集截断到固定前 `n_sched_ue / n_cell` 个 UE。
3. `prg_only_type0` 更适合作为 bitmap-only 对照实验。
   它会固定 Type-0 的 all-active-UE slot 布局，只学习 PRG bitmap / slot assignment，因此在某 cell 关联 UE 数超过 slot 数时会天然丢掉一部分 UE 候选。
4. `exec-mode=gpu` 现在只适合 baseline 或导出模型后的推理评估，不适合 online bridge 训练。
5. 当前训练内的 `ppo_actor_best.pt` 仍按 rollout `goodput` 选优，并依次用更低的 `expiry_drop_rate` 与 `tb_err_rate` 做 tie-break；`objective` 继续保留用于 PPO 优化诊断，但不再直接决定 `ppo_actor_best.pt`。
   - 如果你更关心部署实测，应优先看 `auto-main-eval` 产出的 `ppo_actor_best_eval.pt / model_best_eval.onnx`。
6. 当前 traffic model 已支持“超过最大驻留时间自动过期”的 packet TTL 机制，默认关闭，可通过 `--packet-ttl-tti` 或 `--packet-ttl-ms` 显式开启。
7. 开启 TTL 后，run log / `kpi_summary.json` / online PPO episode curves 会同步输出 `expired_bytes / expired_packets / expiry_drop_rate`；`served_bytes_est` 也会显式扣除 expired bytes，避免把 timeout packet 误记成 served throughput。
8. `packet_delay_*` 统计仍然只覆盖“被完整送达”的 packet；TTL 过期 packet 不进入 packet-delay 成功样本，而是进入 expiry 统计。

## 9. 关于是否需要 packet 失效时间

当前实现现状：

- [`cuMAC/examples/trafficModel/trafficFlows.hpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/trafficModel/trafficFlows.hpp) 里只在队列容量超过 `MAX_BYTES` 时丢弃新到达流量
- [`cuMAC/examples/trafficModel/trafficGenerator.hpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/trafficModel/trafficGenerator.hpp) 里会记录 packet 的 `arrival_tti`
- packet 被服务完成时统计 `packet_delay_*`
- packet 若超过配置 TTL，则会在 traffic model 中直接过期并累计到 `expired_packets / expired_bytes`

当前 TTL 设计：

- TTL 默认关闭，通过 `--packet-ttl-tti` 或 `--packet-ttl-ms` 开启
- 当两者同时设置时，以 `ttl_tti` 为准
- `expired_bytes / expired_packets / expiry_drop_rate` 会进入：
  - run log 的 `TRAFFIC_EXPIRY`
  - `kpi_summary.json` 的 `traffic` 与 `global_kpi`
  - online PPO 的 `online_ppo_metrics.csv`、`online_ppo_episode_metrics.csv` 与训练曲线
- `served_bytes_est` 口径调整为 `accepted - flow_queued - mac_buffer - expired`

推荐做法：

1. 如果当前目标只是“最大化长期成功交付”，可以先保持 TTL 关闭，只看 `goodput`
2. 如果目标包含 latency budget、业务鲜活性，建议开启 TTL，并同时观察 `expiry_drop_rate + packet_delay_p95_ms`
3. 若以后做多业务混合流量，TTL 最好按 traffic class 配置，而不是所有 flow 共用一个全局阈值

## 10. 当前推荐门禁

对于导出后的 `gnnrl_model`，至少建议检查：

- `cluster_sum_throughput_mbps >= baseline * 0.90`
- `residual_buffer_ratio <= baseline * 1.10`
- `scheduled_ratio == 0` 的 UE 数 `<= 2`
- `packet_delay_p95_ms` 不出现灾难性恶化

若要和当前 RR/PF baseline 做一致对比，推荐再额外固定：

- `packet_size_bytes`
- `traffic_arrival_rate`
- `topology_seed`
- `exec_mode`
- `fading_mode`

全部与 RR/PF compare 使用完全相同的配置。
