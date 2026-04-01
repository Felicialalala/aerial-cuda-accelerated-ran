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
- C++ 推理落地：
  - [`cuMAC/examples/customScheduler/GnnRlPolicyRuntime.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/customScheduler/GnnRlPolicyRuntime.cpp)
  - [`cuMAC/examples/customScheduler/CustomUePrgScheduler.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/customScheduler/CustomUePrgScheduler.cpp)

## 3. 当前训练约束

当前训练/桥接/部署要求满足以下约束：

- `alloc_type = 0`
- `n_tot_cell == n_cell`
- `n_active_ue % n_cell == 0`
- `n_sched_ue % n_cell == 0`
- `n_prg = 68`

这意味着当前实现并不只锁死在 `7-cell`。
只要编译期场景满足上面的维度约束，当前主线同样支持：

- 默认 `7cell`：`n_cell=7`、`n_active_ue=56`、`n_sched_ue=56`、`n_edges=42`、`action_alloc_len=476`
- 当前 `3cell`：`n_cell=3`、`n_active_ue=24`、`n_sched_ue=24`、`n_edges=6`、`action_alloc_len=204`

补充说明：

- offline dataset 只支持 `alloc_type=0`
- online PPO 只支持 `alloc_type=0`
- `gnnrl_model` C++ runtime 目前也只支持 `allocType=0`
- online bridge 当前要求 `n_tot_cell == n_cell`
- online bridge 当前不能和 `exec-mode=gpu` 组合

## 4. 当前状态、动作、奖励与 KPI 口径

### 4.1 Actor 真正接收的观测

当前策略网络输入只有 4 个张量：

- `obs_cell_features: float32[n_cell, 5]`
- `obs_ue_features: float32[n_active_ue, 8]`
- `obs_edge_index: int16[n_edges, 2]`
- `obs_edge_attr: float32[n_edges, 2]`

语义如下。

`obs_cell_features[..., 5]`

1. `cell_load_bytes`
2. `active_ue_count`
3. `mean_wb_sinr_lin`
4. `mean_avg_rate_mbps`
5. `tb_err_rate`

`obs_ue_features[..., 8]`

1. `buffer_bytes`
2. `avg_rate_mbps`
3. `wb_sinr_lin`
4. `cqi`
5. `ri`
6. `tbErrLastActUe`
7. `newDataActUe`
8. `staleSlots`

`obs_edge_attr[..., 2]`

1. `src_load_ratio`
2. `load_diff_ratio`

### 4.2 训练和在线交互额外使用的字段

这些字段不是 actor 输入层，但当前 offline/online 训练都依赖它们：

- `action_mask_ue: bool[n_active_ue]`
- `action_mask_cell_ue: bool[n_cell, n_active_ue]`
- `action_mask_prg_cell: bool[n_cell, n_prg]`
- `reward_scalar`
- `reward_terms`
- `done`
- `next_cell_features`
- `next_ue_features`
- `next_edge_attr`

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

### 4.4 奖励

当前统一奖励是：

- `reward_terms = [throughput_mbps, total_buffer_mb, tb_err_rate, fairness_jain]`
- `reward_scalar = throughput_mbps - 0.05 * total_buffer_mb - 2.0 * tb_err_rate + 0.5 * fairness_jain`

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

### 4.6 `ue_kpi.csv` / `kpi_summary.json` 当前最值得看的字段

当前做策略比较时，建议优先看：

- `global_kpi.cluster_sum_throughput_mbps`
- `global_kpi.ue_throughput_jain`
- `global_kpi.ue_throughput_p5_mbps`
- `global_kpi.residual_buffer_ratio`
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
  uint16_t version;     // 1
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
- `rewardTerms[4]`
- `EnvDims`
- `obs_cell_features`
- `obs_ue_features`
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
  --build-method cmake \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --topology-scenario 3cell \
  --tti 2000 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 0.8 \
  --topology-seed 42 \
  --baseline-scheduler pfq \
  --action-mode prg_only_type0 \
  --init-policy-checkpoint training/gnnrl/checkpoints/m1_bc_seed42/checkpoint_best.pt \
  --online-persistent 1 \
  --rollout-steps 256 \
  --iters 40 \
  --ppo-epochs 6 \
  --minibatch-size 128 \
  --actor-lr 1e-4 \
  --critic-lr 3e-4 \
  --target-kl 0.05 \
  --plot-after-train 1 \
  --out-dir training/gnnrl/checkpoints/m3_online_seed42
```

这里要注意：

- online bridge 训练必须用 `exec-mode=both`
- 如果你当前对齐的是 `3cell + pfq + seed=42 + traffic_arrival_rate=0.8` 基线，就把这些参数原样带进 launcher
- 若直接调用 `ppo_online_train.py`，需要自行保证 compile-time 参数与运行期环境变量完全一致

## 8. 当前限制与不应误解的点

1. 当前 online PPO 虽然输出 UE 选择和 PRG bitmap 两部分动作，但 native Type-0 baseline 自身是“全部 active UE 固定进入 slot，再比较 PRG bitmap 分配”的语义。
2. 现在已经有 `prg_only_type0` 模式可用于严格同口径对比。
   该模式会固定 Type-0 的 all-active-UE slot 布局，只学习 PRG bitmap / slot assignment。
3. `exec-mode=gpu` 现在只适合 baseline 或导出模型后的推理评估，不适合 online bridge 训练。
4. 当前最佳 checkpoint 不能只看 `objective`，还需要回看：
   - `throughput`
   - `residual_buffer_ratio`
   - `packet_delay`
   - `scheduled_ratio`

## 9. 当前推荐门禁

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
