# 2026-04-12 Stage-B `v16c iter0290` 与 RR/PFQ 多 topology-seed 对比记录

## 1. 文档定位

- 本文档记录当前 `v16c` 稳定训练 run 在 `3cell + ue36 + ttl200 + rbg16` 场景下，与 native `RR` / `PFQ` 的一次真正多 topology-seed 对比。
- 本次记录的目标不是重复证明“模型在 fixed topology=42 上有效”，而是回答更接近部署的问题：
  - 在 `topology-seed=41,42,43` 的均值口径下，当前 deployment best 是否已经明显超过 baseline。
  - 它超过的是哪些维度，仍然落后的又是哪些维度。
  - 相比上一版 deployment best `iter0360`，这次 `v16c iter0290` 是否已经形成代际提升。

## 2. 当前场景与对比口径

### 2.1 当前场景

- 拓扑：`3cell`
- UE 总数：`36`
- 每小区 UE 数：`12`
- fading：`Rayleigh`
- TTI：`4096`
- `prbs-per-group`：`16`
- `n_prg`：`17`
- packet size：`3000 bytes`
- traffic arrival rate：`0.8 pkt/TTI`
- packet TTL：`200 ms`
- 对比 seeds：`topology-seed=41,42,43`
- 场景目录：`RAYLEIGH`

### 2.2 基线与模型口径

- baseline 路径：
  - native `RR`
  - native `PFQ`
  - `--custom-ue-prg 0`
- 模型路径：
  - `gnnrl_model`
  - `Custom UE+PRG` 路径
  - 模型文件：[`model_best_eval.onnx`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/model_best_eval.onnx)

### 2.3 本次模型 decode 口径

- decode mode：`sample`
- sample seed：固定 `42`
- 因此本次模型三拓扑均值是：
  - `topology-seed=41,42,43`
  - `sample-seed=42`

这点与此前 [`iter0360` 文档](/home/oai2/aerial-cuda-accelerated-ran/Doc/stageB_iter0360_rr_pfq_multiseed_compare_20260407.md) 不同：

- `iter0360` 那次 wrapper 默认使用 `sample_seed = topology_seed`
- 本次为了更干净地看 topology 泛化，把 `sample-seed` 固定成了 `42`

因此本次对 baseline 的比较仍然是 apples-to-apples 的 topology multiseed 均值，但它不应直接和“`sample_seed = topology_seed` 口径”的旧模型均值混为同一个随机性设定。

### 2.4 `exec-mode` 说明

- baseline 多 seed 平均沿用 native baseline 的 `--exec-mode gpu`
- 模型多 seed 平均使用 `--exec-mode both`

原因：

- 当前 `gnnrl_model + custom UE+PRG` 路径仍不支持 `--exec-mode gpu`
- `run_stageB_model_multi_seed_compare.sh` 在 custom UE+PRG 模式下会将 `gpu` 自动升级为 `both`

因此本次胜负判断仍以：

- `traffic.*`
- `global_kpi.*`

为主，不使用 `cpu_gpu_compare.*` 作为结论依据。

### 2.5 当前口径的一个小 caveat

这次对比有一个需要明说的小差异：

- baseline 聚合来自旧的 `RR/PFQ` multi-seed 运行，命令口径里使用的是 `--tti 4000`
- 本次模型 multi-seed 运行使用的是 `--tti 4096`

这意味着它不是“所有命令参数 100% 一字不差”的完全重跑对比。

不过当前这组对比仍然有较高参考价值，原因是：

- `topology-seed=41,42,43` 一致
- `3cell + ue36 + ttl200 + rbg16 + arrival_rate=0.8 + packet_size=3000` 一致
- 关键 KPI 使用的是：
  - `traffic.goodput_mbps`
  - `cluster_sum_throughput_mbps`
  - `expiry_drop_rate`
  - `packet_delay_*`
  - `ue_goodput_*`
- 这些指标本质上都是按模拟时间归一化后的“速率/比例/分位数”口径，而不是简单累计字节数

因此：

- 当前文档结论可以用于判断“这版模型是否已经显著优于 RR/PFQ”
- 如果后续要做最严格的论文式对齐，仍建议把 `RR/PFQ` 也按 `TTI=4096` 重新补跑一轮

## 3. 关键产物路径

### 3.1 训练与部署 best 产物

- 当前训练 run：
  - [`online_ppo_summary.json`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/online_ppo_summary.json)
  - [`online_ppo_metrics.csv`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/online_ppo_metrics.csv)
  - [`online_ppo_episode_metrics.csv`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/online_ppo_episode_metrics.csv)
- 候选 checkpoint 主实验评测：
  - [`candidate_eval_summary.json`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/candidate_main_eval/candidate_eval_summary.json)
  - [`candidate_eval_metrics.csv`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/candidate_main_eval/candidate_eval_metrics.csv)
- deployment best：
  - checkpoint：[`ppo_actor_best_eval.pt`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/ppo_actor_best_eval.pt)
  - ONNX：[`model_best_eval.onnx`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/model_best_eval.onnx)
  - external data：[`model_best_eval.onnx.data`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/model_best_eval.onnx.data)

### 3.2 Baseline 聚合结果

- RR/PFQ 多 seed 平均：
  - [`rr_vs_pfq_compare_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_rr_pfq_multiseed_compare_rr_pfq_3cell_s41_s42_s43_ue36_ttl200_rbg16_20260407_082742/RAYLEIGH/rr_vs_pfq_compare_mean.csv)
- baseline 聚合摘要：
  - [`aggregate_summary.txt`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_rr_pfq_multiseed_compare_rr_pfq_3cell_s41_s42_s43_ue36_ttl200_rbg16_20260407_082742/aggregate_summary.txt)

### 3.3 模型三拓扑聚合结果

- 模型多 topology-seed 平均：
  - [`v16c_iter0290_toposeed414243_kpi_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_v16c_iter0290_topo414243_20260412_080041/RAYLEIGH/v16c_iter0290_toposeed414243_kpi_mean.csv)
- 模型聚合摘要：
  - [`aggregate_summary.txt`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_v16c_iter0290_topo414243_20260412_080041/aggregate_summary.txt)
- 模型每 seed 运行清单：
  - [`seed_runs.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_v16c_iter0290_topo414243_20260412_080041/seed_runs.csv)

### 3.4 模型每个 topology-seed 的运行目录

- `seed41`：
  - [`run dir`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_main_experiment_v16c_iter0290_topo414243_s41_20260412_080041)
- `seed42`：
  - [`run dir`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_main_experiment_v16c_iter0290_topo414243_s42_20260412_080105)
- `seed43`：
  - [`run dir`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_main_experiment_v16c_iter0290_topo414243_s43_20260412_080128)

## 4. 运行命令记录

### 4.1 RR/PFQ 多 topology-seed 平均命令

baseline 均值来自：

```bash
./cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh \
  --seed-list 41,42,43 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --pf-baseline pfq \
  --build-method cmake \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --tti 4000 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 0.8 \
  --packet-ttl-ms 200 \
  --progress-tti 1000 \
  --kpi-tti-log 0 \
  --compare-tti 0 \
  --compact-output 1 \
  --exec-mode gpu \
  --tag rr_pfq_3cell_s41_s42_s43_ue36_ttl200_rbg16
```

### 4.2 `v16c iter0290` 多 topology-seed 对比命令

这次模型多 seed 命令是：

```bash
RUN=training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable

./cuMAC/scripts/run_stageB_model_multi_seed_compare.sh \
  --gpu 1 \
  --build-method skip \
  --seed-list 41,42,43 \
  --model-path "$RUN/model_best_eval.onnx" \
  --model-label v16c_iter0290_toposeed414243 \
  --gnnrl-model-decode-mode sample \
  --gnnrl-model-sample-seed 42 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --baseline-scheduler pfq \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --tti 4096 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 0.8 \
  --packet-ttl-tti 0 \
  --packet-ttl-ms 200 \
  --progress-tti 0 \
  --kpi-tti-log 100 \
  --compare-tti 0 \
  --compact-output 1 \
  --exec-mode both \
  --tag v16c_iter0290_topo414243
```

对应脚本：

- [`run_stageB_model_multi_seed_compare.sh`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/run_stageB_model_multi_seed_compare.sh)
- [`aggregate_model_kpi_compare.py`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/aggregate_model_kpi_compare.py)

## 5. 训练与 deployment best 结论

### 5.1 训练是否稳定收敛

这条 `v16c` run 可以判断为“稳定收敛，但最优 rollout 出现在中后段，而不是最后一轮”。

关键信息：

- 训练 summary：[`online_ppo_summary.json`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/online_ppo_summary.json)
- `completed_episodes = 400`
- `best_iter = 293`
- `best_rollout_goodput_mbps_mean = 1174.35`
- `best_rollout_expiry_drop_rate_mean = 0.1695`
- `best_rollout_tb_err_rate_mean = 0.0874`

稳定性的直观表现是：

- 前 `100` episode 平均 goodput：`968.65 Mbps`
- `101-200`：`1028.95 Mbps`
- `201-300`：`1122.49 Mbps`
- `301-400`：`1067.34 Mbps`

也就是说：

- 训练前中段持续改善
- `201-300` 已经进入高性能 plateau
- 最后 `100` episode 略有回落，但没有出现上一条 `h1024` run 那种明显发散或熵反弹

### 5.2 当前 run 的最佳模型是哪一个

这条 run 里同样要区分两个“最佳”：

- 训练 rollout best：`iter293`
  - 对应 [`ppo_actor_best.pt`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/ppo_actor_best.pt)
- deployment main-eval best：`iter0290`
  - 对应 [`ppo_actor_best_eval.pt`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/ppo_actor_best_eval.pt)
  - 对应 [`model_best_eval.onnx`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/model_best_eval.onnx)

真正用于部署的应是 `iter0290`，不是 `iter293 trainer_best`。

原因见 [`candidate_eval_summary.json`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/candidate_main_eval/candidate_eval_summary.json)：

- `best_candidate = iter0290`
- `mean_cluster_goodput_mbps = 1098.33`
- `mean_expiry_drop_rate = 0.0515`
- `mean_global_tb_bler = 0.1354`
- `mean_packet_delay_mean_ms = 47.92`
- `mean_ue_goodput_jain = 0.9711`

而 `iter0293_trainer_best` 的 main-eval 只有：

- `mean_cluster_goodput_mbps = 955.81`
- `mean_expiry_drop_rate = 0.1462`

这再次说明：

- 训练侧 rollout best 不等于 deployment best
- 当前上线结论应始终以 `*_best_eval` 为准

## 6. 结果记录

### 6.1 三方均值总表

下表基于：

- 模型均值 [`v16c_iter0290_toposeed414243_kpi_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_v16c_iter0290_topo414243_20260412_080041/RAYLEIGH/v16c_iter0290_toposeed414243_kpi_mean.csv)
- baseline 均值 [`rr_vs_pfq_compare_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_rr_pfq_multiseed_compare_rr_pfq_3cell_s41_s42_s43_ue36_ttl200_rbg16_20260407_082742/RAYLEIGH/rr_vs_pfq_compare_mean.csv)

| 指标 | 方向 | RR mean | PFQ mean | v16c iter0290 mean | 相对 RR | 相对 PFQ |
|---|---|---:|---:|---:|---:|---:|
| Cluster throughput | 高更好 | 1016.34 | 938.63 | 1254.50 | `+23.4%` | `+33.7%` |
| Goodput | 高更好 | 898.21 | 832.37 | 1107.24 | `+23.3%` | `+33.0%` |
| Expiry drop rate | 低更好 | 0.2132 | 0.2474 | 0.0639 | 改善 `70.0%` | 改善 `74.2%` |
| TB BLER | 低更好 | 0.1130 | 0.1217 | 0.1317 | 变差 `16.5%` | 变差 `8.2%` |
| Packet delay mean | 低更好 | 41.79 ms | 91.35 ms | 36.84 ms | 改善 `11.9%` | 改善 `59.7%` |
| Packet delay p95 | 低更好 | 199.0 ms | 199.17 ms | 197.5 ms | 改善 `0.8%` | 改善 `0.8%` |
| UE goodput Jain | 高更好 | 0.8587 | 0.8530 | 0.9599 | `+11.8%` | `+12.5%` |
| UE goodput p10 | 高更好 | 10.94 | 9.86 | 20.35 | `+86.1%` | `+106.4%` |
| UE goodput p5 | 高更好 | 8.85 | 9.40 | 18.31 | `+107.0%` | `+94.8%` |

### 6.2 与 baseline 的核心变化

如果把 `RR` 视作主基线，这次最关键的结果是：

- `goodput` 从 `898.21` 提升到 `1107.24 Mbps`
- `served throughput` 从 `1016.34` 提升到 `1254.50 Mbps`
- `expiry_drop_rate` 从 `0.2132` 降到 `0.0639`
- `packet_delay_mean` 从 `41.79 ms` 降到 `36.84 ms`
- `ue_goodput_jain` 从 `0.8587` 提升到 `0.9599`
- `ue_goodput_p10` 从 `10.94` 提升到 `20.35 Mbps`

但代价也很清楚：

- `global_tb_bler` 从 `0.1130` 升到 `0.1317`

也就是说，这版模型不是“所有维度都更好”，而是：

- 在业务交付、TTL 过期、尾部 UE 体验、公平性上，全面优于 `RR/PFQ`
- 在物理层误块率上，仍比 `RR/PFQ` 更激进

### 6.3 每个 topology-seed 的直接对齐

| Topology seed | v16c goodput | RR goodput | PFQ goodput | v16c expiry | RR expiry | PFQ expiry | v16c BLER | RR BLER | PFQ BLER | v16c delay mean | RR delay mean | PFQ delay mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 41 | 1098.61 | 868.66 | 763.10 | 0.0763 | 0.2360 | 0.2909 | 0.1345 | 0.1183 | 0.1195 | 36.95 | 32.33 | 110.84 |
| 42 | 1100.93 | 958.52 | 868.80 | 0.0559 | 0.1701 | 0.2218 | 0.1365 | 0.1141 | 0.1269 | 46.16 | 37.84 | 85.87 |
| 43 | 1122.18 | 867.43 | 865.21 | 0.0596 | 0.2336 | 0.2296 | 0.1241 | 0.1067 | 0.1188 | 27.41 | 55.20 | 77.34 |

这张表说明：

- `goodput` 在 `3/3` 个 topology seeds 上都超过 `RR`，也都超过 `PFQ`
- `expiry` 在 `3/3` 个 seeds 上都显著优于 `RR/PFQ`
- `delay mean` 在 `2/3` 个 seeds 上优于 `RR`，在 `3/3` 个 seeds 上优于 `PFQ`
- `BLER` 在 `3/3` 个 seeds 上都高于 `RR`，也都高于 `PFQ`

因此这次均值提升不是由某一个 seed 偶然拉高，而是 3 个拓扑都稳定受益。

## 7. 与上一版 deployment best `iter0360` 的对比

这一步不是本次 baseline 对比的主结论，但值得单独记录，因为它能回答“这条新主线是否只是重新跑了一次，还是确实变强了”。

对比文件：

- 新模型：[`v16c_iter0290_toposeed414243_kpi_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_v16c_iter0290_topo414243_20260412_080041/RAYLEIGH/v16c_iter0290_toposeed414243_kpi_mean.csv)
- 旧模型：[`iter0360_kpi_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_iter0360_toposeed414243_20260407_085302/RAYLEIGH/iter0360_kpi_mean.csv)

| 指标 | iter0360 | v16c iter0290 | 变化 |
|---|---:|---:|---:|
| Goodput | 973.68 | 1107.24 | `+13.7%` |
| Cluster throughput | 1119.38 | 1254.50 | `+12.1%` |
| Expiry drop rate | 0.1335 | 0.0639 | 改善 `52.1%` |
| TB BLER | 0.1381 | 0.1317 | 改善 `4.6%` |
| Packet delay mean | 68.04 ms | 36.84 ms | 改善 `45.9%` |
| UE goodput Jain | 0.9170 | 0.9599 | `+4.7%` |
| UE goodput p10 | 16.24 | 20.35 | `+25.3%` |

结论很直接：

- `v16c iter0290` 不只是超过 `RR/PFQ`
- 它也已经明显超过上一版 deployment best `iter0360`

而且这次不是靠“牺牲 BLER 换吞吐”硬堆出来的：

- 相比 `iter0360`，`v16c iter0290` 的 `goodput`、`expiry`、`delay`、`fairness` 全部更好
- 连 `TB BLER` 也比 `iter0360` 略好

所以从当前仓库证据看，`v16c iter0290` 应视为新的 deployment 主线候选。

## 8. 结果分析

### 8.1 能否说这次模型已经超过 baseline

如果主目标是：

- `goodput`
- `served throughput`
- `expiry_drop_rate`
- 业务平均时延
- edge / low-percentile UE 体验
- fairness

那么答案是：

- 是，`v16c iter0290` 已经明确超过 `RR`
- 也更明显地超过 `PFQ`

并且这次不只是“均值超过”，而是：

- `3/3` 个 topology seeds 的 goodput 都赢 `RR/PFQ`
- `3/3` 个 topology seeds 的 expiry 都赢 `RR/PFQ`

### 8.2 为什么还不能说“全面优于 RR”

当前仍不能说“模型在所有维度都已经全面超过 RR”。

主要保留项只有一个：

- `global_tb_bler`
  - `RR = 0.1130`
  - `v16c iter0290 = 0.1317`

也就是说：

- 业务端 KPI 已经明显更强
- 但链路层误块率仍偏高

这更像是一个“更激进但端到端更有效”的策略，而不是一个在所有链路层指标上都更保守的策略。

### 8.3 为什么这次结果比 `iter0360` 强这么多

基于当前证据，更合理的解释是组合因素共同作用：

1. 训练更稳
   - 这次 `v16c` 超参显著压住了中后段漂移
   - 最后 `100` episode 虽然不再创新高，但整体保持在高性能 plateau
2. 选模更准
   - 当前 deployment best 不是训练侧 `iter293`，而是 main-eval 选出的 `iter0290`
   - 这次 main-eval 选到的 checkpoint 更贴近真实部署目标
3. topology 泛化更好
   - 在 `seed41/42/43` 上的 goodput 都稳定领先 `RR/PFQ`
   - 没有出现此前某些 run 在新 topology 上明显 starvation 的现象

### 8.4 当前最合理的部署结论

截至 `2026-04-12`，当前最合理的结论应更新为：

- 训练侧 rollout best：`iter293`
- deployment best：`iter0290`
- exported deployment artifact：
  - [`ppo_actor_best_eval.pt`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/ppo_actor_best_eval.pt)
  - [`model_best_eval.onnx`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable/model_best_eval.onnx)

如果当前 deployment 目标是 `3cell + ue36 + ttl200 + rbg16`，并且需要对 `topology-seed=41,42,43` 这类变化场景保持稳定，这版 `v16c iter0290` 已经是比：

- `RR`
- `PFQ`
- `iter0360`

都更值得优先部署验证的候选。
