# 2026-04-07 Stage-B `iter0360` 与 RR/PFQ 多 topology-seed 对比记录

## 1. 文档定位

- 本文档记录 `iter0360` 在当前 `3cell + ue36 + ttl200 + rbg16` 场景下，与 `RR` / `PFQ` 的一次真正 apples-to-apples 多 seed 对比。
- 这里的关键变化不是“又跑了一次 baseline”，而是把模型也改成了和 baseline 一样的 `topology-seed=41,42,43` 均值口径。
- 这份记录用于回答两个问题：
  - `iter0360` 在与 `RR/PFQ` 相同的多 topology-seed 平均口径下，是否已经超过 baseline。
  - 它超过的是哪些维度，仍然落后的又是哪些维度。

## 2. 当前场景与对比口径

### 2.1 当前场景

- 拓扑：`3cell`
- UE 总数：`36`
- 每小区 UE 数：`12`
- fading：`Rayleigh`
- TTI：`4000`
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
  - 模型文件：[`iter0360.onnx`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16b_joint_ttl200_rbg16_ue36_blankaware_prg8_i500_eval/candidate_main_eval/onnx/iter0360.onnx)
- 模型 decode：
  - `sample`
  - 本次 wrapper 没有显式传 `--gnnrl-model-sample-seed`
  - 因此使用“`sample_seed = topology_seed`”的默认规则，即：
    - `seed41 -> sample41`
    - `seed42 -> sample42`
    - `seed43 -> sample43`

### 2.3 这次口径为什么比之前更对齐

此前 `candidate_eval_summary.json` 里的 `best_eval=iter0360`，是：

- 固定 `topology-seed=42`
- 只对 `sample-seed=41,42,43` 求均值

而本次文档记录的是：

- `RR/PFQ`：`topology-seed=41,42,43` 求均值
- `iter0360`：同样对 `topology-seed=41,42,43` 求均值

所以这次结果和 baseline 的比较是对齐的，但它和此前 `best_eval=iter0360 mean goodput=907.51 Mbps` 不是同一类均值，不能直接横向比较。

### 2.4 `exec-mode` 说明

- baseline 多 seed 平均沿用了 native baseline 的 `--exec-mode gpu`
- 模型多 seed 平均最终使用了 `--exec-mode both`

原因：

- 当前 `gnnrl_model + custom UE+PRG` 路径不支持 `--exec-mode gpu`
- 初次尝试使用 `gpu` 时运行失败
- 因此模型正式对比改成了 `both`

这也意味着本次胜负判断仍然应该以：

- `traffic.*`
- `global_kpi.*`

为主，不使用 `cpu_gpu_compare.*` 作为胜负依据。

## 3. 关键产物路径

### 3.1 Baseline 平均

- RR/PFQ 多 seed 平均：
  - [`rr_vs_pfq_compare_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_rr_pfq_multiseed_compare_rr_pfq_3cell_s41_s42_s43_ue36_ttl200_rbg16_20260407_082742/RAYLEIGH/rr_vs_pfq_compare_mean.csv)

### 3.2 模型平均

- `iter0360` 多 topology-seed 平均：
  - [`iter0360_kpi_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_iter0360_toposeed414243_20260407_085302/RAYLEIGH/iter0360_kpi_mean.csv)
- RR / PFQ / `iter0360` 三方统一对比表：
  - [`rr_vs_pfq_vs_iter0360_compare_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_iter0360_toposeed414243_20260407_085302/RAYLEIGH/rr_vs_pfq_vs_iter0360_compare_mean.csv)

### 3.3 运行与聚合清单

- 模型聚合摘要：
  - [`aggregate_summary.txt`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_iter0360_toposeed414243_20260407_085302/aggregate_summary.txt)
- 模型每 seed 运行清单：
  - [`seed_runs.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_iter0360_toposeed414243_20260407_085302/seed_runs.csv)
- 模型每场景每 seed 清单：
  - [`scenario_seed_manifest.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_iter0360_toposeed414243_20260407_085302/scenario_seed_manifest.csv)

## 4. 运行命令记录

### 4.1 单 seed baseline 命令

这是当前 baseline 对比的单 seed 入口命令：

```bash
./cuMAC/scripts/run_stageB_rr_pf_compare.sh \
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
  --topology-seed 42 \
  --progress-tti 1000 \
  --kpi-tti-log 0 \
  --compare-tti 0 \
  --compact-output 1 \
  --exec-mode gpu \
  --tag rr_pfq_3cell_seed42_ue36_ttl200_rbg16
```

### 4.2 RR/PFQ 多 topology-seed 平均命令

这次 RR/PFQ 均值来自下面的 wrapper：

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

对应脚本：

- [`run_stageB_rr_pf_multi_seed_compare.sh`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh)
- [`aggregate_rr_pf_compare.py`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/aggregate_rr_pf_compare.py)

### 4.3 `iter0360` 多 topology-seed 对比命令

最终成功跑通的 `iter0360` 多 seed 命令如下：

```bash
./cuMAC/scripts/run_stageB_model_multi_seed_compare.sh \
  --model-path training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16b_joint_ttl200_rbg16_ue36_blankaware_prg8_i500_eval/candidate_main_eval/onnx/iter0360.onnx \
  --model-label iter0360 \
  --baseline-mean-csv output/stageB_rr_pfq_multiseed_compare_rr_pfq_3cell_s41_s42_s43_ue36_ttl200_rbg16_20260407_082742/RAYLEIGH/rr_vs_pfq_compare_mean.csv \
  --seed-list 41,42,43 \
  --build-method cmake \
  --topology-scenario 3cell \
  --total-ue-count 36 \
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
  --exec-mode both \
  --tag iter0360_toposeed414243
```

对应脚本：

- [`run_stageB_model_multi_seed_compare.sh`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/run_stageB_model_multi_seed_compare.sh)
- [`aggregate_model_kpi_compare.py`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/aggregate_model_kpi_compare.py)

### 4.4 模型运行注意事项

- `gnnrl_model + custom UE+PRG` 不支持 `--exec-mode gpu`
- 因此该 wrapper 当前会将 `gpu` 自动升级为 `both`
- 该 wrapper 默认 `decode_mode=sample`
- 若未显式传 `--gnnrl-model-sample-seed`，则默认使用 `sample_seed = topology_seed`

## 5. 结果记录

### 5.1 均值结果总表

下表来自 [`rr_vs_pfq_vs_iter0360_compare_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_iter0360_toposeed414243_20260407_085302/RAYLEIGH/rr_vs_pfq_vs_iter0360_compare_mean.csv)。

| 指标 | 方向 | RR mean | PFQ mean | iter0360 mean | iter0360 相对 RR | iter0360 相对 PFQ |
|---|---|---:|---:|---:|---:|---:|
| Served throughput | 高更好 | 1016.34 | 938.63 | 1119.38 | +103.04 Mbps | +180.75 Mbps |
| Goodput | 高更好 | 898.21 | 832.37 | 973.68 | +75.47 Mbps | +141.31 Mbps |
| Expiry drop rate | 低更好 | 0.2132 | 0.2474 | 0.1335 | -0.0797 | -0.1139 |
| TB BLER | 低更好 | 0.1130 | 0.1217 | 0.1381 | +0.0250 | +0.0163 |
| Packet delay mean | 低更好 | 41.79 ms | 91.35 ms | 68.04 ms | +26.25 ms | -23.31 ms |
| Packet delay p50 | 低更好 | 0.50 ms | 72.67 ms | 19.83 ms | +19.33 ms | -52.83 ms |
| Packet delay p90 | 低更好 | 193.50 ms | 198.33 ms | 197.00 ms | +3.50 ms | -1.33 ms |
| Served buffer ratio | 高更好 | 0.7344 | 0.6790 | 0.8103 | +0.0759 | +0.1312 |
| Backlog-free UE ratio | 高更好 | 0.3056 | 0.0370 | 0.0185 | -0.2870 | -0.0185 |
| UE throughput p5 | 高更好 | 10.26 | 10.85 | 18.39 | +8.14 Mbps | +7.54 Mbps |
| UE goodput p10 | 高更好 | 10.94 | 9.86 | 16.24 | +5.30 Mbps | +6.38 Mbps |
| Scheduled ratio mean | 高更好 | 0.8573 | 0.6239 | 0.1550 | -0.7023 | -0.4689 |
| PRG utilization | 高更好 | 1.0000 | 1.0000 | 0.9996 | -0.0004 | -0.0004 |

### 5.1.1 模型均值相对 RR/PFQ 均值的关键指标对比

为方便直接判断强弱，下面把 `iter0360` 相对 `RR/PFQ` 的关键指标变化单独展开。

说明：

- 对于“高更好”指标：
  - 正值表示 `iter0360` 更好
  - 百分比按 `(iter0360 / baseline - 1)` 计算
- 对于“低更好”指标：
  - 负绝对差表示 `iter0360` 更好
  - 文中的“改善/变差”按低更好语义解释

#### 5.1.1.1 吞吐与业务完成类

| 指标 | iter0360 相对 RR | iter0360 相对 PFQ | 结论 |
|---|---:|---:|---|
| Served throughput | `+103.04 Mbps`，`+10.1%` | `+180.75 Mbps`，`+19.3%` | 对两条 baseline 都明显领先 |
| Goodput | `+75.47 Mbps`，`+8.4%` | `+141.31 Mbps`，`+17.0%` | 这是本次最核心的正向结果 |
| Expiry drop rate | `-0.0797`，改善 `37.4%` | `-0.1139`，改善 `46.0%` | TTL 过期控制明显优于 RR/PFQ |
| Served buffer ratio | `+0.0759`，`+10.3%` | `+0.1312`，`+19.3%` | 被真正服务掉的业务比例最高 |

#### 5.1.1.2 可靠性与时延类

| 指标 | iter0360 相对 RR | iter0360 相对 PFQ | 结论 |
|---|---:|---:|---|
| TB BLER | `+0.0250`，变差 `22.2%` | `+0.0163`，变差 `13.4%` | 可靠性仍明显落后于 RR，也落后于 PFQ |
| Packet delay mean | `+26.25 ms`，变差 `62.8%` | `-23.31 ms`，改善 `25.5%` | 均值时延介于 RR 和 PFQ 之间 |
| Packet delay p50 | `+19.33 ms`，约 `39.7x` | `-52.83 ms`，改善 `72.7%` | 中位时延远差于 RR，但明显优于 PFQ |
| Packet delay p90 | `+3.50 ms`，变差 `1.8%` | `-1.33 ms`，改善 `0.7%` | 高分位时延与 RR 接近，但仍略差 |
| Backlog-free UE ratio | `-0.2870`，变差 `93.9%` | `-0.0185`，变差 `50.0%` | 结束时清空 backlog 的 UE 比例最低 |

#### 5.1.1.3 Edge 体验与调度行为类

| 指标 | iter0360 相对 RR | iter0360 相对 PFQ | 结论 |
|---|---:|---:|---|
| UE throughput p5 | `+8.14 Mbps`，`+79.3%` | `+7.54 Mbps`，`+69.4%` | edge UE served throughput 大幅领先 |
| UE goodput p10 | `+5.30 Mbps`，`+48.5%` | `+6.38 Mbps`，`+64.7%` | 低分位 UE goodput 明显改善 |
| Scheduled ratio mean | `-0.7023`，变差 `81.9%` | `-0.4689`，变差 `75.2%` | 模型不是靠更频繁调度 UE 获胜 |
| PRG utilization | `-0.0004`，约 `-0.037%` | `-0.0004`，约 `-0.037%` | 仍接近打满 PRG，没有明显 blanking |

#### 5.1.1.4 一句话总结

- 如果优先看 `goodput + served throughput + expiry`，`iter0360` 相对 `RR/PFQ` 都是清晰领先。
- 如果优先看 `BLER + 低时延 + backlog 清空能力`，`RR` 仍然更强。

### 5.2 `iter0360` 自身均值与波动

下表来自 [`iter0360_kpi_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_iter0360_toposeed414243_20260407_085302/RAYLEIGH/iter0360_kpi_mean.csv)。

| 指标 | iter0360 mean | std | min | max |
|---|---:|---:|---:|---:|
| Goodput | 973.68 Mbps | 39.78 | 938.24 | 1029.23 |
| Served throughput | 1119.38 Mbps | 33.72 | 1086.89 | 1165.85 |
| Expiry drop rate | 0.1335 | 0.0227 | 0.1035 | 0.1582 |
| TB BLER | 0.1381 | 0.0027 | 0.1343 | 0.1405 |
| Packet delay mean | 68.04 ms | 5.95 | 60.20 | 74.63 |
| Packet delay p50 | 19.83 ms | 4.25 | 14.00 | 24.00 |
| UE throughput p5 | 18.39 Mbps | 1.90 | 16.60 | 21.02 |
| UE goodput p10 | 16.24 Mbps | 1.43 | 15.10 | 18.26 |

### 5.3 每个 topology-seed 的详细记录

下表将 `iter0360` 与对应 seed 的 `RR/PFQ` 单次结果直接对齐：

| Topology seed | iter0360 goodput | RR goodput | PFQ goodput | iter0360 expiry | RR expiry | PFQ expiry | iter0360 BLER | RR BLER | PFQ BLER | iter0360 delay mean | RR delay mean | PFQ delay mean | 对 RR goodput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 41 | 1029.23 | 868.66 | 763.10 | 0.1035 | 0.2360 | 0.2909 | 0.1343 | 0.1183 | 0.1195 | 69.29 | 32.33 | 110.84 | 胜 |
| 42 | 938.24 | 958.52 | 868.80 | 0.1582 | 0.1701 | 0.2218 | 0.1405 | 0.1141 | 0.1269 | 60.20 | 37.84 | 85.87 | 负 |
| 43 | 953.55 | 867.43 | 865.21 | 0.1389 | 0.2336 | 0.2296 | 0.1393 | 0.1067 | 0.1188 | 74.63 | 55.20 | 77.34 | 胜 |

## 6. 结果分析

### 6.1 能否说 `iter0360` 已经超过 baseline

如果主目标是：

- `goodput`
- `served throughput`
- `expiry_drop_rate`
- `served_buffer_ratio`
- cell-edge / low-percentile UE throughput

那么答案是：

- 是，`iter0360` 已经明显超过 `RR`
- 也明显超过 `PFQ`

支撑点：

- `goodput` 比 `RR` 高 `+8.4%`，比 `PFQ` 高 `+17.0%`
- `served throughput` 比 `RR` 高 `+10.1%`
- `expiry_drop_rate` 比 `RR` 低 `37.4%`，比 `PFQ` 低 `46.0%`
- `UE throughput p5` 比 `RR` 高约 `79%`
- `UE goodput p10` 比 `RR` 高约 `48.5%`

### 6.2 为什么还不能说“全面超过 RR”

如果标准是“全面优于 RR”，当前还不能这么说。

主要原因：

- `BLER` 仍明显高于 `RR`
  - `iter0360 = 0.1381`
  - `RR = 0.1130`
  - 相对高约 `22.2%`
- `packet_delay_mean` 仍明显差于 `RR`
  - `68.04 ms vs 41.79 ms`
- `packet_delay_p50` 也远高于 `RR`
  - `19.83 ms vs 0.50 ms`
- `backlog_free_ue_ratio` 很低
  - `0.0185 vs RR 0.3056`

因此：

- 在“吞吐 + TTL 过期 + edge fairness”维度，`iter0360` 是当前三者最强
- 在“可靠性 + 低时延”维度，`RR` 仍然更好

### 6.3 每 seed 行为说明了什么

逐 seed 结果说明这次均值提升不是单一 seed 偶然拉高，但也不是“每个 seed 都赢 RR”：

- `seed41`
  - `iter0360` 明显赢 `RR`
  - goodput 提升很大
  - 过期率显著更低
  - 但 `BLER` 与时延仍差于 `RR`
- `seed42`
  - `iter0360` 的 goodput 略低于 `RR`
  - 但过期率仍优于 `RR`
  - `BLER` 和时延继续劣于 `RR`
- `seed43`
  - `iter0360` 再次明显赢 `RR`
  - 过期率显著更低
  - `BLER` 与时延仍落后于 `RR`

所以这次均值结论更准确的表达是：

- `iter0360` 在 `3` 个 topology seeds 中有 `2` 个 seed 的 goodput 超过 `RR`
- `3` 个 seed 的 expiry 都优于 `RR`
- `3` 个 seed 的 BLER 都差于 `RR`
- `3` 个 seed 的 packet delay mean 都差于 `RR`

### 6.4 为什么这次均值比早先 `best_eval=iter0360` 高很多

此前聊天中引用的 `best_eval=iter0360 mean goodput=907.51 Mbps`，来自：

- 固定 `topology-seed=42`
- `sample-seed=41,42,43` 均值

而本次文档里的 `973.68 Mbps` 来自：

- `topology-seed=41,42,43`
- 且每个 topology seed 对应同名 `sample-seed`

因此这两个均值不是一回事。

这次均值更高，主要不是“模型突然变强了”，而是：

- 口径变成了多 topology-seed 平均
- `seed41` 和 `seed43` 对 `iter0360` 更友好
- 它们把均值整体抬高了

这也再次说明：

- 固定 `topology-seed=42` 的 candidate-eval 均值不能直接拿来和 `RR/PFQ` 的 topology multiseed baseline 均值硬比

### 6.5 如何理解 `scheduled_ratio` 和 `PRG utilization`

这两个指标本次不应当被当作第一主指标。

原因：

- `RR/PFQ` 走的是 native `Type-0` baseline 路径
- `iter0360` 走的是 `custom UE+PRG` 路径
- 两条路径下 `scheduled_ratio` 的语义并不完全等价

不过它们仍然提供了两个有用信号：

- `scheduled_ratio_mean` 很低，说明模型不是通过“更频繁地把 UE 都点一遍”来赢得 goodput
- `prg_utilization_ratio ≈ 1.0`，说明这版 `iter0360` 仍然几乎把 PRG 打满，没有学出明显的 blanking 行为

因此当前更像是：

- 模型在“高 served throughput + 更低 TTL 过期 + 更强 edge fairness”上已经做得比 baseline 更激进且更有效
- 但它还没有进入 `RR` 那种“低 BLER + 低中位时延”的稳定可靠性区间

## 7. 最终结论

### 7.1 简短结论

- 如果当前项目的第一目标是 `cluster goodput` 和 `TTL-aware service`，`iter0360` 已经是这组对比里最好的策略。
- 如果目标是“全面超过 RR”，当前还做不到，因为 `BLER` 和低时延表现仍明显差于 `RR`。

### 7.2 更准确的表述

最适合写进实验结论的话术是：

> 在 `3cell + ue36 + ttl200ms + rbg16 + Rayleigh` 场景下，`iter0360` 在 `topology-seed=41,42,43` 的均值口径上，已经显著超过 `RR` 和 `PFQ` 的 `goodput / served throughput / expiry_drop_rate / edge throughput` 表现；但其 `BLER` 和 packet-level latency 仍然落后于 `RR`，因此目前应视为“吞吐与 TTL 服务质量最优”，而非“全面最优”。

### 7.3 后续建议

- 如果下一轮目标是继续稳住 `goodput` 优势，同时逼近 `RR`：
  - 优先压 `BLER`
  - 其次压 `packet_delay_mean / p50`
  - 关注 `backlog_free_ue_ratio`
- 如果下一轮目标只是拿到更强的“业务完成率”结果：
  - 当前 `iter0360` 已经足够作为新的强候选 baseline

## 8. 备注

- 这次对比只有 `3` 个 topology seeds，因此这里说的“显著超过”是工程意义上的明显幅度，不是严格统计学显著性结论。
- 全量字段请直接以这两个文件为准：
  - [`iter0360_kpi_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_iter0360_toposeed414243_20260407_085302/RAYLEIGH/iter0360_kpi_mean.csv)
  - [`rr_vs_pfq_vs_iter0360_compare_mean.csv`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_model_multiseed_compare_iter0360_toposeed414243_20260407_085302/RAYLEIGH/rr_vs_pfq_vs_iter0360_compare_mean.csv)
