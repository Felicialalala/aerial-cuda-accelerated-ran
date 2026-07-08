# Stage-B Boundary 智能算法实验报告

本文系统梳理当前 Stage-B 智能调度代码、baseline 结果和 HGraphPPO 迭代记录。评估不只看系统 goodput，也同时关注 packet 服务质量和 UE 维度体验：包括 per-UE goodput/throughput 的 p5、p10、Jain 公平性、UE-macro packet service rate、UE-macro packet delay、backlog-free UE ratio 等。除特别说明外，场景固定为：

```text
3cell / 36 UE / cell_radius=500m / site_spacing=1000m
ue-placement=coop_boundary
cell-assoc-mode=strongest
shadow_seed=legacy_shared_rng
Rayleigh + SVD + Type-0 bitmap
packet_size=3000 bytes
arrival_rate=1.0 pkt/TTI/UE
TTL=200 ms
horizon=4000 TTI
```

## 1. 结论摘要

当前“部署测试最好结果”必须分口径看，不能只按一条 goodput 均值排序：

- 单 seed 正式部署最好结果是 `HGraphPPO v9c best / iter4`，seed41，t4000，argmax + greedy。它不是靠填满 PRG，而是在 `86.73%` PRG util/fill 下拿到 `1531.197 Mbps` goodput，比 PFQ reference 高 `+10.662 Mbps`；同时 TB BLER `11.387%`，expiry `0`，UE goodput p5 `40.022 Mbps`，packet p95 delay `2.5 ms`，R_pkt `22.847 Mbps`。
- 10-seed 正式部署证据仍来自 v9c。full-run 10-seed 尚未超过 PFQ：`1518.248 Mbps` vs PFQ `1520.536 Mbps`，差 `-2.287 Mbps`，且 expiry/p95 tail 被少数 seed 拖住。但去掉前 `1000 TTI` 冷启动后，v9c seed41-50 mean goodput 为 `1530.558 Mbps`，比 PFQ full-run 高 `+10.022 Mbps`，mean PRG util 仅 `88.226%`。
- rate-aware 正式部署较好的三 seed 结果是 `HGraphPPO v10 rate2 iter24`，seed41-43，warm1000。full-run goodput `1516.414 Mbps`，R_pkt `10.559 Mbps`，per-packet mean rate `28.753 Mbps`；warmup-excluded evaluator goodput `1528.135 Mbps`。
- v19 是后续 packet-tail/coverage quota 方向，在历史实验记录中主要是 online rollout/pilot；本轮补充跑出的 seed41-50 checkpoint eval 可以作为资源效率补充证据，但不能覆盖 v9c/v10-v11 这条正式部署主线。它的价值是低 PRG utilization，短板是 UE-macro R_pkt、packet tail 和 UE tail rate 未全面超过 PFQ。

单 seed 正式部署最好模型：

```text
HGraphPPO v9c best / iter4
checkpoint:
training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_prgeff_greedy_i12_r512_from_v9blast_thread1/ppo_actor_best.pt
evaluation:
training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_i12r512_best_argmax_greedy_coop_boundary_r500_t4000_w0
```

10-seed 去冷启动部署证据：

```text
HGraphPPO v9c threshold best, seed41-50
checkpoint:
training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_prgeff_greedy_i12_r512_from_v9blast_thread1/ppo_actor_threshold_best.pt
evaluation:
training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50_v9c_i12r512_best_threshold_argmax_greedy_coop_boundary_r500_t4000_w0
```

rate-aware 三 seed 正式部署模型：

```text
HGraphPPO v10 rate2 iter24
checkpoint:
training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v10_rate2_greedy_i24_r512_from_v9cbest_thread1/candidate_checkpoints/ppo_actor_iter_0024.pt
evaluation:
training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s43_v10_rate2_iter24_argmax_greedy_coop_boundary_r500_t4000_w1000
```

原因：

- 当前 3cell rate1 下，`packet_effective_*` 的硬上界是 `48 Mbps`；系统级 offered/goodput 的期望上界是 `1.728 Gbps`。日志里 `offered_mbps` 偶尔到 `~1730 Mbps`，是 Poisson 到包在 4000 TTI 内的随机浮动。
- v9c seed41 best 的正式部署证据最干净：goodput、served、BLER、expiry、UE goodput p5、packet p95 delay、R_pkt 都优于 PFQ reference，且 PRG util/fill 低很多。
- v9c seed41-50 full-run 均值没有过 PFQ，但 post-1000 均值过 PFQ，说明主要问题不是学不到高效调度，而是冷启动和少数 seed 的 tail/expiry 风险。
- 在 seed 41-43、忽略前 1000 TTI 的 evaluator 统计中，v10 iter24 的 goodput 为 `1528.135 Mbps`，TB err 为 `11.094%`，PRG fill/util 约 `89.203%`。
- 用同一 seed 41-43 的 active RRQ/PFQ full-run baseline 做可落地 KPI 对比时，v10 iter24 相比 PFQ goodput `+1.091 Mbps`，TB BLER `-0.250 pp`，packet delay mean `-1.313 ms`，p95 delay `-8.0 ms`；UE 维度上，UE goodput p5 `-0.062 Mbps`、p10 `+0.483 Mbps`、Jain `+0.000160`，整体公平性接近 PFQ 但边缘 UE p5 没有明显赢；同时 expiry `+0.186 pp`、UE-macro packet rate `-4.742 Mbps`、PRG utilization `-10.736 pp`。
- 原始 RR/PF 已在容器内补跑。seed41-43 matched 结果显示，原始 PF 在 boundary 下 goodput 只有 `902.517 Mbps`、expiry `31.574%`、UE goodput p5 `10.266 Mbps`，明显不可作为强 reference；原始 RR goodput `1480.561 Mbps`，也低于 RRQ/PFQ/HGraph。
- 本轮补充 v19 seed41-50 checkpoint eval 后，v19 的资源效率证据成立：full-run goodput `1518.480 Mbps`，PRG utilization `88.041%`，相对 PFQ 少用 `10.482 pp` PRG。但 UE p5/p10 没有明显优于 PFQ，UE-macro R_pkt 和 p95 delay 明显弱于 PFQ，因此只作为资源效率补充，不作为当前正式部署最好主结论。

重要口径说明：

- boundary 原始 RR/PF 已按容器环境补跑，输出在 `output/stageB_rr_pf_multiseed_compare_boundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50`。此前宿主机直接运行失败，是动态库环境不完整导致。
- `aggregate_model_kpi_compare.py` 在读取 `rrq_vs_pfq_compare_mean.csv` 时会把 `rrq_mean` 映射到输出列 `rr_mean`，因此 HGraph compare 文件名里的 `rr_vs_pfq_vs_hgraph` 实际应读作 `RRQ vs PFQ vs HGraph`。
- 本报告没有把 uniform RR/PF 结果混入 boundary 主表。uniform RR/PF 只能作为机制背景，不能替代 boundary RR/PF。
- 原始 RR/PF 补跑必须进入 Aerial 容器并激活虚拟环境：

```bash
./cuPHY-CP/container/attach_aerial.sh
cd /opt/nvidia/cuBB
source .venv/bin/activate
```

## 2. 代码与实验链路

### 2.1 Native baseline 链路

核心入口：

- `cuMAC/scripts/run_stageB_main_experiment.sh`
- `cuMAC/scripts/run_stageB_rr_pf_compare.sh`
- `cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh`
- `cuMAC/scripts/run_stageB_baseline_eval_suite.sh`
- `cuMAC/scripts/rr_vs_pf_compare.py`
- `cuMAC/scripts/aggregate_rr_pf_compare.py`

核心 C++/CUDA：

- `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
  - 注入 Stage-B 场景参数。
  - Type-0 下先用 `populateType0AllUeSelection()` 将所有 active UE 放入可调度集合。
  - 输出 `TRAFFIC_KPI`、`TRAFFIC_PKT_DELAY`、`TRAFFIC_PKT_RATE`、`UE_KPI`、`UE_PKT_DELAY` 等 KPI。
  - 支持 `CUMAC_PACKET_DELAY_SAMPLES=1` 导出逐包时延样本。
- `cuMAC/src/4T4R/roundRobinScheduler.cu`
  - 原始 RR Type-0 PRG bitmap 分配路径。
- `cuMAC/src/4T4R/multiCellScheduler.cu`
  - 原始 PF 与 PFQ GPU 路径。
  - 当前 PFQ 保留 queue-aware buffer weight、demand cap、intra-TTI decay。
- `cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`
  - CPU reference / KPI 对照路径。

当前 Type-0 语义不是旧式完整“UE selection + PRG allocation”两阶段，而是所有 active UE 都进入候选，RR/PF/RRQ/PFQ 的差异主要落在 PRG bitmap 分配策略。

### 2.2 HGraphPPO 链路

核心入口：

- `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh`
- `cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh`
- `training/stageb_hgraph/online_ppo.py`
- `training/stageb_hgraph/eval_checkpoint_online.py`
- `cuMAC/scripts/aggregate_model_kpi_compare.py`

模型与特征：

- `training/stageb_hgraph/feature_snapshot.py`
  - 生成 cell / UE / PRG 特征。
- `training/stageb_hgraph/edge_generator.py`
  - 构建 Cell-Cell、Cell-UE、Cell-PRG、UE-PRG、PRG-PRG 边。
  - 正式实验要求 `candidate_mode=post_eq_topk`。
- `training/stageb_hgraph/model.py`
  - `StageBHGraphPolicy`，cell/UE/PRG template encoder + hetero message passing + Type-0 action head。
- `training/stageb_hgraph/graph_ops.py`
  - Type-0 joint action decode、demand-cap、budgeted selector 支持。
- `cuMAC/examples/onlineTrainBridge/*`
  - online observation / action bridge。
  - v14 后协议升级，导出 true per-UE packet diagnostics。

### 2.3 指标框架

本报告把指标分为三层，避免只用单一 goodput 做 checkpoint 选择：

| 层级 | 代表指标 | 作用 |
|---|---|---|
| 系统容量 | `traffic.goodput_mbps`、`traffic.served_mbps_est`、TB BLER、PRG utilization | 衡量总吞吐和频谱资源使用效率 |
| packet 服务质量 | `packet_delay_mean/p90/p95`、`R_pkt`、per-packet rate、expiry drop | 衡量业务包从入队到完成服务的 residence time 和超时风险 |
| UE 维度体验 | average UE goodput/throughput、UE goodput p5/p10、Jain、UE-macro packet rate/delay、backlog-free UE ratio | 衡量边缘 UE、弱 UE、跨 UE 公平性和 per-UE 服务节奏 |

其中 UE 维度速率有两类，不应混读：

- `global_kpi.ue_goodput_p5/p10/jain`：基于每个 UE 的 PHY 成功 goodput，适合判断用户体验和公平性。
- `traffic.ue_macro_packet_effective_service_rate_mbps`：先算每个 UE 的 packet effective service rate，再对 UE 做算术平均；这是 UE-weighted 的 packet 服务速率，能暴露少数 UE 长等待/慢服务问题。

因此，一个模型即使系统 goodput 高，也可能因为 UE p5/p10、UE-macro packet rate 或 UE queue-delay tail 变差而不能算“稳健更优”。

## 3. Baseline 结果

### 3.1 Active boundary RRQ/PFQ gate

模型评测使用的 active baseline：

```text
output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50
```

10-seed mean：

| 指标 | RRQ | PFQ | PFQ - RRQ |
|---|---:|---:|---:|
| served Mbps | 1707.590 | 1722.060 | +14.471 |
| goodput Mbps | 1513.633 | 1520.536 | +6.903 |
| TB BLER | 11.555% | 11.778% | +0.224 pp |
| expiry drop | 0.609% | 0.017% | -0.592 pp |
| R_pkt | 9.835 Mbps | 8.972 Mbps | -0.863 |
| per-packet rate mean | 37.783 Mbps | 34.377 Mbps | -3.405 |
| packet delay mean | 7.114 ms | 4.690 ms | -2.424 |
| packet delay p95 | 56.078 ms | 25.650 ms | -30.428 |
| UE goodput p5 | 37.589 Mbps | 39.524 Mbps | +1.935 |
| UE goodput p10 | 39.263 Mbps | 40.132 Mbps | +0.869 |
| UE goodput Jain | 0.996117 | 0.998592 | +0.002475 |
| UE throughput p5 | 43.430 Mbps | 46.175 Mbps | +2.745 |
| UE throughput p10 | 45.624 Mbps | 46.760 Mbps | +1.136 |
| UE throughput Jain | 0.997187 | 0.999523 | +0.002336 |
| UE-macro R_pkt | 31.728 Mbps | 30.425 Mbps | -1.303 |
| UE-macro per-packet rate | 37.208 Mbps | 34.276 Mbps | -2.932 |
| UE-macro packet delay mean | 9.205 ms | 4.913 ms | -4.292 |
| backlog-free UE ratio | 52.222% | 47.222% | -5.000 pp |
| PRG utilization | 93.019% | 98.523% | +5.504 pp |

结论：PFQ 是当前 active gate 的主 reference。它的 goodput、expiry、UE p5/p10、公平性、UE-macro packet delay 和 PRG utilization 均优于 RRQ；RRQ 只在 UE-macro packet rate、per-packet rate、backlog-free UE ratio 等部分 packet/service-rhythm 指标上更高。这说明 RRQ 更容易把一部分 UE 的包快速清空，但 tail 和整体公平性不如 PFQ 稳。

### 3.2 Full KPI / packet-delay suite

逐包时延分布文档使用的 full KPI suite：

```text
output/stageB_baseline_eval_suite_fullkpi_seed41_50
```

该 suite 的 boundary RRQ/PFQ 10-seed mean：

| 指标 | RRQ | PFQ | PFQ - RRQ |
|---|---:|---:|---:|
| goodput Mbps | 1494.513 | 1516.222 | +21.710 |
| TB BLER | 11.635% | 11.823% | +0.188 pp |
| expiry drop | 1.282% | 0.083% | -1.199 pp |
| packet delay mean | 12.362 ms | 7.261 ms | -5.102 |
| packet delay p95 | 94.700 ms | 41.850 ms | -52.850 |
| UE goodput p5 | 34.829 Mbps | 38.954 Mbps | +4.124 |
| UE goodput p10 | 36.998 Mbps | 39.885 Mbps | +2.887 |
| UE goodput Jain | 0.992767 | 0.998177 | +0.005409 |
| UE-macro R_pkt | 28.415 Mbps | 27.859 Mbps | -0.556 |
| UE-macro per-packet rate | 33.297 Mbps | 31.407 Mbps | -1.890 |
| UE-macro packet delay mean | 16.395 ms | 7.791 ms | -8.605 |
| PRG utilization | 94.350% | 98.598% | +4.248 pp |

该 suite 同时有 uniform RR/PF 结果，但没有 boundary RR/PF；本报告另外在容器内补跑了 boundary RR/PF，见下一节。报告中的 boundary 模型主对比不混用 uniform RR/PF。从 UE 维度看，PFQ 对 RRQ 的优势比系统 goodput 更清楚：UE p5/p10 和 Jain 都显著改善；但 RRQ 在 UE-macro packet effective rate 上仍略高，说明两类算法的服务节奏不同。

### 3.3 Boundary 原始 RR/PF 补跑

按容器口径补跑：

```text
output/stageB_rr_pf_multiseed_compare_boundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50
```

10-seed mean：

| 指标 | RR | PF | PF - RR |
|---|---:|---:|---:|
| served Mbps | 1666.277 | 1086.953 | -579.324 |
| goodput Mbps | 1467.894 | 940.790 | -527.104 |
| TB BLER | 11.889% | 12.872% | +0.982 pp |
| expiry drop | 1.287% | 29.484% | +28.197 pp |
| R_pkt | 1.112 Mbps | 0.256 Mbps | -0.857 |
| per-packet rate mean | 28.189 Mbps | 14.544 Mbps | -13.644 |
| UE-macro R_pkt | 22.224 Mbps | 6.180 Mbps | -16.045 |
| UE-macro per-packet rate | 26.969 Mbps | 8.734 Mbps | -18.235 |
| packet delay mean | 24.510 ms | 95.127 ms | +70.617 |
| packet delay p95 | 151.750 ms | 199.450 ms | +47.700 |
| average UE goodput | 40.775 Mbps | 26.133 Mbps | -14.642 |
| UE goodput p5 | 34.234 Mbps | 11.009 Mbps | -23.226 |
| UE goodput p10 | 35.607 Mbps | 12.399 Mbps | -23.208 |
| UE goodput Jain | 0.992659 | 0.842600 | -0.150059 |
| backlog-free UE ratio | 43.611% | 15.278% | -28.333 pp |
| PRG utilization | 99.998% | 100.000% | +0.002 pp |

结论：在 boundary 场景下，原始 PF 几乎满 PRG 使用，但 UE tail 和 packet tail 严重恶化，p95 packet delay 接近 TTL 上限，expiry 接近 `29.5%`。原始 RR 明显好于原始 PF，但仍显著弱于 RRQ/PFQ，尤其 R_pkt、packet delay、UE p5/p10 都落后。这也说明 RRQ/PFQ 的 queue-aware 改良不是小修，而是 boundary 稳态可用性的关键。

## 4. 智能算法迭代

| 阶段 | 核心变化 | 代表结果 | 结论 |
|---|---|---:|---|
| seed41 i70 / v8 早期 | HGraphPPO 初始长训，偏高 fill | seed41 goodput 1512.368，BLER 12.460% | 训练能动作闭环，但 high-risk fill 导致 BLER 偏高，低于 PFQ |
| v9 reliability gate | 加 reliability-gated goodput，限制 TB err | seed41 best4 deployment 1514.988，BLER 12.166% | 训练态有提升，但 deployment 仍输 PFQ |
| v9b stricter gate | 更硬 TB gate，greedy/budgeted decode | seed41 last 1514.967；slack0/nosafe override 1526.139 | 证明 decoder 的 demand slack/safe fill 是关键变量 |
| v9c PRG efficiency | 对齐少发高效策略，强化 harmful packing 惩罚 | seed41 best 1531.197，BLER 11.387%，p95 2.5 ms；10-seed post-1000 1530.558 | 单 seed 正式部署最好；10-seed full-run 被冷启动/tail 拖住 |
| v10 rate2 | goodput + packet-rate 第二主线，保存 iter20/24 | iter24 w1000 seed41-43 eval goodput 1528.135 | 当前 balanced 智能 baseline |
| v11 rate-tail gate | checkpoint gate 加 R_pkt / per-packet rate / delay 硬门 | iter20 w1000 seed41-43 eval goodput 1525.170 | seed41 gate 有效，但不能解决 seed42/43 service-time collapse |
| v12 multi-seed rate | 训练 seed 41/42/43 list-cycle | last w1000 eval goodput 1526.673 | 多 seed 方向正确，但训练窗口受冷启动误导 |
| v12b steady warmup | rollout warmup 1000，训练看稳态 | seed42 R_pkt 从 6.140 到 6.801，seed43 未改善 | warmup 必要，但 reward/selector 压力仍不足 |
| v12c tail rescue | tail/rescue reward + selector bias | seed42 R_pkt 到 7.387，seed43 约 2.520 | 对 seed42 有效，seed43 仍是 blocker |
| v13 UE-rate proxy | 引入 UE goodput-rate proxy | seed43 goodput 可到 1540.356，但 R_pkt 降到 1.235 | goodput proxy 不能替代 packet residence time |
| v14 true packet-rate | bridge 导出 true UE packet diagnostics | smoke 通过；seed43 v14/v14b p10 packet-rate 仍塌陷 | 诊断口径正确，但 selector 仍要改 |
| v15-v18 tail feature/quota | tail 特征、strict target、thin/thick quota | v17 seed43 goodput 1542.374，UE p10 0.592 | quota 有效但覆盖/厚度不稳 |
| v19 coverage quota | top8 * 2 PRG coverage quota | seed43 pilot UE p10 packet-rate >1 Mbps；补充 eval goodput 1518.480，PRG util 88.041% | 后续资源效率/coverage 探索；不替代 v9c/v10 正式部署主线 |

## 5. 当前最好模型对比

### 5.1 选择依据

本报告把“最好”拆成四个口径：

- `v9c best / iter4, seed41`：单 seed 正式部署最好，goodput、served、BLER、expiry、UE p5、packet delay tail、R_pkt 同时非常漂亮。
- `v9c threshold best, seed41-50 post-1000`：10-seed 去冷启动最好证据，post-1000 goodput 超 PFQ `+10.022 Mbps`，但 full-run 被冷启动和少数 seed tail 拖住。
- `v10 rate2 iter24, seed41-43 warm1000`：rate-aware 三 seed 较好正式部署，R_pkt/per-packet 指标比 v9c 10-seed full-run 更适合作为 rate-aware baseline。
- `v19 coverage quota`：后续机制方向和本轮补充资源效率评估，不作为当前正式部署最好主结论。

### 5.2 单 seed 正式部署最好：v9c best / iter4

对比口径：

- model：`ppo_actor_best.pt`，seed41，t4000，argmax + greedy。
- 输出目录：`training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_i12r512_best_argmax_greedy_coop_boundary_r500_t4000_w0`
- PFQ reference：同一 active boundary PFQ compare 口径。

| 指标 | HGraph v9c | vs PFQ reference |
|---|---:|---:|
| goodput | `1531.197 Mbps` | `+10.662 Mbps` |
| served | `1730.643 Mbps` | `+8.583 Mbps` |
| global TB BLER | `11.387%` | `-0.391 pp` |
| PRG util / fill | `86.730%` | `-11.793 pp` |
| expiry | `0` | 优于 PFQ `0.0173%` |
| average UE goodput | `42.533 Mbps` | `+0.296 Mbps` |
| UE goodput p5 | `40.022 Mbps` | `+0.497 Mbps` |
| UE goodput p10 | `40.961 Mbps` | `+0.829 Mbps` |
| UE goodput Jain | `0.999094` | `+0.000502` |
| UE throughput p5 | `46.995 Mbps` | `+0.820 Mbps` |
| packet delay mean | `1.050 ms` | `-3.640 ms` |
| packet p90 delay | `2.0 ms` | `-9.0 ms` |
| packet p95 delay | `2.5 ms` | `-23.15 ms` |
| packet effective rate R_pkt | `22.847 Mbps` | `+13.875 Mbps` |
| per-packet mean rate | `31.906 Mbps` | `-2.472 Mbps` |
| UE-macro packet delay mean | `1.049 ms` | `-3.864 ms` |

解读：

- 这是目前最干净的“部署通过”证据：不靠填满 PRG，而是用更低 fill 得到更高 goodput、更低 BLER、更低 delay。
- `packet_effective_*` 在 rate1 下硬上界是 `48 Mbps`，v9c 的 per-packet mean rate `31.906 Mbps` 约为理想上界的 `66.5%`；R_pkt `22.847 Mbps` 说明真实 packet residence-time 也明显优于 PFQ/RRQ。
- UE 维度上，v9c 的 p5/p10/Jain 都优于 PFQ reference，这一点比系统 goodput 更能说明 seed41 结果质量。

### 5.3 10-seed 部署证据：v9c full-run 与 post-1000

full-run 10 seed 其实还没有过 PFQ：

| 指标 | HGraph v9c full | PFQ full | 差值 |
|---|---:|---:|---:|
| mean goodput | `1518.248 Mbps` | `1520.536 Mbps` | `-2.287 Mbps` |
| served | `1715.513 Mbps` | `1722.060 Mbps` | `-6.547 Mbps` |
| global TB BLER | `11.377%` | `11.778%` | `-0.401 pp` |
| expiry | `0.285%` | `0.017%` | `+0.268 pp` |
| PRG utilization | `87.973%` | `98.523%` | `-10.550 pp` |
| UE goodput p5 | `38.472 Mbps` | `39.524 Mbps` | `-1.052 Mbps` |
| UE goodput p10 | `40.220 Mbps` | `40.132 Mbps` | `+0.088 Mbps` |
| UE goodput Jain | `0.997839` | `0.998592` | `-0.000754` |
| packet p95 delay | `37.650 ms` | `25.650 ms` | `+12.000 ms` |
| R_pkt | `8.503 Mbps` | `8.972 Mbps` | `-0.469 Mbps` |

去掉前 `1000 TTI` 冷启动后，按 HGraph `checkpoint_eval.log` 的等长 `500 TTI` 窗口重聚合：

| 指标 | HGraph full 4000 | HGraph post-1000 | PFQ full 4000 | post-1000 HGraph vs PFQ |
|---|---:|---:|---:|---:|
| mean goodput | `1518.248 Mbps` | `1530.558 Mbps` | `1520.536 Mbps` | `+10.022 Mbps` |
| mean TB err | `11.553%` | `10.996%` | n/a | n/a |
| mean PRG util | `87.973%` | `88.226%` | `98.523%` | `-10.298 pp` |
| mean blank PRG / TTI | `6.134` | `6.005` | n/a | n/a |

说明：

- PFQ baseline 现有产物只有 full-run KPI 汇总，没有同粒度 post-1000 窗口日志，因此 post-1000 表中的 PFQ 对照仍是 full-run `1520.536 Mbps`。
- 这个口径下，v9c 10-seed 去冷启动后已经超过 PFQ `+10.022 Mbps`，同时少用约 `10.3 pp` PRG。
- full-run 失败主要来自冷启动和少数 seed tail/expiry：设计日志中记录 post-1000 对 PFQ full-run 是 `7/10` 胜出，落后 seed 为 `43`、`45`、`48`。

### 5.4 Rate-aware 三 seed：v10 iter24 warm1000

v10 iter24 是当前 rate-aware 正式部署较好的三 seed 结果：

| 指标 | HGraph v10 iter24, seed41-43 |
|---|---:|
| full-run goodput | `1516.414 Mbps` |
| full-run TB BLER | `11.519%` |
| expiry | `0.226%` |
| R_pkt | `10.559 Mbps` |
| per-packet mean rate | `28.753 Mbps` |
| UE-macro R_pkt | `25.834 Mbps` |
| packet delay mean | `5.757 ms` |
| packet p95 delay | `37.500 ms` |
| UE goodput p5 | `38.792 Mbps` |
| UE goodput p10 | `40.139 Mbps` |
| UE goodput Jain | `0.998237` |
| PRG utilization | `88.944%` |
| warmup-excluded evaluator goodput | `1528.135 Mbps` |
| warmup-excluded TB err | `11.094%` |
| warmup-excluded PRG util | `89.203%` |

相对 PFQ seed41-43，v10 iter24 的 goodput `+1.091 Mbps`、TB BLER `-0.250 pp`、packet delay mean `-1.313 ms`、p95 delay `-8.0 ms`、UE goodput p10 `+0.483 Mbps`、Jain `+0.000160`；但 UE goodput p5 `-0.062 Mbps`，UE-macro R_pkt `-4.742 Mbps`。

### 5.5 v19 的当前定位

v19 是 packet-tail coverage quota 的后续方向。历史实验记录中它主要是 online rollout/pilot：seed43 上 UE p10 true packet-rate 已经能稳定到 `>1 Mbps`，但那不是正式 t4000 deployment gate。

本轮补充跑出的 v19 seed41-50 checkpoint eval 可以作为资源效率观察：

| 指标 | PFQ active | HGraph v19 i3 | v19 - PFQ |
|---|---:|---:|---:|
| full-run goodput | `1520.536 Mbps` | `1518.480 Mbps` | `-2.056 Mbps` |
| TB BLER | `11.778%` | `11.407%` | `-0.371 pp` |
| PRG utilization | `98.523%` | `88.041%` | `-10.482 pp` |
| UE goodput p5 | `39.524 Mbps` | `38.815 Mbps` | `-0.709 Mbps` |
| UE goodput p10 | `40.132 Mbps` | `40.065 Mbps` | `-0.067 Mbps` |
| UE throughput p5 | `46.175 Mbps` | `45.000 Mbps` | `-1.175 Mbps` |
| R_pkt | `8.972 Mbps` | `8.170 Mbps` | `-0.802 Mbps` |
| UE-macro R_pkt | `30.425 Mbps` | `23.927 Mbps` | `-6.498 Mbps` |
| packet p95 delay | `25.650 ms` | `51.100 ms` | `+25.450 ms` |

结论：v19 的资源效率很有价值，但它没有在 UE tail rate 和 packet service tail 上超过 PFQ，因此不能替代 v9c/v10-v11 作为“当前正式部署最好结果”。

### 5.6 原始 RR/PF 补跑状态

本轮已按用户给定环境完成 boundary 原始 RR/PF 10-seed 补跑。

运行环境：

- 先进入容器：`./cuPHY-CP/container/attach_aerial.sh`
- 容器内仓库路径：`/opt/nvidia/cuBB`
- 激活虚拟环境：`source .venv/bin/activate`
- 输出目录：`output/stageB_rr_pf_multiseed_compare_boundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50`

实际命令：

```bash
./cuPHY-CP/container/attach_aerial.sh
cd /opt/nvidia/cuBB
source .venv/bin/activate

./cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh \
  --seed-list 41,42,43,44,45,46,47,48,49,50 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --tti 4000 \
  --cell-radius 500 \
  --cell-assoc-mode strongest \
  --ue-placement coop_boundary \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --prbs-per-group 16 \
  --precoding svd \
  --packet-delay-samples 1 \
  --reference-baseline rr \
  --pf-baseline pf
```

宿主机 shell 环境缺少旧二进制依赖的 `libhdf5_serial_cpp.so.103`、`libhdf5_serial.so.103`、CUDA/TensorRT 等动态库，不能直接补跑。本报告纳入的 RR/PF 数字均以容器内 `/opt/nvidia/cuBB` 路径下的输出为准。

补跑过程中部分原始 PF seed 出现 CPU/GPU per-UE throughput CDF perf-check WARN/FAIL，但每个 seed 都完成全部 4000 TTI，并写出 `kpi_summary.json`、`ue_kpi.csv` 和 per-seed compare 文件；聚合表使用这些已完成 KPI。

## 6. 对 PFQ 的横向对比

按用户要求，除最好模型外，其余智能模型只和 PFQ 对比。这里使用各自 compare 文件中的 PFQ active gate。`w1000` 行同时给出 warmup-excluded evaluator goodput。

| 模型/部署 | seeds | full-run goodput | vs PFQ | TB BLER | expiry | R_pkt | delay mean | p95 | eval goodput after w1000 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v9 best4 | 41 | 1514.988 | -5.547 | 12.166% | 0.000% | 17.339 | 1.384 | 5.0 | 1514.988 | BLER 偏高，输 PFQ |
| v9b best | 41 | 1511.768 | -8.768 | 12.245% | 0.000% | 15.892 | 1.510 | 6.0 | 1511.768 | high fill harmful |
| v9b last | 41 | 1514.967 | -5.569 | 12.116% | 0.000% | 15.139 | 1.585 | 6.5 | 1514.967 | 仍输 PFQ |
| v9b last slack0/nosafe | 41 | 1526.139 | +5.604 | 11.733% | 0.000% | 22.533 | 1.065 | 2.5 | 1526.139 | decoder override 证明少发高效 |
| v9c best | 41 | 1531.197 | +10.662 | 11.387% | 0.000% | 22.847 | 1.050 | 2.5 | 1531.197 | 最强单 seed |
| v9c last | 41 | 1530.625 | +10.089 | 11.602% | 0.000% | 22.478 | 1.068 | 2.5 | 1530.624 | 单 seed 稳定 |
| v9c threshold 10-seed | 41-50 | 1518.248 | -2.287 | 11.377% | 0.285% | 8.503 | 5.853 | 37.65 | 1530.558 | full-run 被冷启动/tail 拖住，post-1000 过 PFQ |
| v10 iter20 w1000 | 41-43 | 1514.965 | -5.570 | 11.666% | 0.191% | 10.175 | 5.792 | 38.83 | 1526.088 | 稳态有增益，全程受冷启动影响 |
| v10 iter24 w1000 | 41-43 | 1516.414 | -4.122 | 11.519% | 0.226% | 10.559 | 5.757 | 37.50 | 1528.135 | balanced 智能 baseline |
| v11 iter20 w1000 | 41-43 | 1515.117 | -5.419 | 11.558% | 0.236% | 10.394 | 5.866 | 42.33 | 1525.170 | rate-tail gate 未改善跨 seed |
| v12 last w1000 | 41-43 | 1514.837 | -5.699 | 11.640% | 0.170% | 10.183 | 5.916 | 44.33 | 1526.673 | multi-seed 训练仍需稳态对齐 |
| v19 i3 w1000 supplement | 41-50 | 1518.480 | -2.056 | 11.407% | 0.201% | 8.170 | 6.693 | 51.10 | 1531.037 | 补充资源效率证据；不作为当前部署最好主结论 |

注：`vs PFQ` 使用 active boundary PFQ 10-seed mean `1520.536 Mbps` 作为横向 reference。v10/v11/v12 的 seeds 为 41-43，所以这些行的差值用于方向比较；v9c post-1000 是 HGraph 窗口重聚合结果，PFQ 没有同源 post-1000 统计，因此仍对 PFQ full-run。

## 7. 机制分析

### 7.1 为什么 v9c 单 seed 强但 full-run 被拖住

v9c 的关键成功在于少发高效：

- `safeFill=0`、低 demand slack 让模型避免把 marginal PRG 塞给高风险 UE。
- seed41 中 PRG utilization 降到约 `86.73%`，但 goodput 反而高于 PFQ。
- BLER 从 PFQ 参考约 `11.78%` 降到 `11.39%`，说明 harmful packing 被抑制。

但 10-seed threshold 结果显示：

- full-run goodput 均值低于 PFQ，`1518.248 Mbps` vs `1520.536 Mbps`。
- 去掉前 `1000 TTI` 后，HGraph mean goodput 升到 `1530.558 Mbps`，对 PFQ full-run 为 `+10.022 Mbps`。
- R_pkt 从 seed41 的 `22.847 Mbps` 降到 10-seed 的 `8.503 Mbps`。
- p95 delay 从 `2.5 ms` 扩大到 `37.65 ms`。

这说明模型学到的少发策略有价值，而且稳态容量已经能超过 PFQ；真正短板是冷启动、seed43/45/48 的 tail/expiry，以及跨 seed packet service rhythm。

### 7.2 v10 与 v19 的定位

v10 把 packet effective rate 作为第二主线：

- seed41 训练 absolute best iter14 goodput 达 `1567.854 Mbps`，但 R_pkt 只有 `21.262 Mbps`。
- iter24 goodput 不如 iter14，但 R_pkt 和 per-packet rate 更好，因此更适合作为 balanced/rate candidate。
- 跨 seed w1000 评估中，iter24 比 iter20 goodput 更高、TB err 更低、R_pkt 更高。

v10 的价值在于 balanced：

- seed41-43 matched full-run 中，它对 PFQ 有 `+1.091 Mbps` goodput、`-0.250 pp` TB BLER、`-8.0 ms` p95 delay、UE goodput p10 `+0.483 Mbps`。
- 但它只有 seed41-43 证据，且 UE-macro packet-rate、per-packet mean rate、backlog-free UE ratio 低于 PFQ/RRQ。

v19 的价值在于探索资源效率和 coverage quota：

- seed41-50 full-run 中，它对 PFQ goodput 仅 `-2.056 Mbps`，TB BLER `-0.371 pp`，PRG utilization `-10.482 pp`。
- `goodput / PRG utilization` 比 PFQ 高约 `11.75%`，说明它更接近“少用频谱资源拿到接近同等吞吐”的目标。
- 但 UE goodput p5/p10 没有明显优于 PFQ，UE-macro R_pkt 和 packet delay p95 明显落后。

因此，当前正式部署最好主线仍是 v9c/v10-v11：v9c 负责 goodput/时延/UE tail 的最好证据，v10 负责 rate-aware 三 seed 证据；v19 只作为后续资源效率和 packet-tail quota 的补充方向。

### 7.3 v11-v19 的主线

v11-v12 证明单 seed gate 不足：

- v11 在 seed41 内可以筛出 R_pkt `>23 Mbps`、weighted delay 约 `1 ms` 的候选。
- 但 seed42/43 的 R_pkt 仍塌陷，goodput/TB err 看起来可用，packet service-time 已经失败。

v12b/v12c 证明 warmup 对齐是必要条件：

- 加 `rollout_warmup_steps=1000` 后训练看到稳态窗口。
- tail/rescue 对 seed42 有改善，但 seed43 仍是 blocker。

v13-v14 证明必须用 true packet residence-time：

- UE goodput-rate proxy 在 seed43 看起来很好，但真实 R_pkt 会降到 `1.235 Mbps`。
- v14 协议导出 true UE packet diagnostics 后，诊断路径正确，但 reward/selector 仍需要更直接的 tail service 约束。

v15-v19 的方向是显式 packet-tail quota：

- v17 的 thin quota 证明“被服务次数”不是问题，问题是服务太薄。
- v18 thick quota 改善第一轮，但覆盖范围不稳。
- v19 coverage quota 用 top8 * 2 PRG 改善覆盖，seed43 pilot 能保持 UE p10 packet-rate 大于 `1 Mbps` 多轮。
- 本轮补充 seed41-50 checkpoint eval 后，v19 有资源效率证据：它用 `88.041%` PRG utilization 达到 `1518.480 Mbps` full-run goodput，BLER 也低于 PFQ。
- 但 v19 的 seed43/45/48 仍暴露 packet-rate collapse 和 delay tail；它还不能替代 v9c/v10-v11 的正式部署最好结论。

## 8. 后续建议

1. 固化 boundary 原始 RR/PF 结果。
   - 本轮已在容器内完成 RR/PF boundary 10-seed full KPI。
   - 建议把 `output/stageB_rr_pf_multiseed_compare_boundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50` 纳入后续 baseline suite。
   - 若要发版报告，建议复跑一次无 WARN 的 RR/PF 容器任务，或在报告中保留 CPU/GPU per-UE CDF WARN 说明。

2. 修复聚合脚本命名。
   - `aggregate_model_kpi_compare.py` 不应把 `rrq_mean` 写入 `rr_mean`。
   - 建议输出 `reference_label`，并生成 `rrq_vs_pfq_vs_<model>`。

3. 为 native baseline 增加 warmup-excluded 统计。
   - 当前 HGraph evaluator 有 `eval.stats_warmup_steps=1000`。
   - Native RRQ/PFQ/RR/PF 只有 full-run KPI；要公平比较冷启动，应增加 per-TTI goodput/BLER/PRG/packet 统计或复用 online evaluator。

4. 继续把 v9c/v10-v11 作为正式部署主线，v19 作为机制探索线。
   - v9c 下一步应优先修冷启动和 seed43/45/48 tail/expiry，而不是推翻少发高效策略。
   - v10-v11 下一步应继续提高 UE-macro R_pkt 和 per-packet mean rate，同时保住 v10 的 goodput/BLER/delay 优势。
   - v19 下一轮 gate 不应只看 goodput/utilization，还要同时约束 UE goodput p5/p10、UE-macro R_pkt、packet delay p95、weighted delay、expiry。

5. Checkpoint 选择不要只看 goodput。
   - goodput 接近容量上限时不够敏感。
   - 必须同时看 R_pkt、per-packet rate、UE-macro packet-rate、UE p5/p10 goodput、UE p10 true packet-rate、PRG utilization、weighted delay 和 expiry。
   - 建议新增 `goodput / PRG utilization` 或等价资源效率指标，作为 goodput 饱和阶段的主排序指标之一。

## 9. 数据来源

主要结果：

- active boundary RRQ/PFQ gate:
  `output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50/RAYLEIGH/rrq_vs_pfq_compare_mean.csv`
- boundary RR/PF container rerun:
  `output/stageB_rr_pf_multiseed_compare_boundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50/RAYLEIGH/rr_vs_pf_compare_mean.csv`
- full KPI packet-delay suite:
  `output/stageB_baseline_eval_suite_fullkpi_seed41_50/suite_aggregate/boundary_rrq_pfq_metrics.csv`
- packet-delay distribution report:
  `Doc/stageB_packet_delay_distribution_analysis.md`
- current Stage-B config:
  `Doc/current_stageB_effective_configuration.md`
- HGraphPPO design/training log:
  `Doc/stageB_7cell_tar_gnn_pfq_design.md`
- v9c single-seed best deployment:
  `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_i12r512_best_argmax_greedy_coop_boundary_r500_t4000_w0/RAYLEIGH/rr_vs_pfq_vs_hgraph_s41_v9c_i12r512_best_coop_boundary_r500_t4000_w0_compare_mean.csv`
- v9c seed41-50 threshold deployment:
  `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50_v9c_i12r512_best_threshold_argmax_greedy_coop_boundary_r500_t4000_w0/RAYLEIGH/rr_vs_pfq_vs_hgraph_s41_v9c_i12r512_best_threshold_same_coop_boundary_r500_t4000_w0_compare_mean.csv`
- v10 iter24 evaluation:
  `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s43_v10_rate2_iter24_argmax_greedy_coop_boundary_r500_t4000_w1000/RAYLEIGH/rr_vs_pfq_vs_hgraph_v10_rate2_iter24_rate_w1000_coop_boundary_r500_t4000_w1000_compare_mean.csv`
- v19 seed41-50 supplementary evaluation:
  `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s50_v19_packet_tail_i3_argmax_greedy_rerun_skip_coop_boundary_r500_t4000_w1000/RAYLEIGH/rr_vs_pfq_vs_hgraph_v19_packet_tail_i3_coop_boundary_r500_t4000_w1000_compare_mean.csv`
- v19 pilot:
  `training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s43_v19_packet_tail_quota_coverage_i3_r512_from_v19last_pilot/online_ppo_metrics.csv`
