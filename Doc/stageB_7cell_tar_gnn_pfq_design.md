# Stage-B HGraphPPO Training Plan

> 定位：本文是 HGraphPPO / 7cell 迁移的设计与历史训练日志。当前实验结论、模型排序和报告口径以 [`stageB_boundary_intelligent_algorithm_experiment_report.md`](./stageB_boundary_intelligent_algorithm_experiment_report.md) 为准。

本文是 Stage-B 智能调度当前执行主文档。7-cell RRQ/PFQ 压力扫描继续保留为长期目标背景，但 2026-06-22 之后的实际训练入口先固定在
`3cell + ISD1000 + coop_boundary + strongest association + no explicit shadow_seed`。旧 `coop_center/ISD500` 和固定 `shadow_seed=41` 的结果只作为机制参考，不再作为 HGraphPPO 的训练 gate。

## 1. 当前决策

### 1.1 Active Training Scenario

当前口径：

| 项 | 当前执行值 |
|---|---|
| topology | `3cell` / `3cell_triangle` |
| UE | `36`, 即 `12 UE/cell` |
| cell radius / site spacing | `500 m` / `1000 m` |
| UE placement | `coop_boundary` |
| association | `strongest` |
| shadow fading seed | 不传 `--shadow-seed`，日志应显示 `shadow_seed=legacy_shared_rng` |
| Voronoi clip | `1` |
| packet size | `3000 bytes` |
| arrival rate | `1.0 pkt/TTI/UE` |
| TTL | `200 ms` |
| horizon | native gate 使用 `4000 TTI` |
| RBG | `16 PRB/group`, `17 PRG/cell` |
| precoding | `svd` |
| channel | Rayleigh |
| native baselines | `RRQ` and `PFQ` |

选择这个口径的原因：

- `ISD1000` 方便和旧 uniform 下智能算法的小幅增益对照。
- strongest association 是合理系统行为，不能用 nominal association 人为制造不真实 hard case。
- 不设 `shadow_seed` 后，新 uniform run 与旧 run 基本一致，说明 legacy shared RNG 是当前最适合复现旧对照的口径。
- `coop_boundary` 比 uniform 的 goodput 低约 `11-12 Mbps`，能提供轻度 hard case；同时没有把问题推到 ISD500/nominal 这种过窄诊断分支。

当前保留的 baseline 证据：

| 场景 | 输出目录 |
|---|---|
| uniform, strongest, no shadow seed | `output/stageB_rrq_pfq_multiseed_compare_uniform_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50` |
| coop_boundary, strongest, no shadow seed | `output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50` |
| uniform vs boundary 汇总 | `output/stageB_isd1000_legacyshadow_uniform_vs_boundary_rrq_pfq_detailed_compare.md` |

下一轮 baseline 复核升级为“完整 KPI + raw packet delay distribution”口径。除 `goodput_mbps` 外，正式比较表必须同时包含：

- `traffic.served_mbps_est` / `traffic.goodput_mbps`
- `traffic.packet_effective_service_rate_mbps`
- `traffic.packet_effective_service_rate_per_packet_mean_mbps`
- `traffic.ue_macro_packet_effective_service_rate_mbps`
- `traffic.ue_macro_packet_effective_service_rate_per_packet_mean_mbps`
- `traffic.packet_delay_mean_ms` / `p50` / `p90` / `p95` / `max`
- `traffic.ue_macro_packet_delay_mean_ms`
- `global_kpi.ue_goodput_p5_mbps` / `ue_goodput_jain`
- `global_kpi.global_tb_bler` / `traffic.expiry_drop_rate` / `global_kpi.prg_utilization_ratio`

raw packet delay 样本必须开启 `--packet-delay-samples 1`。每个 scenario run 会保留：

```text
packet_delay_samples.csv
columns: ue_id,packet_index,delay_ms
```

multi-seed 聚合会额外输出：

```text
RAYLEIGH/packet_delay_samples_manifest.csv
columns: scenario,seed,baseline,csv,rows,exists
```

这样后续画 “delay <= x ms 占比 / tail bucket 占比 / CDF” 时，不需要重新跑仿真，只要按 manifest 读取对应 baseline 的 raw sample CSV。

完整 baseline suite 入口：

```bash
./cuMAC/scripts/run_stageB_baseline_eval_suite.sh \
  --build-method phase4 \
  --seed-list 41,42,43,44,45,46,47,48,49,50 \
  --tag-prefix fullkpi_seed41_50
```

该 suite 固定跑三组对照并做 seed41-50 平均：

| suite leg | placement | reference | compare |
|---|---|---|---|
| `uniform_rrq_pfq` | `uniform` | `rrq` | `pfq` |
| `boundary_rrq_pfq` | `coop_boundary` | `rrq` | `pfq` |
| `uniform_rr_pf` | `uniform` | `rr` | `pf` |

suite 完成后会自动生成跨 leg 汇总：

```text
suite_aggregate/uniform_four_baseline_metrics.csv
suite_aggregate/uniform_four_baseline_metrics.json
suite_aggregate/boundary_rrq_pfq_metrics.csv
suite_aggregate/packet_delay_samples_manifest_all.csv
```

其中 `uniform_four_baseline_metrics.*` 将 `rr`、`pf`、`rrq`、`pfq` 放在同一张表里，并给出 `pf_minus_rr`、`pfq_minus_rrq`、`rr_minus_rrq`、`rr_minus_pfq`、`pf_minus_rrq`、`pf_minus_pfq` 的 seed-paired delta 均值，专门用于检查基础 RR/PF 与 queue-aware RRQ/PFQ 的差距。

默认参数保持当前 active baseline 口径：`3cell`、`36 UE`、`cell_radius=500`、`strongest association`、不传 `shadow_seed`、`packet_size=3000 bytes`、`arrival_rate=1.0 pkt/TTI/UE`、`TTL=200 ms`、`4000 TTI`、`16 PRB/group`、`svd`、`Rayleigh`。

10-seed mean 对比：

| 指标 | Uniform RRQ | Uniform PFQ | Boundary RRQ | Boundary PFQ | Boundary-Uniform RRQ | Boundary-Uniform PFQ |
|---|---:|---:|---:|---:|---:|---:|
| served Mbps | `1708.141` | `1724.174` | `1707.590` | `1722.060` | `-0.552` | `-2.114` |
| goodput Mbps | `1525.710` | `1531.572` | `1513.633` | `1520.536` | `-12.077` | `-11.037` |
| TTL expiry | `0.602%` | `0.038%` | `0.609%` | `0.017%` | `+0.008 pp` | `-0.020 pp` |
| packet p95 delay | `67.950 ms` | `21.700 ms` | `56.078 ms` | `25.650 ms` | `-11.873 ms` | `+3.950 ms` |
| TB BLER | `11.267%` | `11.390%` | `11.555%` | `11.778%` | `+0.288 pp` | `+0.388 pp` |
| PRG utilization | `88.784%` | `94.238%` | `93.019%` | `98.523%` | `+4.235 pp` | `+4.285 pp` |
| UE goodput p5 | `37.312` | `39.754` | `37.589` | `39.524` | `+0.277` | `-0.229` |

训练目标先按 boundary PFQ 定 gate：

```text
primary:
  traffic.goodput_mbps > 1520.536 Mbps on seed41 deployment gate
  seed41-50 mean should beat PFQ by >= 0.3% after a candidate passes seed41

hard guards:
  global_tb_bler <= PFQ + 0.2 percentage point
  expiry_drop_rate <= PFQ + 0.05 percentage point
  ue_goodput_p5_mbps >= PFQ - 0.2 Mbps

secondary:
  packet p95 delay should not regress by more than 10 ms
  packet effective service rate / UE-macro rate are second-mainline gates once seed41 proves goodput competitiveness
  pending packets remain diagnostics unless they correlate with rate/tail failures
```

### 1.1.1 Rate as the Second Mainline

2026-06-24 起，`goodput_mbps` 仍是第一主线，但 packet service rate 升级为第二主线，用来判断在容量不容易大幅抬高时，HGraph 是否实际改善了业务到包服务效率。

当前 3cell active traffic 上界：

```text
packet_size = 3000 bytes
slot = 0.5 ms
arrival = 1 pkt/TTI/UE
UE = 36

single-packet min-delay service rate cap = 3000 * 8 / 0.5ms = 48 Mbps
per-UE offered rate = 48 Mbps
aggregate offered/goodput cap = 36 * 48 = 1728 Mbps
```

因此速率指标分两类读：

| 指标 | 用途 |
|---|---|
| `traffic.packet_effective_service_rate_mbps` | 总体完成包的聚合服务速率，偏 capacity/throughput |
| `traffic.packet_effective_service_rate_per_packet_mean_mbps` | 已完成包逐包均值服务速率，偏 latency-aware service quality |
| `traffic.ue_macro_packet_effective_service_rate_mbps` | UE 宏平均后的聚合速率，防止少数 UE 贡献过度主导 |
| `traffic.ue_macro_packet_effective_service_rate_per_packet_mean_mbps` | UE 宏平均后的逐包服务速率，和公平性/尾包服务一起看 |

v10 rate2 的训练策略：

```text
base = v9c/v9d reliability deployment semantics
best_metric = rate_guarded_goodput
hard gates = BLER / expiry / service_p10 / unserved UE / backlog_p90
score = goodput
      + aggregate packet-rate bonus
      + per-packet mean-rate bonus
      - packet-delay penalty
      - BLER/expiry/backlog penalties
```

已落地入口：

```text
script preset:
  goodput_delay_rate_simple_risk_v10_rate2_greedy_fast
  goodput_delay_rate_simple_risk_v10_rate2_sample_fast

reward fields:
  packet_per_packet_rate_weight
  packet_per_packet_rate_scale_mbps

metrics fields:
  rollout_packet_effective_service_rate_per_packet_mean_mbps_mean
  rollout_packet_per_packet_rate_bonus_mean
  rollout_packet_per_packet_rate_weighted_bonus_mean
  best_packet_per_packet_rate_bonus_mbps_equiv
```

运行要求：训练/验证必须先进入容器，再激活虚拟环境。

```bash
./cuPHY-CP/container/attach_aerial.sh
cd /opt/nvidia/cuBB
source .venv/bin/activate
```

seed41 代表性长训已完成：

```text
run_dir = training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v10_rate2_greedy_i24_r512_from_v9cbest_thread1
warm_start = training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_prgeff_greedy_i12_r512_from_v9blast_thread1/ppo_actor_best.pt
```

长训结果：

```text
completed_iterations = 24 / 24
hard-pass iters = 2, 3, 4, 5, 7-24
hard-pass count = 22 / 24
score/absolute best iter = 14
candidate checkpoints = iter4, iter8, iter12, iter16, iter20, iter24
```

Hard-pass 均值：

```text
goodput = 1546.718 Mbps
TB err = 0.10697
R_pkt = 22.624 Mbps
R_pkt per-packet mean = 31.430 Mbps
weighted delay = 1.074 ms
```

候选 checkpoint：

| iter | goodput | TB err | R_pkt | R_pkt per-packet mean | weighted delay | hard gate | 备注 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | `1543.920` | `0.10880` | `22.495` | `31.134` | `1.081 ms` | pass | early stable |
| 8 | `1544.071` | `0.11408` | `22.980` | `31.596` | `1.054 ms` | pass | rate-preserving |
| 12 | `1546.982` | `0.10696` | `21.919` | `30.829` | `1.107 ms` | pass | goodput leaning |
| 16 | `1563.676` | `0.10381` | `22.636` | `31.308` | `1.152 ms` | pass | balanced |
| 20 | `1554.694` | `0.09832` | `23.071` | `31.632` | `1.003 ms` | pass | balanced/rate candidate |
| 24 | `1544.577` | `0.10111` | `23.331` | `32.027` | `1.018 ms` | pass | saved rate candidate |

候选路径：

```text
score best:
  ${run_dir}/ppo_actor_absolute_best.pt
  iter14, goodput=1567.854, R_pkt=21.262, R_pkt_per=30.486

balanced/rate candidates:
  ${run_dir}/candidate_checkpoints/ppo_actor_iter_0020.pt
  ${run_dir}/candidate_checkpoints/ppo_actor_iter_0024.pt
```

读数：长训后半段没有发散；iter20/24 证明在 seed41 上可以找到同时保留 goodput 与 packet-rate 的候选。但 score-best iter14 虽然 goodput 最高，`R_pkt` 和逐包速率反而偏低，说明 `rate` 指标确实不是 goodput 的重复读数。

启动命令摘要：

```bash
CUDA_VISIBLE_DEVICES=0 ./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --ppo-preset goodput_delay_rate_simple_risk_v10_rate2_greedy_fast \
  --build-method skip \
  --tti 4000 \
  --topology-scenario 3cell \
  --cell-radius 500 \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 41 \
  --topology-seed-mode fixed \
  --baseline-scheduler pfq \
  --ue-placement coop_boundary \
  --ue-voronoi-clip 1 \
  --online-persistent 1 \
  --episode-horizon 512 \
  --rollout-steps 512 \
  --iterations 24 \
  --seed 42 \
  --device auto \
  --init-checkpoint training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_prgeff_greedy_i12_r512_from_v9blast_thread1/ppo_actor_best.pt \
  --out-dir training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v10_rate2_greedy_i24_r512_from_v9cbest_thread1 \
  --candidate-checkpoint-interval 4 \
  --candidate-checkpoint-require-gate 1 \
  --print-every-iters 1 \
  --sim-env CUMAC_CELL_ASSOC_MODE=strongest \
  --sim-env CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1
```

长训收敛判断：

```text
must pass:
  best_hard_gate_pass = 1
  rollout_tb_err_rate_mean <= 0.117
  rollout_expiry_reward_rate_mean <= 0.010
  rollout_recent100_service_rate_p10_mean >= 0.70

rate checks:
  rollout_packet_effective_service_rate_mbps_mean should improve versus v9c/PFQ reference
  rollout_packet_effective_service_rate_per_packet_mean_mbps_mean should not fall while aggregate R_pkt rises
  final eval must include UE-macro packet-rate metrics
```

跨 seed 部署验证：

```text
command family = run_stageB_hgraph_checkpoint_multi_seed_compare.sh
decode = argmax
deployment selector = greedy
horizon = 4000 TTI
stats warmup = 1000 TTI
seeds = 41, 42, 43 preliminary cross-seed
```

`iter20` warmup=1000 evaluator stats:

| seed | goodput | TB err | R_pkt | R_pkt per-packet mean | weighted delay | expiry |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | `1538.002` | `0.11338` | `22.309` | `31.147` | `1.090 ms` | `0.00000` |
| 42 | `1538.090` | `0.11272` | `7.164` | `27.623` | `4.067 ms` | `0.00000` |
| 43 | `1502.172` | `0.11137` | `1.884` | `25.848` | `15.350 ms` | `0.00139` |
| mean | `1526.088` | `0.11249` | `10.452` | `28.206` | `6.836 ms` | `0.00046` |

`iter24` warmup=1000 evaluator stats:

| seed | goodput | TB err | R_pkt | R_pkt per-packet mean | weighted delay | expiry |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | `1542.119` | `0.11094` | `22.955` | `31.747` | `1.056 ms` | `0.00000` |
| 42 | `1540.548` | `0.11122` | `7.542` | `27.645` | `3.812 ms` | `0.00000` |
| 43 | `1501.737` | `0.11067` | `1.872` | `25.843` | `15.467 ms` | `0.00173` |
| mean | `1528.135` | `0.11094` | `10.789` | `28.411` | `6.778 ms` | `0.00058` |

部署结论：

- `iter24` 比 `iter20` 更贴近 rate objective，seed41 上 `R_pkt=22.955`、逐包均值 `31.747`，goodput 也高于已知 v9c seed41 formal goodput。
- 但 seed42/43 暴露明显的服务节奏退化：goodput 和 TB err 看起来仍可接受，`R_pkt`、逐包均值和 weighted delay 已经严重变差。
- 因此 rate 指标应当成为第二主线，且必须进入 checkpoint 选择 gate；只用 goodput/TB/expiry 仍可能选出跨 seed 不稳的策略。
- 当前 v10 rate2 不是最终可部署版本；下一轮应把 per-seed rate/tail 指标做成硬门或强惩罚，例如 `R_pkt_per_packet_mean` 下界、UE-macro packet-rate 下界、weighted delay/p95 上界、expiry seed-wise 上界，并考虑 multi-seed/domain-randomized PPO。

v11 rate-tail gate follow-up：

```text
preset:
  goodput_delay_rate_simple_risk_v11_rate_tail_greedy_fast
  goodput_delay_rate_simple_risk_v11_rate_tail_sample_fast

best_metric = rate_tail_guarded_goodput

extra hard gates on training rollout:
  best_hard_packet_rate_min_mbps = 22.5
  best_hard_packet_per_packet_rate_min_mbps = 31.5
  best_hard_packet_delay_max_ms = 1.25

warm_start:
  v10 iter24 candidate checkpoint
```

v11 这轮的目的不是直接扩大容量，而是验证“把 rate/tail 写进 checkpoint 选择”能否避免 v10 的跨 seed 速率塌陷。代码侧已经加入：

- `rate_tail_guarded_goodput` checkpoint metric。
- `--best-hard-packet-rate-min-mbps`
- `--best-hard-packet-per-packet-rate-min-mbps`
- `--best-hard-packet-delay-max-ms`
- summary/CSV 中的 hard-gate pass/fail 字段。

seed41 长训：

```text
run_dir = training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v11_rate_tail_greedy_i24_r512_from_v10i24_thread1
init_checkpoint = training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v10_rate2_greedy_i24_r512_from_v9cbest_thread1/candidate_checkpoints/ppo_actor_iter_0024.pt
completed_iterations = 24 / 24
gate-pass iters = 11, 20, 23
absolute_best_iter = 20
threshold_best_iter = 20
saved rate-tail candidate = candidate_checkpoints/ppo_actor_iter_0020.pt
```

Gate-pass 均值：

```text
goodput = 1531.530 Mbps
TB err = 0.111779
R_pkt = 23.147 Mbps
R_pkt per-packet mean = 31.634 Mbps
weighted delay = 1.045 ms
```

Top eligible snapshots：

| iter | score | goodput | TB err | R_pkt | R_pkt per-packet mean | weighted delay |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | `1553.565` | `1540.646` | `0.10739` | `23.014` | `31.506` | `1.050 ms` |
| 23 | `1550.546` | `1539.305` | `0.11364` | `23.656` | `31.874` | `1.020 ms` |
| 11 | `1522.652` | `1514.640` | `0.11430` | `22.770` | `31.522` | `1.065 ms` |

v11 iter20 warmup=1000 cross-seed evaluator stats：

```text
run_dir = training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s43_v11_rate_tail_iter20_argmax_greedy_coop_boundary_r500_t4000_w1000
checkpoint = training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v11_rate_tail_greedy_i24_r512_from_v10i24_thread1/candidate_checkpoints/ppo_actor_iter_0020.pt
```

| seed | goodput | TB err | R_pkt | R_pkt per-packet mean | weighted delay | expiry | blank |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 41 | `1541.367` | `0.11060` | `23.218` | `31.855` | `1.043 ms` | `0.00000` | `0.13487` |
| 42 | `1535.468` | `0.11392` | `7.152` | `27.608` | `4.223 ms` | `0.00000` | `0.08267` |
| 43 | `1498.675` | `0.11158` | `1.880` | `25.923` | `15.560 ms` | `0.00186` | `0.10569` |
| mean | `1525.170` | `0.11203` | `10.750` | `28.462` | `6.942 ms` | `0.00062` | `0.10774` |

v10/v11 preliminary mean 对比：

| checkpoint | goodput | TB err | R_pkt | R_pkt per-packet mean | weighted delay | expiry |
|---|---:|---:|---:|---:|---:|---:|
| v10 iter20 | `1526.088` | `0.11249` | `10.452` | `28.206` | `6.836 ms` | `0.00046` |
| v10 iter24 | `1528.135` | `0.11094` | `10.789` | `28.411` | `6.778 ms` | `0.00058` |
| v11 iter20 | `1525.170` | `0.11203` | `10.750` | `28.462` | `6.942 ms` | `0.00062` |

v11 结论：

- v11 证明了 seed41 内的 rate-tail gate 可以筛出 `R_pkt > 23 Mbps`、weighted delay 约 `1 ms` 的候选。
- 但 gate 只来自 seed41 rollout，不能阻止 seed42/43 的跨 seed service-time collapse。seed42/43 的 goodput 和 TB err 仍然“看起来可用”，但 `R_pkt`、逐包速率和 weighted delay 明确失败。
- 这进一步确认：rate 指标不是 goodput 的重复指标，而是 goodput 接近容量上限时最敏感的第二主线。goodput 衡量成功 bit 总量，`R_pkt` 近似衡量 `delivered_bits / packet_system_time`；当包被拖长服务时，goodput 可以保持，`R_pkt` 会快速下降。
- v11 不进入 41-50 正式扩展。下一版应把跨 seed 本身纳入训练和 checkpoint 选择，而不是只在单 seed 上加更硬的 gate。

下一轮算法建议：

```text
v12 target = multi-seed rate-constrained PPO

train seeds:
  start with 41,42,43 list-cycle or short per-iteration seed rotation

checkpoint gate:
  seed-wise goodput/TB/expiry still required
  seed-wise R_pkt lower bound
  seed-wise R_pkt per-packet mean lower bound
  seed-wise weighted delay / p95 delay upper bound
  add UE-macro packet-rate lower bound when the eval wrapper exports it directly

reward/selector:
  keep goodput as first mainline
  keep BLER/expiry as reliability guard
  add stronger anti-service-clumping term based on packet system time / HOL age
  tune tail bias against seed42/43, not only seed41
```

v12 multi-seed rate gate 已完成一轮长训：

```text
run_dir = training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s43_v12_multiseed_rate_greedy_i24_r512_from_v11i20_thread1
init_checkpoint = v11 iter20 candidate
train seeds = 41,42,43 list-cycle
best_metric = multi_seed_rate_guarded_goodput
completed_iterations = 24 / 24
candidate_checkpoints = none
```

v12 训练窗口没有 warmup，512-step rollout 被冷启动影响。训练 CSV 显示 seed42/43 的 `R_pkt` 比 v11 deployment 好看，但 warmup=1000 部署验证没有兑现：

| checkpoint | seed | goodput | TB err | R_pkt | R_pkt per-packet mean | weighted delay | expiry |
|---|---:|---:|---:|---:|---:|---:|---:|
| v12 last deploy | 41 | `1538.494` | `0.11263` | `22.463` | `31.247` | `1.081 ms` | `0.00000` |
| v12 last deploy | 42 | `1539.888` | `0.11162` | `7.272` | `27.576` | `4.050 ms` | `0.00000` |
| v12 last deploy | 43 | `1501.638` | `0.11179` | `1.807` | `25.843` | `15.904 ms` | `0.00106` |

v12 结论：multi-seed checkpoint gate 是对的，但训练统计口径必须和部署一致；否则训练会优化冷启动阶段的 rate，部署稳态仍然保留 seed42/43 的 service-time collapse。

v12b steady-rate pilot 已完成：

```text
run_dir = training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s43_v12b_steady_rate_greedy_i6_r512_from_v12last_pilot
init_checkpoint = v12 ppo_actor_last.pt
rollout_warmup_steps = 1000
rollout_steps = 512
episode_horizon = 1512
completed_iterations = 6 / 6
candidate_checkpoints = none
```

v12b pilot 指标：

| iter | seed | goodput | TB err | R_pkt | R_pkt per-packet mean | weighted delay | hard fails |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41 | `1518.675` | `0.11897` | `22.480` | `30.994` | `1.077 ms` | `2` |
| 2 | 42 | `1538.174` | `0.11432` | `6.140` | `27.623` | `4.453 ms` | `3` |
| 3 | 43 | `1480.457` | `0.11138` | `2.745` | `26.299` | `9.750 ms` | `5` |
| 4 | 41 | `1524.524` | `0.11225` | `22.149` | `30.781` | `1.096 ms` | `5` |
| 5 | 42 | `1533.568` | `0.11819` | `6.801` | `27.782` | `3.980 ms` | `5` |
| 6 | 43 | `1500.554` | `0.10970` | `2.708` | `25.906` | `10.043 ms` | `4` |

v12b 结论：

- `--rollout-warmup-steps` 是必要修正：训练窗口现在看见的是部署稳态，不再被冷启动 rate 误导。
- 但 v12b 不应直接长训。seed42 第二轮只从 `6.140 -> 6.801 Mbps`，仍低于 v11 deployment `7.152 Mbps`；seed43 从 `2.745 -> 2.708 Mbps`，没有改善。
- 当前瓶颈不像单纯 checkpoint gate 问题，更像 selector/reward 对尾部 UE 的服务节奏压力不够。v12b 的 tail/rescue bias 只在 selector 上生效，`stageb_blankaware` reward 仍没有直接奖励 tail potential delta / rescue credit。

v12c next：

```text
preset:
  goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_greedy_fast
  goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_sample_fast

kept from v12b:
  rollout_warmup_steps = 1000
  rollout_steps = 512
  train seeds = 41,42,43 list-cycle
  best_metric = multi_seed_rate_guarded_goodput

new pressure:
  keep packet-rate / per-packet-rate / delay in stageb_blankaware
  add tail_potential_delta and expiry_potential_delta shaping to stageb_blankaware
  add rescue_credit / rescue_miss shaping
  increase budgeted tail/rescue logit bias for steady seed42/43

pilot rule:
  run 6 iters first, do not start long train unless seed42 R_pkt exceeds v11 and seed43 R_pkt/delay both improve
```

v12c steady-tail-rescue pilot 结果：

```text
run_dir = training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s43_v12c_steady_tail_rescue_greedy_i6_r512_from_v12last_pilot
init_checkpoint = v12 ppo_actor_last.pt
rollout_warmup_steps = 1000
rollout_steps = 512
status = completed 6 / 6; no eligible multi-seed checkpoint
candidate_checkpoints = none
```

| iter | seed | goodput | TB err | R_pkt | R_pkt per-packet mean | weighted delay | tail bias selected | rescue bias selected | hard fails |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41 | `1529.651` | `0.11328` | `22.429` | `31.090` | `1.082 ms` | `0.0210` | `0.0029` | `1` |
| 2 | 42 | `1533.063` | `0.11734` | `6.435` | `27.604` | `4.117 ms` | `0.0312` | `0.0022` | `2` |
| 3 | 43 | `1484.229` | `0.11117` | `2.483` | `25.772` | `10.374 ms` | `0.0590` | `0.0037` | `4` |
| 4 | 41 | `1525.855` | `0.11482` | `22.708` | `31.206` | `1.069 ms` | `0.0214` | `0.0029` | `4` |
| 5 | 42 | `1531.791` | `0.11844` | `7.387` | `27.606` | `3.571 ms` | `0.0294` | `0.0021` | `4` |
| 6 | 43 | `1483.508` | `0.11105` | `2.520` | `25.793` | `10.358 ms` | `0.0600` | `0.0038` | `3` |

v12c 结论：

- 强化 tail/rescue selector bias 没有损坏 seed41：seed41 reliability 比 v12b 首轮更好，`R_pkt` 基本持平。
- seed42 第二轮是积极信号：`R_pkt=7.387 Mbps`，高于 v12b 第二轮 `6.801 Mbps`，也高于 v11 deployment `7.152 Mbps`；但 TB err `0.11844` 接近上限。
- seed43 仍是 blocker：第二轮 `R_pkt=2.520 Mbps`、weighted delay `10.358 ms`，没有达到 v12b 第一轮 `2.745 Mbps / 9.750 ms`，虽然优于 v11 的 `1.880 Mbps / 15.560 ms`。
- multi-seed window 最差 seed 仍被 seed43 限制，`best_multi_seed_min_packet_rate_mbps=2.483`、`best_multi_seed_max_packet_delay_ms=10.374`，因此无候选 checkpoint。
- tail/rescue bias 的 selected mean 上升，但绝对幅度仍低，说明当前特征/目标对“哪个 UE 的包服务时间被拉长”没有足够直接的 credit assignment。

下一步不直接长训 v12c。应把 seed42 的正向变化保留为参考，但主线改成显式 rate/fairness 约束：

```text
v13 direction:
  export or reconstruct UE-macro packet effective service rate in online rollout
  add seed-wise UE-macro packet-rate lower bound to multi_seed_rate_guarded_goodput
  add per-UE low-rate / high-system-time auxiliary target, not only PRG tail bias
  consider selector target based on delivered_bits / packet_system_time delta by UE
  keep v12b warmup alignment as mandatory
```

v13 UE-rate-proxy implementation and pilot:

```text
preset = goodput_delay_rate_simple_risk_v13_ue_rate_proxy_greedy_fast
new rollout fields =
  rollout_ue_macro_rate_proxy_mbps_mean
  rollout_ue_p10_rate_proxy_mbps_mean
  rollout_ue_min_rate_proxy_mbps_mean
  rollout_ue_rate_proxy_active_count_mean
proxy definition = demand-active recent100 actual_goodput_bytes_mean * 8 / 0.5ms
```

说明：当前 online bridge 暂未导出真实 per-UE packet system time，因此 v13 先实现明确命名的 `ue_*_rate_proxy`，避免把 goodput-rate proxy 误认为 `traffic.ue_macro_packet_effective_service_rate_mbps`。

smoke：

```text
run_dir = training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v13_ue_rate_proxy_greedy_i1_w32_r32_smoke
init_checkpoint = v12 ppo_actor_last.pt
rollout_warmup_steps = 32
rollout_steps = 32
status = passed; code path and CSV fields verified
```

smoke 读数：`ueRateM=15.91 Mbps`、`ueRateP10=1.92 Mbps`。该窗口太短，只用于验证字段和 reward/gate 路径。

seed43 steady pilot：

```text
run_dir = training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s43_v13_ue_rate_proxy_greedy_i3_r512_from_v12last_pilot
init_checkpoint = v12 ppo_actor_last.pt
rollout_warmup_steps = 1000
rollout_steps = 512
iterations = 3
best_multi_seed_window_iters = 3
best_multi_seed_min_covered_seeds = 1
```

| iter | goodput | TB err | R_pkt | R_pkt per-packet mean | weighted delay | UE macro proxy | UE p10 proxy | UE min proxy | hard fails |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `1495.849` | `0.11410` | `5.286` | `29.500` | `4.932 ms` | `41.346` | `35.729` | `32.523` | `2` |
| 2 | `1519.003` | `0.11330` | `2.157` | `29.785` | `12.573 ms` | `41.829` | `35.923` | `32.881` | `3` |
| 3 | `1540.356` | `0.09894` | `1.235` | `30.017` | `21.241 ms` | `42.653` | `37.174` | `32.895` | `2` |

v13 结论：

- UE goodput-rate proxy 能反映“每 UE 解码 bit 供给”是否均衡，但不能替代 packet service-time。seed43 中 UE proxy 全部通过 gate，真实 `R_pkt` 却从 `5.286 -> 1.235 Mbps`，weighted delay 从 `4.932 -> 21.241 ms`。
- 这说明当前 seed43 的失败不是简单 per-UE decoded-throughput starvation，而是 packet residence time / backlog age 被拉长。goodput 和 UE proxy 可以同时好看，packet service-rate 仍然塌陷。
- 因此 v13 不应直接长训。下一轮应优先把 native `traffic.ue_macro_packet_effective_service_rate_mbps` / per-UE packet system-time 暴露到 online rollout，或构造基于 packet_system_time delta 的 UE-local target；在此之前，UE goodput proxy 只能作为辅助诊断，不应作为主 gate。

v14 true packet-rate implementation:

```text
preset =
  goodput_delay_rate_simple_risk_v14_true_packet_rate_greedy_fast
  goodput_delay_rate_simple_risk_v14_true_packet_rate_sample_fast

protocol =
  OnlineProtocol version 10 -> 11
  add STATE_FLAG_HAS_ACTUAL_UE_PACKET_DIAGNOSTICS

bridge export =
  actual_ue_packet_delivered_packets
  actual_ue_packet_pending_packets
  actual_ue_packet_delivered_bits
  actual_ue_packet_system_time_ms
  actual_ue_packet_service_rate_mbps
  actual_ue_packet_service_rate_per_packet_mean_mbps

Python reconstruction =
  per-UE cumulative delivered_bits/system_time deltas
  recent100_packet_rate_mbps = delta_bits / delta_system_time_ms / 1000
  demand-active macro / p10 / min packet rate
```

该实现把 v13 的 `ue_*_rate_proxy` 保留下来做诊断，但 v14 gate/reward 使用新的 true packet fields：

```text
rollout_ue_macro_packet_rate_mbps_mean
rollout_ue_p10_packet_rate_mbps_mean
rollout_ue_min_packet_rate_mbps_mean
rollout_ue_macro_packet_delay_ms_mean
rollout_ue_p90_packet_delay_ms_mean
rollout_actual_ue_packet_diagnostics_available_mean
```

业务理想上界仍是：

```text
packet_size = 3000 B
TTI = 0.5 ms
arrival = 1 packet / TTI / UE
per-UE packet service-rate upper bound = 3000 * 8 / 0.5ms = 48 Mbps
36UE aggregate offered-rate upper bound = 1728 Mbps
```

v14 smoke：

```text
run_dir = training/stageb_hgraph/runs/online_ppo_3cell_v14_true_packet_rate_smoke
build_method = phase4
topology_seed = 41
rollout_warmup_steps = 16
rollout_steps = 16
iterations = 1
status = passed; C++ rebuild, protocol v11 payload, reward/gate CSV columns verified
```

smoke 关键读数：

| metric | value |
|---|---:|
| packet diagnostics available | `1.0` |
| UE macro true packet rate | `47.361 Mbps` |
| UE p10 true packet rate | `45.517 Mbps` |
| UE min true packet rate | `42.888 Mbps` |
| UE macro packet delay | `0.508 ms` |
| UE p90 packet delay | `0.529 ms` |
| aggregate packet rate | `46.256 Mbps` |
| per-packet mean packet rate | `47.068 Mbps` |

注意：这个 smoke 只覆盖短窗口和 seed41，不能代表性能收益；它的价值是证明 native per-UE packet residence-time 口径已经进入 online training loop。下一步应先用典型 blocker seed43 做 3-6 iter pilot，再决定是否启动长训：

```text
init_checkpoint = v12 ppo_actor_last.pt 或 v11/v12c 中更稳的 rate checkpoint
seed = 43 first
rollout_warmup_steps = 1000
rollout_steps = 512
best_multi_seed_window_iters = 1
best_multi_seed_min_covered_seeds = 1

pilot pass condition:
  R_pkt and weighted delay both improve vs v12b/v12c seed43
  UE p10 true packet rate rises without TB err / expiry regression
  proxy rate may be ignored unless it contradicts true packet-rate diagnostics
```

seed43 true packet-rate pilots:

```text
common:
  init_checkpoint = training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s43_v12_multiseed_rate_greedy_i24_r512_from_v11i20_thread1/ppo_actor_last.pt
  topology_seed = 43
  rollout_warmup_steps = 1000
  rollout_steps = 512
  episode_horizon = 1512
  tti = 2000
```

| version | core change | iter used | goodput | TB err | R_pkt | R_pkt per-packet | weighted delay | UE macro pkt-rate | UE p10 pkt-rate | hard fails | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v14 | true packet-rate only in reward/gate | 2 | `1525.886` | `0.1096` | `2.181` | `30.306` | `12.216 ms` | `22.443` | `0.327` | `4` | reject: UE p10 collapses after update |
| v14b | stronger true packet tail reward/gate | 2 | `1535.316` | `0.1065` | `2.796` | `30.890` | `9.425 ms` | `23.006` | `0.411` | `4` | reject: delay improves, p10 still collapses |
| v15 | append true packet-tail UE features into candidate selector | 2 | `1519.912` | `0.1090` | `2.305` | `29.548` | `11.101 ms` | `21.563` | `0.349` | `4` | reject: feature visible, but 2 Mbps selector target too soft |
| v16 | strict packet-tail selector target: 12 Mbps, heavier first-service/bias | 2 | `1538.276` | `0.1052` | `3.573` | `30.310` | `7.081 ms` | `22.208` | `0.563` | `2` | not enough for long train; direction improves but p10 still below gate |
| v17 | explicit per-UE packet-tail quota, top8 * 1 PRG | 2 | `1542.374` | `0.1035` | `3.769` | `30.467` | `6.820 ms` | `22.431` | `0.592` | `2` | reject: quota active, but one PRG per tail UE is too thin |
| v18 | thick packet-tail quota, top5 * 3 PRG + completion bonus | 2 | `1527.247` | `0.1073` | `2.958` | `28.547` | `8.419 ms` | `20.703` | `0.696` | `3` | reject: thick quota helps first iter, but not stable enough |
| v19 | coverage packet-tail quota, top8 * 2 PRG | 2 | `1529.153` | `0.1089` | `4.222` | `30.535` | `6.362 ms` | `22.361` | `1.735` | `1` | continue: UE p10/delay gate holds in 2-iter pilot |
| v19-last | v19 last checkpoint, 3-iter stability check | 3 | `1545.881` | `0.0983` | `3.921` | `29.874` | `6.866 ms` | `21.788` | `1.218` | `1` | candidate: 3/3 p10 > 1 Mbps, but hard gate/cost still need cleanup |

v15-v18 diagnostic:

- v15 confirmed the true packet-tail features are wired into deployment selector: `tailBiasSel` rose to `0.175` and `rescueBiasSel` to `0.048` at iter2. But UE p10 packet-rate still dropped to `0.349 Mbps`, so reward+bias is not sufficient.
- v16 lifted selector packet target from `2 Mbps` to `12 Mbps` and increased first-service pressure. It improved seed43 iter2 goodput, BLER, `R_pkt`, and weighted delay versus v15, but UE p10 packet-rate still fell from `1.415 -> 0.563 Mbps`.
- v17 moved from soft bias to explicit tail quota. It selected about `5.19` tail-quota PRGs per step and kept goodput/TB strong at iter2, but per-UE p10 packet-rate still fell to `0.592 Mbps`. Per-UE diagnostics showed the worst cell-1 UEs had `recent100_served_tti=100/100`, so the failure is thin service, not total starvation.
- v18 tested the thin-service hypothesis by reserving about `13.8 -> 15.0` quota PRGs per step. Iter1 was excellent for the rate second-mainline: UE p10 packet-rate `2.701 Mbps`, UE min `1.231 Mbps`, packet delay `3.012 ms`, and `R_pkt=8.326`. Iter2 still fell to UE p10 `0.696 Mbps` and delay `8.419 ms`, with worst UEs again `served100=100/100` but only `0.55-0.81 Mbps` recent packet-rate and `160-314 KB` backlog.
- v19 lowered the tail-quota threshold and spread roughly the same quota budget over more UEs (`active_ue_count ~= 7.8`, selected quota `~15`). This fixed the v18 top5 coverage issue: the 2-iter pilot ended at UE p10 `1.735 Mbps`, and the from-last 3-iter stability run kept UE p10 at `4.821 -> 1.552 -> 1.218 Mbps`.
- v19 still has two caveats before cross-seed deployment: one extreme UE can remain very poor (`recent100_packet_rate=0.154 Mbps`, HOL `159 ms` at from-last iter3), and the guarded best gate remains `0/1` because at least the later system-level `R_pkt` rows are below the `7.2 Mbps` hard gate.
- v19 cost is high but better than v18: from-last stability iterations were about `568 / 566 / 563 s`, versus v18 up to `698 s`. Any long train should keep this in mind or optimize the deterministic selector/logp replay path.

Current decision:

```text
do not cross seed yet
v19 coverage-quota is the first candidate that keeps seed43 UE p10 packet-rate > 1 Mbps for multiple consecutive pilots
next long seed43 run should start from v19-last, while separately tightening the remaining R_pkt/min-UE tail risk and reducing runtime
```

v19 implemented preset:

```text
selector:
  preset = goodput_delay_rate_simple_risk_v19_packet_tail_quota_coverage_greedy_fast
  tail quota min_score = 0.35
  tail quota topk_ues = 8
  tail quota prgs_per_ue = 2
  tail quota bonus/order_bonus = 7.0 / 5.0
  completion bonus = 0.50
  usage penalty = 0.18

training/eval gate:
  keep goodput as primary
  require seed43 UE p10 packet-rate to stay above 1 Mbps in 2-3 iter pilot before long train
  require weighted packet delay <= 8 ms and TB err <= 0.120
  before cross-seed, require a longer seed43 window to avoid the remaining min-UE collapse

performance:
  keep rescue burst deterministic/inference-side if possible, so PPO logp does not repeatedly replay the heavy selector
  cache per-candidate packet-tail and completion scores for the rollout step
  avoid high-cost completion bonus in the PPO update unless it is part of the deployed deterministic action
```

正式 41-50 验证命令模板；只有当 41-43 preliminary 过 rate/tail gate 后再扩展：

```bash
./cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh \
  --build-method skip \
  --seed-list 41,42,43,44,45,46,47,48,49,50 \
  --checkpoint-path training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v10_rate2_greedy_i24_r512_from_v9cbest_thread1/candidate_checkpoints/ppo_actor_iter_0024.pt \
  --model-label hgraph_v10_rate2_iter24_rate_w1000 \
  --baseline-mean-csv output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50/RAYLEIGH/rrq_vs_pfq_compare_mean.csv \
  --decode-mode argmax \
  --eval-episode-horizon 4000 \
  --eval-stats-warmup-steps 1000 \
  --parallel-seeds 1 \
  --compare-output-dir training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s50_v10_rate2_iter24_argmax_greedy \
  --tti 4000 \
  --topology-scenario 3cell \
  --cell-radius 500 \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --baseline-scheduler pfq \
  --ue-placement coop_boundary \
  --ue-voronoi-clip 1 \
  --exec-mode both \
  --sim-env CUMAC_CELL_ASSOC_MODE=strongest \
  --sim-env CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1
```

### 1.2 Feature Sufficiency Audit

当前 HGraph 输入特征基本够开始训练，不需要先扩 schema。关键输入已经覆盖三类信号：

| 层级 | 当前字段 | 训练价值 |
|---|---|---|
| Cell `[C,5]` | load bytes, active UE count, mean WB SINR, mean avg rate, TB err rate | cell-level load and reliability context |
| UE `[U,12]` | buffer, avg rate, WB SINR, CQI, RI, last TB err, new data, stale slots, HOL delay, TTL slack, recent sched ratio, recent goodput deficit | 队列压力、尾包风险、公平 debt、链路状态 |
| PRG `[C*P,12]` | top1 SINR, top2 gap, previous assignment, reuse ratio, neighbor top1 stats, same-PRG conflict, ICI proxy, postEq p10/mean, backlog-weighted expected goodput, low-SINR HOL/TTL weight | per-PRG 机会、跨 cell 同 PRG 冲突、低 SINR 尾包风险 |
| UE-PRG edge | per-PRG quality, urgency, debt, risk, optional PFQ anchor, optional 24-dim candidate risk features | 把 UE 队列状态和 PRG 物理机会连起来 |

足够开始训练的原因：

- `CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1` 后，`E_UP` 使用真实 per-UE/per-PRG postEq SINR，而不是 shared WB pool。
- `coop_boundary + strongest` 下 PFQ 已经接近满 PRG，增益空间主要来自更低 BLER 的 PRG/UE 选择和更好的尾包服务时机；这些信号都已在 UE、PRG、UE-PRG edge 中出现。
- `recent_scheduled_ratio` 和 `recent_goodput_deficit_norm` 能抑制只追高 SINR UE 的偏置。
- `low_sinr_hol_ttl_weight` 专门给低 SINR 且 HOL/TTL 压力大的 UE 暴露 rescue 机会。

仍需 sanity 验证的点：

- `candidate_mode` 必须是 `post_eq_topk` 或由日志证明 `post_eq_sinr` 已导出；如果退化到 `shared_cell_pool`，本轮训练不算正式结果。
- `candidate_coverage_ratio` 要接近 `1.0`，否则模型看不到足够 UE-PRG 可行动作。
- `prg_post_eq_p10_sinr_db_mean`、`prg_post_eq_mean_sinr_db_mean`、`prg_backlog_weighted_expected_goodput_mbps_mean` 必须在 TTI 间有动态范围。
- HGraph wrappers 当前不显式暴露 `--cell-assoc-mode` 和 `--shadow-seed`；主脚本默认 `cell_assoc_mode=strongest` 且 shadow 为空。每次 sanity 和 PPO 都必须从 simulator log 核查：

```text
cell_assoc_mode=strongest
shadow_seed=legacy_shared_rng
ue_placement=coop_boundary
cell_radius_m=500 site_spacing_m=1000
```

### 1.3 Algorithm Structure Audit

当前结构总体合理，建议继续走 HGraphPPO candidate action head，而不是回到 slot-only 或 PFQ mimic：

| 模块 | 当前判断 |
|---|---|
| graph schema | `Cell / UE / PRG` 三类节点正好对应调度决策的层次；`PRG-PRG` 同 local index 边能表达跨 cell reuse risk |
| message passing | 2-3 层异构 message layer 足够覆盖 3cell 一跳邻区；先不需要更深网络 |
| action head | `candidate` head 更适合当前问题，因为最终动作是 `(cell, PRG) -> UE/blank`，slot-only 会把关键 per-PRG SINR 信号绕远 |
| budgeted selector | 仍然需要，避免同一 UE 被过量 PRG 选择后被 native demand cap 裁掉 |
| PFQ anchor | 正式 PPO 先关掉 `include_pfq_score_anchor=0`；PFQ 可作为 baseline 和 teacher diagnostic，不作为部署 score anchor |
| reward | 不再用 completed-packet delay 做主评分；goodput 是主目标，BLER/expiry/backlog 是 hard guard，delay 是解释指标 |

当前结构的主要风险：

- boundary PFQ 只比 uniform PFQ 低 `11.0 Mbps`，提升目标很窄，单 seed 噪声可能掩盖真实收益。
- `packet_delay_mean/p95` 只统计完成包，救回 old packet 会自然抬高 completed-packet delay；过强 delay penalty 会鼓励“不救老包”。
- 如果 candidate width 太窄，模型可能看不到边缘 UE 的 rescue 机会；第一版建议 `all_demand_ues` 或 `topk_diversity + extra`，不要只取 top4 high-SINR。

## 1.4 HGraphPPO Training Plan

### Step 0: baseline is locked

不再扩 ISD500，也不再补固定 `shadow_seed=41`。当前两个 native baseline 已足够进入 HGraph sanity：

```text
output/stageB_rrq_pfq_multiseed_compare_uniform_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50
output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50
```

### Step 1: graph sanity, seed41

目标是确认 action mask、postEq SINR、candidate coverage 和 runtime labels 都正确。

```bash
CUDA_VISIBLE_DEVICES=1 ./cuMAC/scripts/run_stageB_hgraph_online_sanity.sh \
  --build-method skip \
  --tti 4000 \
  --topology-scenario 3cell \
  --cell-radius 500 \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 41 \
  --topology-seed-mode fixed \
  --baseline-scheduler pfq \
  --ue-placement coop_boundary \
  --ue-voronoi-clip 1 \
  --episode-horizon 512 \
  --max-steps 512 \
  --episodes 1 \
  --action-source random_legal \
  --random-blank-prob 0.25 \
  --sim-env CUMAC_CELL_ASSOC_MODE=strongest \
  --sim-env CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1 \
  --out-dir training/stageb_hgraph/runs/online_sanity_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41
```

Pass 条件：

```text
mask_ok_all = true
candidate_mode = post_eq_topk
post_eq_layer_dim > 0
mean_prg_candidate_edges > 0
candidate coverage near full demand set
sim log confirms strongest + legacy_shared_rng + r500 + coop_boundary
```

### Step 2: short PPO smoke, seed41

先跑 `30` iter，不 warm-start，不 PFQ anchor。目标是验证 reward、candidate decode、budgeted selector 和 native accept path 没有结构性问题。

```bash
CUDA_VISIBLE_DEVICES=1 ./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --ppo-preset goodput_delay_rate_simple_risk_v8_seq_budgeted_sample_fast \
  --build-method skip \
  --tti 4000 \
  --topology-scenario 3cell \
  --cell-radius 500 \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 41 \
  --topology-seed-mode fixed \
  --baseline-scheduler pfq \
  --ue-placement coop_boundary \
  --ue-voronoi-clip 1 \
  --online-persistent 1 \
  --episode-horizon 512 \
  --rollout-steps 512 \
  --iterations 30 \
  --action-head-mode candidate \
  --candidate-mode all_demand_ues \
  --max-prg-candidates 12 \
  --include-pfq-score-anchor 0 \
  --include-candidate-risk-features 1 \
  --candidate-seq-state-dim 8 \
  --candidate-budgeted-select-mode sample \
  --hidden-dim 192 \
  --message-layers 3 \
  --device cuda \
  --lr 1.0e-4 \
  --actor-lr 8.0e-5 \
  --critic-lr 3.0e-4 \
  --entropy-coef 0.004 \
  --target-kl 0.05 \
  --update-epochs 2 \
  --minibatch-size 256 \
  --best-metric goodput \
  --best-packet-delay-penalty-mbps 0 \
  --candidate-checkpoint-interval 5 \
  --candidate-checkpoint-require-gate 0 \
  --early-stop-patience 0 \
  --sim-env CUDA_VISIBLE_DEVICES=0 \
  --sim-env CUMAC_CELL_ASSOC_MODE=strongest \
  --sim-env CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1 \
  --out-dir training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_i30_smoke
```

Smoke 观察项：

```text
rollout_candidate_coverage_ratio_mean
rollout_native_reject_* and python_preflight_* reject counts
rollout_goodput_mbps_mean trend
rollout_tb_err_rate_mean
rollout_expiry_drop_rate_mean
rollout_candidate_budgeted_changed_count_mean
candidate_success_prob_mae
candidate_argmax/deploy keep ratio
```

### Step 3: seed41 deployment gate

Smoke 中每 `5` iter 保存的候选 checkpoint 只形成 shortlist。真正选择 checkpoint 必须用 `4000 TTI` greedy deployment：

```text
decode_mode = argmax
candidate_budgeted_select_mode = greedy
episode_horizon / eval horizon = 4000
same scenario labels: coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41
```

Pass gate：

```text
goodput > boundary PFQ seed41
TB BLER <= boundary PFQ seed41 + 0.2 pp
expiry <= boundary PFQ seed41 + 0.05 pp
UE goodput p5 >= boundary PFQ seed41 - 0.2 Mbps
native reject diagnostics clean
```

### Step 4: extend training only after seed41 passes

如果 seed41 deployment gate 通过，再跑：

```text
iterations = 100
topology_seed_mode = list_cycle
seed_list = 41,42,43,44,45
candidate_checkpoint_interval = 10
```

然后只评估入围 checkpoint：

```text
seeds 41-50
same native baseline directory:
  output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50
primary comparison:
  traffic.goodput_mbps, global_tb_bler, expiry_drop_rate, ue_goodput_p5, packet p95
```

### Step 5: only then move toward 7cell

只有当 `3cell coop_boundary ISD1000 strongest` 的 10-seed mean 稳定超过 PFQ，才进入 7cell sanity/PPO。7cell 仍采用后文的 rate=0.85 locked baseline 做长期迁移目标。

## 1.5 2026-06-23 HGraph PPO v9/v9b Findings

本轮重点从单纯 goodput reward 转向 reliability-gated goodput，并确认 deployment decoder 的 PRG 使用策略是关键变量。

### v9/v9b reward 结论

v9 新增 reliability-gated goodput：

```text
tb_err_target = 0.116
tb_err_goodput_gate_weight = 12
best_metric = reliability_guarded_goodput
```

v9 seed41 `i15/r128` 训练态去掉冷启动后 goodput 已经略高于 PFQ，但 `t4000` deployment gate 仍失败：

| checkpoint | train/eval 口径 | goodput Mbps | vs PFQ | TB BLER |
|---|---|---:|---:|---:|
| v9 best4 train post-cold | r128 rollout | `1521.319` | `+0.783` | `0.11979` |
| v9 best4 deployment | t4000 argmax/greedy | `1514.988` | `-5.547` | `0.12166` |

v9b 将 reliability gate 收紧：

```text
tb_err_target = 0.115
tb_err_goodput_gate_weight = 20
best_hard_tb_err_max = 0.117
rollout_steps = 512
decode path = greedy/budgeted
```

v9b `i20/r512` 从 v9 best warm-start 后，训练后段明显收敛，不再是早期偶然 best：

| 口径 | goodput Mbps | vs PFQ | TB err | hard gate |
|---|---:|---:|---:|---:|
| all 20 iter | `1521.562` | `+1.027` | `0.11992` | `6/20` |
| post-cold iter2-20 | `1526.493` | `+5.957` | `0.11750` | `6/19` |
| last10 | `1529.406` | `+8.871` | `0.11543` | `5/10` |
| last5 | `1532.685` | `+12.149` | `0.11272` | `5/5` |
| best iter16 | `1540.739` | `+20.203` | `0.11552` | pass |

但原始 v9b deployment 使用 checkpoint 内的 `candidate_decode_demand_slack_bytes=9000` 与 `safe_fill_max_per_cell=1`，导致 fill 约 `0.983`，仍有 high-risk marginal PRG 被发送。t4000 结果：

| deployment | goodput Mbps | vs PFQ | global TB BLER | vs PFQ BLER | fill | expiry |
|---|---:|---:|---:|---:|---:|---:|
| v9b best/iter16 | `1511.768` | `-8.768` | `0.12245` | `+0.466 pp` | `0.9830` | `0` |
| v9b last/iter20 | `1514.967` | `-5.569` | `0.12116` | `+0.338 pp` | `0.9838` | `0` |

结论：v9b reward 能学到后段可靠策略，但原始 deployment decoder 过度追求高 fill，full-run BLER 仍高于 PFQ。

### PRG usage sweep

在不改模型的前提下，用 v9b `ppo_actor_last.pt` 做 deployment-aligned decoder sweep：

| deployment override | goodput Mbps | vs PFQ | global TB BLER | vs PFQ BLER | UE goodput p5 | fill | p95 delay |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | `1514.967` | `-5.569` | `0.12116` | `+0.338 pp` | `39.346` | `0.9838` | `6.5 ms` |
| `slack=3000`, `safe_fill=0` | `1521.625` | `+1.090` | `0.11840` | `+0.062 pp` | `39.594` | `0.9322` | `4.0 ms` |
| `slack=0`, `safe_fill=0` | `1526.139` | `+5.604` | `0.11733` | `-0.045 pp` | `39.657` | `0.8699` | `2.5 ms` |

关键发现：

- 少用 PRG 显著降低 BLER，并且没有牺牲 goodput；`slack=0/safe_fill=0` 反而比 PFQ 多 `+5.604 Mbps`。
- 原始 fill `0.984` 中存在 harmful packing：边际 PRG 增加 served throughput，但更多 TB error 抵消了 goodput。
- v9c 不应继续单纯增加 BLER penalty，而应把 deployment decoder 和 reward 都对齐到 PRG efficiency / harmful packing。

### v9c plan

v9c 固定如下 deployment 对齐设置：

```text
candidate_decode_demand_slack_bytes = 0
safe_fill_max_per_cell = 0
candidate_budgeted_select_mode = greedy for deployment
```

Reward 侧新增显式权重：

```text
prg_efficiency_weight > v9b implicit value
harmful_packing_weight > v9b implicit value
tb_err_goodput_gate_weight 保持 v9b，不继续上调
```

目标是让训练本身学到 `slack=0/safe_fill=0` 的少发高效行为，而不是只依赖 eval-time override。

### v9c implementation and results

v9c 已落地为 preset：

```text
goodput_delay_rate_simple_risk_v9c_prg_efficiency_greedy_fast
```

代码侧变化：

| 文件 | 变化 |
|---|---|
| `training/stageb_hgraph/reward_utils.py` | `compute_stageb_blankaware_reward` 新增显式 `prg_efficiency_weight` / `harmful_packing_weight`，默认 `-1` 保持旧 implicit 系数兼容 |
| `training/stageb_hgraph/online_ppo.py` | CLI/reward_cfg 透传上述两个权重 |
| `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh` | 新增 v9c preset，并固定 `candidate_decode_demand_slack_bytes=0`、`safe_fill_max_per_cell=0` |

v9c preset 关键值：

```text
best_metric = reliability_guarded_goodput
best_tb_err_target = 0.115
best_hard_tb_err_max = 0.117
tb_err_goodput_gate_weight = 20
prg_efficiency_weight = 0.45
harmful_packing_weight = 24
candidate_decode_demand_slack_bytes = 0
safe_fill_max_per_cell = 0
candidate_budgeted_select_mode = greedy
rollout_steps = 512
```

训练 run：

```text
training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_prgeff_greedy_i12_r512_from_v9blast_thread1
```

从 v9b last warm-start 跑 `i12/r512`，summary 选择 `iter4` 为 threshold/absolute best：

| 口径 | goodput Mbps | TB err | fill | hard gate |
|---|---:|---:|---:|---:|
| all 12 iter | `1529.03` | `0.11613` | `0.87482` | `9/12` |
| post-cold iter2-12 | `1536.27` | `0.11272` | `0.87767` | `9/11` |
| last5 | `1533.31` | `0.11267` | `0.87629` | `4/5` |
| last3 | `1533.83` | `0.11091` | `0.88083` | `3/3` |
| best iter4 | `1547.36` | `0.10709` | `0.88056` | pass |

训练观察：

- `safeFill=0` 全程生效，fill 从 v9b 原始 deployment 的 `~0.983` 降到 `~0.87-0.89`。
- 少用 PRG 后 BLER 明显下降，post-cold 平均 TB err `0.11272`，已经低于 PFQ seed41 的 `0.11778`。
- 仍有两个贴边失败点：iter3 `0.11892`、iter9 `0.11814`。iter9 fill 只有 `0.8663` 但 BLER 仍高，说明总 PRG 少用不是充分条件，局部 harmful packing/risky UE-PRG 组合仍需约束。
- reward 后半程下降但 hard metrics 仍好，说明 `harmful_packing_weight=24` 可能略重；不过 deployment gate 先验证，不急于继续 sweep。

v9c seed41 `t4000` deployment gate 使用同训练一致的 decoder：

```text
decode_mode = argmax
candidate_budgeted_select_mode = greedy
candidate_decode_demand_slack_bytes = 0
safe_fill_max_per_cell = 0
```

| checkpoint | goodput Mbps | vs PFQ | global TB BLER | vs PFQ BLER | served Mbps | UE goodput p5 | PRG util/fill | expiry | p95 delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v9c best/iter4 | `1531.197` | `+10.662` | `0.11387` | `-0.391 pp` | `1730.643` | `40.022` | `0.8673` | `0` | `2.5 ms` |
| v9c last/iter12 | `1530.625` | `+10.089` | `0.11602` | `-0.176 pp` | `1730.634` | `40.318` | `0.8696` | `0` | `2.5 ms` |

对应 PFQ seed41 baseline：

```text
goodput = 1520.536 Mbps
global TB BLER = 0.11778
served = 1722.060 Mbps
UE goodput p5 = 39.524 Mbps
PRG utilization = 0.9852
expiry = 0.0001726
p95 delay = 25.65 ms
```

结论：

- v9c 已通过 seed41 deployment gate，且不是靠更高 PRG utilization；它用约 `0.867-0.870` 的 PRG utilization 得到更高 goodput、更低 BLER 和更低 delay。
- `best/iter4` 比 `last/iter12` 更干净：goodput 高 `+0.573 Mbps`，global BLER 低 `0.215 pp`，建议以 `ppo_actor_best.pt` / `ppo_actor_threshold_best.pt` 作为后续 seed41-50 gate 的候选。
- 对“少用 PRG 是否会降低 BLER”的回答是肯定的，但要带限定：全局少用 PRG 能去掉 high-risk marginal transmissions，明显降低 BLER；低 fill 仍可能因局部 harmful packing 越线，所以下一步应继续优化 PRG efficiency / harmful-packing shaping，而不是单纯再加 BLER penalty。

### v9c seed41-50 multi-seed deployment gate

`ppo_actor_best.pt` 与 `ppo_actor_threshold_best.pt` 文件哈希不同，但 actor tensor hash 相同，且同为 `iter4` / `best_score=1545.974`；因此本次 multi-seed gate 用 `ppo_actor_best.pt` 代表二者。

Gate 口径：

```text
seeds = 41..50
decode_mode = argmax
candidate_budgeted_select_mode = greedy
candidate_decode_demand_slack_bytes = 0
safe_fill_max_per_cell = 0
horizon = 4000 TTI
```

输出目录：

```text
training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50_v9c_i12r512_best_threshold_argmax_greedy_coop_boundary_r500_t4000_w0
```

10-seed mean 对比：

| 指标 | RRQ mean | PFQ mean | HGraph v9c mean | HGraph vs PFQ |
|---|---:|---:|---:|---:|
| goodput Mbps | `1513.633` | `1520.536` | `1518.248` | `-2.287` |
| served Mbps | `1707.590` | `1722.060` | `1715.513` | `-6.547` |
| global TB BLER | `11.555%` | `11.778%` | `11.377%` | `-0.401 pp` |
| eval mean TB err | n/a | n/a | `11.553%` | n/a |
| expiry drop rate | `0.609%` | `0.017%` | `0.285%` | `+0.268 pp` |
| PRG utilization | `93.019%` | `98.523%` | `87.973%` | `-10.550 pp` |
| packet p95 delay | `56.078 ms` | `25.650 ms` | `37.650 ms` | `+12.000 ms` |
| UE goodput p10 | `39.263` | `40.132` | `40.220` | `+0.088` |
| scheduled ratio p5 | `63.217%` | `55.777%` | `43.846%` | `-11.931 pp` |

去掉前 `1000 TTI` 后，按 HGraph `checkpoint_eval.log` 的等长 `500 TTI`
窗口重聚合：

| 指标 | HGraph full 4000 | HGraph post-1000 | PFQ full 4000 | post-1000 HGraph vs PFQ |
|---|---:|---:|---:|---:|
| mean goodput Mbps | `1518.248` | `1530.558` | `1520.536` | `+10.022` |
| mean TB err | `11.553%` | `10.996%` | n/a | n/a |
| mean fill / PRG util | `87.973%` | `88.226%` | `98.523%` | `-10.298 pp` |
| mean blank PRG / TTI | `6.134` | `6.005` | n/a | n/a |

说明：PFQ baseline 现有产物只有整段 `TRAFFIC_GOODPUT` / KPI 汇总，没有同粒度的
post-1000 窗口日志；因此上表的 PFQ 对照仍是 full-run `1520.536 Mbps`。在这个口径下，
HGraph 去掉冷启动后 goodput 已超过 PFQ `+10.022 Mbps`，也比自身 full-run 高
`+12.310 Mbps`。按 seed 看，post-1000 HGraph 对 PFQ full-run 是 `7/10` 胜出，
仍落后的 seed 是 `43 (-5.207 Mbps)`、`45 (-17.704 Mbps)`、`48 (-10.055 Mbps)`；
这三个 seed 继续对应 multi-seed gate 的低服务/expiry 风险，而不是单纯 BLER 风险。

Per-seed eval summary:

| seed | goodput Mbps | TB err | fill | expiry | blank PRG | diagnosis |
|---:|---:|---:|---:|---:|---:|---|
| 41 | `1531.197` | `0.11512` | `0.8673` | `0` | `6.77` | pass, seed41 was optimistic |
| 42 | `1521.132` | `0.12178` | `0.9140` | `0` | `4.39` | high fill / high BLER |
| 43 | `1492.835` | `0.11574` | `0.8920` | `0.00105` | `5.51` | low goodput + expiry |
| 44 | `1530.458` | `0.11475` | `0.9179` | `0` | `4.19` | pass |
| 45 | `1506.428` | `0.10370` | `0.7036` | `0.00656` | `15.12` | over-blank / under-fill |
| 46 | `1539.995` | `0.10707` | `0.9588` | `0` | `2.10` | pass despite high fill |
| 47 | `1511.978` | `0.12543` | `0.9278` | `0` | `3.68` | high fill / high BLER |
| 48 | `1507.733` | `0.11309` | `0.8510` | `0.00313` | `7.60` | conservative but low service / expiry |
| 49 | `1524.216` | `0.11573` | `0.8512` | `0` | `7.59` | borderline pass |
| 50 | `1516.508` | `0.12287` | `0.9137` | `0` | `4.40` | high BLER |

结论：

- v9c `best/threshold_best` 未通过 seed41-50 multi-seed gate：HGraph 平均 goodput 仍低于 PFQ `2.287 Mbps`，expiry 和 p95 delay 也比 PFQ 差。
- 但 HGraph 平均 BLER 低于 PFQ，说明失败不是“可靠性不够”这一种；失败 seed 分成两类：`42/47/50` 是 high-fill harmful packing / BLER 越线，`45/48` 是 over-blank / low service / expiry，`43` 是中等 fill 但 goodput/expiry 失败。
- 因此不应把下一步理解为“继续少用 PRG 降 BLER”。少用 PRG 可以降掉一部分高风险传输，但过度 blank 会直接损害 goodput、expiry 和 scheduled-ratio tail。
- 小 sweep 先不全铺 `18/20/24 x 0.35/0.45`。首选诊断点是 `harmful_packing_weight=20, prg_efficiency_weight=0.45`，从当前 best warm-start，并用 `seed41-45 list_cycle` 做短多 seed fine-tune：目标是缓解 `45/48` 这类过空，同时观察 `42` 的 BLER 是否恶化。若 BLER 恶化，再回到 `harmful=24` 并尝试更强的 service/efficiency shaping；若 BLER 稳且 goodput 上升，再试 `harmful=18`。

#### v9c h20/p045 r512 fine-tune 负结果

尝试目录：

```text
training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41s45_v9c_h20_p045_greedy_i15_r512_from_v9cbest_thread1
```

配置要点：

```text
init_checkpoint = v9c ppo_actor_best.pt
seeds = 41,42,43,44,45 list_cycle
iterations = 15 planned, stopped after iter6
rollout_steps = 512
harmful_packing_weight = 20
prg_efficiency_weight = 0.45
tb_err_goodput_gate_weight = 20
best_hard_tb_err_max = 0.117
decode/deploy = argmax + greedy, demand_slack=0, safe_fill=0
```

前 6 个 iter：

| iter | seed | goodput Mbps | TB err | fill | blank PRG | pkt delay | KL | hard gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41 | `1448.119` | `15.260%` | `83.923%` | `8.20` | `1.01 ms` | `2.040` | fail |
| 2 | 42 | `1426.169` | `16.184%` | `86.795%` | `6.73` | `2.05 ms` | `1.441` | fail |
| 3 | 43 | `1413.054` | `15.329%` | `88.304%` | `5.96` | `2.62 ms` | `2.023` | fail |
| 4 | 44 | `1451.866` | `15.376%` | `89.227%` | `5.49` | `1.21 ms` | `2.047` | fail |
| 5 | 45 | `1415.125` | `13.761%` | `69.531%` | `15.54` | `4.81 ms` | `2.252` | fail |
| 6 | 41 | `1443.263` | `15.970%` | `84.911%` | `7.70` | `1.05 ms` | `2.365` | fail |

汇总：

```text
mean goodput = 1432.933 Mbps
mean TB err = 15.313%
max approx KL = 2.365
hard_gate_pass = 0/6
candidate_checkpoint_saved = iter5 only, but hard gate failed
```

结论：

- 这条 run 在第二个 seed cycle 回到 seed41 后仍未恢复，说明不是单个 seed 噪声，而是 warm-start 后 PPO update 过猛。
- `target_kl=0.05` 没能有效限制一次 update 的策略漂移，实际 `approx_kl` 在 `1.44-2.36`，已经远超可接受范围。
- 不应拿 iter5 candidate 做 deployment gate；它 goodput 低、TB err 高、seed45 还出现明显 over-blank。
- 后续如果继续 fine-tune，应先修 PPO 步长而不是继续调 reward：建议 `actor_lr <= 2e-5`、`update_epochs=1`、更小 `clip_eps` 或显式 rollback/skip update on KL，且保留 v9c best 作为当前 deployment 候选。

#### v9d plan: seed45 single-seed stable fine-tune

当前仍有希望的证据：

- 去掉前 `1000 TTI` 后，HGraph seed41-50 mean goodput 为 `1530.558 Mbps`，已高于 PFQ full-run `+10.022 Mbps`。
- HGraph multi-seed mean BLER 低于 PFQ，说明 learned scheduler 确实学到了更可靠的 PRG/UE 选择。
- 主要失败不是单一方向：有 `42/47/50` 的 high-fill/high-BLER，也有 `45/48` 的 over-blank/low-service；继续只加 BLER penalty 会把后者推得更差。

提升空间按优先级：

| 空间 | 现象 | 设计动作 |
|---|---|---|
| PPO 稳定性 | warm-start fine-tune `approx_kl=1.44-2.36`，策略被 1-2 个 update 拉坏 | 使用 `actor_lr=5e-6`、`clip_eps=0.05`、`target_kl=0.01`、`update_epochs=1`，并新增 `target_kl_min_updates` 让 KL early stop 可在第 1 个 minibatch 后触发 |
| cold-start / 前 1000 TTI | full-run 低于 post-1000 `12.310 Mbps` | 训练先看固定 seed 的稳定收敛，再做 deployment eval 时同时报告 full-run 与 post-1000；不把多 seed 全量评估提前 |
| over-blank / low service | seed45 full-run fill `70.36%`、blank `15.12`、expiry `0.656%` | 保持 `safe_fill=0`，但在 selector/ranking 层降低 usage penalty、增加 first-service/tail/rescue 小偏置，鼓励模型自己选安全 PRG |
| harmful packing | seed42/47/50 高 fill 下 BLER 越线 | 保留 reliability-gated goodput、fail-risk ranking、harmful packing 惩罚，但 v9d 不继续加大 BLER penalty |

代表 seed 选择：

```text
primary single-seed training seed = 45
reason = 最典型且最严重的 over-blank/under-fill/expiry 失败点；
         若 seed45 稳定收敛，说明策略有能力补足服务而不靠 safe-fill。
secondary checks after seed45 convergence = seed43/48 for low-service generalization,
                                           seed42/47/50 for high-BLER regression.
```

代码/配置改动：

| 文件 | 改动 |
|---|---|
| `training/stageb_hgraph/online_ppo.py` | 新增 `--target-kl-min-updates`，默认 `4` 保持旧行为；v9d 可设为 `1` |
| `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh` | 新增 `goodput_delay_rate_simple_risk_v9d_stable_sample_fast` / `greedy_fast` preset |

v9d preset 关键值：

```text
LR / actor_lr = 5e-6
critic_lr = 1e-4
clip_eps = 0.05
target_kl = 0.01
target_kl_min_updates = 1
update_epochs = 1
minibatch_size = 128
best_metric = reliability_guarded_goodput
best_hard_tb_err_max = 0.117
candidate_decode_demand_slack_bytes = 0
safe_fill_max_per_cell = 0
candidate_budgeted_usage_penalty = 0.30
candidate_budgeted_first_service_bonus = 0.03
candidate_budgeted_tail_bias_coef = 0.08
candidate_budgeted_rescue_bias_coef = 0.06
prg_efficiency_weight = 0.35
harmful_packing_weight = 20
tb_err_goodput_gate_weight = 18
```

训练判据：

```text
stage 1, seed45 train:
  max approx KL <= 0.03 preferred, hard stop if repeated > 0.05
  rollout goodput should recover toward >= 1500 Mbps
  fill should move from ~70% toward 80-88%, without TB err > 11.7%
  hard gate pass or stable candidate before doing any multi-seed gate

stage 2, seed45 deployment eval:
  run ppo_actor_best.pt / threshold_best.pt / last.pt if present
  greedy argmax, demand_slack=0, safe_fill=0
  report full-run and post-1000 goodput

stage 3:
  only after seed45 is stable, test seed43/48, then seed42/47/50, finally seed41-50.
```

##### v9d seed45 result and diagnosis

seed45 v9d 先验证了 PPO 稳定性修复是有效的，但没有解决 low-service 本体问题。

训练窗口：

| run | iter | goodput Mbps | TB err | fill | blank PRG | hard gate | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| v9d sample from v9c best | 1 | `1400.236` | `0.14578` | `0.7125` | `14.66` | fail | 冷启动/迁移窗口失败 |
| v9d sample from v9c best | 2 | `1515.991` | `0.09584` | `0.7241` | `14.07` | pass | BLER 很健康，但 fill 仍低 |
| v9d sample from v9c best | 3 | `1494.595` | `0.10326` | `0.6939` | `15.61` | pass | 过度 blank 仍在 |
| v9d sample continuation | 1 | `1420.970` | `0.13724` | `0.6925` | `15.68` | fail | 低 fill + 高初始 BLER |
| v9d sample continuation | 2 | `1524.356` | `0.09929` | `0.6975` | `15.43` | pass | 窗口 goodput 可过线，但靠低 BLER，不靠高服务 |
| v9d greedy continuation | 1 | `1419.582` | `0.13760` | `0.6890` | `15.86` | fail | greedy PPO KL/策略漂移仍不稳 |

v9d sample-continuation `ppo_actor_last.pt` 的 deployment-aligned seed45 eval：

```text
decode = argmax
selector = greedy
candidate_decode_demand_slack_bytes = 0
safe_fill_max_per_cell = 0
eval_dir = training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s45_v9d_samplecont2_last_argmax_greedy_coop_boundary_r500_t4000_w0
```

| 口径 | goodput Mbps | vs PFQ seed45 full `1539.279` | TB err | PRG util / fill | expiry | blank PRG | packet delay mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| full `t4000` | `1499.119` | `-40.160` | `0.10372` | `0.6980` | `0.753%` | `15.40` | `17.06 ms` |
| post-1000 windows | `1510.421` | `-28.858` | `~0.0986` | `~0.6996` | n/a | `~15.32` | n/a |

诊断：

- v9d 已把 reliability 拉到安全区：deployment TB err `0.10372`，明显低于 hard gate `0.117`。
- 但 fill 只有 `~0.70`，blank 约 `15.4 PRG/TTI`，比 v9c 多 seed good seed 的 `~0.87-0.89` 少太多。
- 因此当前不是“继续少用 PRG 降 BLER”的阶段，而是要在 BLER 合格时把可服务 PRG 拉回来。
- greedy PPO fine-tune 仍不稳；代表 seed 训练继续用 `sample`，deployment 再用 `argmax + greedy` gate。

##### v9e gated-fill design

v9e 的目标是修 seed45 这种 over-blank / low-service failure，而不是继续单纯加 BLER penalty。

代码/配置改动：

| 文件 | 改动 |
|---|---|
| `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh` | 新增 `goodput_delay_rate_simple_risk_v9e_gated_fill_sample_fast` / `greedy_fast` |

v9e 复用现有 `stageb_blankaware` 的 `util_floor_weight`，含义是：

```text
if TB err <= 0.110:
  fully penalize fill below 0.82
elif 0.110 < TB err < 0.118:
  linearly fade the fill penalty out
else:
  do not push more fill
```

关键参数：

```text
stable PPO:
  actor_lr = 2e-6
  clip_eps = 0.05
  target_kl = 0.01
  target_kl_min_updates = 1
  update_epochs = 1
  minibatch_size = 128

deployment alignment:
  candidate_decode_demand_slack_bytes = 0
  safe_fill_max_per_cell = 0

service/fill recovery:
  util_floor_weight = 12
  util_floor_target = 0.82
  util_floor_tb_err_guard = 0.110
  util_floor_tb_err_softness = 0.008
  candidate_budgeted_usage_penalty = 0.20
  candidate_budgeted_first_service_bonus = 0.05
  tail_bias_coef = 0.10
  rescue_bias_coef = 0.08

reliability:
  best_hard_tb_err_max = 0.117
  tb_err_goodput_gate_weight = 18
  harmful_packing_weight = 18
  prg_efficiency_weight = 0.30
```

第一版 v9e 使用 `actor_lr=5e-6`、`minibatch=128`、较强 decode aux，iter1 出现 `approx_kl=0.200`，超过稳定 fine-tune 目标。第二版尝试 `minibatch=512`，但 512 大 minibatch 内的 argmax-decode auxiliary 过慢，第一轮迟迟无法落盘。因此当前 v9e stable 改为 reward-level gated-fill：`actor_lr=2e-6`、`candidate_argmax_decode_coef=0`、`candidate_decode_goodput_coef=0.03`，保留低步长和 gated util floor，先验证 reward 方向。

训练计划：

```text
1. seed45, sample PPO, r512, i12-i20, warm-start from v9d sample-continuation last.
2. 若 fill 从 ~0.70 拉到 >=0.80 且 TB err <=0.117，再跑 seed45 t4000 deployment gate。
3. 若 seed45 deployment full/post-1000 接近或超过 PFQ，再补 seed43/48 low-service check。
4. high-BLER seed42/47/50 只在 seed45 稳定后验证，避免把方向又调回过度保守。
```

##### v9e/v9f/v9g seed45 diagnostic results

本轮按 seed45 作为代表 seed 做了三步诊断。结论是：智能算法仍有希望，但 seed45 的瓶颈已经从 reward 标量转移到 action/decode 层的 demand/headroom cap。

| version | 关键改动 | iter | goodput Mbps | TB err | fill | blank PRG | no-headroom | KL | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| v9e reward-only | gated util floor, `slack=0` | 2 | `1515.878` | `0.10280` | `0.7004` | `15.28` | n/a | `0.0326` | BLER 稳，但 fill 基本不动 |
| v9e reward-only | gated util floor, `slack=0` | 4 | `1530.059` | `0.10127` | `0.7102` | `14.78` | n/a | `0.0418` | seed45 训练窗口可接近 PFQ，但仍低 fill |
| v9e reward-only | gated util floor, `slack=0` | 5 | `1513.177` | `0.10274` | `0.6979` | `15.41` | n/a | `0.0279` | scalar reward 无法稳定拉升服务率 |
| v9f service-fill | positive-only blank-target anchor, `slack=0` | 2 | `1516.197` | `0.09962` | `0.7144` | `14.56` | `14.56` | `0.0091` | action anchor 稳定 KL，但 no-headroom 仍等于 blank |
| v9f service-fill | positive-only blank-target anchor, `slack=0` | 4 | `1521.578` | `0.10437` | `0.7084` | `14.87` | `14.87` | `0.0114` | blank logit 已压低，仍被 headroom cap 留空 |
| v9g slack1500 | v9f + `candidate_decode_demand_slack_bytes=1500` | 1 | `1396.872` | `0.14927` | `0.7289` | `13.83` | `13.73` | `0.0162` | slack 能解除 cap，但冷启动 BLER 高 |
| v9g slack1500 | v9f + `candidate_decode_demand_slack_bytes=1500` | 2 | `1513.689` | `0.09763` | `0.7467` | `12.92` | `12.92` | `0.0114` | reliability 可恢复，fill 明显高于 v9f |
| v9g slack1500 | v9f + `candidate_decode_demand_slack_bytes=1500` | 4 | `1518.420` | `0.10180` | `0.7361` | `13.46` | `13.46` | `0.0325` | 方向对，但额外 PRG 尚未转化为足够 goodput |

v9g `ppo_actor_best.pt` 做了两次 seed45 deployment check。第一版沿用 `slack=1500`：

```text
decode = argmax
selector = greedy
candidate_decode_demand_slack_bytes = 1500
safe_fill_max_per_cell = 0
stats_warmup_steps = 1000
```

| TTI | goodput Mbps | TB err | fill | blank PRG | 结论 |
|---:|---:|---:|---:|---:|---|
| 500 | `1405.103` | `0.1462` | `0.7331` | `13.61` | cold-start reliability fail |
| 1000 | `1498.171` | `0.1025` | `0.7413` | `13.19` | reliability recovers, goodput still low |
| 1500 | `1474.545` | `0.1033` | `0.7115` | `14.71` | warm section still far below PFQ |

该 deployment 明显不会通过 seed45 PFQ full-run `1539.279 Mbps`，因此在 1500 TTI 后提前停止，不进入 multi-seed。

随后用同一个 checkpoint 改成 `candidate_decode_demand_slack_bytes=3000`、`safe_fill_max_per_cell=0` 做完整 `t4000` probe。这个设置更接近 native PFQ 的 demand slack，也验证了 `slack=0/1500` 过度保守：

| 窗口 | goodput Mbps | TB err | expiry | fill | blank PRG | 结论 |
|---|---:|---:|---:|---:|---:|---|
| full 4000 | `1496.148` | `0.10624` | `0.00711` | `0.7603` | `12.23` | fill 明显高于 slack1500，但 full goodput 仍低于 PFQ seed45 |
| post-1000 | `1510.192` | `0.10030` | `0.00948` | `0.7579` | `12.34` | 去冷启动后接近 boundary PFQ mean `1520.536`，但仍未过 seed45 PFQ |

主实验 traffic/global 口径：

```text
traffic.goodput_mbps = 1496.148
global_tb_bler = 0.10516
traffic.expiry_drop_rate = 1.6871%
ue_goodput_p5/p10/p50 = 30.94 / 36.37 / 42.89 Mbps
```

相对 seed45 PFQ full-run `1539.279 Mbps` 仍低 `43.13 Mbps`。tail UE 仍集中在 cell1 边缘/低服务用户，说明 slack 解除 cap 后，新增 PRG 还没有被稳定转成 packet goodput。

##### v9h/v9h-b seed45 representative training

基于 slack3000 probe，先做了一个代表 seed fine-tune，而不是直接 multi-seed。

v9h 强 service-fill 版：

```text
init = v9g best
candidate_decode_demand_slack_bytes = 3000
safe_fill_max_per_cell = 0
service_fill_weight = 0.75
tail_bias_coef = 0.16
rescue_bias_coef = 0.12
actor_lr = 1e-6
```

v9h-b 温和版：

```text
init = v9g best
candidate_decode_demand_slack_bytes = 3000
safe_fill_max_per_cell = 0
service_fill_weight = 0.30
tail_bias_coef = 0.08
rescue_bias_coef = 0.04
actor_lr = 2e-7
```

训练结果：

| version | iter | goodput Mbps | TB err | expiry | fill | blank PRG | no-headroom | tail/rescue selected | backlog p90 | KL | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v9h strong fill | 2 | `1492.850` | `0.10759` | `0.00000` | `0.7861` | `10.91` | `10.91` | `0.074/0.003` | `0.52 MB` | `0.0597` | fill 提升，但 KL 偏大 |
| v9h strong fill | 4 | `1498.703` | `0.10735` | `0.00026` | `0.7961` | `10.40` | `10.40` | `0.105/0.026` | `1.10 MB` | `0.1080` | 高 fill 没有换成 goodput |
| v9h strong fill | 5 | `1495.491` | `0.10418` | `0.00185` | `0.7852` | `10.96` | `10.96` | `0.129/0.047` | `1.15 MB` | `0.1172` | tail/rescue 继续升，停止该 run |
| v9h-b moderate | 2 | `1497.425` | `0.10198` | `0.00000` | `0.7774` | `11.35` | `11.35` | `0.074/0.002` | `0.52 MB` | `0.0138` | KL 稳定，但 goodput 未恢复 |
| v9h-b moderate | 5 | `1495.284` | `0.10771` | `0.00369` | `0.7784` | `11.30` | `11.30` | `0.125/0.038` | `1.15 MB` | `0.0138` | backlog/expiry 开始恶化 |
| v9h-b moderate | 6 | `1497.619` | `0.10966` | `0.00694` | `0.7765` | `11.40` | `11.40` | `0.134/0.049` | `1.17 MB` | `0.0253` | 未达到继续 gate 条件 |

关键发现：

- v9e reward-only 修复不了 over-blank。即使 util-floor gate 有效，policy 仍停在 `~0.70` fill。
- v9f 的 action-level service-fill anchor 生效：`candidate_blank_target_service_fill_ratio ~= 0.64-0.67`，blank decision logit p50 约 `-1.55`，说明 actor 已经倾向非 blank；但实际 blank/no-headroom 仍高。
- v9f/v9g 中 `candidate_budgeted_no_headroom_count` 基本等于 final blank count，说明瓶颈不是 actor 显式 blank，而是 budgeted decode 的 demand/headroom cap。
- v9g 的 `slack=1500` 能把 fill 从 `~0.70` 拉到 `0.73-0.75`，`slack=3000` 能进一步拉到 `~0.76`，且 post-1000 BLER 可恢复到 `~0.100`。因此 `candidate_decode_demand_slack_bytes=0` 不是正确 deployment alignment；`safe_fill=0` 仍应保留，但 demand slack 至少要允许 native PFQ 同级的 `3000 bytes`。
- “少用 PRG 降 BLER”只解决可靠性，不解决主目标。v9d/v9f 的低 fill 版本 BLER 更低，但 goodput/tail 明显不足；slack3000 的 BLER 仍可控，真正短板是新增 PRG 的边际 packet goodput。
- v9h/v9h-b 表明继续加 service-fill、tail/rescue 或 util floor 会把 PRG 推给低边际 UE：fill 上去了，goodput 没上去，backlog p90 和 expiry 反而升高。
- 因此当前不应进入 multi-seed gate。seed45 还没有稳定超过 PFQ seed45 full-run `1539.279 Mbps`，也没有超过 v9g+slack3000 deployment probe。

下一步算法方向：

```text
v10:
  stop increasing reward penalties/fill anchors
  keep deployment decode at:
    candidate_decode_demand_slack_bytes = 3000
    safe_fill_max_per_cell = 0
  add marginal packet-goodput / packet-completion utility into budgeted decode:
    value(PRG -> UE) should depend on whether the extra PRG helps finish a 3000B packet
    discount extra PRGs that only increase low-MCS backlog service time without completing packets
    keep reliability/risk and harmful-packing penalties as guards
  expose or compute per-candidate features:
    estimated bytes per PRG
    remaining bytes/headroom to next packet completion
    marginal completion probability/value
    per-UE residual demand after already selected PRGs
  train seed45 first from v9g best with r512

train gate:
  require fill ~= 0.75-0.78, not higher by itself
  require TB err <= 0.117 for at least 3 eligible windows
  require rollout goodput trend >= 1515 and no backlog p90 growth before t4000 deployment
  require t4000 seed45 post-1000 goodput > v9g+slack3000 `1510.192` before multi-seed
  no seed43/48 or seed41-50 until seed45 passes this local gate
```

##### v10 implementation note

已实现一个默认关闭的 budgeted selector 参数组：

```text
--candidate-budgeted-completion-bonus
--candidate-budgeted-completion-progress-weight
--candidate-budgeted-completion-max-score
```

实现位置：

| 文件 | 改动 |
|---|---|
| `training/stageb_hgraph/graph_ops.py` | 新增 `_candidate_packet_completion_scores()`，并在 `select_budgeted_candidate_actions()` 中把 completion/progress utility 加进 candidate score |
| `training/stageb_hgraph/online_ppo.py` | PPO rollout、log-prob mirror、update auxiliary decode 都透传 completion 参数，并把 completion candidate/selected diagnostics 写入 rollout summary |
| `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh` | CLI/default/echo/python args 透传 completion 参数 |

v10 seed45 short run 使用：

```text
init = v9g best
candidate_decode_demand_slack_bytes = 3000
safe_fill_max_per_cell = 0
completion_bonus = 0.35
completion_progress_weight = 0.20
tail_bias_coef = 0.04
rescue_bias_coef = 0.02
actor_lr = 2e-7
selector = greedy
```

前两轮 rollout：

| iter | goodput Mbps | TB err | fill | blank PRG | no-headroom | completion selected | completion candidate | KL | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `1389.976` | `0.14907` | `0.7695` | `11.76` | `11.66` | `0.855` | `0.997` | `5.046` | cold-start，高 KL |
| 2 | `1505.016` | `0.10245` | `0.7752` | `11.46` | `11.46` | `0.872` | `1.004` | `5.267` | 比 v9h-b iter2 高约 `+7.6 Mbps`，但 KL 仍失控 |

该 run 已停止，不进入长训。当前判断：

- completion utility 本身能运行，并且 diagnostics 可见。
- 第 2 轮 goodput 改善说明 completion utility 有信号，但 `greedy` budgeted selector 在 PPO update 中用 deterministic log-prob；加入 completion bonus 后，轻微参数变化会导致 selected candidate 翻转，KL 被记成极大值。
- 下一步不要直接 greedy PPO。更稳妥的顺序是：
  1. 先做 selector-only deployment probe：v9g checkpoint + completion bonus sweep `0.10/0.20/0.35`，不更新 actor。
  2. 如果 probe 有收益，再用 `candidate_budgeted_select_mode=sample` 做 PPO fine-tune，或把 greedy log-prob 改为 adjusted-logit soft surrogate。
  3. eval tool 也需要透传 completion 参数后，才能做正式 t4000 deployment gate。

## 2. 7-Cell Baseline Evidence

### 2.1 已完成输出目录

| 场景 | seeds | 输出目录 |
|---|---|---|
| 7cell rate=1.0 | `41-50` | `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_7cell_s41_s50_ue84_ttl200_rbg16_svd_trafficseed_20260601_064006` |
| 7cell rate=0.8 | `41-43` | `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_7cell_s41_s43_ue84_ttl200_rbg16_svd_rate0p8_trafficseed_20260601_075406` |
| 7cell rate=0.85 | `41-43` | `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_7cell_s41_s43_ue84_ttl200_rbg16_svd_rate0p85_trafficseed_20260601_081200` |
| 7cell rate=0.85 locked | `41-50` | `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_7cell_s41_s50_ue84_ttl200_rbg16_svd_rate0p85_trafficseed_20260601_090103` |
| 7cell rate=0.9 | `41-43` | `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_7cell_s41_s43_ue84_ttl200_rbg16_svd_rate0p9_trafficseed_20260601_080659` |

### 2.2 Rate Sweep Summary

| arrival | seed count | PFQ served delta | PFQ goodput delta | PFQ expiry | PFQ p90 | PFQ p95 | PFQ PRG util | 判断 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0.8` | 3 | `+0.14%` | `-0.64%` | `0.815%` | `25.33 ms` | `65.00 ms` | `97.17%` | 偏轻 |
| `0.85` | 3 | `+0.42%` | `-0.06%` | `1.433%` | `48.50 ms` | `109.83 ms` | `97.90%` | 主训练点 |
| `0.85` | 10 | `+0.60%` | `+0.16%` | `0.954%` | `26.55 ms` | `74.35 ms` | `97.12%` | locked baseline |
| `0.9` | 3 | `+0.41%` | `-0.19%` | `2.177%` | `68.50 ms` | `148.17 ms` | `98.46%` | 压力泛化点 |
| `1.0` | 10 | `+2.12%` | `+2.02%` | `2.657%` | `85.00 ms` | `166.70 ms` | `98.66%` | 接近过载 |

解释：

- PFQ 在 7cell 中长期保持高 PRG utilization，但 goodput 不总是高于 RRQ，说明存在 BLER/重传/风险分配空间。
- locked `arrival_rate=0.85` 的 PFQ p95 为 `74.35 ms`，TTL 为 `200 ms`，tail latency 仍有可优化空间；主对照以 `s41-s50` 10-seed 为准。
- 如果新算法只追 served throughput，很难显著超过 PFQ；真正机会在 goodput、tail delay、expiry、packet effective service rate 和 cell-edge goodput 的组合上。

### 2.3 Locked Baseline Artifacts

锁定 baseline 使用容器环境：

```bash
./cuPHY-CP/container/attach_aerial.sh
source .venv/bin/activate
```

已生成主工件：

| 工件 | 路径 |
|---|---|
| mean compare | `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_7cell_s41_s50_ue84_ttl200_rbg16_svd_rate0p85_trafficseed_20260601_090103/RAYLEIGH/rrq_vs_pfq_compare_mean.csv` |
| key metrics | `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_7cell_s41_s50_ue84_ttl200_rbg16_svd_rate0p85_trafficseed_20260601_090103/key_metrics_summary.csv` |
| tail objective | `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_7cell_s41_s50_ue84_ttl200_rbg16_svd_rate0p85_trafficseed_20260601_090103/tail_latency_throughput_weighted_summary.csv` |
| explanation | `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_7cell_s41_s50_ue84_ttl200_rbg16_svd_rate0p85_trafficseed_20260601_090103/tail_latency_throughput_explanation.txt` |
| alignment | `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_7cell_s41_s50_ue84_ttl200_rbg16_svd_rate0p85_trafficseed_20260601_090103/alignment_check.csv` |

Gate A 当前状态：

```text
seeds aligned = 10/10
PFQ goodput = 2893.345 Mbps
PFQ p90/p95 = 26.55 / 74.35 ms
PFQ expiry = 0.954%
PFQ PRG utilization = 97.12%
PFQ p5 UE goodput = 30.014 Mbps
PFQ global TB BLER = 13.991%
```

## 3. Optimization Target

2026-06-05 更新：当前 3cell sequential/budgeted 分支的首要目标是 reliable goodput 超过 PFQ/RRQ，同时不破坏 BLER 和 expiry。packet delay 继续记录，但不再作为主优化目标。

原因是 completed-packet delay 的统计对象是“成功完成包”。当策略救回更多 near-expiry / 长等待包时，成功包集合更老，mean/p95 delay 可能上升；这在 high-load TTL 场景下可能是 goodput 提升的副作用，而不是策略退化。

当前 3cell 主目标：

```text
primary:
  goodput_mbps > PFQ same-scenario baseline
  TB BLER <= reliability gate
  expiry_drop_rate <= PFQ/reference gate

secondary:
  packet_delay_mean_ms / packet_delay_p95_ms after warmup
  packet_effective_service_rate_mbps
  packet_completed_count
  backlog/unserved only as safety guards
```

7cell 长期目标仍是多目标稳定超过 PFQ：

```text
goodput_mbps >= PFQ + 0.3%
packet_delay_p95_ms <= PFQ - 15%
packet_delay_p90_ms <= PFQ - 15%
packet_effective_service_rate_mbps >= PFQ + 10%
expiry_drop_rate <= PFQ
ue_goodput_p5_mbps >= PFQ
global_tb_bler <= PFQ + 0.2 percentage point
prg_utilization_ratio >= 95%
```

主 KPI 优先级：

1. `traffic.goodput_mbps`
2. `traffic.packet_delay_p95_ms`
3. `traffic.packet_effective_service_rate_mbps`
4. `traffic.expiry_drop_rate`
5. `global_kpi.ue_goodput_p5_mbps`
6. `global_kpi.global_tb_bler`
7. `global_kpi.prg_utilization_ratio`

## 4. Algorithm: TAR-GNN-PFQ

名称：

```text
TAR-GNN-PFQ = Tail-Aware Residual GNN over PFQ
```

核心思想：

```text
final_score(cell, prg, ue) = pfq_score(cell, prg, ue) * exp(clip(rl_residual, -rmax, rmax))
```

也就是不从零替代 PFQ，而是让 GNN+RL 学 PFQ 的 residual correction。

这样做的原因：

- PFQ 已经很强，直接从零学完整调度容易训练不稳定。
- residual 学习保留 PFQ 的强 inductive bias，包括 PF、公平、需求上限和 intra-TTI decay。
- GNN 只负责 PFQ 没显式建模的内容：tail urgency、跨小区同 PRG 冲突、BLER 风险、cell-edge debt、goodput risk。

## 5. Graph State Design

### 5.1 Nodes

| 节点 | 数量 | 主要字段 |
|---|---:|---|
| Cell | `C=7` | load bytes, active UE count, mean SINR, mean avg rate, TB err rate, recent PRG utilization |
| UE | `U=84` | buffer bytes, avg rate, wb SINR, CQI, RI, tb_err_last, HOL delay, TTL slack, recent scheduled ratio, goodput deficit |
| PRG | `C*P=119` | top1 SINR, top2 gap, previous assignment, same PRG conflict, ICI proxy, neighbor top1 stats |

### 5.2 Edges

| 边 | 作用 |
|---|---|
| Cell-Cell | 表达邻区干扰、负载差异和协作关系 |
| Cell-UE | 表达 UE 服务归属和合法调度关系 |
| Cell-PRG | 表达 PRG 所属 cell |
| UE-PRG | 表达 UE 在某 PRG 上的候选质量、urgency、debt 和 BLER risk |
| PRG-PRG | 表达跨小区同 local PRG 复用冲突和干扰风险 |

### 5.3 Required New Features

在已有 HGraph 特征基础上，7cell 主线需要优先补齐以下字段：

| 特征 | 粒度 | 用途 |
|---|---|---|
| `hol_delay_ms` | UE | tail urgency |
| `ttl_slack_ms` | UE | expiry risk |
| `expiry_risk = sigmoid((HOL - 0.75*TTL)/tau)` | UE | reward 和 action scorer |
| `recent_goodput_deficit_norm` | UE | cell-edge / long-term debt |
| `estimated_prg_headroom` | UE/TTI | 防止同 UE 过量拿 PRG 后被 demand cap 裁掉 |
| `same_prg_neighbor_load` | PRG | 跨小区同 index PRG 冲突 |
| `expected_bler_proxy` | UE-PRG | goodput-aware scorer |
| `pfq_score` | UE-PRG | residual anchor |

## 6. Action And Decode Design

### 6.1 Two-Level Action

动作保持 Type-0 约束下的两层结构：

1. `slot -> UE`
   - 每个 cell 的 Type-0 slot 选择 UE。
   - 模型输出 UE priority residual。
   - PFQ/association mask 仍负责合法性。

2. `PRG -> scheduled UE`
   - 每个 `(cell, PRG)` 在本 cell 已选 UE 或 packed candidates 中选最终 UE。
   - 模型输出 PRG-UE residual 或 risk penalty。
   - decode 阶段执行 demand cap、headroom、Type-0 bitmap 合法化。

### 6.2 Residual Scoring

每个 candidate 的 deployment score：

```text
service_value = log1p(predicted_bytes)
tail_bonus = w_tail * expiry_risk + w_hol * normalize(HOL_delay)
edge_bonus = w_edge * goodput_deficit_norm
risk_penalty = w_bler * expected_bler + w_ici * same_prg_conflict + w_over * headroom_overuse

rl_residual = GNN(cell, ue, prg, edges)

score = log(pfq_score + eps)
      + service_value
      + tail_bonus
      + edge_bonus
      - risk_penalty
      + clip(rl_residual, -rmax, rmax)
```

### 6.3 Decode Guard

部署必须保持硬 guard：

- 无 backlog UE 不可选。
- 非本 cell UE 不可选。
- PRG 不满足 Type-0 bitmap 合法性不可选。
- UE 当前 TTI 已接近 headroom 时，后续 PRG 降权或 hard-mask。
- 如果所有 candidate 都非法，才允许 blank。

推荐第一版：

```text
rmax = 0.5
headroom mode = hard cap
demand slack = 9000 bytes for the physical-success PPO preset
candidate forcing = on
```

## 7. Reward Design

目标不是“把 PRG 填满”，也不是单独把 completed-packet delay 压低，而是：

```text
primary:
  high system goodput
  low expiry under a reliability gate
  high cell-edge / low-percentile UE service quality without sacrificing goodput

secondary:
  packet delay mean/p95 after warmup
  high goodput per active PRG
  low harmful same-frequency reuse
  physical success before aggressive reuse
```

当前 3cell high-load TTL 场景下，packet delay 应作为 secondary/guard 指标。若把 `packet_delay_mean_ms` 作为强主惩罚，策略可能倾向于不救老包或让尾包过期，从而让“成功完成包”的 delay 下降，但 reliable goodput 和 expiry 变差。这与项目目标相反。

因此 reward/checkpoint 的推荐优先级调整为：

```text
score first:
  goodput

hard gate:
  TB BLER
  expiry
  pathological backlog/unserved

soft/report:
  packet_delay_mean_ms
  packet_delay_p95_ms
  packet_effective_service_rate_mbps
```

因此主线 reward 改为 `Goodput-Delay-UE-Rate v1`：

```text
R_v1 =
  + goodput_term
  + active_prg_goodput_efficiency_term
  + packet_effective_service_rate_term
  + packet_completion_term
  - tb_err_term
  - expiry_drop_term
  - harmful_packing_term
  - aged_backlog_debt_term
  - unserved_demand_ue_term
  - service_gap_p10_term
```

当前代码中 `stageb_blankaware` 已支持这些分量。对应到 `--ppo-preset goodput_delay_ue_rate_v1`：

```text
reward_mode = stageb_blankaware

goodput_weight = 0.08
tb_err_weight = 32.0
expiry_weight = 60.0
ici_weight = 1.25
conflict_weight = 1.25
packet_rate_weight = 0.40
packet_completion_weight = 0.10
aged_backlog_weight = 0.35
unserved_ue_weight = 0.60
service_gap_weight = 1.20
service_rate_target = 0.60
util_floor_weight = 0.0
```

在 `reward_utils.py` 的实际尺度下约等价于：

```text
+ 0.0040 * goodput_mbps
+ 0.3125 * active_prg_goodput_efficiency
+ 0.40   * log1p(packet_effective_service_rate_mbps / 10)
+ 0.10   * log1p(packet_completed_count / 36)
- 8.00   * tb_err_rate
- 40.00  * expiry_drop_rate
- 12.50  * harmful_packing_penalty
- 0.35   * aged_backlog_debt_penalty
- 0.60   * unserved_demand_ue_ratio
- 1.20   * max(0, 0.60 - recent100_service_rate_p10)
```

解释：

- `goodput_mbps` 仍是主目标，但先降到和其他 reward component 同一可学习量级，避免纯 Mbps 数值淹没 delay/UE-rate 信号。
- `active_prg_goodput_efficiency = goodput_spectral_efficiency / active_util`，鼓励每个被使用 PRG 真的交付 goodput。
- `harmful_packing_penalty` 惩罚“调度吞吐高、成功 goodput 低、且 PRG/reuse 压力高”的行为，也就是密集复用带来的失败传输。
- 不再奖励 raw PRG utilization，也不再用 `util_floor >= 95%` 当训练 reward。低 PRG utilization 本身可以接受，只要 goodput、delay 和 UE 服务质量达标。
- 如果模型通过 blanking 减少干扰、提高 active PRG yield，并且没有造成 backlog/expiry/service gap，上述 reward 会允许这种策略。
- 如果模型过度 blank 导致 UE 长时间拿不到成功 goodput，`aged_backlog_debt`、`unserved_demand_ue` 和 `service_gap_p10` 会把它拉回来。
- p5 UE goodput 暂不直接进入在线 reward，而是在固定拓扑 eval gate 中硬检查；当前训练侧用 `recent100_service_rate_p10` 和 `recent100_debt_score_p90` 作为低分位 UE 服务质量代理。

当前 no-teacher follow-up 使用 `--ppo-preset goodput_delay_ue_rate_v2_no_teacher`：

```text
teacher_mode = none
teacher_aux_coef = 0.0
include_pfq_score_anchor = 0
candidate_pfq_anchor_logit_coef = 0.0
reward_mode = stageb_blankaware

goodput_weight = 0.10
tb_err_weight = 36.0
expiry_weight = 120.0
ici_weight = 1.15
conflict_weight = 1.15
urgent_backlog_weight = 0.75
backlog_debt_weight = 0.25
packet_rate_weight = 0.45
packet_completion_weight = 0.12
aged_backlog_weight = 0.80
unserved_ue_weight = 2.00
service_gap_weight = 8.00
service_rate_target = 0.75
util_floor_weight = 0.0
```

v2 的重点不是回到 PFQ teacher，而是让 PPO 从 post-eq SINR、PRG conflict/ICI、success/decode-goodput 辅助目标中学习，同时把 expiry、actual unserved backlog、recent100 debt p90 和 service-p10 gap 提到更高权重。
在 `reward_utils.py` 的实际尺度下，v2 约等价于：

```text
+ 0.0050 * goodput_mbps
+ 0.2875 * active_prg_goodput_efficiency
+ 0.45   * log1p(packet_effective_service_rate_mbps / 10)
+ 0.12   * log1p(packet_completed_count / 36)
- 9.00   * tb_err_rate
- 80.00  * expiry_drop_rate
- 11.50  * harmful_packing_penalty
- 0.25   * actual_unserved_backlog_debt_penalty
- 0.80   * aged_backlog_debt_penalty
- 2.00   * unserved_demand_ue_ratio
- 8.00   * max(0, 0.75 - recent100_service_rate_p10)
```

由于 v2 到 20-30 iter 后出现 tail service/debt/expiry 持续恶化，后续改用更少分量的
`--ppo-preset goodput_tailguard_v3_no_teacher`：

```text
reward_mode = stageb_tailguard_simple
teacher_mode = none
include_pfq_score_anchor = 0
candidate_pfq_anchor_logit_coef = 0.0

R_v3 =
  + 1.00 * goodput_norm
  - 1.00 * reliability_penalty
  - 1.25 * tail_pressure

goodput_norm = clip(goodput_mbps / 3000, 0, 1.25)
reliability_penalty = tb_err_rate + 6.0 * expiry_drop_rate
tail_pressure = max(
  service_gap_pressure,
  current_unserved_backlog_pressure,
  recent100_debt_p90_pressure,
  no_service_streak_p90_pressure,
  actual_unserved_ue_pressure
)
```

v3 的目标是避免 v2 中 packet-rate、packet-completion、active-efficiency、packing、fairness、多个 backlog 子项并列相加后造成训练信号稀释。
它只保留三个顶层训练信号；tail 子项用 `max` 聚合，含义是“只要最差尾部压力很高，tail penalty 就不能被其它尾部子项平均掉”。

后续 v3 到 30 iter 的观察显示：分量简化后比 v2 更稳一些，但 expiry/debt/no-service 的信用分配仍然不够直接。
因此新增 `--ppo-preset goodput_tailcredit_v4_no_teacher`：

```text
reward_mode = stageb_tailcredit_v4
teacher_mode = none
include_pfq_score_anchor = 0
candidate_pfq_anchor_logit_coef = 0.0

R_v4 =
  + 1.00 * goodput_norm
  - 1.00 * reliability_penalty
  + 1.00 * clip(tail_potential_before - tail_potential_after, -0.75, 0.75)
  - 0.15 * tail_potential_after

goodput_norm = clip(goodput_mbps / 3000, 0, 1.25)
reliability_penalty = tb_err_rate + 6.0 * expiry_drop_rate
tail_potential = 0.65 * topk_tail_risk_mean + 0.35 * max_tail_risk
```

v4 的核心不是把 tail/service 变成 hard constraint，而是把它变成可归因的软信号：

- reward 用 pre-action/post-action 的 `tail_potential` 差值，让“这一步是否降低了高风险 UE 的 tail pressure”直接进入 advantage；
- candidate budgeted decode 只加小幅 `tail_bias_coef=0.20` 的候选 logit 偏置，保留其它 UE/PRG 动作的探索空间；
- tail risk 由 backlog、TTL slack、recent service/deficit、HOL、no-service streak 等状态构成，并扣除明显高风险 PRG 的候选倾向；
- v4 仍然 no-teacher/no-anchor，避免重新绑定到 PFQ 行为。

2026-06-02 对 v4 `ppo_actor_best.pt` 做 seed41/4000 TTI 评估，确认 best 不是 last：

```text
checkpoint = online_ppo_7cell_fixed_s41_tailcredit_v4_no_teacher_no_anchor_gpu1_i100_r128_20260602_152909/ppo_actor_best.pt
completed_steps = 4000
mean_goodput_mbps = 2763.321
mean_tb_err_rate = 0.1523
mean_expiry_drop_rate = 0.00686
mean_fill_ratio = 0.9655
```

该 best checkpoint 仍然只有约 2.76G，说明 v4 的 tail credit 能稳定 debt/service，但对 expiry-near-drop 的局部信用仍不足。
因此新增 `--ppo-preset goodput_tailcredit_expiry_v5_no_teacher`：

```text
reward_mode = stageb_tailcredit_expiry_v5
teacher_mode = none
include_pfq_score_anchor = 0
best_min_iter = 12

R_v5 =
  + 1.00 * goodput_norm
  - 1.00 * (tb_err_rate + 8.0 * expiry_drop_rate)
  + 1.00 * clip(tail_potential_before - tail_potential_after, -0.75, 0.75)
  - 0.12 * tail_potential_after
  + 1.25 * clip(expiry_potential_before - expiry_potential_after, -0.75, 0.75)
  - 0.35 * expiry_potential_after

expiry_potential = 0.75 * topk_expiry_risk_mean + 0.25 * max_expiry_risk
expiry_risk 重点使用 ttl_slack_ms <= 40ms 的 near-drop 压力，并保留少量 backlog/HOL/service/streak 信息。
```

v5 的原则：

- 保持 v4 的 soft tail credit，不回 teacher，不回 PFQ anchor；
- expiry 信号从单纯全局 drop rate 变成 pre/post potential delta，让服务快过期 UE 的动作更容易拿到 advantage；
- `best_min_iter=12` 让 early 低-expiry 偶然点不能直接成为永久 best；
- candidate decode 只把 tail bias 从 `0.20` 小幅提高到 `0.22`，TTL guard 提到 `30ms`，不把 service/tail 变成硬约束。

### 7.1 v6 Rescue Credit Result

2026-06-02/03 启动 `--ppo-preset goodput_tailcredit_expiry_rescue_v6_no_teacher`：

```text
reward_mode = stageb_tailcredit_expiry_rescue_v6
teacher_mode = none
include_pfq_score_anchor = 0
device = cuda
best_min_iter = 15
candidate_budgeted_rescue_bias_coef = 0.32
candidate_rescue_aux_coef = 0.08
candidate_pairwise_rank_rescue_weight = 0.45
```

v6 的改造目标是让单步动作获得更明确的“救了哪个 UE”的信用：

- `graph_ops.py` 增加 `candidate_prg_rescue_priority_scores`，把低 recent service、低 TTL slack、高 backlog/debt/streak 的 UE 变成 UE-local rescue score；
- candidate budgeted decode 对 rescue score 加小幅 logit bias，保留探索，不做 hard service constraint；
- `reward_utils.py` 在 v5 tail/expiry potential delta 之外加入 `rescue_credit` / `rescue_miss`；
- `online_ppo.py` 增加 single-step rescue 统计、candidate rescue auxiliary target 和 pairwise rescue rank 权重；
- checkpoint 选择使用 `best_min_iter=15`，避免 iter1-3 的低 expiry 偶然点直接成为永久 best。

长训目录：

```text
training/stageb_hgraph/runs/online_ppo_7cell_fixed_s41_tailcredit_expiry_rescue_v6_no_teacher_no_anchor_gpu1_i100_r128_20260602_110742
```

训练在 iter76 触发 early stop：

```text
threshold_best_iter = 16
absolute_best_iter = 16
best_score = 2604.373
completed_iterations = 76 / 100
```

关键窗口指标：

| iter window | goodput Mbps | BLER | expiry | service-p10 | debt-p90 | rescue credit | rescue miss | selected rescue bias |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1-10` | `2708.7` | `0.1645` | `0.0000` | `0.9399` | `111.7 KB` | `0.0869` | `0.0921` | `0.0020` |
| `11-20` | `2760.1` | `0.1445` | `0.0057` | `0.8904` | `364.2 KB` | `0.3121` | `0.3052` | `0.0089` |
| `21-30` | `2784.4` | `0.1438` | `0.0197` | `0.8692` | `466.6 KB` | `0.3305` | `0.3725` | `0.0164` |
| `41-50` | `2801.2` | `0.1450` | `0.0333` | `0.9173` | `421.4 KB` | `0.3496` | `0.3380` | `0.0228` |
| `61-70` | `2823.8` | `0.1435` | `0.0360` | `0.8951` | `408.5 KB` | `0.3519` | `0.3312` | `0.0230` |
| `71-76` | `2833.4` | `0.1403` | `0.0359` | `0.8822` | `407.7 KB` | `0.3630` | `0.3141` | `0.0249` |

best checkpoint 对应 iter16，而不是早期非 eligible 的 iter3：

```text
iter3  score=2803.147, goodput=2840.241, expiry=0.0000, service-p10=0.9233, best_eligible=0
iter16 score=2604.373, goodput=2806.195, expiry=0.0058, service-p10=0.8649, best_eligible=1
iter76 score=2498.194, goodput=2821.637, expiry=0.0361, service-p10=0.8597, best_eligible=1
```

per-UE 最新诊断显示仍存在长期挨饿 UE：

| UE | cell | backlog | recent100 goodput service | recent100 service | debt score |
|---:|---:|---:|---:|---:|---:|
| `33` | `3` | `1085464 B` | `0.34` | `0.39` | `1430576 B` |
| `36` | `6` | `1113616 B` | `0.58` | `0.67` | `948736 B` |
| `56` | `2` | `984024 B` | `0.60` | `0.74` | `799412 B` |
| `71` | `2` | `958000 B` | `0.69` | `0.84` | `601543 B` |

结论：

- v6 不是合格 checkpoint，不建议评估多 seed，也不建议在当前 v6 上继续长训；
- rescue credit/miss 的方向是有信号的，后期 `rescue_credit > rescue_miss`，说明“局部救援”链路可用；
- 失败点在信用密度和动作约束强度：`candidate_rescue_positive_ratio` 长期只有约 `0.5%-0.8%`，`selected_rescue_bias` 后期也只有约 `0.02-0.03`，对具体 UE/PRG 动作的梯度太弱；
- expiry potential 后期接近 `0.89` 并饱和，单纯加全局 expiry/tail pressure 仍会回到“全局标量 reward 信号稀释”的问题；
- 下一版应保留 no-teacher/no-anchor，但把 per-UE action credit 做得更直接：例如按 actual-goodput-served 的 UE-local delta 做 advantage shaping，增加 top tail UE 的局部 counterfactual credit，或引入低分位 service dual / constrained objective，而不是继续简单加大全局 penalty。

### 7.2 GDPO-Style Multi-Reward Normalization

参考：

```text
paper = https://arxiv.org/html/2601.05242v1
repo = https://github.com/NVlabs/GDPO
```

GDPO 的核心启发不是直接把我们的 PPO 替换成 LLM/GRPO 训练，而是避免多目标 reward 在加权求和前就被量级差异淹没。
GDPO 先对每一组 reward component 单独做 group-wise normalization，再按权重聚合 advantage，最后做 batch-wise advantage normalization。

当前 `stageb_blankaware` 的实际问题类似：

```text
goodput_mbps                       数百到数千量级
active_prg_goodput_efficiency      约 10-30 量级
tb_err_rate / expiry_drop_rate     0-1 量级
harmful_packing_penalty            0-1 量级
packet_rate_bonus                  log 尺度
aged_backlog_debt_penalty          log 尺度
service_gap_p10                    0-1 量级
```

现在的实现是先按手工系数把这些项合成一个 scalar reward，再用 scalar running normalization。这样可以训练，
但不同 component 的相对信号容易被 `goodput_mbps` 或当前 batch 的拓扑波动主导。
`goodput_delay_ue_rate_v1` 先做低风险的手工量级均衡：把 `goodput_weight` 从默认 `1.0` 降到 `0.08`，
同时提高 BLER、expiry、service-gap、aged-backlog 权重。`goodput_delay_ue_rate_v2_no_teacher` 在此基础上去掉 teacher/PFQ anchor，
并进一步提高 expiry、actual backlog debt、recent100 debt p90 和 service-p10 gap。后续如果 seed41 曲线仍被单一分量主导，再实现 component-level normalization。

可迁移的 GDPO-style 改造：

```text
reward_components = [
  goodput_delta_or_goodput,
  active_prg_goodput_efficiency,
  packet_rate_bonus,
  packet_completion_bonus,
  -tb_err_rate,
  -expiry_drop_rate,
  -harmful_packing_penalty,
  -aged_backlog_debt_penalty,
  -unserved_demand_ue_ratio,
  -service_gap_p10
]

for each component k:
  z_k = normalize_k(component_k)

reward_scalar = sum_k weight_k * z_k
```

第一版不要直接做完整 GRPO/GDPO，因为当前算法是 PPO + value critic + GAE，不是同 prompt 多 rollout 的 GRPO。
推荐做一个低风险近似：

1. `log_only`：先把每个 reward component 写入 rollout metrics，不改变训练。
2. `running_component_norm`：每个 component 单独维护 running mean/std，归一化后再加权求和。
3. `rollout_component_norm`：固定拓扑下，每个 rollout 内对每个 component 做均值/方差归一化，再求和。
4. 最后仍保留 PPO 的 batch advantage normalization。

关键条件化规则：

```text
reuse / active-PRG efficiency bonus must not compensate high BLER
packet completion bonus only active if goodput success is nonzero
tail bonus should not reward repeated failed transmissions
raw PRG utilization is diagnostic, not an objective
```

这对应 GDPO 论文里的 priority/conditioned reward 思想：容易优化的目标不能越过更硬的正确性目标。
在本项目中，“物理成功”应当是 hard prerequisite，“多调度/高复用”只是成功前提下的效率奖励。

## 8. Training Plan

### Phase 0: Baseline Lock

目的：固定主评测口径。

任务：

- 用 `arrival_rate=0.85` 跑完整 `s41-s50` RRQ/PFQ 10-seed。
- 生成 `key_metrics_summary.csv`、`tail_latency_throughput_*` 和 alignment check。
- 记录 PFQ 的 seed-wise outlier，作为后续 RL 对照。

通过条件：

```text
seeds aligned = 10/10
PFQ PRG util ~= 97-98%
PFQ p95 locked at 74.35 ms
PFQ expiry locked at 0.954%
```

### Phase 1: PFQ Teacher Imitation

目的：先学合法动作、基本 PFQ 节奏和资源利用。

当前 smoke 状态：

```text
7cell online bridge sanity:
  steps = 80
  all_mask_ok = true
  max illegal cross-cell UP edges = 0
  max unschedulable UP edges = 0
  mean E_UP = 900.79
  E_PP = 714 = 119 PRG * 6 neighbors
  post_eq_layer_dim = 4

7cell PFQ teacher imitation:
  steps = 160
  teacher_mode = pfq_passthrough
  action_head_mode = slot
  mean UE acc = 0.447
  mean PRG acc = 0.220
  bad target ratio UE/PRG = 0 / 0
  predicted legal ratio UE/PRG = 1 / 1
  checkpoints written = last + best

7cell PFQ candidate-anchor imitation:
  steps = 40
  teacher_mode = pfq_passthrough
  action_head_mode = candidate
  candidate_pfq_anchor_logit_coef = 1.0
  mean UE acc = 0.635
  mean PRG acc = 0.303
  bad target ratio UE/PRG = 0 / 0
  predicted legal ratio UE/PRG = 1 / 1
  mean PFQ anchor logit = 4.745
  checkpoints written = last + best

7cell PFQ candidate-anchor warm-start:
  seeds = 41,42,43
  steps = 360
  teacher_mode = pfq_passthrough
  action_head_mode = candidate
  candidate_pfq_anchor_logit_coef = 1.0
  mean UE acc = 0.487
  mean PRG acc = 0.409
  bad target ratio UE/PRG = 0 / 0
  predicted legal ratio UE/PRG = 1 / 1
  mean PFQ anchor logit = 4.435
  best_mean_loss = 2.423
  checkpoint = training/stageb_hgraph/runs/online_imitation_7cell_pfq_s41_s43_rate0p85_candidate_anchor_warmstart/hgraph_imitation_best.pt

7cell candidate-anchor deploy smoke:
  seed = 41
  checkpoint = warm-start best
  eval horizon = 200
  decode = argmax + candidate budgeted greedy
  goodput = 2272.951 Mbps
  p90/p95 = 5.5 / 14.5 ms
  expiry = 0
  PRG utilization = 83.315%
  TB BLER = 22.834%
  status = deploy path works, performance not yet competitive
```

数据：

- PFQ online trace。
- 每个 TTI 记录 `pfq_score`、selected UE、selected PRG、headroom、decode keep/drop。

Loss：

```text
L = CE(slot_ue)
  + CE(prg_candidate)
  + BCE(kept_after_decode)
  + BCE(success_or_goodput_positive)
  + MSE(expected_goodput_proxy)
```

通过条件：

```text
PRG utilization >= 95%
invalid/drop after decode low
seed42 smoke goodput within 1% of PFQ
```

### Phase 2A: Goodput-Delay-UE-Rate PPO

目的：先把“接近 PFQ 但 BLER/服务效率仍不足”的 candidate policy 改成显式物理成功学习，
同时让 reward 对齐高 goodput、低 delay 和低分位 UE 服务质量。PRG utilization 只做诊断，不再作为训练目标。

训练节奏改为先固定拓扑收敛，再验证，再泛化：

#### Phase 2A-0: Fixed-Topology Convergence

目的：让 policy 在固定 UE 位置、固定邻区几何和相对稳定的干扰结构下先学会稳定提升。

配置：

```text
topology_seed = 41
topology_seed_mode = fixed
iterations = 50 first, then 100 if curves still improving
rollout_steps = 128 or 256
episode_horizon = rollout_steps for fast PPO iterations
eval_horizon = 4000 only after training checkpoint is selected
init_checkpoint = 7cell PFQ candidate-anchor warm-start best
ppo_preset = goodput_delay_ue_rate_v1
best_metric = latency_guarded_goodput
```

推荐训练命令模板：

```bash
./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --build-method skip \
  --ppo-preset goodput_delay_ue_rate_v1 \
  --topology-scenario 7cell \
  --total-ue-count 84 \
  --fading-mode 0 \
  --tti 4000 \
  --episode-horizon 128 \
  --rollout-steps 128 \
  --iterations 50 \
  --topology-seed-mode fixed \
  --topology-seed 41 \
  --out-dir training/stageb_hgraph/runs/online_ppo_7cell_fixed_s41_gd_ue_rate_v1_i50_r128 \
  --init-checkpoint training/stageb_hgraph/runs/online_imitation_7cell_pfq_s41_s43_rate0p85_candidate_anchor_warmstart/hgraph_imitation_best.pt \
  --teacher-aux-coef 0.15 \
  --teacher-aux-min-coef 0.03 \
  --teacher-aux-decay-iters 30 \
  --actor-lr 5e-5 \
  --critic-lr 3e-4 \
  --entropy-coef 0.005 \
  --update-epochs 2 \
  --minibatch-size 128 \
  --device cpu \
  --baseline-scheduler pfq \
  --prbs-per-group 16 \
  --precoding svd \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 0.85 \
  --packet-ttl-ms 200 \
  --kpi-tti-log 0 \
  --compare-tti 0 \
  --compact-tti-log 1 \
  --exec-mode both
```

观察曲线：

```text
reward_mean
goodput_mbps
tb_err_rate / BLER
expiry_drop_rate
packet_effective_service_rate_mbps
recent100_service_rate_p10
recent100_debt_score_p90_bytes
no_service_streak_p90
harmful_packing_penalty
active_prg_goodput_efficiency
prg_utilization_ratio
blank_ratio / final_blank_count
candidate_success_aux_loss
candidate_argmax_decode_loss
```

通过条件：

```text
seed41 training rollout:
  reward trend not collapsing
  goodput stable or improving over last 10 iterations
  BLER <= old candidate on seed41, target <= 14.2%
  expiry/drop trend not worse than warm-start
  recent100_service_rate_p10 stable or improving, target >= 0.60
  recent100_debt_score_p90_bytes and no_service_streak_p90 not rising
  active_prg_goodput_efficiency improving without harmful_packing growth
  PRG utilization is recorded only; no raw 95% utilization gate
```

#### Phase 2A-1: Fixed-Topology Validation

目的：同一拓扑 `seed=41` 下做 4000 TTI 正式评估，确认训练不是短 horizon 偶然波动。

通过条件：

```text
seed41 eval:
  goodput >= seed41 old candidate, or at minimum >= seed41 v1 physical-success run
  BLER <= seed41 old candidate, target <= 14.2%
  expiry <= PFQ seed41
  p95 <= PFQ seed41
  p5 UE goodput >= PFQ seed41
  no pathological under-service: demand-heavy PRG utilization should not collapse below 85%
```

如果 seed41 固定拓扑都不能超过 old candidate，不进入多 seed 泛化，也不加 E_PP edge_attr。

推荐验证命令模板：

```bash
./cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh \
  --build-method skip \
  --parallel-seeds 1 \
  --seed-list 41 \
  --checkpoint-path training/stageb_hgraph/runs/online_ppo_7cell_fixed_s41_gd_ue_rate_v1_i50_r128/ppo_actor_absolute_best.pt \
  --model-label hgraph_fixed_s41_gd_ue_rate_v1_i50_r128 \
  --decode-mode argmax \
  --eval-episode-horizon 4000 \
  --eval-device cpu \
  --eval-print-every-steps 1000 \
  --eval-candidate-budgeted-select-mode greedy \
  --eval-candidate-budgeted-hard-cap 1 \
  --eval-candidate-decode-demand-slack-bytes 9000 \
  --eval-candidate-headroom-rank-mode phy \
  --topology-scenario 7cell \
  --total-ue-count 84 \
  --fading-mode 0 \
  --tti 4000 \
  --baseline-scheduler pfq \
  --prbs-per-group 16 \
  --precoding svd \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 0.85 \
  --packet-ttl-ms 200 \
  --kpi-tti-log 0 \
  --compare-tti 0 \
  --compact-output 1 \
  --exec-mode both \
  --compare-output-dir training/stageb_hgraph/runs/eval_7cell_fixed_s41_gd_ue_rate_v1_i50_r128
```

固定 seed 验证不要和 10-seed PFQ mean 混比。对照使用：

```text
PFQ seed41 artifact =
output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_7cell_s41_s50_ue84_ttl200_rbg16_svd_rate0p85_trafficseed_20260601_090103/per_seed/s41/compare_summary.txt
```

#### Phase 2A-2: Small Generalization

目的：固定拓扑收敛后，再检查是否只是 seed41 过拟合。

配置：

```text
train or eval seeds = 41,42,43
only after seed41 fixed validation passes
```

通过条件：

```text
s41-s43:
  mean goodput >= old candidate s41-s43
  mean BLER <= old candidate s41-s43
  mean expiry <= PFQ s41-s43
  mean p95 <= PFQ s41-s43
  mean p5 UE goodput >= PFQ s41-s43
  no demand-heavy seed below 85% PRG utilization
```

#### Phase 2A-3: Locked 10-Seed Eval

只有 Phase 2A-2 通过后，才跑 `s41-s50` locked eval。`s41-s50` 是报告口径，不是早期收敛训练口径。

已完成的 10-seed candidate-anchor PPO 对照：

```text
run = training/stageb_hgraph/runs/online_ppo_7cell_pfq_anchor_s41_s50_rate0p85_i10_r128_slack6000_20260601_093946
eval = training/stageb_hgraph/runs/eval_7cell_pfq_anchor_i10_r128_slack6000_s41_s50_t4_p5
goodput = 2874.903 Mbps vs PFQ 2893.345 Mbps, -18.442 Mbps
served = 3369.213 Mbps vs PFQ 3366.910 Mbps, +2.303 Mbps
BLER = 14.425% vs PFQ 13.991%, +0.433 percentage point
PRG utilization = 93.615% vs PFQ 97.117%, -3.501 percentage point
p95 = 69.55 ms vs PFQ 74.35 ms, better
expiry = 0.795% vs PFQ 0.954%, better
p5 UE goodput = 30.471 Mbps vs PFQ 30.014 Mbps, better
```

解释：模型已经能找到高服务量和 tail 改善机会，但 served 到 goodput 的转化被 BLER 损耗吃掉，
同时 budget/headroom 解码留下约 `6.4%` blank。下一步不是盲目加长 PPO，而是先补物理成功学习。

已完成的 physical-success blankaware preset 对照：

```text
--ppo-preset physical_success_blankaware

action_head_mode = candidate
candidate_mode = all_demand_ues
max_prg_candidates = 12
candidate_pfq_anchor_logit_coef = 1.0

candidate_success_aux_coef = 0.05
candidate_success_calib_coef = 0.02
candidate_success_max_calib_coef = 0.01
candidate_pairwise_rank_coef = 0.02
candidate_decode_goodput_coef = 0.02
candidate_argmax_decode_coef = 0.05
candidate_argmax_decode_headroom_rank_mode = phy

reward_mode = stageb_blankaware
packet_rate_weight = 0.20
packet_completion_weight = 0.05
util_floor_weight = 1.5
util_floor_target = 0.95
util_floor_tb_err_guard = 0.155
util_floor_tb_err_softness = 0.03

candidate_budgeted_select_mode = greedy
candidate_budgeted_hard_cap = 1
candidate_decode_demand_slack_bytes = 9000
```

当前下一轮正式 preset：

```text
--ppo-preset goodput_delay_ue_rate_v1

action_head_mode = candidate
candidate_mode = all_demand_ues
max_prg_candidates = 12
candidate_pfq_anchor_logit_coef = 1.0

candidate_success_aux_coef = 0.10
candidate_success_calib_coef = 0.03
candidate_success_max_calib_coef = 0.02
candidate_pairwise_rank_coef = 0.04
candidate_decode_goodput_coef = 0.03
candidate_argmax_decode_coef = 0.08
candidate_argmax_decode_headroom_rank_mode = phy

reward_mode = stageb_blankaware
best_metric = latency_guarded_goodput
goodput_weight = 0.08
tb_err_weight = 32.0
expiry_weight = 60.0
ici_weight = 1.25
conflict_weight = 1.25
packet_rate_weight = 0.40
packet_completion_weight = 0.10
aged_backlog_weight = 0.35
unserved_ue_weight = 0.60
service_gap_weight = 1.20
service_rate_target = 0.60
util_floor_weight = 0.0

candidate_budgeted_select_mode = greedy
candidate_budgeted_hard_cap = 1
candidate_decode_demand_slack_bytes = 9000
```

正式 10-seed 评估结果：

```text
run = training/stageb_hgraph/runs/online_ppo_7cell_phys_success_blankaware_s41_s50_rate0p85_i10_r128_slack9000_20260602
eval = training/stageb_hgraph/runs/eval_7cell_phys_success_blankaware_i10_r128_slack9000_s41_s50_t4_p5_retry1
checkpoint = ppo_actor_absolute_best.pt

goodput = 2864.462 Mbps vs PFQ 2893.345 Mbps, -28.884 Mbps
goodput = 2864.462 Mbps vs previous candidate 2874.903 Mbps, -10.442 Mbps
served = 3361.680 Mbps vs PFQ 3366.910 Mbps, -5.231 Mbps
BLER = 14.478% vs PFQ 13.991%, +0.486 percentage point
BLER = 14.478% vs previous candidate 14.425%, +0.053 percentage point
PRG utilization = 95.585% vs PFQ 97.117%, -1.532 percentage point
PRG utilization = 95.585% vs previous candidate 93.615%, +1.970 percentage point
blank ratio = 4.415% vs previous candidate 6.385%, better
p95 = 72.35 ms vs PFQ 74.35 ms, better
expiry = 0.915% vs PFQ 0.954%, better
p5 UE goodput = 30.041 Mbps vs PFQ 30.014 Mbps, slightly better
```

结论：

- 该 preset 学到了更高 PRG 复用效率：fill/PRG utilization 明显上升，blank 明显下降。
- 但系统 goodput 未提升，主要被 `global_tb_bler` 高于 PFQ 约 `0.486 pp` 和 seed-wise 低复用 outlier 吃掉。
- s47 是主要失败模式：`fill ~= 92.97%`、`BLER ~= 14.60%`、`goodput ~= 2747.71 Mbps`、`unique UE ~= 54.65`。这更像“安全复用候选/风险排序不足”，不是单纯 harmful packing。
- 因此 Phase 2A 未达到净提升 gate；暂不进入 E_PP edge_attr 实装。下一轮先调强物理成功学习和 headroom/risk gate。

10-seed locked 通过条件：

```text
goodput >= previous candidate + 10 Mbps on s41-s50
BLER <= previous candidate - 0.2 percentage point
expiry <= PFQ
p95 <= PFQ
p5 UE goodput >= PFQ
no demand-heavy seed below 85% PRG utilization
```

如果 Phase 2A 有提升，再进入 E_PP edge_attr：

```text
E_PP attr = [
  same_local_prg,
  neighbor_top1_sinr_db,
  neighbor_top2_gap_db,
  neighbor_ici_proxy,
  neighbor_reuse_ratio,
  relative_cell_load
]
```

这样跨小区同频冲突从“有边”升级成“有强弱、有风险”。如果 Phase 2A 没提升，先继续调 success target、
slack/headroom 和 blankaware 权重，不急于扩图。当前建议的下一轮是：

```text
--ppo-preset goodput_delay_ue_rate_v2_no_teacher

teacher_mode: none
include_pfq_score_anchor: 0
candidate_pfq_anchor_logit_coef: 0.0
candidate_success_aux_coef: keep 0.10
candidate_argmax_decode_coef: keep 0.08
candidate_pairwise_rank_coef: keep 0.04
goodput_weight: 0.08 -> 0.10
expiry_weight: 60.0 -> 120.0
backlog_debt_weight: 0.0 -> 0.25
aged_backlog_weight: 0.35 -> 0.80
unserved_ue_weight: 0.60 -> 2.00
service_gap_weight: 1.20 -> 8.00
service_rate_target: 0.60 -> 0.75
util_floor_weight: keep 0.0
eval must keep candidate_headroom_rank_mode = phy
```

目标是把 BLER 拉回 `<=14.2%`，让 active PRG yield、p5 UE goodput、expiry 和 p95 都优于 PFQ/old candidate。
PRG utilization 只确认没有 demand-heavy under-service collapse，不再要求 `>=95%`。

### Phase 2B: Residual PPO

目的：在 PFQ 周围小范围优化。

策略：

- 初始化自 Phase 1。
- 冻结或半冻结 PFQ base score。
- PPO 只更新 residual head 和一部分 graph encoder。
- residual clip 从 `0.2` 逐步放到 `0.5`。

训练环境：

```text
main: arrival_rate=0.85, seed cycle 41-48
validation: seeds 49-50
stress: arrival_rate=0.9, seeds 41-43
light: arrival_rate=0.8, seeds 41-43
```

通过条件：

```text
goodput >= PFQ on validation
p95 delay <= PFQ - 10%
expiry <= PFQ
p5 UE goodput >= PFQ
no demand-heavy under-service collapse
```

### Phase 3: Tail-Focused Fine Tune

目的：把优势从“能跑”变成“tail 明显低于 PFQ”。

改动：

- 提高 `p95_delay_proxy`、`expiry_risk_penalty` 和 `p5_ue_goodput_bonus`。
- 保持 goodput guard：若 rolling goodput 低于 PFQ 超过 `0.5%`，降低 tail penalty。
- 混入 `arrival_rate=0.9` 做压力泛化。

通过条件：

```text
arrival_rate=0.85 10-seed:
  goodput >= PFQ + 0.3%
  p95 <= PFQ - 15%
  R_pkt >= PFQ + 10%

arrival_rate=0.9 3 or 10-seed:
  p95 <= PFQ
  expiry <= PFQ
  no catastrophic goodput drop
```

## 9. Implementation Map

优先修改或扩展这些位置：

| 模块 | 文件 | 工作 |
|---|---|---|
| observation extras | `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp` | 导出 HOL/TTL/headroom/PFQ score/expected BLER proxy |
| PRG features | `cuMAC/examples/network.cu`, `training/stageb_hgraph/feature_snapshot.py` | 补 7cell PRG conflict 和 ICI proxy |
| graph edges | `training/stageb_hgraph/edge_generator.py` | 支持 `C=7,U=84,P=17` 的 dynamic edge packing |
| model | `training/stageb_hgraph/model.py` | 新增 residual PFQ scorer 和 expected-goodput head |
| decode ops | `training/stageb_hgraph/graph_ops.py` | 统一 budgeted/headroom-aware candidate decode |
| reward | `training/stageb_hgraph/reward_utils.py` | 使用 `stageb_blankaware`，通过 `goodput_delay_ue_rate_v1` preset 对齐 goodput/delay/UE-rate |
| PPO | `training/stageb_hgraph/online_ppo.py` | residual clip、PFQ anchor、multi-rate curriculum |
| eval | `training/stageb_hgraph/eval_checkpoint_online.py` | 同口径 RRQ/PFQ/TAR-GNN-PFQ 多 seed评测 |
| export | `training/stageb_hgraph/export_onnx.py` | 7cell/84UE action-mode shape profile |
| C++ runtime | `cuMAC/examples/customScheduler/*` | PFQ residual score injection 和 hard guard |

当前已落地的 PFQ residual anchor 前置改动：

- `training/stageb_hgraph/edge_generator.py`：UE-PRG edge attr 增加第 6 维 `log1p(PFQ-score proxy)`，默认开启。
- `training/stageb_hgraph/model.py`：`up_edge_attr_dim` 默认从 `5` 切到 `6`，message relation 和 candidate head 同步；candidate main logits 显式加入 `candidate_pfq_anchor_logit_coef * pfq_anchor`，形成 `PFQ prior + learned residual`。
- `training/stageb_hgraph/online_imitation.py`：支持 `--action-head-mode candidate`，可直接用 PFQ passthrough teacher warm-start candidate head。
- `training/stageb_hgraph/online_ppo.py`：支持 `--candidate-pfq-anchor-logit-coef`，新 candidate PPO 训练默认使用 PFQ anchor prior。
- `cuMAC/scripts/run_stageB_hgraph_online_imitation.sh`：透传 candidate head 和 PFQ anchor 训练参数。
- `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh`：透传 PPO candidate PFQ anchor 参数。
- `training/stageb_hgraph/export_onnx.py`：fixed action wrapper 和 candidate pack 同步生成第 6 维，并在 action-mode candidate logits 中加入同样的 PFQ anchor prior；旧维度自动裁剪/补零。
- `training/stageb_hgraph/eval_checkpoint_online.py`：按 checkpoint 的 `up_edge_attr_dim` 自动兼容旧 5 维 actor。

当前 PPO smoke：

```text
run = training/stageb_hgraph/runs/online_ppo_7cell_pfq_anchor_smoke_s41_iter1
init = online_imitation_7cell_pfq_s41_s43_rate0p85_candidate_anchor_warmstart/hgraph_imitation_best.pt
iterations = 1
rollout_steps = 64
candidate_budgeted_select_mode = greedy
rollout_reward_mean = 1.254
rollout_goodput = 1457.690 Mbps
rollout_tb_err = 0.504
rollout_blank_ratio = 0.258
candidate_coverage_ratio = 1.0
status = PPO update path works; this is smoke only, not a performance checkpoint
```

## 10. Evaluation Matrix

每个候选 checkpoint 必须按顺序跑，不能在早期直接跳到 `s41-s50`：

| 套件 | 场景 | seeds | 目的 |
|---|---|---|---|
| bridge smoke | rate `0.85` | `41`, short horizon | 快速发现崩溃、非法动作、util 过低 |
| fixed-converge | rate `0.85` | train `41` only | 固定拓扑看 50-100 iteration 是否稳定收敛 |
| fixed-eval | rate `0.85` | eval `41` only, 4000 TTI | 同拓扑正式验证是否超过 PFQ 或 old candidate |
| small-generalization | rate `0.85` | `41,42,43` | 通过 seed41 后再看小范围泛化 |
| main | rate `0.85` | `41-50` | 主报告口径，只在 small-generalization 通过后运行 |
| stress | rate `0.9` | `41,42,43` 或 `41-50` | 压力泛化 |
| light | rate `0.8` | `41,42,43` | 非过载泛化 |
| overload reference | rate `1.0` | `41-50` | 只做压力背景，不作为主训练目标 |

输出必须包含：

- `rrq_vs_pfq_compare_mean.csv` 或对应 model compare mean。
- `key_metrics_summary.csv`。
- `tail_latency_throughput_weighted_summary.csv`。
- `alignment_check.csv`。
- seed-wise summary，尤其标注 seed42 这类 tail outlier。

## 11. Decision Gates

进入下一阶段前必须满足：

### Gate A: Baseline readiness

```text
rate=0.85, RRQ/PFQ 10-seed complete
alignment_check all yes
tail summary generated
```

### Gate B: Imitation readiness

```text
PFQ action imitation legal rate >= 99%
PRG utilization >= 95%
goodput within 1% of PFQ in seed42 smoke
```

### Gate C: RL fixed-topology candidate

```text
seed41 fixed eval:
  goodput >= seed41 old candidate, or at least >= seed41 v1 physical-success run
  BLER <= seed41 old candidate, target <= 14.2%
  p95 <= seed41 PFQ
  expiry <= seed41 PFQ
  p5 UE goodput >= seed41 PFQ
  no pathological under-service: demand-heavy PRG utilization should not collapse below 85%
```

### Gate C2: RL small-generalization

```text
3-seed quick after Gate C:
  mean goodput >= old candidate s41-s43
  mean BLER <= old candidate s41-s43
  mean expiry <= PFQ s41-s43
  mean p95 <= PFQ s41-s43
  mean p5 UE goodput >= PFQ s41-s43
  no demand-heavy seed below 85% PRG utilization
```

### Gate D: Main claim

```text
10-seed main:
  goodput >= PFQ + 0.3%
  p95 <= PFQ - 15%
  R_pkt >= PFQ + 10%
  p5 UE goodput >= PFQ
```

## 12. Why This Is Different From Old "Graph + RL"

旧 HGraph 的问题不是“没有图”，而是图输出和 Type-0/headroom/PFQ 机制没有完全对齐。TAR-GNN-PFQ 的创新点应集中在：

- PFQ-residual learning，而不是从零学调度。
- Expected-goodput scorer，而不是只追 served throughput。
- Decode/headroom-aware target，而不是部署阶段临时裁剪。
- Tail urgency and TTL slack 进入 action scorer。
- Same-PRG inter-cell conflict 进入 PRG/UE-PRG edge。
- Multi-rate curriculum：`0.85` 主训，`0.9` 压力泛化，`0.8` 轻载泛化。

这条路线的目标很明确：不和 PFQ 比谁更会填 PRG，而是比谁能在同等业务负载下交付更多成功 goodput、给低分位 UE 更稳定的服务，同时显著降低 tail delay。

## 13. Historical Mechanism Log: 2026-06-04 3cell v5 Relative-Margin / Windowed-Expiry

以下内容保留作机制演化记录，不是当前 HGraphPPO 训练 gate。当前正式口径以上方
`ISD1000/r500 + coop_boundary + strongest association + legacy shadow RNG` 为准。

本轮为 `3cell + ISD500 + coop_center + rate1.0 + 36UE + RBG16 + TTL200 + SVD`
固定 seed41 的机制验证，主要目的是在不继续复杂化 reward 的前提下，检查 candidate ranking 特征是否足够表达同 PRG 内的物理风险和边际 goodput。

### 13.1 已落地改动

- `edge_generator.py`：candidate risk feature 从原始 risk proxy 扩展为同 PRG 内相对 margin：
  - SINR vs PRG 内 mean；
  - SINR vs nearest/best competitor；
  - fail-risk vs mean；
  - fail-risk rank percentile；
  - expected-goodput/value vs mean；
  - expected-goodput/value rank percentile。
- `online_ppo.py`：新增 windowed expiry reward。
  - 原始 C++/env 仍保留累计 `expiry_drop_rate` 作为 KPI；
  - reward 只使用最近 `expiry_reward_window_steps=1000` TTI 的过期字节率；
  - 避免已经过期、策略无法挽回的历史包继续污染后续 advantage。
- `run_stageB_hgraph_online_ppo.sh`：新增/使用 `goodput_delay_rate_simple_risk_v5_relative_margin_fast`，保持 reward 简洁：
  - 主目标仍是 goodput；
  - 约束项仍是 TB error、expiry、packet rate/completion、delay/service proxy；
  - 不引入显式“同 UE 重复 PRG”硬惩罚。

### 13.2 训练与正式评估

训练目录：

```text
training/stageb_hgraph/runs/online_ppo_3cell_isd500_coopcenter_rate1_posteq_s41_simple_risk_v5_relative_margin_winexp1000_rebuild_stochastic_i100_r128_l2_e1_20260604_062006
```

训练配置要点：

```text
iterations = 100
rollout_steps = 512
decode = stochastic/sample
expiry_reward_window_steps = 1000
expiry_reward_offered_bytes_per_tti = 108000
candidate_risk_feature_dim = 14
up_edge_attr_dim = 19
```

训练内现象：

- `iter100` 出现短窗口高点：`goodput=1480.444 Mbps`, `TB err=12.524%`, `expR=3.254%`。
- 但后 20/40 轮均值仍在平台区间：
  - tail20：约 `1458.5 Mbps`, `TB err=12.89%`, `expR=3.43%`；
  - tail40：约 `1458.8 Mbps`, `TB err=12.89%`, `expR=3.45%`。
- 结论：windowed expiry 修正了 reward 单调下滑问题，但短窗口 high-goodput spike 不能直接当作部署收益。

正式 seed41 eval：

```text
checkpoint = ppo_actor_absolute_best.pt / ppo_actor_last.pt
decode = sample
horizon = 4000 TTI
stats_warmup_steps = 1000
stats_count = 3000
```

两份 checkpoint 的部署输出完全一致：

| metric | warmup 后 3000 TTI |
|---|---:|
| goodput | `1458.097 Mbps` |
| TB err | `13.012%` |
| expiry drop | `0.646%` |
| PRG util | `98.274%` |
| blank ratio | `1.267%` |
| unique UE / TTI | `29.460` |

全 4000 TTI C++ KPI：

| metric | value |
|---|---:|
| cluster goodput | `1445.798 Mbps` |
| global TB BLER | `13.331%` |
| expiry drop | `1.540%` |
| packet delay p90 / p95 | `109.5 / 193.0 ms` |
| UE goodput p5 / p10 | `32.352 / 34.246 Mbps` |
| MAC buffer bytes | `9,420,528` |

### 13.3 经验判断

- 不建议继续单纯拉长 v5 训练。
  - 训练后段没有稳定上升，只是偶发 512-window spike；
  - 正式 4000 TTI eval 后，goodput 接近 PFQ seed41 gate，但 delay p95 明显劣化。
- reward 不应继续复杂化。
  - windowed expiry 已解决“累计历史过期包不可逆拖累 reward”的硬伤；
  - 继续堆 reward 权重容易把问题变成 checkpoint 选择噪声。
- 下一步应补 candidate 级直接输入，而不是只靠全局/UE 侧 proxy：
  - TTL slack / near-expiry；
  - packet age / HOL；
  - estimated PRG headroom / one-PRG 后剩余需求；
  - recent service gap / deficit。

### 13.4 v6 TTL/Headroom Candidate Feature

新增 preset：

```text
--ppo-preset goodput_delay_rate_simple_risk_v6_ttl_headroom_fast
```

v6 不改 reward，只扩展 candidate edge_attr：

```text
candidate_risk_feature_dim = 24
up_edge_attr_dim = 29  # no PFQ anchor
```

新增 10 个 candidate-direct context feature：

| feature | 含义 |
|---|---|
| `hol_norm` | `HOL / 200ms`，packet age 直接输入 |
| `ttl_known` | TTL slack 是否有效 |
| `ttl_near_40` | `ttl_slack <= 40ms` 的 near-expiry 压力 |
| `ttl_critical_20` | `ttl_slack <= 20ms` 的 critical pressure |
| `age_to_deadline` | `HOL / (HOL + TTL)` |
| `recent_service_gap` | `1 - recent_scheduled_ratio` |
| `recent_deficit` | recent goodput deficit |
| `decode_headroom_need` | `(backlog + 9000B) / est_bytes_per_prg` 归一化 |
| `marginal_headroom_need` | `(backlog + 3000B) / est_bytes_per_prg` 归一化 |
| `after_one_prg_margin` | 分配 1 个 PRG 后是否仍有剩余可服务需求 |

预期效果：

- 模型不再只能从 `urgency/debt` 间接推 TTL 和 age；
- 高 backlog UE 仍可合法拿多个 PRG，`after_one_prg_margin` 会告诉模型“再给一个 PRG 是否仍有需求”；
- 对快过期且 headroom 充足的 UE，candidate scorer 可以在同 PRG 内做更有对应性的排序；
- 如果 v6 仍在正式 eval 中出现 `p95 ~= TTL`，下一步应考虑 deployment/checkpoint selection 或 sequential candidate state，而不是再调 scalar reward。

### 13.5 v6 TTL/Headroom 训练与正式评估

训练目录：

```text
training/stageb_hgraph/runs/online_ppo_3cell_isd500_coopcenter_rate1_posteq_s41_simple_risk_v6_ttl_headroom_winexp1000_rebuild_stochastic_i100_r128_l2_e1_20260604_095053
```

训练配置沿用 v5 的简单 reward 与 windowed expiry，只改 candidate 输入：

```text
iterations = 100
rollout_steps = 512
decode = stochastic/sample
expiry_reward_window_steps = 1000
candidate_risk_feature_dim = 24
up_edge_attr_dim = 29
include_pfq_score_anchor = 0
```

训练内摘要：

| window | goodput | TB err | expR | expiry drop | PRG util | blank |
|---|---:|---:|---:|---:|---:|---:|
| tail5 | `1465.075 Mbps` | `12.891%` | `3.314%` | `3.223%` | `98.457%` | `1.084%` |
| tail10 | `1462.202 Mbps` | `12.935%` | `3.267%` | `3.222%` | `98.404%` | `1.061%` |
| tail20 | `1460.459 Mbps` | `12.910%` | `3.253%` | `3.219%` | `98.387%` | `1.030%` |
| tail40 | `1460.372 Mbps` | `12.906%` | `3.309%` | `3.210%` | `98.427%` | `1.015%` |
| all100 | `1459.400 Mbps` | `12.922%` | `3.192%` | `2.761%` | `98.280%` | `1.256%` |

`absolute_best_iter = 100`，`threshold_best_iter = 100`，训练内 `iter100` 为：

```text
goodput = 1481.849 Mbps
TB err = 12.46%
expR = 3.09%
PRG util = 98.65%
```

`ppo_actor_absolute_best.pt`、`ppo_actor_threshold_best.pt`、`ppo_actor_last.pt`
文件 checksum 不同，但直接比较 `actor_state` 的 233 个 tensor 后确认权重完全一致；三者正式 eval 输出也完全一致。

正式 seed41 eval：

```text
checkpoint = ppo_actor_absolute_best.pt / ppo_actor_threshold_best.pt / ppo_actor_last.pt
decode = sample
sample_seed = 41
horizon = 4000 TTI
stats_warmup_steps = 1000
stats_count = 3000
```

评估目录：

```text
training/stageb_hgraph/runs/eval_3cell_s41_v6_ttl_headroom_winexp1000_sample_warm1000_20260604_130756
```

warmup 后 3000 TTI：

| metric | v6 |
|---|---:|
| goodput | `1456.157 Mbps` |
| TB err | `13.022%` |
| expiry drop | `0.602%` |
| PRG util | `98.429%` |
| blank ratio | `1.097%` |
| unique UE / TTI | `29.598` |

全 4000 TTI C++ KPI：

| metric | value |
|---|---:|
| cluster goodput | `1443.686 Mbps` |
| global TB BLER | `13.386%` |
| expiry drop | `1.437%` |
| packet delay p90 / p95 | `111.5 / 192.5 ms` |
| UE goodput p5 / p10 | `32.951 / 34.259 Mbps` |
| PRG util | `98.189%` |
| MAC buffer bytes | `9,720,080` |

和 v5 的同口径 warmup eval 对比：

| metric | v5 | v6 | delta |
|---|---:|---:|---:|
| goodput | `1458.097` | `1456.157` | `-1.940 Mbps` |
| TB err | `13.012%` | `13.022%` | `+0.010 pp` |
| expiry drop | `0.646%` | `0.602%` | `-0.044 pp` |
| PRG util | `98.274%` | `98.429%` | `+0.155 pp` |
| blank ratio | `1.267%` | `1.097%` | `-0.170 pp` |
| unique UE / TTI | `29.460` | `29.598` | `+0.139` |

经验判断：

- v6 的 TTL/age/headroom direct feature 是健康改动：没有造成 fill collapse，反而略提升 util、降低 blank 和 expiry。
- 但 v6 没把 goodput 推过 v5/PFQ seed41 gate；formal eval 显示 `iter100` 的 `1481.8 Mbps` 仍主要是 512-window spike。
- 过期窗口 reward 是必要的：训练后期 expiry 不再被累计历史过期包单调拖垮，reward 曲线可比较。
- 单纯继续长训的收益不确定。当前瓶颈更像 ranking/selection 的 reliability-goodput tradeoff，而不是缺少更多 scalar reward。
- 下一步如果继续优化，优先顺序应是：
  - checkpoint selection 更保守地约束 TB err / formal eval proxy；
  - candidate ranking loss 适度提高 reliability/risk margin，而不是新增复杂 reward；
  - 研究 sequential candidate state，让同 TTI 内已分配 PRG 对后续 candidate 的 headroom/risk 可见。
