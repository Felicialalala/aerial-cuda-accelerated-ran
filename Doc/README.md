# Stage-B 文档入口

本文是 Stage-B 文档导航和当前结论索引。详细实验数字集中放在实验报告与配置说明中，历史设计、会议记录和长流水日志放入 `archive/`，避免多个文档重复维护同一结论。

## 当前结论

当前主线是 `3cell coop_boundary + strongest association + legacy shared shadow RNG` 下的 HGraphPPO 智能调度评估，长期目标仍是迁移到 `7cell + 84UE + rate0.85`。

当前正式场景：

- `3cell / 36UE / cell_radius=500m / site_spacing=1000m`
- `ue-placement=coop_boundary`
- `cell-assoc-mode=strongest`
- 不显式传 `--shadow-seed`，日志应显示 `shadow_seed=legacy_shared_rng`
- `Rayleigh + SVD + Type-0 bitmap`
- `packet_size=3000 bytes`
- `traffic_arrival_rate=1.0 pkt/TTI/UE`
- `TTL=200 ms`
- `horizon=4000 TTI`

当前报告结论：

- 单 seed 正式部署最好结果是 `HGraphPPO v9c best / iter4`，seed41，t4000，argmax + greedy。
- 10-seed full-run 尚未稳定超过 PFQ，但 v9c 去掉前 `1000 TTI` 冷启动后 mean goodput 超过 PFQ full-run。
- `v10 rate2 iter24` 是当前 rate-aware 三 seed 口径下较稳的正式部署候选。
- `v19` 主要作为低 PRG utilization / coverage quota 的资源效率补充证据，不替代 v9c/v10 作为当前主结论。
- 原始 RR/PF boundary 10-seed 已补跑；原始 PF 在该场景下严重劣化，RRQ/PFQ 才是有意义的 native reference。

## 当前文档

- [`stageB_boundary_intelligent_algorithm_experiment_report.md`](./stageB_boundary_intelligent_algorithm_experiment_report.md)：当前主报告，覆盖 baseline、HGraphPPO 迭代、最好模型选择、横向对比和后续建议。
- [`current_stageB_effective_configuration.md`](./current_stageB_effective_configuration.md)：Stage-B 当前生效配置、RR/RRQ/PF/PFQ 口径、KPI 定义和脚本入口。
- [`stageB_packet_delay_distribution_analysis.md`](./stageB_packet_delay_distribution_analysis.md)：基线逐包时延分布与 RR/RRQ/PF/PFQ 机制解释。
- [`stageB_7cell_tar_gnn_pfq_design.md`](./stageB_7cell_tar_gnn_pfq_design.md)：HGraphPPO/7cell 迁移设计与历史训练日志。当前只作为设计和机制记录，最新结论以主报告为准。
- [`figures/stageB_coop_placement_schematic.png`](./figures/stageB_coop_placement_schematic.png)：coop boundary 放置示意图。

## 关键数据入口

- active boundary RRQ/PFQ gate:
  `output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50`
- boundary RR/PF container rerun:
  `output/stageB_rr_pf_multiseed_compare_boundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50`
- full KPI / packet-delay suite:
  `output/stageB_baseline_eval_suite_fullkpi_seed41_50`
- v9c single-seed best deployment:
  `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_i12r512_best_argmax_greedy_coop_boundary_r500_t4000_w0`
- v9c 10-seed threshold deployment:
  `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50_v9c_i12r512_best_threshold_argmax_greedy_coop_boundary_r500_t4000_w0`
- v10 iter24 rate-aware evaluation:
  `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s43_v10_rate2_iter24_argmax_greedy_coop_boundary_r500_t4000_w1000`
- v19 supplementary evaluation:
  `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s50_v19_packet_tail_i3_argmax_greedy_rerun_skip_coop_boundary_r500_t4000_w1000`

## 代码入口

- `cuMAC/scripts/run_stageB_main_experiment.sh`：单场景/矩阵实验、编译期参数注入、KPI 汇总入口。
- `cuMAC/scripts/run_stageB_rr_pf_compare.sh`：RR/RRQ 与 PF/PFQ 单 seed 对比。
- `cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh`：RR/RRQ 与 PF/PFQ 多 seed 聚合对比。
- `cuMAC/scripts/run_stageB_baseline_eval_suite.sh`：full KPI baseline suite。
- `cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh`：固定 HGraph checkpoint 多 seed 评测。
- `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh`：HGraph online PPO 训练入口。
- `training/stageb_hgraph/`：当前优先维护的 sparse entity graph 调度栈。

## 归档索引

归档目录只保留上下文价值，不再代表当前结论。

- [`archive/platform/`](./archive/platform/)：Aerial 平台能力、天线/precoding、PRG/RBG、状态字段审计。
- [`archive/hgraph_design/`](./archive/hgraph_design/)：HGraph 设计过程、v33 训练/评测收口材料。
- [`archive/hgraph_experiments/`](./archive/hgraph_experiments/)：HGraph v34-v50 多轮优化、训练、评估流水日志和 3cell retrospective。
- [`archive/legacy_gnnrl/`](./archive/legacy_gnnrl/)：旧 GNNRL online/offline/部署说明与 checklist。
- [`archive/plans/`](./archive/plans/)：早期总体规划、UE/PRG 接管路线、历史 RL+GNN MVP。
- [`archive/meeting_notes/`](./archive/meeting_notes/)：第一次/第二次会后确认材料。
- [`archive/code_audit/`](./archive/code_audit/)：原始 PF call-chain 等历史代码审计。

## 保留与清理规则

保留：

- 当前报告直接引用的聚合目录、checkpoint eval 目录和 baseline suite。
- 含 `ppo_actor*.pt`、`hgraph_imitation_*.pt`、ONNX、`*_compare_mean.*`、`aggregate_summary.txt`、`eval_summary.json`、`graph_sanity_summary.json` 的目录。
- 能解释当前结论的历史机制结果。

删除：

- Python cache 和空目录。
- failed launch、空 manifest、只有 wrapper log 而没有 KPI/summary 的目录。
- 已被正式结果替代的 smoke / skip / probe 目录。
- 命名与实际场景或 horizon 不匹配、且没有文档引用价值的结果。

`output/` 是生成目录。长期引用结果时优先保留聚合目录、报告中的数据来源和复现命令，不保留所有单次运行目录。
