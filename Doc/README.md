# Doc 目录索引与当前口径

本文档用于避免历史实验记录和当前主线混在一起阅读。

## 当前权威口径

- [`current_stageB_effective_configuration.md`](./current_stageB_effective_configuration.md)：Stage-B 当前运行配置、KPI 口径、RRQ/PFQ baseline 入口。
- [`第二次会后确认.md`](./第二次会后确认.md)：packet-level rate、packet delay、SVD 容量提升、PFQ demand-cap 解释。
- [`stageB_hybrid_scheduler_graph_variable_plan.md`](./stageB_hybrid_scheduler_graph_variable_plan.md)：HGraph 当前实现和 v33 训练/评测收口。

## 平台能力说明

- [`aerial_platform_antenna_directionality_and_mimo_precoding.md`](./aerial_platform_antenna_directionality_and_mimo_precoding.md)：天线方向性、4T4R SVD、64T64R beamforming 能力边界。
- [`aerial_platform_rbg_selection_and_resource_mapping_completeness.md`](./aerial_platform_rbg_selection_and_resource_mapping_completeness.md)：PRG/RBG、Type-0/Type-1、资源映射边界。
- [`aerial_platform_candidate_state_field_audit.md`](./aerial_platform_candidate_state_field_audit.md)：候选状态字段和当前 bridge/调度主线的可用性核对。

## GNNRL 历史与兼容说明

- [`stageB_gnn_rl_online_training_design.md`](./stageB_gnn_rl_online_training_design.md)：旧 GNNRL online/offline 实现参考；当前主实验优先看 HGraph。
- [`stageB_gnn_rl_training_implementation_tasklist.md`](./stageB_gnn_rl_training_implementation_tasklist.md)：GNNRL readiness checklist，保留作兼容参考。
- [`stageB_rl_gnn_scheduler_mvp_to_publish_plan.md`](./stageB_rl_gnn_scheduler_mvp_to_publish_plan.md)：历史归档，不代表当前实现状态。

## 规划与历史材料

- [`cluster_cell_intelligent_scheduling_implementation_plan.md`](./cluster_cell_intelligent_scheduling_implementation_plan.md)：早期总体实施计划。
- [`ue_prg_takeover_multidomain_plan.md`](./ue_prg_takeover_multidomain_plan.md)：UE selection + PRG 接管的早期路线。
- [`pf_call_chain_677099a_20260415.md`](./pf_call_chain_677099a_20260415.md)：原始 PF call-chain 历史分析。
- [`空间域确认点.md`](./空间域确认点.md)：第一次会后口头汇报稿，保留为历史材料；其中 no-precoding 口径已被当前 SVD 主线覆盖。

## 当前保留的关键实验产物

- RRQ/PFQ 10-seed baseline：
  `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s50_ue36_ttl200_rbg16_svd_20260506_091443`
- RRQ/PFQ 3-seed baseline 历史对照：
  `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s42_s43_ue36_ttl200_rbg16_svd_20260501_030142`
- HGraph v33 3-seed evaluation：
  `output/stageB_hgraph_v33_best_iter86_s41_s42_s43_ue36_ttl200_rbg16_svd_warmup1000`
- 当前保留 HGraph 训练 run：
  `training/stageb_hgraph/runs/online_ppo_seed42_svd_rawbridge_joint_v33_capSlack3000_simpleExecutor_ablation`
- 当前保留 GNNRL 兼容 checkpoint：
  `training/gnnrl/checkpoints/m3_online_ppo_3cell_pfq_fixedseed42_v16c_joint_ttl200_rbg16_ue36_blankaware_prg8_i400_eval_h1024_stable`
