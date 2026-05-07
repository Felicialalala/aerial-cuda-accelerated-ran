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
- HGraph action-mode ONNX/C++ seed42 smoke：
  `output/stageB_main_experiment_hgraph_action_s42_warmup1000_20260507_021558`

## 2026-05-07 HGraph 部署进展

固定 `3cell/36UE/17PRG` 的 HGraph 部署路径已经从“导出 logits 后由 C++ 解码”推进到“ONNX 直接输出最终动作”：

- 导出脚本：`training/stageb_hgraph/export_onnx.py`
- 一致性校验：`training/stageb_hgraph/check_onnx_consistency.py`
- 推荐模型：`training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx`
- ONNX action 输出：
  - `action_ue_select [1,36]`
  - `action_prg_alloc [1,51]`
- C++ runtime 保留最终合法化：
  - Type-0 slot/PRG 顺序合法化
  - UE demand-cap
  - 同 cell fallback
  - strict metadata preflight，防止 `3cell/36UE/17PRG` 模型误跑到默认 `7cell/56UE` 配置

seed42 action-mode smoke 结果：

| 指标 | HGraph action s42 | PFQ s42 | RRQ s42 |
|---|---:|---:|---:|
| `traffic.served_mbps_est` | 1736.670 | 1734.346 | 1725.653 |
| `traffic.goodput_mbps` | 1555.805 | 1548.486 | 1548.063 |
| `global_kpi.global_tb_bler` | 0.105984 | 0.108009 | 0.108990 |
| `global_kpi.ue_goodput_jain` | 0.998669 | 0.998616 | 0.998790 |
| `global_kpi.ue_goodput_p5_mbps` | 40.209 | 39.604 | 40.306 |
| `global_kpi.prg_utilization_ratio` | 0.603208 | 0.906142 | 0.866892 |
| `traffic.packet_delay_mean_ms` | 2.353 | 0.692 | 0.685 |
| `traffic.packet_delay_p95_ms` | 6.500 | 2.000 | 1.500 |

结论口径：

- action-mode 已经证明最终动作路径基本可用，吞吐、goodput、BLER、公平性和 p5 goodput 均达到或略优于 PFQ seed42。
- 当前策略形态是“低 PRG 利用率、低调度频率、单次集中服务”。因此吞吐不差，但 packet delay 高于 PFQ/RRQ。
- `warmup1000` 目前只是运行标签和总 TTI 设计；主实验 KPI 仍按全程累计。严格排除前 1000 TTI 还需要补正式 `--stats-warmup-tti 1000` 统计路径。

## HGraph 固定部署路径

固定 `3cell/36UE/17PRG` 的主部署导出已支持两种 ONNX ABI：

- `logits`：输出 `ue_logits/prg_logits`，保留旧 C++ masked decode。
- `action`：额外输入 `obs_post_eq_sinr`，直接输出 `action_ue_select/action_prg_alloc`，C++ 只做 Type-0 合法化、demand-cap 和 fallback。

推荐 action-mode 命令：

```bash
.venv/bin/python training/stageb_hgraph/export_onnx.py --output-mode action --check
./cuMAC/scripts/run_stageB_main_experiment.sh \
  --build-method cmake \
  --custom-ue-prg 1 --custom-policy gnnrl_model \
  --model-path training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx \
  --gnnrl-model-output-mode action \
  --gnnrl-model-use-post-eq-input 1 \
  --gnnrl-model-decode-mode argmax \
  --gnnrl-model-strict 1 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --tti 4000 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --exec-mode both
```
