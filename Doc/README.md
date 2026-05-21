# Stage-B 项目主文档

本文是当前项目的主入口。历史设计、会议记录和平台审计已移入 [`archive/`](./archive/)，避免和当前 Stage-B 主线混读。

## 当前结论

当前主线收敛到：

- 场景：`3cell + 36UE + RBG16 + Rayleigh + TTL200ms + 4T4R + SVD`
- 业务：`packet_size_bytes=3000`，`traffic_arrival_rate=1.0 pkt/TTI`
- baseline：RRQ/PFQ 以 `s41-s50` 10-seed 结果为主参考
- 智能调度：优先看 `training/stageb_hgraph/`，旧 `training/gnnrl/` 保留为 bridge/replay/ONNX 兼容参考
- 部署形态：HGraph v33 已支持固定 `3cell/36UE/17PRG` 的 action-mode ONNX，C++ runtime 做最终合法化、demand-cap 和 fallback

最新可引用口径：

- RRQ/PFQ 10-seed baseline：
  `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s50_ue36_ttl200_rbg16_svd_20260506_091443`
- HGraph v33 3-seed evaluation：
  `output/stageB_hgraph_v33_best_iter86_s41_s42_s43_ue36_ttl200_rbg16_svd_warmup1000`
- HGraph action-mode seed42 smoke：
  `output/stageB_main_experiment_hgraph_action_s42_warmup1000_20260507_021558`
- RRQ/PFQ 3-seed baseline：
  `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s42_s43_ue36_ttl200_rbg16_svd_20260501_030142`，仅作为历史对照，不建议报告中作为主 baseline

详细配置仍以 [`current_stageB_effective_configuration.md`](./current_stageB_effective_configuration.md) 为准。

下一步 HGraph 优化方案和当前算法结构详见：

- [`stageB_hgraph_next_optimization_plan.md`](./stageB_hgraph_next_optimization_plan.md)

## 代码入口

当前 Stage-B 运行入口：

- `cuMAC/scripts/run_stageB_main_experiment.sh`：单场景/矩阵实验、编译期参数注入、KPI 汇总入口
- `cuMAC/scripts/run_stageB_rr_pf_compare.sh`：RRQ/PFQ 单 seed 对比
- `cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh`：RRQ/PFQ 多 seed 聚合对比
- `cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh`：固定 HGraph checkpoint 多 seed 评测
- `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh`：HGraph online PPO 训练入口
- `cuMAC/scripts/run_stageB_hgraph_online_sanity.sh`：HGraph online bridge/图特征 sanity

当前模型代码边界：

- `training/stageb_hgraph/`：当前优先维护的 sparse entity graph 调度栈
- `training/stageb_hgraph/export_onnx.py`：固定形状 ONNX 导出，推荐 `--output-mode action`
- `training/stageb_hgraph/check_onnx_consistency.py`：PyTorch/ONNX 一致性检查
- `training/gnnrl/`：旧 GNNRL 栈，主要保留 online bridge、replay、旧 checkpoint 和兼容实现

当前 KPI/对比脚本：

- `cuMAC/scripts/summarize_stageA_kpi.py`
- `cuMAC/scripts/summarize_stageB_matrix.py`
- `cuMAC/scripts/rr_vs_pf_compare.py`
- `cuMAC/scripts/aggregate_model_kpi_compare.py`

这些脚本当前已覆盖 packet delay、packet effective service rate，以及 UE-macro packet delay/rate 这类更适合公平性解释的口径。

## HGraph 部署状态

推荐 action-mode ONNX：

- `training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx`
- `training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx.metadata.json`

action-mode 输入输出：

- 输入：`obs_cell_features [1,3,5]`、`obs_ue_features [1,36,12]`、`obs_prg_features [1,3,17,8]`、`obs_post_eq_sinr [1,36,17]`、mask 与 edge metadata
- 输出：`action_ue_select [1,36]`、`action_prg_alloc [1,51]`

seed42 action-mode smoke 结论：

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

当前判断：action path 已经可用，吞吐、goodput、BLER、公平性和 p5 goodput 达到或略优于 PFQ seed42；主要缺口是 packet delay 和服务节奏，策略表现为低 PRG 利用率、低调度频率、单次集中服务。

注意：`warmup1000` 目前只是运行标签/实验布局，主 KPI 仍按全程累计。严格排除 warmup 还需要补正式 `--stats-warmup-tti 1000` 统计路径。

## 推荐复现实验

RRQ/PFQ 10-seed baseline：

```bash
./cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh \
  --build-method cmake \
  --topology-scenario 3cell \
  --seed-list 41,42,43,44,45,46,47,48,49,50 \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --tti 4000
```

HGraph action-mode C++ smoke：

```bash
.venv/bin/python training/stageb_hgraph/export_onnx.py --output-mode action --check

./cuMAC/scripts/run_stageB_main_experiment.sh \
  --build-method cmake \
  --custom-ue-prg 1 \
  --custom-policy gnnrl_model \
  --model-path training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx \
  --gnnrl-model-output-mode action \
  --gnnrl-model-use-post-eq-input 1 \
  --gnnrl-model-decode-mode argmax \
  --gnnrl-model-strict 1 \
  --topology-scenario 3cell \
  --topology-seed 42 \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --exec-mode both \
  --tti 5000 \
  --tag hgraph_action_s42_warmup1000
```

## 归档索引

归档目录只保留上下文价值，不再代表当前结论。

- [`archive/platform/`](./archive/platform/)：Aerial 平台能力、天线/precoding、PRG/RBG、状态字段审计
- [`archive/hgraph_design/`](./archive/hgraph_design/)：HGraph 设计过程、v33 训练/评测收口材料
- [`archive/legacy_gnnrl/`](./archive/legacy_gnnrl/)：旧 GNNRL online/offline/部署说明与 checklist
- [`archive/plans/`](./archive/plans/)：早期总体规划、UE/PRG 接管路线、历史 RL+GNN MVP
- [`archive/meeting_notes/`](./archive/meeting_notes/)：第一次/第二次会后确认材料
- [`archive/code_audit/`](./archive/code_audit/)：原始 PF call-chain 等历史代码审计

## 清理规则

`output/` 是忽略目录，只保留有代表性的聚合结果和当前 smoke；单 seed 原始 `stageB_main_experiment_*` 目录属于中间产物，可按需重跑。若后续要长期引用某个结果，优先保留聚合目录和生成命令，而不是堆积所有单次运行目录。
