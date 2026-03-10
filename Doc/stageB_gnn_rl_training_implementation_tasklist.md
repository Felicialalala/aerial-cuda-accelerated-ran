# Stage-B GNN+RL 训练实施计划与任务清单

> 进行状态更新（2026-03-10）：
> - M0：已完成
> - M1：已完成
> - M2：训练闭环已完成（可训练/可导出），算法效果仍需继续优化
> - M3：接入已完成（含 TensorRT 插件修复），需按 600 TTI 持续回归
> - M4：未完成
> - M5：未开始

## 1. 目标与范围

- 目标：将当前 `--custom-policy gnnrl` 从启发式打分升级为可训练的 GNN+RL 调度器，同时保持 Stage-B 输出口径不变（继续产出 `ue_kpi.csv` / `stageB_kpi_matrix.csv`）。
- 范围：优先支持 `4T4R + Type-0 + 7cell/56UE + Rayleigh/CDL` 的 Stage-B 主流程。
- 非目标（MVP 阶段）：暂不引入严格 HOL delay、TE/DI、信息瓶颈正则；这些在增强阶段增量加入。

---

## 2. 当前现状（实现前共识）

- 当前 `gnnrl` 并非 RL 训练模型，而是 C++ 侧加权打分策略：
  - `cuMAC/examples/customScheduler/CustomUePrgScheduler.cpp`
  - `cuMAC/examples/customScheduler/CustomUePrgScheduler.h`
- 当前 Stage-B 已支持自定义调度入口，但没有 transition replay 持久化与训练闭环：
  - `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
  - `cuMAC/scripts/run_stageB_main_experiment.sh`

---

## 3. 总体路线（M0~M5）

## M0 数据闭环（必须先做）

- 新增 per-TTI transition 导出（obs/mask/action/reward/next_obs/done）。
- 保证导出不影响现有 KPI 统计流程。

## M1 模型闭环（可离线训练）

- 实现 Cell 图编码 + UE/PRG 动作头。
- 先用 baseline 轨迹做行为克隆（BC）预热，防止 RL 冷启动崩盘。

## M2 RL 训练闭环（Masked PPO）

- 使用动作掩码约束合法动作（buffer=0 不可调度等）。
- 在线/离线混合训练，先单场景，再多 seed。

## M3 推理部署（接入 Stage-B）

- 导出 ONNX/TorchScript，C++ 侧调用推理。
- 推理失败或非法动作时自动回退启发式策略。

## M4 回归评估（与 baseline 对比）

- 固定 `seed=42` 回归，跑 baseline 与 gnnrl 同口径对比。
- 保证无大规模 `scheduled_ratio=0` UE。

## M5 增强版（论文向）

- 引入严格 HOL delay、TE/DI 边剪枝、信息瓶颈正则、跨场景泛化评估。

---

## 4. 详细任务清单（按文件落地）

## 4.1 M0：Transition Replay 导出

- [x] 在 `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp` 增加 replay writer 生命周期管理：
  - 初始化（run start）
  - 每 TTI 采样写入
  - 结束 flush + index 文件
- [x] 新增模块目录 `cuMAC/examples/rlReplay/`：
  - `ReplayWriter.h/.cpp`（二进制/npz/parquet 三选一，推荐二进制+json meta）
  - `FeaturePack.h/.cpp`（将 C 结构体打包为训练特征）
  - `RewardBuilder.h/.cpp`（按既定 reward 公式计算）
- [x] replay 样本字段定义（固定 schema）：
  - `obs.cell_features`
  - `obs.ue_features`
  - `obs.edge_index / edge_attr`
  - `action.ue_select`
  - `action.prg_alloc`
  - `action_mask`
  - `reward.scalar + reward_terms`
  - `next_obs`
  - `done`
  - `meta`（tti, seed, profile, ds, policy）
- [x] 在 `cuMAC/scripts/run_stageB_main_experiment.sh` 增加开关参数：
  - `--replay-dump 0|1`
  - `--replay-dir <path>`
  - 导出对应环境变量供 `main.cpp` 使用。

说明：`FeaturePack/RewardBuilder` 当前以 `ReplayWriter` 内部实现为主，尚未拆分为独立文件，但功能链路已可用。

## 4.2 M1：GNN 模型与 BC 预训练

- [x] 新增训练代码目录 `training/gnnrl/`：
  - `dataset.py`（读取 replay）
  - `model.py`（Cell encoder + UE/PRG policy heads）
  - `masks.py`（动作 mask 处理）
  - `bc_train.py`（行为克隆训练）
  - `export_onnx.py`（模型导出）
- [x] 模型输入设计：
  - Cell 节点：load / active UE / mean SINR / BLER / util
  - UE 节点：`wbSinr`, `avgRatesActUe`, `bufferSize`, `tbErrLastActUe`, stale
  - Edge：邻区耦合强度 + 邻区负载
- [x] 模型输出设计（两阶段）：
  - Head-A：每 cell 的候选 UE 评分 logits
  - Head-B：每 `cell x PRG` 的 `UE/NO_TX` logits
- [x] BC 训练流程：
  - 用 baseline actions 监督训练
  - 先收敛到合法动作率 > 99%
  - 保存 `checkpoint_best.pt` 与 `model.onnx`

## 4.3 M2：Masked PPO 训练

- [x] 新增 `ppo_train.py`：
  - 支持 GAE、clip ratio、entropy、value loss
  - 动作采样前应用 mask（非法动作 logits=-inf）
- [x] 奖励函数（MVP）：
  - `+ alpha * sum_throughput`
  - `+ beta * edge_ue_throughput_proxy`
  - `- gamma * residual_buffer`
  - `- delta * interference_collision_proxy`
  - `- eta * unfairness_proxy`
- [x] 训练编排：
  - 单 seed（42）冒烟
  - 多 seed（42/43/44）稳定性
  - 每 N iter 执行离线 eval 并保存最优模型

说明：M2 当前“训练闭环可运行”，但离线训练策略收益尚未达到预期，需在 M4 前继续调参/扩充数据。

## 4.4 M3：C++ 推理接入

- [x] 在 `cuMAC/examples/customScheduler/` 增加推理适配器：
  - `GnnRlPolicyRuntime.h/.cpp`
  - 输入：FeaturePack
  - 输出：`setSchdUePerCellTTI` + `allocSol`
- [x] 在 `CustomUePrgScheduler.cpp` 增加策略分支：
  - `legacy`（保留）
  - `gnnrl_heuristic`（当前启发式，保留）
  - `gnnrl_model`（新模型推理）
- [x] 增加回退机制：
  - 推理超时、返回非法 action、mask 全空 -> fallback 到 `legacy`
- [x] 在 `run_stageB_main_experiment.sh` 增加参数：
  - `--custom-policy gnnrl_model`
  - `--model-path <onnx>`
  - `--policy-timeout-ms <int>`

说明：已完成 TensorRT 插件修复（`initLibNvInferPlugins` + `nvinfer_plugin` 链接），`gnnrl_model` 冒烟运行不再报插件缺失并可正常出 KPI。

## 4.5 M4：评估与回归

- [ ] 使用当前 Stage-B 汇总脚本保持同口径输出：
  - `ue_kpi.csv`
  - `kpi_summary.json`
  - `stageB_kpi_matrix.csv`
- [ ] 新增对比脚本 `cuMAC/scripts/compare_stageB_runs.py`：
  - baseline vs gnnrl_model 自动对比
  - 关键指标红线检查
- [ ] 红线告警（失败即判不通过）：
  - `scheduled_ratio==0` UE 数 > 2
  - `cluster_sum_throughput < baseline * 0.90`
  - `residual_buffer_ratio > baseline * 1.10`

## 4.6 M5：增强项（第二阶段）

- [ ] 严格 HOL delay 埋点：
  - traffic queue 中新增 packet timestamp
  - 导出 UE 级 HOL p50/p95/p99
- [ ] TE/DI 边剪枝：
  - 从 replay 离线计算边重要度
  - 训练时动态裁剪 edge
- [ ] 信息瓶颈正则：
  - 在 message passing 加 KL 正则项

---

## 5. 里程碑与验收标准

## 里程碑 A（M0 完成）

- 状态：已完成
- 可生成 replay 文件，且 Stage-B 结果文件不受影响。

## 里程碑 B（M1 完成）

- 状态：已完成
- BC 模型可离线训练并导出 ONNX。

## 里程碑 C（M2+M3 完成）

- 状态：已完成（接入维度）
- `--custom-policy gnnrl_model` 可跑通 Stage-B 冒烟，不崩溃。
- 备注：性能收益需在 M4 回归阶段进一步确认。

## 里程碑 D（M4 完成）

- 状态：未完成
- 相同 seed 下满足：
  - `Cluster Sum Throughput >= baseline * 0.90`
  - `UE Throughput Jain >= baseline`
  - `scheduled_ratio_p5` 明显高于当前启发式 `gnnrl_fix`
  - `scheduled_ratio==0` UE 数接近 0（<=2）

---

## 6. 运行与验证命令模板

## 6.1 baseline 对照

```bash
./cuMAC/scripts/run_stageB_main_experiment.sh \
  --build-method cmake \
  --tti 600 \
  --custom-ue-prg 0 \
  --topology-seed 42 \
  --tag baseline
```

## 6.2 采集 replay（示例）

```bash
./cuMAC/scripts/run_stageB_main_experiment.sh \
  --build-method cmake \
  --tti 600 \
  --custom-ue-prg 1 \
  --custom-policy legacy \
  --topology-seed 42 \
  --replay-dump 1 \
  --replay-dir output/replay_seed42 \
  --tag replay_legacy
```

## 6.3 推理模型对比（示例）

```bash
./cuMAC/scripts/run_stageB_main_experiment.sh \
  --build-method cmake \
  --tti 600 \
  --custom-ue-prg 1 \
  --custom-policy gnnrl_model \
  --model-path training/gnnrl/checkpoints/model.onnx \
  --topology-seed 42 \
  --tag gnnrl_model
```

---

## 7. 风险与缓解

- 风险：训练策略出现“强者通吃”，大量 UE 长期不调度。
  - 缓解：mask + 份额惩罚 + min-service 约束，且设置红线门禁。
- 风险：推理输出非法动作导致运行异常。
  - 缓解：C++ 侧强校验 + fallback 到 legacy。
- 风险：训练与部署特征不一致导致性能失真。
  - 缓解：统一 `FeaturePack` 代码路径，训练和推理共享同一 schema。

---

## 8. 建议执行顺序（两周版本）

- 第 1-3 天：M0 replay 导出 + schema 固化
- 第 4-6 天：M1 BC 模型 + ONNX 导出
- 第 7-10 天：M2 PPO 训练 + 初步对比
- 第 11-12 天：M3 C++ 推理接入 + fallback
- 第 13-14 天：M4 回归报告 + 指标门禁
