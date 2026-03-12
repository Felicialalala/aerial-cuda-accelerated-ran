# Stage-B GNNRL 离线训练实施文档

## 1. 文档定位

- 本文档只覆盖离线训练链路：`Replay -> BC -> PPO(offline) -> ONNX -> Stage-B 推理回归`。
- 在线交互训练（PPO + Aerial 实时环境）单独见 `Doc/stageB_gnn_rl_online_training_design.md`。

## 2. 当前状态（2026-03-10）

- M0 replay 导出：已完成。
- M1 BC 训练：已完成，可稳定得到可用 warm-start。
- M2 离线 PPO：已完成实现，训练稳定性较之前显著提升。
- M3 推理接入：已完成（TensorRT 插件链路已修复）。
- M4 回归评估：已完成首轮闭环通过，`gnnrl_model` 在 Rayleigh 基线场景上已超过 PF baseline（见 2.1）。

### 2.1 最新闭环结果（2026-03-10 09:53，Rayleigh，`tti=600`，`seed=42`）

- baseline：`cluster_sum_throughput_mbps=211.221094`
- gnnrl_model：`cluster_sum_throughput_mbps=239.736332`（较 baseline `+13.5%`）
- 关键副指标同步改善：
- `residual_buffer_ratio`: `0.4304 -> 0.3425`
- `ue_throughput_p50_mbps`: `3.1148 -> 4.2873`
- `queue_delay_p95_ms`: `29216 -> 14894`

### 2.2 当前“推荐配置”（已验证）

- 推理保护参数（建议固化）：
- `CUMAC_GNNRL_MODEL_NO_UE_BIAS=0.30`
- `CUMAC_GNNRL_MODEL_MIN_SCHED_RATIO=0.75`
- `CUMAC_GNNRL_MODEL_NO_PRG_BIAS=0.20`
- `CUMAC_GNNRL_MODEL_MIN_PRG_RATIO=0.90`
- `CUMAC_GNNRL_MODEL_MAX_PRG_SHARE_PER_UE=0.20`

说明：

- `NO_UE/NO_PRG` 偏置 + 最小分配比例用于抑制“全 NO_CLASS”坍塌。
- `MAX_PRG_SHARE_PER_UE` 用于抑制“少数 UE 吃满 PRG”坍塌。

### 2.3 基线场景/布局参考文档（已检查）

- 最新基线参考（建议优先看）：`Doc/stageB_baseline_scenario_reference.md`
- 历史大场景配置梳理（含 19-site 版本说明）：`Doc/current_stageB_effective_configuration.md`
- 在线训练设计（与场景接口相关）：`Doc/stageB_gnn_rl_online_training_design.md`

## 3. 离线链路实现现状

- Replay v2 已包含 `action_mask_cell_ue`，mask/标签一致性问题已修复。
- UE 头已加入 slot 可辨识性（global/local slot embedding），不再卡在 `val_ue_acc≈0.125`。
- PPO 训练已修复以下实现问题：
- critic 输入状态归一化（dataset 统计均值/方差）。
- reward 归一化后再做 GAE/returns。
- actor/critic 分离优化器与分离梯度裁剪。
- value loss 支持 huber，缓解 critic 大残差冲击。

## 4. 最新训练结果解读

### 4.1 BC（`lr=1e-4`）

- 峰值 `val_ue_acc` 约 `0.403`（epoch 18），较旧版 `~0.125` 明显提升。
- `best_val_loss` 出现在 epoch 22，末段虽有回落，但 `checkpoint_best.pt` 可规避末段抖动影响。

### 4.2 PPO（修复后参数）

- 本次 50 iter 汇总（`ppo_summary.json`）：
- `best_iter=26`，`best_objective=0.0601`。
- `value_loss` 从早期约 `9.34` 下降到末段最低约 `2.71`。
- `clipfrac` 不再长期锁死 0，后半段均值约 `0.111`，说明策略已发生有效更新。
- 仍有波动风险：末段部分 iter `clipfrac` 偏高（最高约 `0.48`），`approx_kl` 偶发尖峰，需结合回归 KPI 选模型。

结论：

- 当前 PPO 已从“几乎不学习”切换到“可学习但波动偏大”。
- 进入 M4 评估时，应优先比较 `ppo_actor_best.pt` 与 `m1_bc/checkpoint_best.pt` 的闭环 KPI，而不是只看训练 objective。

## 5. 标准命令（绝对路径 + replay 自动探测）

```bash
# 0) 绝对路径变量（如仓库不在 /opt/nvidia/cuBB，请改 ROOT）
ROOT="/opt/nvidia/cuBB"
RUN_SCRIPT="${ROOT}/cuMAC/scripts/run_stageB_main_experiment.sh"
INSPECT_SCRIPT="${ROOT}/cuMAC/scripts/inspect_rl_replay.py"
BC_SCRIPT="${ROOT}/training/gnnrl/bc_train.py"
PPO_SCRIPT="${ROOT}/training/gnnrl/ppo_train.py"
EXPORT_SCRIPT="${ROOT}/training/gnnrl/export_onnx.py"

TS="$(date +%Y%m%d_%H%M%S)"
TAG_REPLAY="replay_m4_seed42_v2"
TAG_BASELINE="baseline_m4_seed42_v2"
TAG_GNNRL="gnnrl_m4_seed42_v2"
REPLAY_BASE="${ROOT}/output/replay_m4_seed42_v2_${TS}"
CKPT_BASE="${ROOT}/training/gnnrl/checkpoints/m4_seed42_v2_${TS}"
mkdir -p "${CKPT_BASE}"

# 0.1) 代码目录一致性检查（重要）
# 如果下面两个路径不一致，说明 /opt/nvidia/cuBB 与当前编辑目录不是同一份代码，不会自动同步。
echo "PWD_REAL=$(readlink -f "$PWD")"
echo "ROOT_REAL=$(readlink -f "${ROOT}")"

# 0.2) 手动重编 multiCellSchedulerUeSelection（修改 customScheduler 后建议先执行）
BUILD_DIR="${ROOT}/build.$(uname -m)"
if [[ -d "${BUILD_DIR}" ]]; then
  cmake --build "${BUILD_DIR}" --target multiCellSchedulerUeSelection -j"$(nproc)"
else
  echo "[WARN] ${BUILD_DIR} 不存在，回退到 phase4 构建脚本"
  "${ROOT}/testBenches/phase4_test_scripts/build_aerial_sdk.sh" --build_dir "${BUILD_DIR}"
fi

# 1) 采集 replay（native PF 监督来源，推荐）
"${RUN_SCRIPT}" \
  --build-method cmake \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --tti 600 \
  --custom-ue-prg 0 \
  --topology-seed 42 \
  --replay-dump 1 \
  --replay-dir "${REPLAY_BASE}" \
  --compact-output 0 \
  --tag "${TAG_REPLAY}"

# 可选：确认本次 replay 确实是 native PF 路径
LATEST_REPLAY_RUN="$(ls -dt "${ROOT}"/output/stageB_main_experiment_"${TAG_REPLAY}"_* | head -n1)"
grep -En "native UE selection \\+ PRG allocation|Using CPU multi-cell PF" \
  "${LATEST_REPLAY_RUN}/RAYLEIGH/run.log"

# 2) 自动探测 replay 目录
CANDIDATE_A="${REPLAY_BASE}/RAYLEIGH"
LATEST_RUN_BASE="$(ls -dt "${ROOT}"/output/stageB_main_experiment_"${TAG_REPLAY}"_* 2>/dev/null | head -n1 || true)"
CANDIDATE_B=""
if [[ -n "${LATEST_RUN_BASE}" ]]; then
  CANDIDATE_B="${LATEST_RUN_BASE}/RAYLEIGH/replay"
fi

if [[ -f "${CANDIDATE_A}/rl_replay_meta.json" ]]; then
  REPLAY_DIR="${CANDIDATE_A}"
elif [[ -n "${CANDIDATE_B}" && -f "${CANDIDATE_B}/rl_replay_meta.json" ]]; then
  REPLAY_DIR="${CANDIDATE_B}"
else
  echo "[ERROR] replay meta not found"
  echo "  - ${CANDIDATE_A}"
  echo "  - ${CANDIDATE_B}"
  exit 1
fi

echo "Resolved REPLAY_DIR=${REPLAY_DIR}"

# 3) 完整性与多样性检查（建议作为硬门禁）
python3 "${INSPECT_SCRIPT}" --replay-dir "${REPLAY_DIR}" | tee "${CKPT_BASE}/replay_inspect.txt"

if command -v rg >/dev/null 2>&1; then
  GATE_CMD='rg -q "PRG labels highly concentrated|UE labels are highly deterministic"'
else
  GATE_CMD='grep -Eq "PRG labels highly concentrated|UE labels are highly deterministic"'
fi

if eval "${GATE_CMD} \"${CKPT_BASE}/replay_inspect.txt\""; then
  echo "[ERROR] replay action diversity gate failed; please re-collect replay."
  exit 2
fi

# 4) BC 重训
python3 "${BC_SCRIPT}" \
  --replay-dir "${REPLAY_DIR}" \
  --out-dir "${CKPT_BASE}/m1_bc" \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-4 \
  --seed 42

# 5) PPO 重训（离线）
python3 "${PPO_SCRIPT}" \
  --replay-dir "${REPLAY_DIR}" \
  --init-policy-checkpoint "${CKPT_BASE}/m1_bc/checkpoint_best.pt" \
  --out-dir "${CKPT_BASE}/m2_ppo" \
  --iters 50 \
  --ppo-epochs 6 \
  --minibatch-size 128 \
  --actor-lr 1e-4 \
  --critic-lr 3e-4 \
  --target-kl 0.05 \
  --normalize-state 1 \
  --normalize-reward 1 \
  --value-loss huber \
  --value-huber-beta 10.0 \
  --seed 42

# 6) 导出 ONNX（PPO best）
python3 "${EXPORT_SCRIPT}" \
  --checkpoint "${CKPT_BASE}/m2_ppo/ppo_actor_best.pt" \
  --out "${CKPT_BASE}/m2_ppo/model.onnx" \
  --opset 18

# 7) baseline 回归
"${RUN_SCRIPT}" \
  --build-method cmake \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --tti 600 \
  --custom-ue-prg 0 \
  --topology-seed 42 \
  --compact-output 0 \
  --tag "${TAG_BASELINE}"

# 8) gnnrl_model 回归（启用解码防坍塌保护）
CUMAC_GNNRL_MODEL_NO_UE_BIAS=0.30 \
CUMAC_GNNRL_MODEL_MIN_SCHED_RATIO=0.75 \
CUMAC_GNNRL_MODEL_NO_PRG_BIAS=0.20 \
CUMAC_GNNRL_MODEL_MIN_PRG_RATIO=0.90 \
CUMAC_GNNRL_MODEL_MAX_PRG_SHARE_PER_UE=0.20 \
"${RUN_SCRIPT}" \
  --build-method cmake \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --tti 600 \
  --custom-ue-prg 1 \
  --custom-policy gnnrl_model \
  --model-path "${CKPT_BASE}/m2_ppo/model.onnx" \
  --topology-seed 42 \
  --compact-output 0 \
  --tag "${TAG_GNNRL}"

# 9) 验证是否真的走到 gnnrl_model 推理路径（非 fallback）
LATEST_GNNRL_RUN="$(ls -dt "${ROOT}"/output/stageB_main_experiment_"${TAG_GNNRL}"_* | head -n1)"
GNNRL_LOG="${LATEST_GNNRL_RUN}/RAYLEIGH/run.log"
if command -v rg >/dev/null 2>&1; then
  rg -n "\\[GNNRL_MODEL\\] init|CustomUePrgScheduler config: policy=gnnrl_model" "${GNNRL_LOG}"
else
  grep -En "\\[GNNRL_MODEL\\] init|CustomUePrgScheduler config: policy=gnnrl_model" "${GNNRL_LOG}"
fi

# 如果仍出现“少数UE吃满PRG”，可进一步下调：
#   CUMAC_GNNRL_MODEL_MAX_PRG_SHARE_PER_UE=0.12~0.20

# 10) 快速检查每小区每TTI平均已选UE槽位（避免被低slot hint限流）
python3 - <<'PY' "${LATEST_GNNRL_RUN}/RAYLEIGH/ue_kpi.csv"
import csv, collections, sys
p=sys.argv[1]
by_sum=collections.defaultdict(int)
by_n=collections.defaultdict(int)
with open(p,newline='') as f:
    rd=csv.DictReader(f)
    for r in rd:
        try:
            cid=int(r['cell_id']); int(r['ue_id'])
        except:
            continue
        by_sum[cid]+=int(r['scheduled_tti_count'])
        by_n[cid]+=1
for cid in sorted(by_sum):
    # 600 comes from --tti 600 in the standard command.
    print(f"cell{cid}: avg_slots_per_tti={by_sum[cid]/600.0:.3f}, mean_sched_ratio={by_sum[cid]/(600.0*by_n[cid]):.3f}")
PY
```

## 6. M4 评估建议

- 至少比较三条策略：`baseline`、`BC best`、`PPO best`。
- 红线门禁建议：
- `cluster_sum_throughput >= baseline * 0.90`
- `scheduled_ratio==0` UE 数 `<= 2`
- `residual_buffer_ratio <= baseline * 1.10`
- 当前单 seed（42）已通过并超过 baseline，下一步应做 `seed=41/42/43` 与 `TDL_PRBG` 复验。
- 若 PPO 在 KPI 上不稳，优先用 BC best 部署，继续离线调参。

## 7. 风险与缓解

- 风险：PPO 后半段更新过猛导致策略波动。
- 缓解：降低 `actor-lr`，并结合 KL/clipfrac 选择 checkpoint。
- 风险：离线目标与在线部署目标不完全一致。
- 缓解：后续转在线交互训练，并保留离线 BC 作为 warm-start。
