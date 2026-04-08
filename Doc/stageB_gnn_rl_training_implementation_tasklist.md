# Stage-B GNNRL 当前 Readiness Checklist

## 1. 结论

当前仓库已经具备以下可实际使用的训练主线：

- replay v2 导出
- offline BC
- offline PPO
- ONNX 导出
- `gnnrl_model` C++ 推理落地
- online bridge
- online PPO launcher

但“现在能不能直接开训”取决于两个前置条件是否满足：

1. 当前 Python 环境里已经安装 `torch`
2. 你已经拿到 replay dump，或者准备走 online bridge

仅有 `rr_vs_pf_compare.csv` / `rr_vs_pfq_compare.csv` 还不够，因为它们只是 KPI 汇总，不是训练样本。

## 2. 当前已就绪能力

### 2.1 离线路径

已可用：

- `run_stageB_main_experiment.sh --replay-dump 1` 生成 replay v2
- `training/gnnrl/bc_train.py` 读取 replay 做 BC
- `training/gnnrl/ppo_train.py` 基于 replay 做 masked PPO
- `training/gnnrl/export_onnx.py` 导出 ONNX

当前约束：

- 仅支持 `alloc_type=0`
- 仅支持 `n_tot_cell == n_cell`
- 推荐优先用 `--action-mode prg_only_type0` 做与当前 Type-0 baseline 同口径训练

### 2.2 在线路径

已可用：

- `cuMAC/scripts/run_stageB_online_train.sh`
- `training/gnnrl/ppo_online_train.py`
- `training/gnnrl/aerial_env.py`
- `cuMAC/examples/onlineTrainBridge/*`

当前约束：

- online bridge 只能配合 `exec-mode=both`
- 不能配合 `exec-mode=gpu`
- 仅支持 `alloc_type=0`
- 仅支持 `n_tot_cell == n_cell`

### 2.3 3-cell 场景

当前主线已支持 `3cell`，包括：

- replay 采集
- offline BC / PPO
- online launcher

前提是编译期和运行期参数保持一致。

## 3. 你当前这组结果的直接判断

你现在已有：

- `3cell`
- `pfq` baseline compare
- `fading-mode=0`
- `tti=2000`
- `packet-size-bytes=3000`
- `traffic-arrival-rate=0.8`
- `topology-seed=42`
- `exec-mode=gpu`

这说明：

- baseline 路径已经跑通
- 当前 KPI 口径已经稳定
- 可以据此冻结 RL 训练场景

但它还不等于“可以立刻开始训练”，因为还缺：

- replay 数据文件，或者
- online PPO 所需的 `torch` 运行环境

## 4. 推荐开训顺序

### 4.1 先补 replay

如果你想先做离线训练，推荐先按当前场景重新采 replay：

```bash
./cuMAC/scripts/run_stageB_main_experiment.sh \
  --topology-scenario 3cell \
  --baseline-scheduler pfq \
  --build-method cmake \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --tti 2000 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 0.8 \
  --topology-seed 42 \
  --exec-mode gpu \
  --replay-dump 1 \
  --compact-output 0 \
  --tag replay_3cell_pfq_seed42
```

然后检查：

```bash
python3 cuMAC/scripts/inspect_rl_replay.py \
  --replay-dir output/<your_run>/RAYLEIGH/replay
```

### 4.2 再做 BC / offline PPO

推荐先从同口径模式开始：

```bash
python3 training/gnnrl/bc_train.py \
  --replay-dir output/<your_run>/RAYLEIGH/replay \
  --out-dir training/gnnrl/checkpoints/m1_bc_3cell_seed42 \
  --action-mode prg_only_type0
```

```bash
python3 training/gnnrl/ppo_train.py \
  --replay-dir output/<your_run>/RAYLEIGH/replay \
  --init-policy-checkpoint training/gnnrl/checkpoints/m1_bc_3cell_seed42/checkpoint_best.pt \
  --out-dir training/gnnrl/checkpoints/m2_ppo_3cell_seed42 \
  --action-mode prg_only_type0
```

### 4.3 需要在线训练时再走 online PPO

```bash
./cuMAC/scripts/run_stageB_online_train.sh \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --baseline-scheduler pfq \
  --build-method skip \
  --prbs-per-group 16 \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --tti 4000 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 0.8 \
  --packet-ttl-ms 200 \
  --topology-seed 42 \
  --exec-mode both \
  --topology-seed-mode fixed \
  --online-persistent 1 \
  --episode-boundary-mode trainer \
  --episode-horizon 1024 \
  --rollout-steps 1024 \
  --action-mode joint \
  --reward-mode goodput_reliability_blankaware \
  --ppo-epochs 6 \
  --minibatch-size 256 \
  --entropy-coef 0.003 \
  --actor-lr 2e-4 \
  --critic-lr 1e-4 \
  --target-kl 0.02 \
  --candidate-save-every-iters 20 \
  --candidate-save-start-iter 200 \
  --auto-main-eval 1 \
  --eval-build-method skip \
  --eval-decode-mode sample \
  --eval-sample-seeds 41,42,43 \
  --eval-candidate-limit 16 \
  --out-dir training/gnnrl/checkpoints/m3_online_ppo_3cell_seed42
```

## 5. 当前仍未完全闭环的地方

- 奖励里还没有直接纳入 packet-level delay
- 训练后的“统一 compare 总结”还没有完全收敛成一条覆盖所有 baseline / sample-seed / tag 约定的总脚本，但主线已经支持候选 checkpoint 自动导出并用主实验选 `deployment-best`
- 训练内 `ppo_actor_best.pt` 仍是 rollout 口径；如果要部署，checkpoint 选择仍建议结合主实验 KPI 回看，不要只看 `objective` 或 rollout `goodput`
- repo 没有自带 `torch` 安装与锁版本方案

## 6. 现在最重要的门禁

开始训练前，至少确认：

- `python3 -c "import torch"` 能通过
- replay 目录里存在 `rl_replay_meta.json`
- replay inspect 结果 `size_match: true`
- 训练时固定 `topology-scenario`、`fading-mode`、`packet-size-bytes`、`traffic-arrival-rate`、`topology-seed`
- 与当前 baseline 做同口径比较时，优先使用 `joint`
- 对固定部署目标，优先使用 tuned PPO 参数，不要退回 launcher 默认值
- 如果训练要直接产出可部署模型，优先开启 `--candidate-save-every-iters` 与 `--auto-main-eval 1`
