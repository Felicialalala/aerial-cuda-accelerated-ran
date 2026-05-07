# Stage-B Sparse Entity Graph Stack

This directory is the parallel first-version graph stack for the new Stage-B scheduler work.

Current retained reference run: `training/stageb_hgraph/runs/online_ppo_seed42_svd_rawbridge_joint_v33_capSlack3000_simpleExecutor_ablation`.

It is intentionally separate from `training/gnnrl/`:

- reuse the Stage-B simulator scenario/build flow
- do not reuse the old GNN+RL model/training scripts as the implementation base
- keep the first-version scope fixed to:
  - `Cell / UE / P_{i,g}` nodes
  - `Cell-Cell / Cell-UE / Cell-PRG / UE-PRG / PRG-PRG` edges
  - template encoders implemented as structured MLPs
  - one later hetero-graph message-passing layer

## Current module boundary

- `feature_snapshot.py`
  - builds the first-version `cell_features / ue_features / prg_features / static_meta`
  - current HGraph input layout is now:
    - `cell_features[C,5]`
    - `ue_features[U,12]`
    - `prg_features[C*P,8]`
  - consumes existing Stage-B observation dicts:
    - `obs_cell_features`
    - `obs_ue_features`
    - `obs_prg_features`
    - optional `post_eq_sinr` with shape `[active_ue, prg, layer]`
    - `action_mask_ue`
    - `action_mask_cell_ue`
    - `action_mask_prg_cell`
- `template_encoders.py`
  - `CellTemplateEncoder`
  - `UeTemplateEncoder`
  - `PrgTemplateEncoder`
  - these are structured encoders, not layer-internal GNNs
- `edge_generator.py`
  - builds `E_CC / E_CU / E_CP / E_UP / E_PP`
  - keeps `Cell-PRG` as static affiliation edges
  - supports two `UE-PRG` modes:
    - post-eq candidate edges when raw `post_eq_sinr` is exported by the online bridge
    - shared-pool candidate edges when only compact observation features are available
- `mask_checker.py`
  - validates graph legality and mask alignment
- `sanity_logger.py`
  - emits per-TTI graph statistics for static/dynamic sanity checks
- `graph_ops.py`
  - still keeps `UE-PRG` packing utilities for graph-side heuristic labeling/debug
  - now primarily provides `joint` Type-0 target/decode helpers for `slot -> UE` and `PRG -> slot`
- `model.py`
  - `StageBHGraphPolicy`
  - uses `Cell / UE / PRG` template encoders plus one sparse hetero message-passing stack
  - exposes a `joint` Type-0 action head:
    - `slot -> UE`
    - `PRG -> slot`
- `teacher.py`
  - supports:
    - `pfq_passthrough` from the online bridge
    - `heuristic_graph` as a fallback/debug teacher
- `reward_utils.py`
  - reward-term decoding from the online bridge
  - first PPO shaping helper for the Stage-B v1 reward
- `online_ppo.py`
  - first online PPO fine-tuning entry for the sparse entity graph stack
  - supports warm-start from the imitation checkpoint
  - also supports direct PPO from scratch without imitation initialization
- `export_onnx.py` / `check_onnx_consistency.py`
  - fixed `3cell/36UE/17PRG` deployment export and PyTorch-vs-ONNX validation
  - `--output-mode logits` keeps the legacy `ue_logits/prg_logits` ABI
  - `--output-mode action` adds `obs_post_eq_sinr` and emits `action_ue_select/action_prg_alloc`
  - action-mode C++ finalization handles Type-0 legality, demand-cap, and fallback

## Reuse strategy

Recommended reuse boundary:

1. Keep `cuMAC/scripts/run_stageB_main_experiment.sh` as the canonical Stage-B scenario/build/run entry.
2. Use the new independent launcher:
   - `cuMAC/scripts/run_stageB_hgraph_online_sanity.sh`
3. Keep this stack separate from `cuMAC/scripts/run_stageB_online_train.sh` and `training/gnnrl/*`.
4. Use the current online bridge observation path first, because it already exposes `obs_prg_features`.

Current validated runtime boundary:

- fixed seed + `online_persistent=1`
  - the simulator is now reused as one continuous online stream across PPO iterations
  - this is the recommended first formal-training path
- `--sim-env`
  - may be repeated
  - all simulator env entries are preserved, including:
    - `CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1`
    - reward mode
    - TTL
    - topology/runtime overrides

## Independent online entry

The new online entry point is:

```bash
./cuMAC/scripts/run_stageB_hgraph_online_sanity.sh \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 42 \
  --baseline-scheduler pfq \
  --episode-horizon 400 \
  --max-steps 400 \
  --episodes 1 \
  --action-source noop
```

This launcher:

- reuses Stage-B scenario/build arguments
- builds the simulator through `run_stageB_main_experiment.sh --build-only 1`
- launches a separate sparse-graph sanity runner instead of PPO
- defaults to `--precoding svd` so HGraph runs align with the current SVD baseline unless you override it
- enables raw `postEqSinr` export for HGraph through `CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1`; this switches `E_UP` from `shared_cell_pool` to real per-PRG Top-K UE candidates
- writes:
  - `graph_sanity_tti_stats.jsonl`
  - `graph_sanity_summary.json`

The per-TTI sanity rows now include dynamic PRG feature statistics, so you can
inspect time variation directly from the first online sanity run:

- `prg_prev_assigned_ratio`
- `prg_top1_sinr_db_mean/p10/p50/p90`
- `prg_top2_gap_db_mean/p50/p90`
- `prg_neighbor_max_top1_sinr_db_mean/p50/p90`
- `prg_same_prg_conflict_ratio_mean/p90/max`
- `prg_ici_proxy_mean/p90/max`
- `candidate_mode`
- `post_eq_layer_dim`

To validate action-history-dependent PRG features such as `prevPrgAssigned`,
switch the sanity runner from `noop` to `random_legal`:

```bash
./cuMAC/scripts/run_stageB_hgraph_online_sanity.sh \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 42 \
  --baseline-scheduler pfq \
  --episode-horizon 400 \
  --max-steps 400 \
  --episodes 1 \
  --action-source random_legal \
  --random-blank-prob 0.25 \
  --random-seed 1234
```

Current action-source choices:

- `noop`
  - keeps all slots / PRGs blank
  - best for pure graph legality sanity
- `random_legal`
  - fills legal slots with random associated live UEs
  - assigns each legal PRG to a random same-cell slot with configurable blank probability
  - useful for validating `prevPrgAssigned` and other action-history-dependent PRG features

## Important current caveat

Current replay v2 does not persist `obs_prg_features`.

That means:

- graph sanity based on the full `Cell / UE / PRG` state is easiest to start from the online bridge
- offline imitation with PRG nodes will either need:
  - a small replay schema upgrade to write PRG features, or
  - the current online PFQ teacher passthrough path

This caveat is a data-interface gap, not a blocker for the overall architecture.

## First warm-start training path

The first runnable training path now exists as a separate online imitation entry:

```bash
./cuMAC/scripts/run_stageB_hgraph_online_imitation.sh \
  --build-method cmake \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 42 \
  --baseline-scheduler pfq \
  --episode-horizon 400 \
  --max-steps 400 \
  --episodes 10 \
  --hidden-dim 128 \
  --message-layers 2 \
  --lr 3e-4 \
  --out-dir training/stageb_hgraph/runs/online_imitation_seed42
```

Current warm-start scope:

- model side:
  - `CellTemplateEncoder`
  - `UeTemplateEncoder`
  - `PrgTemplateEncoder`
  - sparse hetero encoder on `Cell-Cell / Cell-UE / Cell-PRG / UE-PRG / PRG-PRG`
  - `joint` Type-0 action head (`slot -> UE` + `PRG -> slot`)
- training side:
  - online imitation
  - teacher forcing through the bridge
  - native `PFQ teacher passthrough` as the recommended warm-start source
  - heuristic graph teacher as an optional fallback/debug source

Important current boundary:

- first warm start now prefers the bridge-exported PFQ reference action itself
- the heuristic graph teacher is kept mainly for ablations/debug when you explicitly select `--teacher-mode heuristic_graph`
- this keeps the first warm-start path aligned with the native baseline while preserving the independent HGraph codebase

## PPO fine-tuning path

After warm start is verified, the next entry point is:

```bash
./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --build-method cmake \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 42 \
  --baseline-scheduler pfq \
  --episode-horizon 400 \
  --rollout-steps 400 \
  --iterations 50 \
  --hidden-dim 128 \
  --message-layers 2 \
  --init-checkpoint training/stageb_hgraph/runs/online_imitation_seed42/hgraph_imitation_best.pt \
  --out-dir training/stageb_hgraph/runs/online_ppo_seed42
```

Current PPO boundary:

- actor:
  - reuses `StageBHGraphPolicy`
  - samples in the `joint` Type-0 action space (`slot -> UE`, then `PRG -> slot`)
- critic:
  - uses pooled graph embeddings from the actor encoder
- reward:
  - supports `raw_env`
  - supports `stageb_v1` shaping:
    - goodput positive term
    - TB error / expiry penalties
    - conflict / ICI penalties
    - urgent backlog penalty
  - supports `stageb_blankaware` shaping:
    - mirrors the latest `gnnrl` blank-aware reward intent
    - goodput positive term
    - active-PRG goodput efficiency bonus
    - TB error / expiry penalties
    - harmful packing penalty based on `throughput-goodput`, utilization, and reuse
    - small fairness tie-breaker

Why PPO is still the first recommended RL step here:

- it fits the current online bridge and on-policy rollout loop naturally
- it handles masked discrete actions cleanly once the policy is factorized by PRG token
- it benefits directly from the imitation checkpoint as a warm start
- it is usually a safer first online baseline than more sample-efficient but more brittle off-policy alternatives

What PPO is not claiming here:

- it is not guaranteed to be the globally best RL algorithm for this problem
- it is the most practical first RL continuation for the current first-version stack

## Direct PPO path

At the current code stage, imitation is optional rather than mandatory.

The recommended direct-PPO path is:

- keep the current HGraph encoder stack
- keep the `joint` Type-0 action head:
  - `slot -> UE`
  - `PRG -> slot`
- keep `post_eq_topk` candidate generation enabled
- start from fixed-seed direct PPO first
- only move to multi-seed after the fixed-seed run is stable

Recommended fixed-seed smoke / first formal run:

```bash
CUDA_VISIBLE_DEVICES=1 ./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --build-method skip \
  --tti 4000 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 42 \
  --topology-seed-mode fixed \
  --baseline-scheduler pfq \
  --online-persistent 1 \
  --episode-horizon 512 \
  --rollout-steps 512 \
  --iterations 20 \
  --ue-prg-topk 6 \
  --hidden-dim 192 \
  --message-layers 3 \
  --device cuda \
  --lr 1.5e-4 \
  --actor-lr 1.0e-4 \
  --critic-lr 3.0e-4 \
  --weight-decay 1e-5 \
  --entropy-coef 0.004 \
  --target-kl 0.05 \
  --update-epochs 2 \
  --minibatch-size 256 \
  --reward-mode stageb_blankaware \
  --rollout-log-every-steps 500 \
  --update-log-every-minibatches 2 \
  --best-metric goodput \
  --early-stop-patience 0 \
  --sim-env CUDA_VISIBLE_DEVICES=0 \
  --out-dir training/stageb_hgraph/runs/online_ppo_seed42_svd_blankaware_gpu_split_fix
```

This GPU split means:

- trainer runs on GPU1 through outer-shell `CUDA_VISIBLE_DEVICES=1`
- simulator runs on GPU0 through `--sim-env CUDA_VISIBLE_DEVICES=0`

This is process-level split execution, not one PPO update being data-parallel over two GPUs.

Current online PPO logging / checkpoint behavior:

- every `rollout-log-every-steps` TTI, PPO prints one rollout heartbeat line
- every `update-log-every-minibatches` minibatches, PPO prints one PPO-update heartbeat line
- every PPO iteration, PPO prints:
  - rollout summary
  - `collect_time_s`
  - `update_time_s`
  - `iter_time_s`
- checkpoints:
  - `ppo_actor_last.pt`
  - `ppo_actor_best.pt`
  - `ppo_actor_threshold_best.pt`
  - `ppo_actor_absolute_best.pt`
  - `ppo_train_last.pt`
  - `ppo_train_best.pt`
  - `ppo_train_threshold_best.pt`
  - `ppo_train_absolute_best.pt`
- `ppo_actor_best.pt` and `ppo_train_best.pt` are backward-compatible aliases of the threshold-best checkpoints
- threshold-best means:
  - a checkpoint is only promoted when it improves the selected best metric by at least `early-stop-min-delta`
  - this is the checkpoint family used by early stop
- absolute-best means:
  - the true peak of the selected best metric, regardless of `early-stop-min-delta`
  - this is the recommended checkpoint family for final offline / multi-seed evaluation
- best-checkpoint selection is controlled by `--best-metric {reward,goodput}`
- recommended current defaults:
  - `--best-metric goodput`
  - `--early-stop-patience 60`
  - `--early-stop-min-delta 1.0`
  - `--early-stop-warmup-iters 50`
- if the immediate goal is to retrain a clean new model under the current validated Stage-B setup, the recommended choice is:
  - keep the training/evaluation traffic aligned:
    - `3000 bytes / packet`
    - `1 pkt / TTI`
    - `TTL = 200 ms`
    - `3cell`
    - `36 UE`
    - `precoding = svd`
  - disable early stop:
    - `--early-stop-patience 0`
- early stop is optional and disabled when `--early-stop-patience 0`

## Absolute Best vs Threshold Best

At the current code stage, the PPO runner tracks two notions of "best":

- `threshold_best`
  - used for early-stop semantics
  - only updates when `score > threshold_best_score + early_stop_min_delta`
- `absolute_best`
  - updates whenever `score > absolute_best_score`
  - ignores `early_stop_min_delta`

This solves the common case where:

- training really does reach a slightly higher goodput later
- but the improvement is smaller than `early_stop_min_delta`
- so threshold-best stays at an earlier iteration while absolute-best moves forward

For final model reporting and RRQ/PFQ comparison, prefer:

- `ppo_actor_absolute_best.pt`

For reproducing the exact early-stop-selected checkpoint, use:

- `ppo_actor_threshold_best.pt`

## Multi-Seed Best-Checkpoint Evaluation

To evaluate one fixed HGraph checkpoint with the same multi-topology-seed averaging style used by RRQ/PFQ, use:

The wrapper default decode path is now `argmax`, which matches the deterministic deployment-style policy. Use `--decode-mode sample` only for stochastic-policy ablation.

```bash
./cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh \
  --build-method skip \
  --seed-list 41,42,43 \
  --checkpoint-path training/stageb_hgraph/runs/online_ppo_seed42_svd_rawbridge_joint_v33_capSlack3000_simpleExecutor_ablation/ppo_actor_absolute_best.pt \
  --model-label hgraph_v33_absbest \
  --baseline-mean-root output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s50_ue36_ttl200_rbg16_svd_20260506_091443 \
  --decode-mode argmax \
  --eval-episode-horizon 4000 \
  --eval-device cuda \
  --eval-print-every-steps 500 \
  --sim-env CUDA_VISIBLE_DEVICES=0 \
  --eval-env CUDA_VISIBLE_DEVICES=1 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --exec-mode both \
  --precoding svd \
  --baseline-scheduler pfq \
  --compact-output 1 \
  --compact-tti-log 1 \
  --progress-tti 0 \
  --kpi-tti-log 100 \
  --compare-tti 0 \
  --tti 4000 \
  --compare-output-dir output/stageB_hgraph_v33_absbest_multiseed_compare
```

This wrapper:

- runs the canonical Stage-B main experiment once per topology seed
- uses `--online-bridge 1` and drives actions from the chosen checkpoint
- collects the real per-seed `kpi_summary.json`
- aggregates the final HGraph mean/std/min/max with the same manifest-style flow used by the RRQ/PFQ comparison scripts
- should use the same traffic/scenario parameters as training when your first goal is model-quality validation
- should only switch to a different traffic profile such as `2000 bytes / 0.5 pkt` if you explicitly want apples-to-apples comparison against a baseline produced under that different traffic configuration

## Fixed-Shape ONNX/TRT Deployment Export

For the first real-time deployment path, export the retained HGraph actor as a fixed `3cell / 36UE / 17PRG` ONNX model:

```bash
.venv/bin/python training/stageb_hgraph/export_onnx.py --output-mode action --check
```

This writes:

- `training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx`
- `training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx.metadata.json`

The preferred deployment ABI is now `action` mode. It moves the Python evaluator's masked argmax and Type-0 action composition into the exported graph, then lets C++ do only final legality/demand/fallback cleanup.

- inputs:
  - `obs_cell_features [1,3,5]`
  - `obs_ue_features [1,36,12]`
  - `obs_prg_features [1,3,17,8]`
  - `obs_post_eq_sinr [1,36,17]`
  - `obs_edge_index [1,6,2]`
  - `obs_edge_attr [1,6,2]`
  - `action_mask_ue [1,36]`
  - `action_mask_cell_ue [1,3,36]`
- outputs:
  - `action_ue_select [1,36]`
  - `action_prg_alloc [1,51]`

The legacy `logits` ABI is still available when needed:

```bash
.venv/bin/python training/stageb_hgraph/export_onnx.py --output-mode logits --check
```

That path writes `hgraph_3cell_36ue_17prg_argmax.onnx` and keeps the old `ue_logits/prg_logits` outputs for C++ masked decode experiments.

To re-check an already exported model:

```bash
.venv/bin/python training/stageb_hgraph/check_onnx_consistency.py --output-mode action
```

To run it through the C++ runtime path, pass the action ONNX as the `gnnrl_model` policy model:

```bash
./cuMAC/scripts/run_stageB_main_experiment.sh \
  --build-method skip \
  --custom-ue-prg 1 \
  --custom-policy gnnrl_model \
  --model-path training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx \
  --gnnrl-model-output-mode action \
  --gnnrl-model-use-post-eq-input 1 \
  --gnnrl-model-decode-mode argmax \
  --gnnrl-model-strict 1 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --exec-mode both \
  --tti 4000
```

For the seed42 smoke run with a 1000-TTI warm-start horizon and 4000 measured-horizon intent, use 5000 total TTI:

```bash
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
  --compact-output 0 \
  --tag hgraph_action_s42_warmup1000
```

Important caveat: the current Stage-B main KPI summary still accumulates over the full run. The `warmup1000` tag and `--tti 5000` layout are useful for controlled runs, but strict warmup-excluded KPI still requires a future `--stats-warmup-tti 1000` counter/snapshot path.

Latest seed42 action-mode smoke result:

| Metric | HGraph action s42 | PFQ s42 | RRQ s42 |
|---|---:|---:|---:|
| `traffic.served_mbps_est` | 1736.670 | 1734.346 | 1725.653 |
| `traffic.goodput_mbps` | 1555.805 | 1548.486 | 1548.063 |
| `global_kpi.global_tb_bler` | 0.105984 | 0.108009 | 0.108990 |
| `global_kpi.ue_goodput_jain` | 0.998669 | 0.998616 | 0.998790 |
| `global_kpi.ue_goodput_p5_mbps` | 40.209 | 39.604 | 40.306 |
| `global_kpi.prg_utilization_ratio` | 0.603208 | 0.906142 | 0.866892 |
| `traffic.packet_delay_mean_ms` | 2.353 | 0.692 | 0.685 |
| `traffic.packet_delay_p95_ms` | 6.500 | 2.000 | 1.500 |

Interpretation: the final action path is now functionally credible. It matches or slightly beats PFQ on throughput, goodput, BLER, fairness, and p5 goodput for seed42, but it does so with much lower PRG utilization and scheduling frequency. The remaining deployment-quality gap is therefore mostly packet-delay/service-regularity, not aggregate rate.

## Latest validated smoke status

The latest fixed-seed smoke has already validated the following:

- simulator-side configuration is aligned:
  - `online raw postEqSinr export=on`
  - `online reward mode=goodput_reliability_blankaware`
  - `precoding=svd`
  - `TTL = 200 ms`
  - topology placement and tx pattern are aligned with the intended Stage-B scenario
- PPO-side runtime is aligned:
  - `posteqL = 4`
  - persistent stream shows `continue` instead of relaunching every iteration
- latest smoke tail reached approximately:
  - `goodput = 1500.84 Mbps`
  - `tb_err = 0.092`
  - `blank = 0.046`
  - `fill = 0.954`

This means the current first-version stack has passed the key smoke gates needed for formal fixed-seed training.

## Current recommendation

The current recommendation is to retrain one new model from scratch under the validated Stage-B training configuration, then evaluate the final `absolute_best` checkpoint with three topology seeds under the same traffic setting.

Recommended rebuild:

```bash
./cuMAC/scripts/run_stageB_main_experiment.sh \
  --build-only 1 \
  --build-method cmake \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --topology-seed 42 \
  --baseline-scheduler pfq \
  --exec-mode both \
  --precoding svd
```

Recommended fresh formal training:

```bash
CUDA_VISIBLE_DEVICES=1 ./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --build-method skip \
  --tti 4000 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --precoding svd \
  --topology-seed 42 \
  --topology-seed-mode fixed \
  --baseline-scheduler pfq \
  --online-persistent 1 \
  --episode-horizon 1024 \
  --rollout-steps 1024 \
  --iterations 300 \
  --ue-prg-topk 6 \
  --hidden-dim 192 \
  --message-layers 3 \
  --dropout 0.0 \
  --device cuda \
  --lr 1.5e-4 \
  --actor-lr 1.0e-4 \
  --critic-lr 3.0e-4 \
  --weight-decay 1e-5 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.2 \
  --value-coef 0.5 \
  --entropy-coef 0.004 \
  --target-kl 0.05 \
  --update-epochs 4 \
  --minibatch-size 256 \
  --reward-mode stageb_blankaware \
  --rollout-log-every-steps 500 \
  --update-log-every-minibatches 2 \
  --best-metric goodput \
  --early-stop-patience 0 \
  --print-every-iters 1 \
  --sim-env CUDA_VISIBLE_DEVICES=0 \
  --out-dir training/stageb_hgraph/runs/online_ppo_seed42_svd_rawbridge_joint_v33_capSlack3000_simpleExecutor_ablation
```

Recommended matching evaluation style:

```bash
./cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh \
  --seed-list 41,42,43 \
  --checkpoint-path training/stageb_hgraph/runs/online_ppo_seed42_svd_rawbridge_joint_v33_capSlack3000_simpleExecutor_ablation/ppo_actor_absolute_best.pt \
  --model-label hgraph_v33_absbest \
  --baseline-mean-root output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s50_ue36_ttl200_rbg16_svd_20260506_091443 \
  --build-method skip \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --tti 4000 \
  --prbs-per-group 16 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --progress-tti 1000 \
  --kpi-tti-log 0 \
  --compare-tti 0 \
  --compact-output 1 \
  --exec-mode both \
  --precoding svd \
  --decode-mode argmax \
  --eval-episode-horizon 4000 \
  --eval-device cuda \
  --eval-print-every-steps 500 \
  --sim-env CUDA_VISIBLE_DEVICES=0 \
  --eval-env CUDA_VISIBLE_DEVICES=1 \
  --compare-output-dir output/stageB_hgraph_v33_absbest_multiseed_compare
```
