# Stage-B HGraph Scalable PPO Design

> Status: HGraph trainer design note. It remains useful for PPO scaling and action-semantics decisions; current retained artifacts are indexed in `Doc/README.md`.

Date: `2026-04-27`

## 1. Why change the trainer

Current Stage-B online PPO is sampler-bound, not learner-bound.

- Typical iteration profile:
  - `collect ~= 35s to 38s`
  - `update ~= 1s to 2s`
- This means improving optimizer micro-steps alone will not materially shorten wall-clock training.
- The first scaling priority should be increasing simulator/bridge collection parallelism while keeping the policy update path on-policy and stable.

## 1.1 Current action logic

The current HGraph stack no longer uses the older candidate-local control path as
the main action interface.

Current runtime semantics are:

- keep the hetero graph encoder:
  - `Cell / UE / PRG` nodes
  - `E_CC / E_CU / E_CP / E_UP / E_PP` edges
- feed raw bridge features as the main node inputs:
  - `cell = 5`
  - `ue = 12`
  - `prg = 8`
  - `cell-cell edge = 2`
- keep `UE-PRG` candidate edges as graph context only
  - they improve `PRG` node message passing and rollout diagnostics
  - they no longer define the PPO action classes
- use a direct joint Type-0 action head:
  - `slot -> UE`
  - `PRG -> slot`
- use the sampled `slot -> UE` action to derive the legal slot mask for the
  `PRG -> slot` head during PPO rollout/update
- decode directly back to native bridge actions:
  - `action_ue_select[n_sched_ue]`
  - `action_prg_alloc[n_prg * n_cell]`

This is intentionally closer to `training/gnnrl/`:

- the graph encoder is richer than the older GNN+RL stack
- but the action semantics are now again aligned with the native Stage-B Type-0
  scheduler interface

Practical implication:

- `candidate_coverage`, `cand_ue`, and `demand_ue` remain useful diagnostics
- but PPO no longer samples `PRG -> candidate_k`
- the previous candidate-local remap / fallback path is no longer part of the
  main training loop

## 1.2 Online bridge timing contract

The online bridge must interpret an action with the exact masks that were sent
with the observation that produced that action.

The validated contract is:

- C++ sends observation `S_t` with `action_mask_ue_t`
- Python samples Type-0 action `A_t` from `S_t`
- C++ applies `A_t` using the saved `action_mask_ue_t`, not a freshly rebuilt
  mask after the next traffic arrival update
- after the next response `S_{t+1}` is sent, C++ updates the pending action mask
  to `action_mask_ue_{t+1}`

This fixed the previous online mismatch where `trafSvc->Update()` introduced new
packets before action application, causing the same old action to be decoded with
a different Type-0 slot layout. The regression guards are:

- `rollout_python_preflight_wrong_cell_slot_count_mean == 0`
- `rollout_native_reject_wrong_cell_slot_count_mean == 0`
- `rollout_native_slot_count_abs_delta_sum_mean == 0`
- `rollout_native_cell_start_abs_delta_sum_mean == 0`
- `rollout_native_slot_to_cell_mismatch_count_mean == 0`
- `rollout_planned_native_prg_gap_count_mean == 0`

Temporary auto-repair logic that guessed whether PRG actions were native-major or
cell-major has been removed. Native Type-0 PRG actions are defined only as:

```text
linear_idx = prg_idx * n_cell + cell_idx
```

## 2. Recommended roadmap

### Phase A: In-repo synchronous vector PPO

This is the recommended next implementation target.

- Keep the current HGraph actor / critic code with the direct joint Type-0 action
  head.
- Run `N` bridge-backed simulator workers in parallel.
- Each worker keeps one persistent Aerial online bridge stream.
- The learner process batches actor inference across all workers on one GPU.
- Rollout collection is synchronized every `rollout_fragment_len` TTIs per worker.
- PPO update uses the concatenated on-policy batch from all workers.

Why this should be the first choice:

- Smallest migration risk.
- Preserves current reward shaping and the now-simpler direct Type-0 action
  interface.
- Directly attacks the current bottleneck: simulator wait time.
- Easy to compare against the current single-bridge baseline.

### Phase B: RLlib backend for large-scale runs

Use Ray RLlib when the goal shifts from single-node iteration speed to larger-scale orchestration.

- Candidate use case:
  - `>= 8` simulator copies
  - multi-GPU learner(s)
  - checkpoint/resume and cluster scheduling
  - experiment sweeps with stronger fault tolerance
- Preferred RLlib algorithm:
  - start with PPO if strict on-policy behavior is required
  - move to APPO when rollout latency dominates and slight policy lag is acceptable

### Phase C: TorchRL collector backend

TorchRL is the lighter-weight PyTorch-native alternative.

- Good fit when we want:
  - to keep almost all current PyTorch code
  - multiprocessing collectors without adopting Ray
  - a gradual transition from the current trainer
- Recommendation:
  - start with `MultiSyncDataCollector`
  - only evaluate async collectors after the synchronous vector path is stable

## 3. Phase A implementation plan

### 3.1 New runtime components

Add:

- `training/stageb_hgraph/parallel_runner.py`
  - owns `N` simulator processes
  - owns `N` `BridgeEnvClient` connections
  - exposes `reset_all()`, `step_selected(indices, actions)`, `active_envs()`
- `training/stageb_hgraph/vector_rollout.py`
  - collects `[num_envs, fragment_len]` transitions
  - stores per-env bootstrap value and done flags
- `training/stageb_hgraph/parallel_buffer.py`
  - flattens env-major rollouts into PPO minibatches

### 3.2 Main trainer changes

Extend `online_ppo.py` or add `online_ppo_parallel.py`.

- New args:
  - `--num-envs`
  - `--rollout-fragment-len`
  - `--env-reset-stagger`
  - `--num-topology-workers`
  - `--vectorized-inference-batch`
- Collection loop:
  1. gather current observations from active workers
  2. build graphs in batch
  3. run one batched forward pass
  4. sample `slot -> UE` and `PRG -> slot`
  5. decode direct Type-0 per-env actions
  6. step all workers
  7. append transitions
  8. bootstrap value on the final next-state batch

### 3.3 Scheduling details

- Use one learner GPU for the policy.
- Keep simulator processes on CPU unless the Aerial bridge path explicitly benefits from a different placement.
- Keep `UE-PRG` sparse edges in the graph batch because they still improve PRG
  context, but do not treat them as the sampled action space.
- Keep `topology_seed_mode=list_cycle|sequential` at the worker level so the vector trainer sees diverse trajectories every update.

### 3.4 PPO math

- Total batch size becomes:
  - `train_batch = num_envs * rollout_fragment_len`
- Keep PPO synchronized:
  - no replay
  - no stale-policy training data
- Keep:
  - GAE
  - state normalization
  - reward normalization
  - Huber critic
  - KL early stop

### 3.5 Validation target

Success criteria for Phase A:

- wall-clock iteration time drops materially vs single-env trainer
- update time remains a small fraction of total time
- mean goodput is not worse than single-env PPO after equal environment steps
- variance across seeds decreases

## 4. RLlib mapping

If we build a Ray backend later, map the current trainer as follows.

### 4.1 Environment

- Wrap the Aerial bridge as one RLlib single-agent environment instance.
- Observation should remain a structured dict, not a flattened vector.
- Action should be a structured composite action:
  - `ue_action_class`
  - `prg_action_class`

### 4.2 Model / RLModule

- Reuse the current HGraph actor in a custom RLModule.
- Keep sparse graph construction inside the module forward.
- Keep the action semantics direct:
  - `slot -> UE`
  - `PRG -> slot`
- Decode to bridge actions by flattening native Type-0 tensors, not by
  candidate-local remap logic inside the learner loss.

### 4.3 PPO vs APPO choice

- RLlib PPO:
  - better semantic match to the current trainer
  - simpler correctness story
- RLlib APPO:
  - better when environment sampling is the dominant wall-clock cost
  - acceptable only if we explicitly tolerate policy lag

## 5. TorchRL mapping

### 5.1 Best first target

Use `MultiSyncDataCollector`.

- one collector worker per bridge simulator
- current actor runs as the shared policy
- collected tensors are concatenated into the existing PPO update path

### 5.2 Why not async first

The current problem is still on-policy PPO with simulator-dominated wall-clock time.

- Async collection introduces policy-lag semantics.
- That is a worse first migration than synchronous multi-worker collection.

## 6. Recommendation

Recommended order:

1. Finish and validate the current direct Type-0 HGraph trainer.
2. Implement Phase A synchronous multi-bridge PPO in-repo.
3. If `num_envs` scaling is still insufficient, add an RLlib APPO backend for large runs.
4. Keep TorchRL as the lower-friction alternative when we want multiprocessing collectors without moving to Ray.

## 7. Expected impact

Given the current `collect >> update` profile, the most realistic path to faster training is:

- more environment copies
- batched inference
- synchronized on-policy PPO updates

This should deliver a much larger wall-clock gain than further tuning single-env PPO hyperparameters.

## 8. Fix2 diagnosis and local refactor plan

This section records the diagnosis from
`online_ppo_seed42_svd_rawbridge_joint_v13_fix2`.

### 8.1 What the run shows

The run does not yet prove that leaving PRGs blank is intrinsically better than
serving high-backlog UEs.

The strongest signal is a mismatch between planned allocation and native
execution:

- explicit policy blanking at iter 48 is almost zero:
  - `rollout_blank_ratio_mean ~= 0.0008`
- Python decode still drops many PRGs:
  - `rollout_post_decode_drop_ratio_mean ~= 0.1243`
  - `rollout_final_blank_ratio_mean ~= 0.1251`
- native bridge utilization is much lower than Python planned fill:
  - `rollout_action_fill_ratio_mean ~= 0.8749`
  - `rollout_prg_utilization_ratio_mean ~= 0.5706`

This means the current per-UE CSV should not be interpreted as actual service.
It is a planned-service view computed before the bridge executes the action.

### 8.2 Main logic risks

1. Per-UE diagnostics are computed before `env.step()`.
   - `assigned_prg_count` and `served_this_tti` reflect Python decoded intent.
   - They do not include native Type-0 legality filtering, MCS failures, TB
     errors, or actual goodput.

2. The PRG head is not conditioned on the sampled `slot -> UE` action.
   - The model builds PRG slot context from `argmax(ue_logits_main)`.
   - PPO rollout/update masks PRG actions using the sampled/stored UE action.
   - Therefore the PRG head can score a slot as if it contained UE A, while the
     bridge executes it with UE B.

3. The direct Type-0 factorization creates a hard slot bottleneck.
   - A demand UE must first survive the `slot -> UE` action.
   - Repeated UE selections are collapsed by the slot-selection mask.
   - PRGs cannot directly rescue a high-backlog UE that was not selected into a
     usable slot.

4. `stageb_blankaware` has no direct actual-unserved backlog debt term.
   - It optimizes goodput / efficiency / reliability / packing and a Jain
     fairness tie-breaker.
   - High backlog with low recent service is not directly penalized unless it
     also affects those aggregate metrics.

5. The rollout conflict / ICI logging used the wrong PRG feature indices.
   - PRG feature indices `4/5` are neighbor SINR features.
   - The intended `same_prg_conflict_ratio` and `ici_proxy` are `6/7`.

### 8.3 Four-stage repair plan

Stage 1 makes the diagnostics trustworthy:

- add optional bridge payload fields for actual per-UE diagnostics:
  - actual PRG count per UE
  - served bytes per UE
  - goodput bytes per UE
  - TB tx count per UE
  - TB error count per UE
- parse those fields in the Python bridge client
- rename rollout per-UE action-side fields to `planned_*`
- write actual fields beside them when available
- add rollout summary metrics for planned/native fill gap and actual unserved
  backlog

Stage 2 fixes the actor action semantics:

- split the direct joint head into two phases:
  - compute `ue_logits`
  - build PRG slot context from the sampled/stored `ue_action_class`
- use the same path in rollout, PPO update, and evaluation
- keep the old argmax prior only as an inference fallback when no UE action is
  provided

Stage 3 reduces the slot bottleneck:

- add a deterministic sampled-UE repair pass before PRG sampling
- within each cell, fill invalid / duplicate / null selected slots with
  demand-active UEs ranked by:
  - no current selected slot
  - backlog bytes
  - recent service deficit
  - stale / HOL urgency
- keep PRG actions masked to the repaired slots so PPO log-probs remain
  evaluated against the executed `ue_action_class`

Stage 4 adds a bounded debt signal:

- compute actual unserved backlog after `env.step()` when actual diagnostics are
  available
- add a bounded negative debt term to `stageb_blankaware`
- keep the goodput objective dominant
- expose the debt term in rollout summaries so it can be tuned instead of
  hidden inside normalized reward

### 8.4 Validation checklist

After these changes, compare short runs with the same topology seed:

1. baseline `fix2` checkpoint under the new diagnostics
2. conditioned PRG head only
3. conditioned PRG head plus slot repair
4. conditioned PRG head plus slot repair plus debt reward

Required metrics:

- `planned_action_fill_ratio`
- `native_prg_utilization_ratio`
- `planned_native_fill_gap`
- actual unserved demand UE count
- actual unserved backlog bytes
- p10 recent service rate
- goodput / TB error / expiry

If native utilization rises while goodput stays flat or improves, the previous
blank PRG was mostly action-path loss. If native utilization rises and goodput
falls with higher TB error, then there is a real boundary where leaving risky
PRGs blank can be useful.

### 8.5 Implemented local changes

The first implementation pass covers the four stages above:

- bridge protocol v6 now carries optional actual per-UE diagnostics
- Python bridge parsing exposes those actual arrays in `info`
- per-UE CSV keeps planned action-side fields and adds actual native fields
- rollout summaries include actual diagnostics availability, actual unserved
  backlog, and planned/native PRG gap
- the HGraph PRG head can condition on the sampled/stored UE action instead of
  the argmax UE prior
- rollout sampling uses a deterministic slot coverage repair before PRG
  sampling
- PPO update evaluates PRG log-probs through the same sampled-UE-conditioned
  path
- `stageb_blankaware` adds a bounded actual-unserved backlog debt term
- rollout conflict / ICI logging now reads PRG feature indices `6/7`

### 8.6 v14 observation and follow-up

The first all-stages run
`online_ppo_seed42_svd_rawbridge_joint_v14` did not improve goodput. The new
diagnostics explain why:

- actual per-UE diagnostics were active:
  - `rollout_actual_ue_diagnostics_available_mean = 1.0`
- coverage repair became very aggressive:
  - `rollout_coverage_repair_count_mean ~= 10-13` slots per TTI
  - `rollout_planned_unserved_demand_ue_mean < 1`
- native execution still discarded a large fraction of planned PRGs:
  - `rollout_planned_native_prg_gap_ratio_mean ~= 0.38-0.44`
- actual unserved backlog remained high:
  - several MB per TTI on average

So v14 mostly converted the policy into "try to cover almost every demand UE",
but native execution still accepted only part of that plan and goodput dropped.
The next experiments should therefore be staged and conservative rather than
turning repair/debt on by default.

New control knobs:

- `--slot-repair-max-per-cell`
  - default `0`, disabled
  - try `1` or `2` first
- `--slot-repair-min-backlog-bytes`
  - default `1.0`
- `--backlog-debt-weight`
  - default `0.0`, disabled
  - try `0.02` to `0.05` only after conditioned-head-only improves or matches
    baseline

Recommended next order:

1. conditioned PRG head only:
   - `--slot-repair-max-per-cell 0`
   - `--backlog-debt-weight 0.0`
2. conservative repair:
   - `--slot-repair-max-per-cell 1`
   - `--backlog-debt-weight 0.0`
3. repair plus weak debt:
   - `--slot-repair-max-per-cell 1`
   - `--backlog-debt-weight 0.02`
