# Stage-B HGraph Runs Index

This directory contains generated training, sanity, and checkpoint-evaluation artifacts.

## Current Keep Policy

Keep a run directory when it has at least one of:

- a usable checkpoint such as `ppo_actor_best.pt`, `ppo_actor_last.pt`, or an exported ONNX model
- a gate/evaluation summary with KPI CSV/JSON files
- a documented baseline or mechanism result referenced from `Doc/stageB_7cell_tar_gnn_pfq_design.md`
- a reusable topology or sanity artifact that explains a current experiment

Delete a run directory when it is clearly one of:

- smoke/skip/build-only output superseded by a formal run
- failed launch output with no checkpoint and no summary
- Python cache such as `__pycache__`
- a mislabelled run where the directory name does not match the actual scenario or horizon

Cleanup note, 2026-07-08:

- removed Python cache directories under `training/`
- removed empty `candidate_checkpoints/` placeholders
- removed failed eval directories with empty manifests and only `checkpoint_eval.log` / `main_experiment.log`
- removed unreferenced smoke/probe output directories after their formal full-run or rerun result existed
- kept checkpoint, sanity, imitation, aggregate KPI, and report-referenced evaluation directories

## Current Scenario Naming

New formal runs should include the scenario in the run name:

```text
coopboundary_assocstrongest_legacyshadow_r<cell_radius>_t<horizon>_w<warmup>_s<seed>
```

Example:

```text
online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_i100_r192
```

`legacyshadow` means no `--shadow-seed` is passed and simulator logs show `shadow_seed=legacy_shared_rng`.

`coop_center`, fixed `shadow41`, `ISD500`, and `coop_boundary nominal` runs are mechanism references only. The next formal training target is `coop_boundary + strongest association + ISD1000 + legacy shadow`.
