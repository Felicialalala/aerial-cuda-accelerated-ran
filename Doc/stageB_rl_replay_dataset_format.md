# Stage-B RL Replay 数据集格式（M0）

## 1. 输出目录结构

当开启 replay 导出后，每个场景会生成：

- `rl_replay_records.bin`：按固定 record 布局顺序写入的二进制样本流。
- `rl_replay_meta.json`：维度、record 大小、record 数量、策略/场景元信息。
- `rl_replay_schema.json`：字段顺序、dtype、shape 说明。

默认输出路径：

- 未指定 `--replay-dir`：`<scenario_out_dir>/replay`
- 指定 `--replay-dir <base>`：`<base>/<SCENARIO>/`

---

## 2. 导出开关

`run_stageB_main_experiment.sh` 新增：

- `--replay-dump 0|1`
- `--replay-dir <path>`

运行时环境变量：

- `CUMAC_RL_REPLAY_DUMP`
- `CUMAC_RL_REPLAY_DIR`
- `CUMAC_RL_REPLAY_SCENARIO`

---

## 3. 单条 record 布局

`rl_replay_records.bin` 内每条 transition 按以下顺序写入：

1. `tti` (`int32`)
2. `done` (`uint8`) + 3 字节 padding
3. `reward_scalar` (`float32`)
4. `reward_terms` (`float32[4]`)
5. `obs_cell_features` (`float32[n_cell, 5]`)
6. `obs_ue_features` (`float32[n_active_ue, 8]`)
7. `obs_edge_index` (`int16[n_edges, 2]`)
8. `obs_edge_attr` (`float32[n_edges, 2]`)
9. `action_ue_select` (`int32[n_sched_ue]`, invalid 为 `-1`)
10. `action_prg_alloc` (`int16[action_alloc_len]`)
11. `action_mask_ue` (`uint8[n_active_ue]`)
12. `action_mask_prg_cell` (`uint8[n_cell, n_prg]`)
13. `next_cell_features` (`float32[n_cell, 5]`)
14. `next_ue_features` (`float32[n_active_ue, 8]`)
15. `next_edge_attr` (`float32[n_edges, 2]`)

说明：

- `action_alloc_len`：
  - Type-0：`totNumCell * nPrgGrp`
  - Type-1：`2 * nSchedUe`
- `reward_terms` 顺序：
  - `throughput_mbps`
  - `total_buffer_mb`
  - `tb_err_rate`
  - `fairness_jain`

---

## 4. 快速检查命令

```bash
python3 cuMAC/scripts/inspect_rl_replay.py --replay-dir <replay_dir>
```

该脚本会检查：

- `meta/schema/bin` 是否齐全
- `record_bytes * record_count` 与二进制文件大小是否一致
- 输出关键维度信息

