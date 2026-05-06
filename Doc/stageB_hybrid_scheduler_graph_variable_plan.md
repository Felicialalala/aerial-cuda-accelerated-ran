# Stage-B 混合智能调度规划文档

## 1. 文档目标

本文先聚焦你提出的混合智能调度框架里的第一个落地点:

- 在当前 `Stage-B` 场景下, 先把“哪些数据可以作为图谱变量”梳理清楚
- 明确这些变量来自哪里
- 明确它们属于哪一类
- 明确它们现在是否已经进入调度网络
- 明确它们是“可直接用”“轻改 bridge 可用”还是“必须新增 hook”

这里不把目标做成“大而全知识图谱”, 而是服务于一个“小而硬、能直接约束调度网络”的结构先验图。

## 2. 当前 Stage-B 场景确认

这份文档优先以当前保留的代表性结果目录对应场景为准, 即:

- RRQ/PFQ 10-seed baseline: `output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s50_ue36_ttl200_rbg16_svd_20260506_091443`
- HGraph v33 3-seed evaluation: `output/stageB_hgraph_v33_best_iter86_s41_s42_s43_ue36_ttl200_rbg16_svd_warmup1000`

当前场景不是通用默认 `7cell + 68 PRG` 主线, 而是一个更贴近你当前分析工作的 `3cell` 变体:

| 项目 | 当前值 |
| --- | --- |
| 拓扑 | `3cell_triangle` |
| 协同小区数 | `3` |
| 外环干扰小区 | `0` |
| UE 数 | `12 UE/cell`, 共 `36 UE` |
| 天线 | `4T4R` |
| 分配方式 | `Type-0 bitmap allocation` |
| 预编码 | `SVD` |
| 信道 | `Rayleigh` |
| SCS | `30 kHz` |
| slot 时长 | `0.5 ms` |
| PRB 组大小 | `16 PRB/RBG` |
| PRG 数 | `17` |
| 业务流 | `3000 bytes/packet`, `1 pkt/TTI` |
| TTL | `400 TTI = 200 ms` |
| UE 放置 | `uniform + Voronoi clip` |
| 发射模式 | `omni` |
| 执行模式 | `gpu-only` |

这点很重要: 你的“结构先验图”首版应先对齐这个 `3cell + 36UE + 17PRG` 场景, 不要一开始就按通用 `7cell + 68PRG` 口径把图做得过大。

## 2.1 当前已验证实现状态

截至当前代码版本, 首版 HGraph 已经完成以下实现收口:

- 三类节点:
  - `Cell`
  - `UE`
  - `cell-local PRG token P_{i,g}`
- 五类边:
  - `Cell-Cell`
  - `Cell-UE`
  - `Cell-PRG`
  - `UE-PRG`
  - `PRG-PRG`
- 层内模板图继续只作为共享模板编码器实现, 不额外实例化成层内小图
- 当前实际输入维度已经扩展为:
  - `cell_features[C,6]`
  - `ue_features[U,10]`
  - `prg_features[C*G,9]`
- 当前动作头已经收敛为联合 Type-0 语义:
  - 先 `slot -> UE`
  - 再 `PRG -> slot`
- `postEqSinr` 已经通过 online bridge 接入, `UE-PRG` 候选边已可稳定运行在 `post_eq_topk` 模式
- fixed seed + `online_persistent=1` 下, online PPO 已支持跨 iteration 续跑同一 simulator 流
- `stageb_blankaware` 奖励已经接入, 并与 simulator 侧 `goodput_reliability_blankaware` 模式对齐
- trainer / simulator 已支持分卡运行:
  - trainer on GPU1
  - simulator on GPU0
- PPO 当前已区分两类最佳 checkpoint:
  - `absolute_best`: 真实全局最优
  - `threshold_best`: 受 `early_stop_min_delta` 约束、用于早停判定的最优
- 当前已具备与 RRQ/PFQ 多随机拓扑种子平均口径一致的 checkpoint 评测入口:
  - `run_stageB_hgraph_checkpoint_multi_seed_compare.sh`

这意味着当前首版已经不再只是“图结构规划”, 而是具备了进入正式 PPO 训练的工程基础。

### 2.2 当前训练与评测边界

截至当前版本, HGraph PPO 的训练与评测边界建议明确如下:

- 正式训练默认建议:
  - `best_metric = goodput`
  - `early_stop_patience = 60`
  - `early_stop_min_delta = 1.0`
  - `early_stop_warmup_iters = 50`
- `threshold_best` 适合回答:
  - “按早停规则, 哪个 checkpoint 应被认为是最优停止点”
- `absolute_best` 适合回答:
  - “整个训练轨迹里, 真实达到过的最好 goodput 对应哪个 checkpoint”
- 首版最终与 RRQ/PFQ 做多 seed 对比时, 建议优先使用:
  - `ppo_actor_absolute_best.pt`
- 为保持历史脚本兼容:
  - `ppo_actor_best.pt` 当前仍保留, 但语义上等同于 `threshold_best` 的兼容别名
- 若当前目标是先判断“HGraph 模型本身有没有学对”, 则测试配置应先与训练保持一致:
  - `packet_size_bytes = 3000`
  - `traffic_arrival_rate = 1.0 pkt/TTI`
  - `packet_ttl_ms = 200`
  - `topology_scenario = 3cell`
  - `total_ue_count = 36`
  - `precoding = svd`
- 只有在明确要与另一条不同流量口径的 RRQ/PFQ 结果逐项对齐时, 才建议切换到例如 `2000 bytes / 0.5 pkt/TTI` 的另一套配置
- HGraph 在线评测当前必须使用:
  - `exec_mode = both`
  - 因为 `online-bridge + custom UE/PRG` 接管路径依赖 host-side bridge/socket 协调, 当前不能走原生 `gpu-only` 闭环

### 2.3 当前推荐训练与测试命令

若要在当前保留口径上继续训练/复现, 建议使用与 v33 一致的 `SVD + 3000B + 1 pkt/TTI + TTL200ms` 配置:

编译:

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

正式训练:

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

下面这段是 v33 checkpoint 的三种子评测样式；当前 RRQ/PFQ baseline 已是 10-seed，若要形成最终 HGraph 结论，应进一步补齐 HGraph 的 10-seed 评测或明确声明 HGraph 仍是 3-seed 证据。

三随机拓扑种子测试示例:

```bash
./cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh \
  --seed-list 41,42,43 \
  --checkpoint-path training/stageb_hgraph/runs/online_ppo_seed42_svd_rawbridge_joint_v33_capSlack3000_simpleExecutor_ablation/ppo_actor_best.pt \
  --model-label hgraph_v33_best_iter86 \
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
  --compare-output-dir output/stageB_hgraph_v33_best_iter86_s41_s42_s43_ue36_ttl200_rbg16_svd_warmup1000
```

### 2.4 2026-04-29 Online Type-0 时序错配修复

本轮诊断确认, 早期 `v17-v23` 中大量“planned 有 PRG, native 没接受”的问题不是策略主动留空, 也不是 PRG flatten 顺序错误, 而是 online bridge 的动作-状态时序错配:

- Python 的 `slot -> UE / PRG -> slot` 动作基于上一条 `StepRsp` 中的 `action_mask_ue`
- C++ 下一轮先执行 `trafSvc->Update()`, 新包到达后重新生成 `onlineMaskUe`
- 旧动作被新的 `onlineMaskUe` 构造出的 Type-0 slot layout 解释, 导致 cell1/cell2 大量 `wrong_cell_slot`

已保留的正确修复:

- C++ 在发送 reset/step response 后保存 `onlinePendingActionMaskUe`
- 下一次收到 Python action 时, `applyOnlineActionToSchedule()` 使用这份 pending mask 构造 Type-0 slot layout
- step response 发送成功后, 再把 `onlinePendingActionMaskUe` 更新为下一状态的 `nextMaskUe`
- Python 侧保留 preflight reject 诊断, C++ 侧保留 native reject/layout 诊断, 作为长训回归守卫

已清理的过时逻辑:

- 删除 C++ `applyType0SampledActionToSchedule()` 中按 cell-major/native 两种顺序猜测并自动改写 PRG action 的临时防护
- 删除 Python `_finalize_decoded_type0_action()` 中按 cell-major 猜测并自动转置 action 的临时防护
- 当前 PRG action 只接受明确 native Type-0 顺序: `linear_idx = prg_idx * n_cell + cell_idx`

`v24_pending_mask_fix` 短测已验证以下指标归零:

| 指标 | 期望 |
| --- | --- |
| `rollout_python_preflight_wrong_cell_slot_count_mean` | `0` |
| `rollout_native_reject_wrong_cell_slot_count_mean` | `0` |
| `rollout_native_slot_count_abs_delta_sum_mean` | `0` |
| `rollout_native_cell_start_abs_delta_sum_mean` | `0` |
| `rollout_native_slot_to_cell_mismatch_count_mean` | `0` |
| `rollout_planned_native_prg_gap_count_mean` | `0` |

后续所有正式训练都应先检查这些守卫列。如果其中任何一项重新非零, 说明 bridge/action/layout 又发生错配, 不应继续解释策略优劣。

#### 历史长训命令: pending mask fix 后的干净基线

下面命令对应的 `v25_pending_mask_fix_long` run 已经从本地清理，保留这里只用于说明 pending-mask 修复后的关键参数组合；当前保留 checkpoint 目录是 `online_ppo_seed42_svd_rawbridge_joint_v33_capSlack3000_simpleExecutor_ablation`。

```bash
CUDA_VISIBLE_DEVICES=1 ./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --build-method cmake \
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
  --candidate-mode all_demand_ues \
  --ue-prg-topk 6 \
  --ue-prg-diversity-extra 2 \
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
  --normalize-state 1 \
  --normalize-reward 1 \
  --normalize-adv 1 \
  --max-grad-norm 1.0 \
  --value-loss huber \
  --value-huber-beta 10.0 \
  --reward-mode stageb_blankaware \
  --goodput-weight 1.0 \
  --tb-err-weight 8.0 \
  --expiry-weight 6.0 \
  --conflict-weight 1.0 \
  --ici-weight 1.0 \
  --urgent-backlog-weight 0.5 \
  --rollout-log-every-steps 500 \
  --update-log-every-minibatches 0 \
  --best-metric goodput \
  --early-stop-patience 40 \
  --early-stop-warmup-iters 30 \
  --print-every-iters 1 \
  --sim-env CUDA_VISIBLE_DEVICES=0 \
  --out-dir training/stageb_hgraph/runs/<historical_v25_pending_mask_fix_long>
```

## 3. 可用性分层

为避免把“已经进了当前 RL 状态”的量和“内核里有但还没导出”的量混在一起, 这里先给出统一分层:

- `T0`: 已进入当前 online observation, 现在就能喂给调度网络
- `T1`: API/CPU 内存里已有, 只需扩 bridge/codec, 不需要改底层 kernel
- `T2`: 平台内部有中间量, 但需要新增 hook 或导出逻辑
- `T3`: 主要用于离线评估、奖励、校准或回退阈值拟合, 不宜直接当当前 TTI 的主状态

## 4. 可用数据源与变量池梳理

### 4.1 图骨架与结构先验类

这是你“结构先验约束”最应该先落下来的部分, 因为它直接定义:

- 哪些边是合法的
- 哪些动作是可行的
- 哪些依赖关系本来就不应该让网络自己瞎学

| 变量族 | 具体字段/信号 | 来源 | 粒度 | 层级 | 建议角色 |
| --- | --- | --- | --- | --- | --- |
| 小区数/总小区数 | `nCell`, `totNumCell` | `cumacCellGrpPrms` / `cuMAC/src/api.h` | cluster | `T1` | 定义 cell graph 大小 |
| Cell-UE 归属 | `cellAssocActUe` | `cumacCellGrpPrms` | `cell x active_ue` | `T1` | 最硬的结构边, 直接约束 `cell -> UE` 合法关系 |
| PRG 可用掩码 | `prgMsk` | `cumacCellGrpPrms` | `cell x prg` | `T1` | 约束动作空间 |
| 当前 online mask | `action_mask_cell_ue`, `action_mask_prg_cell` | `OnlineFeatureCodec` | `cell x ue`, `cell x prg` | `T0` | 现成结构输入 |
| 小区拓扑形状 | `3cell_triangle`, `7cell_hex` 等 | `network.cu` + 运行脚本 | cluster | `T1` | cell-cell 固定邻接先验 |
| UE 空间放置规则 | `ue_placement`, `Voronoi clip`, `bs_tx_pattern` | `run_stageB_main_experiment.sh`, `network.cu` | scene | `T1` | 空间先验, 用于解释干扰边 |
| 编译期结构参数 | `nPrbGrp`, `nBsAnt`, `nUeAnt`, `allocType`, `precodingScheme` | `parameters.h`, `cumacCellGrpPrms` | scene | `T1` | 定义图规模与动作语义 |

结论:

- `cellAssocActUe`
- `action_mask_cell_ue`
- `action_mask_prg_cell`
- 固定 cell-cell 邻接关系

这四类量已经足以构成首版“小而硬”的图骨架。

### 4.2 UE 业务、队列与时延紧迫度类

这是“调度收益”与“安全回退”都强相关的变量族, 也是你后续 uncertainty gating 最容易落地的一组变量。

| 变量族 | 具体字段/信号 | 来源 | 粒度 | 层级 | 建议角色 |
| --- | --- | --- | --- | --- | --- |
| UE backlog | `bufferSize` | `cumacCellGrpUeStatus` | `ue` | `T0` | UE 节点核心特征 |
| HOL delay | `hol_age_tti -> hol_delay_ms` | `TrafficService::GetPacketHeadStats` -> `OnlineObservationExtrasBuilder` | `ue` | `T0` | 延迟紧迫度 |
| TTL slack | `ttl_slack_tti -> ttl_slack_ms` | 同上 | `ue` | `T0` | 安全/回退触发的重要信号 |
| pending bytes | `pending_bytes`, `hol_bytes` | `PacketHeadSummary` | `ue` | `T1` | 更细粒度 queue 结构 |
| 业务到达配置 | `packet_bytes`, `arrival_rate_pkt_per_tti`, `ttl_tti`, `ttl_ms` | 运行脚本 + traffic model | scene / ue-flow | `T1` | 业务先验 |
| 流量累计统计 | `generated_bytes`, `accepted_bytes`, `flow_queued_bytes` | `TrafficService`, `main.cpp` | `ue`, cluster | `T3` | 校准与分析 |
| backlog-based queue delay | `queue_delay_est_ms` | `main.cpp` KPI | `ue`, cluster | `T3` | 安全阈值与离线分析 |
| packet delay 统计 | `packet_delay_mean/p90/p95/max` | `trafficGenerator.hpp`, `main.cpp` | `ue`, cluster | `T3` | fallback 阈值拟合 |
| packet expiry | `expired_bytes`, `expired_packets`, `expiry_drop_rate` | `TrafficService`, `main.cpp` | `ue`, cluster | `T3` | 极端风险判据 |

这里尤其值得强调:

- `bufferSize` 是当前最可靠的 backlog proxy
- `hol_delay_ms` 和 `ttl_slack_ms` 已经在当前 online 主线上
- `ttl_slack_ms` 非常适合后续做“极端风险强制回退”

如果只做首版最小图, 这组里优先级最高的是:

- `bufferSize`
- `hol_delay_ms`
- `ttl_slack_ms`

### 4.3 信道、干扰与链路可行性类

这组变量决定“某个 UE 值不值得调度”以及“某个 PRG 复用是否危险”, 是图谱里最接近无线因果结构的一部分。

| 变量族 | 具体字段/信号 | 来源 | 粒度 | 层级 | 建议角色 |
| --- | --- | --- | --- | --- | --- |
| 宽带 SINR | `wbSinr` | `cumacCellGrpPrms` | `ue x layer` | `T0` | UE 可行性主信号 |
| PRG 级后均衡 SINR | `postEqSinr` | `cumacCellGrpPrms` | `ue x prg x layer` | `T1` | PRG 局部质量主信号 |
| 原始跨小区信道 | `estH_fr_actUe`, `estH_fr_actUe_prd` | `cumacCellGrpPrms`, `network.cu` | `ue x prg x cell x ant` | `T2` | 更强的结构先验/边特征 |
| 宽带接收功率 proxy | `rxSigPowDB` | `network.cu` | `cell x ue` | `T1` | 邻区耦合、静态干扰强度 proxy |
| 噪声方差 | `sigmaSqrd` | `cumacCellGrpPrms` | scalar | `T1` | 信道标准化/可靠性口径 |
| 发射功率参数 | `Pt_Rbg`, `Pt_rbgAnt`, `bsTxPower_perAntPrg` | `network.cu`, `cumacCellGrpPrms` | scene / cell | `T1` | 信号强度解释项 |
| BLER target | `blerTargetActUe` | `cumacCellGrpPrms` | `ue` | `T1` | 可作为风险门限先验 |
| CQI/RI | `cqiActUe`, `riActUe` | `cumacCellGrpUeStatus` | `ue` | `T1` | 若当前路径有效可补充 UE 可行性 |
| 精确 ICI | `multiCellSinrCal` 内部干扰协方差 | `multiCellSinrCal.cu` | `ue x prg` 内部量 | `T2` | 后续增强版边特征 |

当前已经进入 online 主线的“轻量干扰 proxy”也非常重要:

| 当前 PRG 特征 | 来源 | 层级 | 含义 |
| --- | --- | --- | --- |
| `top1SinrDb` | `OnlineObservationExtrasBuilder` | `T0` | 当前 `cell x prg` 最优候选质量 |
| `top2GapDb` | 同上 | `T0` | 本小区候选分离度 |
| `neighborMaxTop1SinrDb` | 同上 | `T0` | 邻区对同 PRG 的最强竞争强度 |
| `neighborMeanTop1SinrDb` | 同上 | `T0` | 邻区平均竞争强度 |
| `samePrgConflictRatio` | 同上 | `T0` | 同 PRG 跨小区冲突 proxy |
| `iciProxy` | 同上 | `T0` | 轻量干扰风险 proxy |

这部分其实已经非常接近你想要的依赖图:

- 邻区同频占用/竞争强度上升
- 本 PRG 干扰风险上升
- BLER 风险上升
- goodput 风险上升

区别只在于:

- 现在是 feature proxy
- 还没有被显式声明成“结构依赖约束”

### 4.4 调度器内部状态、动作与历史类

这组变量对“让不确定性真正控制决策”非常关键, 因为它们决定当前状态是一个“正常可控态”还是“已经积累了危险历史”的状态。

| 变量族 | 具体字段/信号 | 来源 | 粒度 | 层级 | 建议角色 |
| --- | --- | --- | --- | --- | --- |
| 长期平均速率 | `avgRatesActUe` | `cumacCellGrpUeStatus` | `ue` | `T0` | PF/PFQ 公平性债务 |
| 最近是否被调度 | `lastSchdSlotActUe`, `staleSlots` | `cumacCellGrpUeStatus` + `OnlineFeatureCodec` | `ue` | `T0` | 服务陈旧度 |
| 上次 TB 是否出错 | `tbErrLastActUe` | `cumacCellGrpUeStatus` | `ue` | `T0` | 可靠性记忆 |
| 新传/重传 | `newDataActUe` | `cumacCellGrpUeStatus` | `ue` | `T0` | HARQ/风险语义 |
| 当前 MCS | `mcsSelSol` | `cumacSchdSol` | `sched_ue` | `T1` | 动作后果解释变量 |
| 当前层数 | `layerSelSol` | `cumacSchdSol` | `sched_ue` | `T1` | 高层数风险 |
| 当前 PRG 分配 | `allocSol`, `setSchdUePerCellTTI` | `cumacSchdSol` | `cell x prg` / `sched_ue` | `T1` | 动作历史 |
| 上次发送摘要 | `allocSolLastTx`, `mcsSelSolLastTx`, `layerSelSolLastTx` | `cumacCellGrpUeStatus` | `ue` | `T1` | 历史上下文 |
| 近期调度比例 | `recent_scheduled_ratio` | `OnlineObservationExtrasBuilder` | `ue` | `T0` | 短时公平性债务 |
| 近期 goodput 欠账 | `recent_goodput_deficit_norm` | `OnlineObservationExtrasBuilder` | `ue` | `T0` | 谁被持续亏待 |
| 上一 TTI 是否占用该 PRG | `prevPrgAssigned` | `OnlineObservationExtrasBuilder` | `cell x prg` | `T0` | 复用惯性 |
| PF 内部 metric | `pfMetricArr`, `pfPredBytesArr` | `cumacSchdSol` | `cell x prg x slot` | `T1` | 可作为 teacher / 保守 RL 参考 |

这组变量的价值在于:

- 它们让图不只是“信道图”, 而是“调度状态图”
- 它们可以自然支撑 conservative action set
- 它们也很适合做 epistemic uncertainty 的 OOD 检测输入

### 4.5 结果、安全监测与校准类

这组变量不一定都应该直接送入当前 TTI 的策略网络, 但它们是后续 uncertainty gating 和 fallback 设计的核心标注源。

| 变量族 | 具体字段/信号 | 来源 | 粒度 | 层级 | 建议角色 |
| --- | --- | --- | --- | --- | --- |
| 当 TTI served bytes | `servedBytesThisTti` | `main.cpp` | `ue` | `T1` | 在线 reward / 短时监测 |
| 当 TTI goodput bytes | `goodputBytesThisTti` | `main.cpp` | `ue` | `T1` | 在线 reward / calibration |
| 当 TTI TB error rate | `onlineTbErrRate` | `main.cpp` + `tbErrLast` | cluster | `T1` | 风险监测 |
| PRG 利用率 | `prgUtilizationRatio` | `main.cpp` | cluster | `T1/T3` | “是否故意少复用/留空” 监测 |
| PRG reuse ratio | `prgReuseRatio` | `main.cpp` | cluster | `T1/T3` | 干扰密度 proxy |
| run-level goodput | `traffic.goodput_mbps` | KPI 汇总 | cluster | `T3` | 离线比较 |
| run-level served rate | `traffic.served_mbps_est` | KPI 汇总 | cluster | `T3` | 离线比较 |
| run-level BLER | `global_kpi.global_tb_bler` | compare summary | cluster | `T3` | 风险阈值拟合 |
| packet 服务速率 | `traffic.packet_effective_service_rate_mbps` | KPI 汇总 | cluster | `T3` | 长尾/队列风险标签 |
| packet delay 分位数 | `packet_delay_p90/p95` | KPI 汇总 | `ue`, cluster | `T3` | fallback 阈值拟合 |
| UE 公平性 | `ue_*_jain`, `scheduled_ratio_*` | KPI 汇总 | cluster | `T3` | 保守策略评价 |

这组量最适合做三件事:

1. 离线评估与 reward 设计参考
2. uncertainty calibration label
3. fallback 阈值与安全规则拟合

## 5. 图结构最终收敛方案

为兼顾知识表达能力、在线实现成本与当前 `Type-0 bitmap` 动作语义的一致性, 本文将图结构最终收敛为:

- 层内共享变量关系模板图
- 层间稀疏实体图

这不是把所有变量拆成一张全局超大图, 而是把“变量依赖关系”和“实体之间谁影响谁”分开建模:

- 层内模板图负责表达同一类实体内部变量之间的依赖关系
- 层间实体图负责表达 Cell、UE、PRG 之间的稀疏耦合、候选关系和冲突关系

### 5.1 双层建模总览

该方案包含两层:

1. 第一层是层内共享变量关系模板图
   - 所有 Cell 实体共享同一套 Cell 模板图
   - 所有 UE 实体共享同一套 UE 模板图
   - 所有 PRG 实体共享同一套 PRG 模板图
2. 第二层是层间稀疏实体图
   - 节点包括 Cell、UE、cell-local PRG token
   - 边只保留和当前调度语义强相关的稀疏关系

这样做的好处是:

- 比简单堆叠特征更能体现结构先验
- 比全局变量大图复杂度低很多
- 比纯实体异质图更名副其实, 因为知识不仅体现在实体类型, 也体现在实体内部变量依赖

### 5.2 层内共享变量关系模板图

层内模板图只描述“同类实体内部的变量依赖”, 不直接描述跨实体拓扑。

#### 5.2.1 Cell 模板图

每个 Cell 实体内部建立 5 个变量节点:

- `cell_load_bytes`
- `active_ue_count`
- `mean_wb_sinr_lin`
- `cell_tb_err_ema`
- `reuse_density`

其中:

- `reuse_density` 首版优先用 cell-local 近似量, 如 `prevPrgAssigned` 的 cell 均值、该 cell 的 `samePrgConflictRatio` 均值或 `iciProxy` 均值
- 不建议直接使用 cluster 级 `prgReuseRatio` 作为单个 Cell 模板输入
- `cell_tb_err_ema` 建议使用最近若干 TTI 的 EMA, 而不是瞬时 TB error rate

Cell 模板图的首版有向依赖定义为:

- `active_ue_count -> cell_load_bytes`
- `cell_load_bytes -> reuse_density`
- `cell_load_bytes -> cell_tb_err_ema`
- `reuse_density -> cell_tb_err_ema`
- `mean_wb_sinr_lin -> cell_tb_err_ema`

这组关系表达的是:

- 活跃 UE 数推高负载压力
- 负载压力提高复用密度倾向
- 复用密度和平均链路质量共同影响短时错误率

#### 5.2.2 UE 模板图

每个 UE 实体内部建立 8 个变量节点:

- `buffer_bytes`
- `hol_delay_ms`
- `ttl_slack_ms`
- `wb_sinr_lin`
- `avg_rate_mbps`
- `tbErrLastActUe`
- `recent_scheduled_ratio`
- `recent_goodput_deficit_norm`

UE 模板图分成三条主链:

紧迫度链:

- `buffer_bytes -> hol_delay_ms`
- `hol_delay_ms -> ttl_slack_ms`
- `buffer_bytes -> ttl_slack_ms`

可行性/风险链:

- `wb_sinr_lin -> tbErrLastActUe`
- `tbErrLastActUe -> risk_flag`
- `wb_sinr_lin -> feasibility_score`

公平性/债务链:

- `avg_rate_mbps -> recent_goodput_deficit_norm`
- `recent_scheduled_ratio -> recent_goodput_deficit_norm`

这里的 `risk_flag` 和 `feasibility_score` 不要求首版作为外显观测字段存在, 可以视为模板图内部的隐含语义节点或中间语义读出。首版实现中, 它们不额外实例化为图节点, 而作为模板编码器内部的隐语义通道或 readout。

模板图编码后, UE 侧建议读出三类隐含语义:

- `urgency embedding`
- `feasibility embedding`
- `service debt embedding`

#### 5.2.3 PRG 模板图

每个 `PRG_{i,g}` 实体内部建立 7 个变量节点:

- `top1SinrDb`
- `top2GapDb`
- `neighborMaxTop1SinrDb`
- `neighborMeanTop1SinrDb`
- `samePrgConflictRatio`
- `iciProxy`
- `availability_mask`

PRG 模板图的首版依赖定义为:

- `top1SinrDb -> local_value`
- `top2GapDb -> local_value`
- `neighborMaxTop1SinrDb -> conflict_risk`
- `neighborMeanTop1SinrDb -> conflict_risk`
- `samePrgConflictRatio -> conflict_risk`
- `iciProxy -> conflict_risk`
- `availability_mask -> usable_flag`
- `local_value + conflict_risk + usable_flag -> prg_context_embedding`

这里的 `local_value`、`conflict_risk`、`usable_flag`、`prg_context_embedding` 在首版实现中同样不额外实例化为图节点, 而作为模板编码器内部的隐语义通道或 readout。

这一步的核心意义是:

- PRG 不再只是平铺特征向量
- 而是被显式拆成“局部收益”“冲突风险”“是否可用”三部分语义

### 5.3 层间稀疏实体图

在模板图编码完成后, 每个实体得到一个 embedding:

- `h_C(i)`: Cell embedding
- `h_U(k)`: UE embedding
- `h_P(i,g)`: PRG embedding

然后再构建层间稀疏实体图。

#### 5.3.1 节点

- Cell 节点: `C_i`
- UE 节点: `U_k`
- PRG 节点: `P_{i,g}`

这里的 `P_{i,g}` 是 cell-local PRG token, 不是全局共享 PRG 节点。这个设计与当前 `Type-0 bitmap allocation` 的动作语义严格对齐。

#### 5.3.2 固定结构边

- `C_i <-> C_j`: 邻接小区边, 静态
- `C_i <-> U_k`: 由 `cellAssocActUe` 决定
- `C_i <-> P_{i,g}`: 静态资源归属边

其中:

- `prgMsk`
- `action_mask_prg_cell`

作为 PRG 节点特征或边 mask 使用, 不作为每个 TTI 动态增删 `Cell-PRG` 拓扑的依据。

#### 5.3.3 稀疏候选边

- `U_k <-> P_{i,g}`: 只为 Top-K 候选 UE 建边, 建议 `K = 2 ~ 4`

候选生成优先级建议:

1. 若当前可稳定导出 `postEqSinr`, 则优先用它做 `UE-PRG` 候选排序
2. 否则用以下混合排序近似:
   - `wbSinr`
   - `top1SinrDb`
   - `top2GapDb`
   - `samePrgConflictRatio` / `iciProxy`
   - UE `urgency score`

需要额外说明的是:

- `top1SinrDb`、`top2GapDb` 是 PRG 级汇总特征
- 在没有 `postEqSinr(ue, prg, layer)` 时, 它们不能精确区分同一 cell 内不同 UE 对同一 PRG 的相对优先级

因此首版应明确分成两档:

- 方案 A: 若已稳定 bridge `postEqSinr`, 则做真正的 `UE-PRG` Top-K 候选边
- 方案 B: 若当前尚未稳定 bridge `postEqSinr`, 则首版 `UE-PRG` 候选边退化为“cell 内共享候选 UE 集”, 其排序主要由 UE 侧变量决定, PRG 差异由 `P_{i,g}` 的上下文在动作头中体现

#### 5.3.4 冲突边

- `P_{i,g} <-> P_{j,g}`: 仅在邻区且同 index PRG 之间建边

首版冲突边特征建议优先使用:

- `neighborMaxTop1SinrDb`
- `samePrgConflictRatio`
- `iciProxy`

若后续需要更强的静态耦合 proxy, 可再补:

- `rxSigPowDB`

### 5.4 在线计算流程

建议把每个 TTI 的在线推理拆成 4 步, 而不是一句“构图后送入 GNN”。

#### Step 1: 在线取数

继续保留本文当前的 `T0/T1/T2/T3` 分层:

- `T0/T1` 进入当前 TTI 主状态
- `T3` 主要用于离线评估、reward 设计参考、校准和回退阈值拟合, 不进入主图状态

这条边界建议严格保留。

#### Step 2: 层内模板编码

对每个 Cell、UE、PRG 分别运行共享模板 encoder, 得到:

- Cell embedding
- UE embedding
- PRG embedding

#### Step 3: 层间稀疏图传播

在 `Cell-UE-PRG` 稀疏实体图上做消息传递, 得到融合后的上下文表示。

#### Step 4: 联合动作头

当前首版实现中, 动作头已经不再采用单纯的 `PRG-centered` 直接分类, 而是改为与 `Type-0 bitmap` 更一致的联合两阶段动作:

第一阶段:

- `slot -> UE`
- 对每个 cell 内的可调度 slot, 在该 cell 当前所有合法活跃 UE 中选择服务对象

第二阶段:

- `PRG -> slot`
- 对每个 `P_{i,g}`, 只在本 cell 已选 slot 中选择分配对象, 或选择留空

其中:

- `UE-PRG` 候选边及 `postEqSinr` 主要用于增强 PRG 侧上下文与局部可行性判断
- 它们不直接裁掉 `slot -> UE` 这一步的全局合法 UE 动作空间

这样做的目的在于:

- 先决定“当前 TTI 该服务哪些 UE”
- 再决定“PRG 如何映射到已选服务 slot”

这比直接让每个 `PRG_{i,g}` 在 Top-K UE 中独立分类, 更贴近原始 Type-0 调度语义。

### 5.5 首版实现边界说明

首版实现中, 层内共享变量关系模板图主要作为共享模板编码器来实现, 不额外实例化为独立图对象, 也不在层内再单独运行一层 GNN。

也就是说:

- Cell、UE、PRG 三类实体先分别经过各自的结构化模板编码器
- 得到 `h_C(i)`、`h_U(k)`、`h_P(i,g)`
- 随后仅在层间稀疏实体图 `Cell-UE-PRG` 上进行一次异质图消息传递

这样做的目的不是否定“层内模板图”的概念, 而是让首版工程实现与概念结构保持一致:

- 概念上保留层内结构先验
- 实现上用共享模板编码器近似层内模板图
- 真正的图消息传递只发生在层间稀疏实体图

这样可以在不丢失层内结构先验语义的前提下, 显著降低首版实现复杂度与调试成本, 并减少训练不收敛时的定位难度。

### 5.6 复杂度与实现策略

与“所有变量都拆成全局节点”的大图相比, 该方案复杂度大致由两部分组成:

- 模板编码:
  - `O(C * |V_cell| + U * |V_ue| + C * G * |V_prg|)`
- 层间传播:
  - `O(|E_CU| + C * G * K + |E_CC| * G)`

其中:

- `C` 是小区数
- `U` 是 active UE 数
- `G` 是 PRG 数
- `K` 是每个 `PRG_{i,g}` 保留的 Top-K 候选 UE 数

复杂度可控的关键原因是:

- 模板图规模固定且共享
- `UE-PRG` 候选边只保留 Top-K
- 冲突边只保留邻区、同 index PRG

因此, 首版实现策略建议是:

- 不做全局变量大图
- 先做“共享模板图 + 稀疏实体图”
- 先用当前可稳定 bridge 的 `T0/T1` 变量跑通

## 6. 首版训练最小闭环输入集

如果目标是先做一个“结构先验约束 + 稀疏实体图 + RL 训练”能跑通的最小闭环, 首版变量建议按模板图输入组织, 而不是继续按平铺特征列表组织。

### 6.1 首版图骨架

- Cell 节点
- UE 节点
- cell-local PRG 节点 `P_{i,g}`
- `cell -> UE` 合法边: `cellAssocActUe`
- `cell -> PRG` 静态资源归属边
- `cell <-> cell` 固定邻接边: 由 `3cell_triangle` 拓扑给定
- `UE_k <-> P_{i,g}` 候选边
- `PRG_{i,g} <-> PRG_{j,g}` 邻区同 index 冲突边

其中 `prgMsk` 与 `action_mask_prg_cell` 作为 PRG 节点特征或边 mask 使用, 不作为每 TTI 动态改拓扑的依据。

### 6.2 Cell 模板图输入变量

当前已实现首版为 6 维:

- `cell_load_bytes`
- `active_ue_count`
- `mean_wb_sinr_lin`
- `cell_mean_avg_rate_mbps`
- `cell_tb_err_ema`
- `reuse_density`

其中 `reuse_density` 当前采用 cell-local 近似:

- `prevPrgAssigned` 的 cell 均值
- `samePrgConflictRatio` 的 cell 均值
- `iciProxy` 的 cell 均值

### 6.3 UE 模板图输入变量

当前已实现首版为 10 维:

- `buffer_bytes`
- `hol_delay_ms`
- `ttl_slack_ms`
- `wb_sinr_lin`
- `avg_rate_mbps`
- `tbErrLastActUe`
- `recent_scheduled_ratio`
- `recent_goodput_deficit_norm`
- `stale_slots`
- `new_data_flag`

其中:

- 前 8 维对应首版模板主链
- `stale_slots` 与 `new_data_flag` 作为当前已接入的辅助上下文, 有助于表达服务陈旧度与新传语义

### 6.4 PRG 模板图输入变量

当前已实现首版为 9 维:

- `top1SinrDb`
- `top2GapDb`
- `neighborMaxTop1SinrDb`
- `neighborMeanTop1SinrDb`
- `samePrgConflictRatio`
- `iciProxy`
- `reuse_ratio`
- `action_mask_prg_cell`
- `prevPrgAssigned`

其中:

- `action_mask_prg_cell` 在模板图里对应 `availability_mask`
- `prevPrgAssigned` 用于表达短时复用惯性
- `reuse_ratio` 保留来自在线观测的局部复用密度语义

### 6.5 候选边排序的首版建议

若当前 `postEqSinr` 可稳定导出:

- 优先用 `postEqSinr` 做 `UE-PRG` Top-K 候选筛选

当前代码与 smoke 验证已经表明:

- `postEqSinr` 在线导出可以稳定开启
- 当前 `candidate_mode` 已可运行在 `post_eq_topk`
- 当前验证到的 `post_eq_layer_dim = 4`

若当前只走最小 bridge 成本路径:

- 首版 `UE-PRG` 候选边退化为“cell 内共享候选 UE 集”
- 候选排序主要由 UE 侧变量决定, 如 `wbSinr + urgency + service debt`
- PRG 差异由 `P_{i,g}` 的上下文在动作头中体现

## 7. 不建议首版就硬塞进图里的量

以下变量不是没价值, 而是放进首版会让系统复杂度和 bridge 成本显著上升:

- `estH_fr_actUe` 全量原始 CFR
- `multiCellSinrCal` 内部精确 ICI 协方差
- `PMI`
- `Beam overlap`
- `PHR`

不建议首版采用的建图方式还有一条:

- 不建议把所有变量拆成全局节点构造一张超大变量图

首版建议策略是:

- 先用 `wbSinr/postEqSinr + PRG conflict proxy`
- 第二版再引入精确 cross-cell channel 或 ICI hook
- 结构上坚持“共享模板图 + 稀疏实体图”

## 8. 对你的三部件设计的直接支撑关系

本节内容主要作为后续阶段接口预留, 用于首版图结构与训练收敛后接入 uncertainty gating 和安全回退, 不作为首版训练目标。

### 8.1 结构先验约束

最直接可用的数据源:

- `cellAssocActUe`
- `action_mask_cell_ue`
- `action_mask_prg_cell`
- 固定 `cell-cell` 邻接
- `samePrgConflictRatio`
- `iciProxy`

这意味着首版就可以做到:

- 约束动作空间
- 约束信息传播路径
- 显式编码“谁会影响谁”

### 8.2 不确定性感知

最适合进入 epistemic uncertainty 估计的输入子集:

- `bufferSize`
- `hol_delay_ms`
- `ttl_slack_ms`
- `wbSinr`
- `postEqSinr` 派生 PRG 特征
- `tbErrLastActUe`
- `recent_scheduled_ratio`
- `recent_goodput_deficit_norm`

最适合做 calibration label / 风险后验的监测量:

- `onlineTbErrRate`
- `expiry_drop_rate`
- `packet_delay_p90/p95`
- `queue_delay_est_ms`

模板图带来的额外好处是:

- UE 层不确定性可以更明确地区分 `urgency / feasibility / service debt`
- PRG 层不确定性可以更明确地区分 `local value / conflict risk`
- Cell 层不确定性可以更明确地区分 `load / error / reuse state`

### 8.3 安全回退

最容易形成明确规则的信号源:

- 低置信度: epistemic uncertainty 高
- 低增益: RL 预期提升低于阈值
- 高风险: `ttl_slack_ms` 过低 / `tbErrRate` 过高 / `expiry_drop_rate` 上升 / `iciProxy` 过高

也就是说, 当前仓库已经具备把 uncertainty 真正接入决策机制的信号基础, 不需要从零补齐。

## 9. 下一步建议

如果下一步是直接开工实现, 建议优先级明确前置为:

1. 优先完成首版图对象构建、边生成器、mask 校验与 graph sanity check
2. 再完成 imitation / teacher warm start, 验证图结构至少能学到不比 PFQ 随机差的行为
3. 之后再接 PPO 主训练

在继续写 uncertainty gating 与 fallback 文档之前, 更建议先完成上述首版实现闭环, 再根据训练表现继续扩展后续机制设计。

下一份文档建议顺着这份继续写三部分:

1. 首版结构依赖图定义
   - 哪些节点
   - 哪些边
   - 哪些边是硬约束
   - 哪些变量只是中介变量

2. uncertainty gating 设计
   - 哪个 head 估计 epistemic uncertainty
   - 哪些监测量用于 calibration
   - 何时从 full RL 降级到 conservative RL

3. fallback 状态机
   - `RL takeover`
   - `conservative RL`
   - `PF/PFQ fallback`
   - `hard safety fallback`

如果只看“当前 Stage-B 能否支撑你的想法”, 结论是:

- 可以
- 而且最关键的那几类变量已经基本就位
- 首版难点不在“有没有数据”, 而在“把变量分层, 不要一次全塞进网络”

## 10. 最新训练实现收口与验证结论

### 10.1 已完成的最新优化

当前代码已经补入并验证通过的优化包括:

- HGraph 输入从早期的 `5/8/7` 扩展到当前的 `6/10/9`
- 动作头从早期 `PRG-centered` 形式升级为联合 `slot -> UE + PRG -> slot`
- `postEqSinr` 在线导出接入, `UE-PRG` 候选边进入真实 `post_eq_topk` 模式
- fixed seed + `online_persistent=1` 下, PPO 已改为跨 iteration 续跑同一 simulator 流
- `--sim-env` 改为可重复累积, 不再丢失前面的 simulator 环境变量
- `stageb_blankaware` 奖励接入并与 simulator 侧 blank-aware 模式对齐
- 训练设备与 simulator 支持分卡:
  - trainer on GPU1
  - simulator on GPU0
- PPO rollout 阶段已减少一次重复 actor/value 前向
- graph 在 rollout 后可常驻训练设备, 降低 PPO update 中的重复搬运开销

### 10.2 最新 smoke 验证结果

当前 smoke PPO 已观察到以下正面信号:

- 日志已出现 `continue`, 说明 fixed seed persistent 续跑生效
- simulator 日志已确认:
  - `online raw postEqSinr export=on`
  - `online reward mode=goodput_reliability_blankaware`
  - `Topology configuration` 与 `TTL=200 ms` 已对齐
- PPO 日志中 `posteqL=4`, 说明 raw `postEqSinr` 已真正进入当前训练路径
- 最新 smoke 后段已达到:
  - `goodput ≈ 1500.84 Mbps`
  - `tb_err ≈ 0.092`
  - `blank ≈ 0.046`
  - `fill ≈ 0.954`

从工程角度看, 这已经明显优于此前 `posteqL=0`、频繁重启 simulator、reward mode 未真正切换时的错误路径。

### 10.3 是否可以开始正式训练

结论: 可以开始正式训练。

但建议按两步推进:

第一步, 先做固定 seed 的长程正式训练:

- 继续使用:
  - `topology_seed=42`
  - `topology_seed_mode=fixed`
  - `online_persistent=1`
  - `reward_mode=stageb_blankaware`
  - `precoding=svd`
  - `postEqSinr export=on`
- 目标是验证:
  - goodput 是否能稳定维持在接近或达到 PF/PFQ 量级
  - `fill` 是否长期保持在高位
  - `blank` 是否不会再次单调上升
  - `tb_err` 是否仍维持在可接受水平

第二步, 在固定 seed 路径稳定后再进入多 seed:

- 再切到:
  - `seed_list = 41,42,43`
  - 或 sequential / list_cycle 模式
- 目标从“先收敛”切换为“看泛化和稳定性”

也就是说, 当前最合理的判断不是“还要继续修结构”, 而是:

- 图结构首版要求已经达到
- online bridge 与 PPO 主链路已经打通
- 当前可以进入正式训练阶段
- uncertainty gating 与 fallback 仍属于第二阶段, 暂不接入首版主训练

## 11. 包级速率 Reward v10 修改

### 11.1 落地顺序

本轮 reward 修改按以下顺序落地:

1. native traffic 层先导出每 TTI 的包完成统计, 不再用全局累计包速率直接塑形。
2. online bridge 协议从 `VERSION=9` 升级到 `VERSION=10`, reward terms 从 12 扩展到 17。
3. Python reward 解析新增 5 个包级字段, 并在 `stageb_blankaware` 中新增默认关闭的塑形项。
4. shell 训练入口补齐新参数, 保持旧训练命令在默认参数下行为不变。

### 11.2 新增 Reward Terms

`reward_terms[12:17]` 固定为:

- `packet_completed_count`: 当前 TTI 完成的包数
- `packet_delivered_bits`: 当前 TTI 完成包对应的 delivered bits
- `packet_system_time_ms`: 当前 TTI 完成包的系统时间总和
- `packet_effective_service_rate_mbps`: 当前 TTI 聚合包有效服务速率
- `packet_effective_service_rate_per_packet_mean_mbps`: 当前 TTI 逐包有效服务速率均值

### 11.3 新增塑形项

`stageb_blankaware` 新增三个默认关闭的项:

- `--packet-rate-weight`: 奖励每 TTI 包级有效服务速率, 用 `log1p(rate / scale)` 压缩尺度
- `--packet-completion-weight`: 奖励当前 TTI 完成包数, 用 `log1p(count / scale)` 防止大流量下过冲
- `--aged-backlog-weight`: 惩罚 `recent100_debt_score_p90_bytes` 和 `no_service_streak_p90` 组合出的 aged backlog

推荐先从保守配置开始:

```bash
--packet-rate-weight 0.15 \
--packet-completion-weight 0.03 \
--packet-rate-scale-mbps 10.0 \
--packet-completion-scale 36.0 \
--aged-backlog-weight 0.08
```

这个组合的意图是给 PPO 一个“把 PRG 转化为及时完成包”的方向, 但不让包数奖励盖过 goodput、BLER 和 expiry 主目标。

## 12. v27-v33 训练复盘与 simple executor 收口

### 12.1 关键实验结论

最近几轮实验表明, HGraph 的主要瓶颈不是 fallback 规则还不够复杂, 而是执行层把 policy action 改写得太多, 导致 PPO 的 credit assignment 变差。

| 版本 | 主要设置 | 结论 |
|---|---|---|
| v27 | raw bridge + packet reward v10 | best goodput 有潜力, 但 post-decode drop 高, 执行空洞明显 |
| v28 | cap slack + PFQ tutor | goodput 明显下降, BLER 升高, PFQ teacher 与当前 joint action 学习目标不匹配 |
| v29 | capSlack3000 + pure ablation | 最早的稳定基线, goodput 均值约 1566 Mbps, drop 下降但仍偏高 |
| v30 | over-cap keep | 出现 1660+ 单点峰值, 但均值未提升, blank/fairness 变差, 规则层开始替 policy 做决策 |
| v31 | over-cap gate | final blank 降低, 单点峰值继续存在, 但均值仍未超过 v29 |
| v32 | dynamic insert fallback + gate | insert fallback 生效, 但 goodput 均值下降, cap pressure 上升, 复杂 repair 收益为负 |
| v33 | simple executor + no packet/backlog shaping | 当前新基线: 全程 goodput 均值约 1568 Mbps, best iter=86, BLER 和 post-decode drop 改善 |

v33 的核心收益不是单点峰值最高, 而是训练行为更健康:

- best checkpoint 出现在 iter 86, 不再是前几轮的偶然峰值
- `insert_fallback` 和 `over-cap keep` 为 0
- post-decode drop 从 v29 的约 24.7% 降到约 16.6%
- BLER 从 v29 的约 9.86% 降到约 9.42%
- goodput 全程均值成为当前几版中最高

代价也很清楚:

- policy blank 上升到约 16.8%
- fill 降到约 66.6%
- unserved demand UE 上升, fairness/coverage 仍需要后续处理

### 12.2 新基线执行层定义

从 v33 开始, HGraph decode 层固定为 simple executor:

1. policy 先产生 joint action: `slot -> UE` 与 `PRG -> slot`
2. decoder 做基本合法性处理
3. 按 `demand_cap_slack_bytes=3000` 检查 UE PRG cap
4. 若某 PRG 指向的 UE 已超 cap, 只允许 fallback 到同 cell、已经被 policy 选中的、有 cap headroom 的 slot
5. 若 fallback 失败, 直接 blank/drop

明确不再使用以下 repair:

- dynamic insert fallback
- over-cap keep
- over-cap gate
- extra/quality/TB-error gate
- fallback 失败后替 policy 动态选择新 UE

保留的诊断指标:

- `cap_drop_count`
- `fallback_success_count`
- `final_blank_count`
- `blank_ratio`
- `post_decode_drop_ratio`
- `action_fill_ratio`

删除的诊断指标:

- `insert_fallback_success_count`
- `over_cap_keep_count`
- `over_cap_block_count`
- `over_cap_extra_block_count`
- `over_cap_quality_block_count`
- `over_cap_tb_err_block_count`

### 12.3 PPO credit assignment 收口

v33 同时修正了 PPO 记录动作的方式:

- PPO update 使用 policy 原始采样动作的 `logp` 和 action class
- decoder 改写后的执行动作不再伪装成 policy 自己采样的动作
- 因此 cap/drop/blank 的后果会回到原始 policy action 上

这条原则后续不要再破坏。若 decoder 需要做安全改写, 它应该作为环境反馈的一部分, 而不是重写 PPO 认为自己执行过的动作。

### 12.4 当前推荐训练配置

v33 推荐作为新的正式训练基线:

```bash
--candidate-mode all_demand_ues \
--ue-prg-topk 6 \
--ue-prg-diversity-extra 2 \
--max-prg-candidates 12 \
--slot-repair-max-per-cell 0 \
--imitation-episodes 0 \
--teacher-aux-coef 0.0 \
--reward-mode stageb_blankaware \
--packet-rate-weight 0.0 \
--packet-completion-weight 0.0 \
--aged-backlog-weight 0.0 \
--backlog-debt-weight 0.0
```

后续如果继续优化, 优先考虑两个轻量方向:

1. 小幅降低 policy blank, 例如轻量 blank penalty 或 high-quality fill bonus
2. 小幅改善 coverage/fairness, 但不要回到 dynamic insert 或 over-cap keep

暂时不建议恢复 PFQ tutor、dynamic fallback 或 over-cap repair。它们会让执行层重新变成隐式 scheduler, 增加 PPO 学习难度。

### 12.5 v33 多 seed 评测结论固化

`v33_best_iter86` 在 `s41/s42/s43`, `warmup1000`, `3cell + 36UE + TTL200ms + RBG16 + SVD` 口径下的结论应收紧为:

> 注意：下表仍是 v33 与 3-seed RRQ/PFQ baseline 的同口径对比。当前 RRQ/PFQ baseline 已扩展为 10-seed `s41-s50`；若要写最终报告，应补跑 HGraph v33 的 10-seed 对比，或明确声明 v33 表是 3-seed 评测。

> HGraph v33 目前不是“全面超过 PFQ”的策略, 而是一个明显具备 PRG 资源效率优势、且没有牺牲边缘 UE 的新型调度基线。它的问题不是不会调度, 而是有效覆盖略不足, 导致 packet-level 服务效率还没有追上 PFQ。

关键对比:

| 指标 | RRQ | PFQ | HGraph v33 | v33 vs PFQ |
|---|---:|---:|---:|---:|
| `traffic.served_mbps_est` | 1703.83 | 1729.52 | 1730.93 | +0.08% |
| `traffic.goodput_mbps` | 1526.62 | 1541.35 | 1537.41 | -0.26% |
| `global_kpi.cluster_spectral_efficiency_bps_per_hz` | 17.400 | 17.663 | 17.677 | +0.08% |
| `global_kpi.ue_goodput_p5_mbps` | 37.36 | 39.68 | 40.20 | +1.30% |
| `global_kpi.ue_goodput_p10_mbps` | 39.16 | 40.88 | 41.05 | +0.41% |
| `global_kpi.ue_goodput_jain` | 0.9961 | 0.9987 | 0.9990 | 略优 |
| `global_kpi.prg_utilization_ratio` | 0.8659 | 0.9135 | 0.7040 | -22.9% |

这说明 v33 至少学到了三件事:

1. 它会选择更高价值的 `PRG/UE` 组合。PRG utilization 明显低于 PFQ, 但 served rate 和 spectral efficiency 反而略高。
2. 它没有牺牲边缘 UE。UE goodput p5/p10 和 Jain 都略好于 PFQ, 不是靠弱 UE 降级换平均速率。
3. 它还没有把 PRG 效率优势完全转化为 packet-level goodput 优势。goodput 比 PFQ 低 0.26%, packet effective service rate 也低于 PFQ。

按单位 PRG 使用效率看, v33 的优势非常清楚:

| 派生指标 | RRQ | PFQ | HGraph v33 | v33 vs PFQ |
|---|---:|---:|---:|---:|
| served Mbps / PRG util | 1967.64 | 1893.37 | 2458.70 | +29.9% |
| goodput Mbps / PRG util | 1763.00 | 1687.38 | 2183.81 | +29.4% |

因此后续论文/报告里的表述建议是:

- 不说“v33 全面超过 PFQ”
- 说“v33 在更低 PRG 使用率下达到接近或略优的簇级速率、频谱效率和边缘 UE 指标”
- 强调它体现了“不是单纯堆资源, 而是用更少资源获得接近或略优 KPI”的多维资源协同价值
- 在固定 nominal PRG 功率模型下, v33 用更低的数据面发射功率占用, 获得了接近 PFQ 的 goodput 和略高的 served throughput / spectral efficiency, 因而体现出更高的资源-功率联合利用效率

### 12.6 packet-level 问题的精确解释

v33 的 packet 问题不应简单解释成“长尾更差”。当前 `packet_delay_p95` 为:

| 策略 | `traffic.packet_delay_p95_ms` |
|---|---:|
| PFQ | 4.67 |
| HGraph v33 | 4.00 |

v33 的 p95 反而更低, 所以它更像是中低时延区间整体右移:

- PFQ 能让更多 packet 很快完成
- v33 因为调度更稀疏, 一部分 packet 多等 1 个或几个 TTI
- 这会拉高平均 system time, 使 `packet_effective_service_rate_mbps` 吃亏
- 但不是典型的 p95/p99 长尾爆炸

同时, `served_mbps_est` 已经略高于 PFQ, 而 `goodput_mbps` 仍略低于 PFQ, 说明后续诊断不能只归因于 PRG 少, 还需要检查:

| 检查项 | 目的 |
|---|---|
| `packet_delay_p50/p75/p90/p99` | 判断是整体右移还是极端长尾 |
| completed packet count | 判断完成包数量是否已经接近 PFQ |
| expired packet count / residual buffer | 判断是否存在隐性积压 |
| TB BLER / failed TB bytes | 判断 goodput 低于 PFQ 是否来自误块 |
| served-goodput gap | 定位 served estimate 到成功交付之间的损耗 |
| 每 TTI scheduled active-demand UE 数 | 判断有效覆盖是否不足 |
| 每 UE 连续未调度 TTI 数 | 判断是否存在短期 starvation |

### 12.7 下一阶段: simple executor + 轻量学习信号

后续不要回到复杂 repair scheduler。v33 已经证明: 在不依赖复杂后处理的情况下, HGraph 能学出明显不同于 RRQ/PFQ 的资源使用模式。若重新加入复杂 repair, 会带来两个风险:

- 性能归因不清: 不知道提升来自 RL policy, 还是 repair 规则兜底
- 策略退化成规则调度器: RL 只负责粗略打分, 真正调度由 repair 接管

下一阶段应保持:

```text
simple executor + light reward shaping
```

核心不是奖励“多用 PRG”, 而是奖励“多填有效 PRG”。

#### 12.7.1 有效 blank 惩罚

不要直接惩罚所有 blank。只惩罚满足以下条件的 `valid_blank`:

1. 当前 `cell c` 的 `PRG g` 最终 blank
2. 该 cell 存在 active-demand UE
3. 至少一个 UE 的 remaining demand 大于 0
4. 该 UE 在该 PRG 上预计 MCS/TBS/SINR 不太差
5. 分配该 PRG 不会明显违反 cap, 也不会带来高 BLER 风险

奖励项:

```text
r_blank = -lambda_b * valid_blank
```

这个项的目标是让模型少浪费“本来可以有效服务业务”的 PRG, 而不是强迫所有 PRG 填满。

#### 12.7.2 active-demand UE coverage 奖励

当前 v33 `stats_mean_uniq_ue` 约 16.5, 在 36 UE 场景下偏低。它不一定错误, 但结合 packet effective rate 偏低, 说明覆盖可能不足。

建议使用 capped coverage, 只奖励“覆盖到”, 不奖励过度服务:

```text
r_cov = lambda_c * sum_u min(1, servedBytes_u / min(demandBytes_u, B_pkt))
```

这样可以鼓励模型让更多有需求 UE 至少完成一部分包, 避免少数高质量 UE 持续吃掉资源。

#### 12.7.3 packet completion / HOL-age 信号

packet-level effective rate 偏低时, 直接优化对象应是 packet completion 和队列停留时间, 而不是 PRG 数。

优先候选:

```text
r_pkt = lambda_p * completedPackets - lambda_d * sum_p waitTime_p
```

如果当前在线状态里 packet age 还不完整, 可先用以下 proxy:

- UE buffer age / HOL age
- 连续未调度次数
- 上一 TTI 有 demand 但未被调度
- scheduling 后 residual buffer

#### 12.7.4 goodput-served gap / BLER 惩罚

由于 v33 served rate 已经略高于 PFQ, 但 goodput 仍略低, 需要防止 policy 只追逐高 instantaneous served estimate。

可加入轻量 gap 或 failed bytes 惩罚:

```text
r_gap = -lambda_g * (R_served - R_goodput)
r_bler = -lambda_e * failedBytes
```

目标是降低 served-to-goodput 损耗, 而不是把 PRG utilization 拉到 PFQ 的 0.91。

### 12.8 v34 候选实验矩阵

不建议把目标设为“接近 PFQ 的 PRG utilization”。更合理的目标是:

```text
在保持 v33 高 PRG efficiency 的基础上, 将 PRG utilization 从 0.70 轻微推到 0.75-0.82, 观察 goodput 和 packet effective rate 是否上升。
```

建议小范围实验矩阵:

| 实验版本 | 改动 | 主要假设 |
|---|---|---|
| `v33-base` | 当前 simple executor | 固化资源效率型基线 |
| `v34-blank-small` | 小 `valid_blank` penalty | 减少可服务 PRG 空洞 |
| `v34-blank-coverage` | `valid_blank` + active-demand coverage | 提高 packet completion 和 UE 覆盖 |
| `v34-coverage-age` | coverage + HOL-age / no-service proxy | 降低中低时延区间整体右移 |
| `v34-full-light` | blank + coverage + age + BLER/gap | 在不过度加规则的前提下补齐 goodput |

每个实验优先看四个主指标:

1. `traffic.goodput_mbps`
2. `traffic.packet_effective_service_rate_mbps`
3. `global_kpi.prg_utilization_ratio`
4. `global_kpi.ue_goodput_p5_mbps` / `global_kpi.ue_goodput_p10_mbps`

成功标准:

- PRG utilization 不需要接近 PFQ, 只需从约 0.70 上升到 0.75-0.82
- goodput 超过 PFQ
- packet effective service rate 明显向 PFQ 靠近
- UE goodput p5/p10 和 Jain 不下降

这条路线与原始研究目标一致: 以簇级速率、干扰水平、频谱效率等 KPI 为目标, 研究时、频、功、空多维资源协同。v33 的价值已经开始体现为“用更少资源获得接近或略优 KPI”, 下一阶段要做的是把这部分资源效率优势转成更稳定的 packet-level 服务效率。
