# 当前 Stage-B 生效配置与 RR/PF 基线说明

## 1. 文档定位

- 本文档是当前 Stage-B 基线场景的唯一配置参考。
- 重点覆盖当前真实生效的三条主线：
  - `./cuMAC/scripts/run_stageB_main_experiment.sh`
  - `./cuMAC/scripts/run_stageB_rr_pf_compare.sh`
  - `multiCellSchedulerUeSelection` 当前在 `Type-0 bitmap` 下的 native PF/RR 行为
- 旧的独立 baseline 场景说明已经并入本文档。

## 2. 一眼结论

当前 Stage-B 基线已经固化为：

- `7` 个 coordinated cells，`0` 个 outer interferer cells
- `4T4R`
- `30 kHz` SCS
- `68` 个 PRG/RBG，每组 `4 PRB`，总计 `272 PRB`
- `Type-0 bitmap allocation`
- `2.5 GHz`
- `slotDurationConst = 0.5 ms`
- 运行入口默认是 DL

当前最重要的变化不是“又多了一个脚本”，而是：

1. native PF 和极限 RR 都已经能在 `GPU + Type-0 bitmap` 下跑通。
2. `run_stageB_rr_pf_compare.sh` 已经把 RR/PF 对比流程脚本化。
3. 当前 baseline 对比在 `Type-0` 下本质上是“PRG bitmap 分配策略对比”，不是旧语义下完整的 UE selection + scheduler 双模块对比。

## 3. 入口脚本与职责

### 3.1 主实验脚本

入口：

```bash
./cuMAC/scripts/run_stageB_main_experiment.sh
```

脚本职责：

- 将 [`cuMAC/examples/parameters.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/parameters.h) 注入为当前 Stage-B 固定编译期参数
- 编译 `multiCellSchedulerUeSelection`
- 运行单场景或场景矩阵
- 调用 [`cuMAC/scripts/summarize_stageA_kpi.py`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/summarize_stageA_kpi.py) 生成：
  - `kpi_summary.json`
  - `kpi_summary.txt`
  - `ue_kpi.csv`

### 3.2 RR/PF 一键比较脚本

入口：

```bash
./cuMAC/scripts/run_stageB_rr_pf_compare.sh
```

脚本职责：

1. 调 `run_stageB_main_experiment.sh` 跑一次 native RR
2. 再跑一次 native PF
3. 对公共场景目录调用 [`cuMAC/scripts/rr_vs_pf_compare.py`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/rr_vs_pf_compare.py)
4. 输出统一的比较产物

当前这条封装只支持：

- `--custom-ue-prg 0`
- 即 native baseline 路径

## 4. 当前脚本实际注入的关键参数

`run_stageB_main_experiment.sh` 当前会固定注入以下编译期常量：

| 参数 | 当前值 |
|---|---:|
| `numCellConst` | `7` |
| `numCoorCellConst` | `7` |
| `numUePerCellConst` | `8` |
| `numActiveUePerCellConst` | `8` |
| `nBsAntConst` | `4` |
| `nUeAntConst` | `4` |
| `nPrbsPerGrpConst` | `4` |
| `nPrbGrpsConst` | `68` |
| `gpuAllocTypeConst` | `0` |
| `cpuAllocTypeConst` | `0` |
| `slotDurationConst` | `0.5e-3` |
| `scsConst` | `30000.0` |
| `cellRadiusConst` | `500` |

对应代码位置：

- [`cuMAC/scripts/run_stageB_main_experiment.sh`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/run_stageB_main_experiment.sh)
- [`cuMAC/examples/parameters.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/parameters.h)

运行期还会通过环境变量补充：

- `CUMAC_TOPOLOGY_SEED`
- `CUMAC_UE_PLACEMENT_MODE`
- `CUMAC_TRAFFIC_ARRIVAL_RATE`
- `CUMAC_EXEC_MODE`
- `CUMAC_CDL_PROFILE`
- `CUMAC_CDL_DELAY_SPREAD_NS`

## 5. 当前拓扑、业务流与 KPI 口径

### 5.1 拓扑

- 当前只建 `1+6` 六边形 `7-cell` 协调簇
- 没有外环干扰站点
- `siteSpacing = 1000 m`
- 非中心小区朝向簇中心

对应代码：

- [`cuMAC/examples/network.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/network.cu)

### 5.2 流量模型

当前脚本已把旧的“`traffic-rate` 看起来像吞吐率”语义明确改成：

- `--packet-size-bytes`：平均包长，单位 `bytes`
- `--traffic-arrival-rate`：平均到达率，单位 `pkt/TTI`

`multiCellSchedulerUeSelection` 内部现在使用：

- `TrafficType(packet_size_bytes, 0, traffic_arrival_rate)`
- 到达过程为 `Poisson`
- 包级队列会记录 `arrival_tti`
- KPI 汇总会同时给出：
  - backlog 估计时延
  - packet-level delay（mean/p50/p90/p95/max）

对应代码：

- [`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/multiCellSchedulerUeSelection/main.cpp)
- [`cuMAC/examples/trafficModel/trafficGenerator.hpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/trafficModel/trafficGenerator.hpp)
- [`cuMAC/examples/trafficModel/trafficService.hpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/trafficModel/trafficService.hpp)

### 5.3 KPI 解释重点

RR/PF 胜负优先看：

- `traffic.*`
- `global_kpi.*`

不要把以下字段误当成 baseline 胜负指标：

- `throughput.*`
- `cpu_gpu_compare.*`

原因：

- `throughput.*` 是 scheduler 内部长时速率状态
- `cpu_gpu_compare.*` 是 CPU/GPU 一致性检查
- 当 `--exec-mode gpu` 时，独立 CPU reference path 会被关闭，此时 `cpu_gpu_compare` 只剩 shadow bookkeeping 含义

## 6. 当前 native PF/RR 在 Type-0 下的真实行为

这是当前最关键的实现边界。

### 6.1 native baseline 路径

条件：

- `--custom-ue-prg 0`

可选 baseline：

- `--baseline-scheduler pf`
- `--baseline-scheduler rr`

### 6.2 Type-0 下 UE selection 的真实语义

当前 `allocType == 0` 时，native baseline 不再先跑 PF/RR UE selection，再交给 scheduler。

实际走的是：

1. [`populateType0AllUeSelection()`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/multiCellSchedulerUeSelection/main.cpp) 先把 `setSchdUePerCellTTI` 填成“所有 active UE”
2. 后续 PF 或 RR scheduler 只负责把 `PRG bitmap` 分给这些 slot

这意味着：

- 当前 Type-0 baseline 的 PF/RR 对比，本质是“PRG allocation kernel 对比”
- 不是旧 Type-1 语义下完整的“UE selection + PRG allocation”双阶段对比

### 6.3 PF/RR 差异实际落在哪

PF 路径：

- GPU: [`multiCellScheduler`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/multiCellScheduler.cu)
- CPU: [`multiCellSchedulerCpu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/multiCellSchedulerCpu.cpp)

RR 路径：

- GPU: [`multiCellRRScheduler`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/roundRobinScheduler.cu)
- CPU: [`roundRobinSchedulerCpu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/roundRobinSchedulerCpu.cpp)

当前 RR 的 `type-0` kernel 行为是：

- 先按 cell 找关联 UE
- 将 `nPrbGrp` 尽量均匀切给这些 UE
- 余数 PRG 从前往后补
- `type-0 + HARQ` 仍不支持

## 7. `exec-mode` 与当前可用组合

`run_stageB_main_experiment.sh` 当前支持：

- `--exec-mode both`
- `--exec-mode gpu`

含义：

- `both`：跑 GPU 主路径，同时保留独立 CPU reference scheduler path
- `gpu`：关闭独立 CPU reference scheduler path，CPU 侧 KPI 仅做镜像/统计

当前限制：

- `exec-mode=gpu` 不支持 `--custom-ue-prg 1`
- `exec-mode=gpu` 不支持 `--online-bridge 1`

对应代码：

- [`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/multiCellSchedulerUeSelection/main.cpp)

## 8. RR/PF 比较脚本当前产物

以用户当前命令为例：

```bash
./cuMAC/scripts/run_stageB_rr_pf_compare.sh \
  --build-method cmake \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --tti 2000 \
  --custom-ue-prg 0 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 0.2 \
  --topology-seed 42 \
  --exec-mode gpu \
  --compact-output 0 \
  --tag rayleigh_seed42_gpu
```

输出目录：

- [`output/stageB_rr_pf_compare_rayleigh_seed42_gpu_20260327_074709`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_rr_pf_compare_rayleigh_seed42_gpu_20260327_074709)

核心产物：

- `rr_wrapper_run.log`
- `pf_wrapper_run.log`
- `compare_manifest.csv`
- `compare_summary.txt`
- `<SCENARIO>/rr_vs_pf_compare.json`
- `<SCENARIO>/rr_vs_pf_compare.csv`
- `<SCENARIO>/rr_vs_pf_compare.txt`

其中 `rr_vs_pf_compare.*` 会给出：

- 统一指标表
- per-cell delta
- Top-N UE throughput delta
- Top-N UE queue-delay delta

## 9. 当前参考结果（2026-03-27，Rayleigh，seed=42，gpu-only）

场景：

- `RAYLEIGH`
- `tti=2000`
- `packet_size_bytes=3000`
- `traffic_arrival_rate=0.2 pkt/TTI`
- `exec-mode=gpu`

结果来源：

- [`output/stageB_rr_pf_compare_rayleigh_seed42_gpu_20260327_074709/RAYLEIGH/rr_vs_pf_compare.txt`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_rr_pf_compare_rayleigh_seed42_gpu_20260327_074709/RAYLEIGH/rr_vs_pf_compare.txt)

关键结论：

- `cluster_sum_throughput_mbps`：RR `2119.033536`，PF `1247.099712`
- `ue_throughput_jain`：RR `0.996670`，PF `0.736751`
- `residual_buffer_ratio`：RR `1.6518%`，PF `42.0366%`
- `packet_delay_p95_ms`：RR `35.5 ms`，PF `525.5 ms`
- `scheduled_ratio_p5`：RR `99.8%`，PF `53.975%`
- PF 仅在少数可靠性指标略优：
  - `global_tb_bler`：PF `8.115%`，RR `8.472%`
  - `global_tx_success_rate`：PF `91.885%`，RR `91.528%`

因此，按当前这组负载和当前 Type-0 baseline 语义：

- RR 明显优于 PF 的主要方向是吞吐、公平性、残留队列和时延
- PF 的优势只剩轻微的 BLER / success-rate 改善

## 10. 对后续智能算法开发的直接启示

当前 baseline 已经不是旧的 Type-1 连续块版本，而是：

- `Type-0 bitmap`
- `7-cell`
- `4T4R`
- native PF/RR 都可在 GPU 下运行

因此后续 GNN+RL 智能算法的第一阶段，应优先和这个 baseline 保持一致：

1. 先对齐同一场景、同一 traffic 模型、同一 KPI 口径
2. 明确“当前 baseline 对比的是 bitmap 分配策略”这一点
3. 先完成与 RR/PF 的同口径 compare automation，再做更复杂的联合动作空间扩展

## 11. 当前已知限制

- 纯 `7-cell` 模式下，`CDL` 仍不是当前主推荐回归模式，稳态对比建议继续优先用 `Rayleigh/TDL`
- `Type-0 + RR + HARQ` 当前不支持
- `gpu-only` 模式下不要把 `cpu_gpu_compare` 当成独立 baseline 校验
- online bridge 与 custom UE+PRG 还没有跟 `gpu-only` 运行模式打通
