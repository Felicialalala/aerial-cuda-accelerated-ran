# 当前 Stage-B 生效配置与 RRQ/PFQ 基线说明

## 1. 文档定位

- 本文档是当前 Stage-B 基线场景的唯一配置参考。
- 重点覆盖当前真实生效的三条主线：
  - `./cuMAC/scripts/run_stageB_main_experiment.sh`
  - `./cuMAC/scripts/run_stageB_rr_pf_compare.sh`
  - `./cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh`
  - `multiCellSchedulerUeSelection` 当前在 `Type-0 bitmap` 下的 native PF/RR 行为
- 旧的独立 baseline 场景说明已经并入本文档。

## 2. 一眼结论

当前 Stage-B 默认基线已经固化为：

- 默认 `7` 个 coordinated cells，`0` 个 outer interferer cells
- `4T4R`
- `30 kHz` SCS
- `68` 个 PRG/RBG，每组 `4 PRB`，总计 `272 PRB`
- `Type-0 bitmap allocation`
- `2.5 GHz`
- `slotDurationConst = 0.5 ms`
- 运行入口默认是 DL
- UE 采用 serving-cell 覆盖区内均匀撒点
- 默认全向发射，不做 sector design
- 入口脚本现已支持 `--topology-scenario 7cell|3cell`；`3cell` 变体保持同样的全向 + 均匀撒点口径，默认也是每站 `8 UE`

当前分析和报告建议优先使用下面这组“最新代表性 baseline”：

- `3cell + 36UE + RBG16 + Rayleigh + TTL200ms + 4T4R + SVD`
- `packet_size_bytes = 3000`
- `traffic_arrival_rate = 1.0 pkt/TTI`
- topology seeds: `41,42,43,44,45,46,47,48,49,50`
- 结果目录：
  [`output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s50_ue36_ttl200_rbg16_svd_20260506_091443`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s50_ue36_ttl200_rbg16_svd_20260506_091443)

这组 10-seed 结果已经取代早期 3-seed / no-SVD / RR-vs-PF 结果，作为当前 RRQ/PFQ baseline 的主参考。

当前最重要的变化不是“又多了一个脚本”，而是：

1. native PF 和极限 RR 都已经能在 `GPU + Type-0 bitmap` 下跑通。
2. `run_stageB_rr_pf_compare.sh` / `run_stageB_rr_pf_multi_seed_compare.sh` 已经把 RRQ/PFQ 对比流程脚本化。
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

### 3.2 RRQ/PFQ 一键比较脚本

单 seed 入口：

```bash
./cuMAC/scripts/run_stageB_rr_pf_compare.sh
```

多 seed 入口：

```bash
./cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh
```

脚本职责：

1. 调 `run_stageB_main_experiment.sh` 跑一次 native RR
2. 再跑一次 native PF 或 PFQ
3. 对公共场景目录调用 [`cuMAC/scripts/rr_vs_pf_compare.py`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/rr_vs_pf_compare.py)
4. 输出统一的比较产物

当前这条封装只支持：

- `--custom-ue-prg 0`
- 即 native baseline 路径

## 4. 当前脚本实际注入的关键参数

`run_stageB_main_experiment.sh` 当前会固定注入以下编译期常量：

| 参数 | 当前值 |
|---|---:|
| `numCellConst` | 默认 `7`，`--topology-scenario 3cell` 时为 `3` |
| `numCoorCellConst` | 默认 `7`，`--topology-scenario 3cell` 时为 `3` |
| `numUePerCellConst` | 默认 `8`；当前 3cell/36UE benchmark 通过脚本注入为 `12` |
| `numActiveUePerCellConst` | 默认 `8`；当前 3cell/36UE benchmark 通过脚本注入为 `12` |
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
- `CUMAC_UE_VORONOI_CLIP`
- `CUMAC_BS_TX_PATTERN`
- `CUMAC_TRAFFIC_ARRIVAL_RATE`
- `CUMAC_EXEC_MODE`
- `CUMAC_CDL_PROFILE`
- `CUMAC_CDL_DELAY_SPREAD_NS`

## 5. 当前拓扑、业务流与 KPI 口径

### 5.1 拓扑

- 默认建 `1+6` 六边形 `7-cell` 协调簇
- 也支持 `3-cell` 等边三角协同簇，可通过 `--topology-scenario 3cell` 切换
- 没有外环干扰站点
- `siteSpacing = 1000 m`
- UE 默认在 serving-cell 覆盖区内做面积均匀撒点
- Stage-B 默认开启 `Voronoi clip`，把 UE 约束在本小区覆盖区内
- 默认 `bs_tx_pattern = omni`，不再设计扇区朝向或扇区水平增益

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
- MAC packet tracker 会为每个成功进入发送队列的 packet 记录：
  - `packet_id`
  - `packet_size_bytes`
  - `arrival_time`
  - `finish_time`
  - `delivered`
- KPI 汇总会同时给出：
  - backlog 估计时延
  - packet-level delay（mean/p50/p90/p95/max）
  - packet-based average effective service rate：
    - `R_pkt = sum(delivered payload bits) / sum(packet system time)`
    - 当前实现里，统计对象是 MAC packet queue tracker 中“整包出队完成”的 packet
    - 分子只按原始 payload 计一次，不因重传重复累计
    - 分母自然包含排队、等待调度、重传等全过程
    - 为避免低负载时由于离散 TTI 时间戳导致“same-slot 完成 = 0 ms”而把 `R_pkt` 放大失真，当前 packet system time 口径统一取 `max(raw_system_time, 1 slot)`
    - 因此如果 packet 在与入队相同的 slot 内完成，`packet_delay_*` 和 `R_pkt` 分母都按 `1 slot` 计，而不是 `0 ms`
    - 当前输出字段为 `TRAFFIC_PKT_RATE.r_pkt_mbps`
    - 汇总后进入 `traffic.packet_effective_service_rate_mbps` 与 `global_kpi.packet_effective_service_rate_mbps`
  - packet-based per-packet mean effective service rate：
    - 先对每个已完成 packet 计算 `r_p = L_p / packet_system_time_p`
    - 再做算术平均：`R_pkt_mean = (1/N) * sum(r_p)`
    - 当前输出字段为 `TRAFFIC_PKT_RATE.r_pkt_per_packet_mean_mbps`
    - 汇总后进入 `traffic.packet_effective_service_rate_per_packet_mean_mbps` 与 `global_kpi.packet_effective_service_rate_per_packet_mean_mbps`
  - packet count / throughput 类指标：
    - `traffic.packet_delay_served_pkt_count`：MAC packet queue tracker 中“整包出队完成”的 packet 个数
    - `traffic.goodput_mbps = goodput_bytes * 8 / total_time_s / 1e6`
    - `traffic.served_mbps_est = served_bytes_est * 8 / total_time_s / 1e6`
    - 其中 `packet_delay_served_pkt_count` 是“个数”，`goodput_mbps` / `served_mbps_est` 是“速率”，三者单位不同，不能直接比较数值大小

对应代码：

- [`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/multiCellSchedulerUeSelection/main.cpp)
- [`cuMAC/examples/trafficModel/trafficGenerator.hpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/trafficModel/trafficGenerator.hpp)
- [`cuMAC/examples/trafficModel/trafficService.hpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/trafficModel/trafficService.hpp)

### 5.3 KPI 解释重点

RRQ/PFQ 胜负优先看：

- `traffic.*`
- `global_kpi.*`
- 其中 `traffic.packet_effective_service_rate_mbps` / `global_kpi.packet_effective_service_rate_mbps` 适合和 `packet_delay_*` 一起看“当前 tracker 定义下已完成 packet 的平均有效服务能力”
- 其中 `traffic.packet_effective_service_rate_per_packet_mean_mbps` / `global_kpi.packet_effective_service_rate_per_packet_mean_mbps` 是“每包先单独算速率再取平均”的版本，会比 `R_pkt` 更偏向奖励短 system-time packet
- 其中 `packet_delay_*` 与 `packet_effective_service_rate_mbps` 共享同一 packet system time 口径，当前最小粒度为 `1 slot`

### 5.4 `packet_delay_served_pkt_count` 与 `goodput_mbps` 的区别

这两个指标看起来都和“包发送了多少”有关，但它们并不是同一类量：

- `packet_delay_served_pkt_count`：
  - 单位是 `packet`
  - 统计的是 MAC packet queue tracker 里“整包出队完成”的次数
  - 当前实现里，这个完成事件发生在 packet 的最后一个字节被从 MAC buffer / packet tracker 中扣减完时
  - 因此它更接近 `served_mbps_est` 对应的“MAC dequeue / served”口径
- `goodput_mbps`：
  - 单位是 `Mbps`
  - 统计的是 `tbErr == 0` 时累计的 `goodput_bytes`
  - 也就是 PHY 侧成功解码后的有效吞吐率

按当前实现，`packet_delay_served_pkt_count` / `traffic.packet_effective_*` 与 `goodput_mbps` 的完成事件并不相同：

- 前者是 MAC queue tracker 的“出队完成”
- 后者是 PHY 侧的“成功解码完成”

因此更合理的关系是：

- 若包长固定为 `L_pkt_bits`，仿真总时长为 `T`
- 则 `packet_delay_served_pkt_count * L_pkt_bits / T` 更接近 `served_mbps_est`
- 而不是更接近 `goodput_mbps`

以当前这组 `1000 bytes`、`4000 TTI`、`slot=0.5 ms` 的实验为例：

- 总时长 `T = 4000 * 0.5 ms = 2 s`
- 每包 `L_pkt_bits = 1000 * 8 = 8000 bits`
- 若 `packet_delay_served_pkt_count = 14413.67`
- 则等效 MAC dequeue 速率约为 `14413.67 * 8000 / 2 / 1e6 = 57.65 Mbps`
- 这与结果里的 `traffic.served_mbps_est ≈ 57.65 Mbps` 一致
- 而 `traffic.goodput_mbps ≈ 46.49 Mbps` 更低，是因为只有成功解码的 TB 字节才会计入 goodput

什么时候两者会出现较大的差异：

- `BLER / tbErr` 较高时：MAC 已出队的字节不一定都成为 successful goodput，`goodput_mbps` 会明显低于按 packet count 换算得到的 dequeue 速率
- 重传较多时：goodput 只认最终成功解码的有效字节，而 packet delay / `R_pkt` 会把等待与重传时间都算进去
- 包长固定但较小、仿真时间较长时：packet count 可以很大，而 `Mbps` 仍然只是几十，这属于单位差异导致的正常现象
- 包长可变时：单纯 packet count 更不能直接映射到 goodput，因为它不反映每个 packet 携带的 bit 数
- 低负载且 backlog 很小时：`packet_delay_served_pkt_count` 可能接近 offered packet 总数，但 `goodput_mbps` 仍由“总成功 bit / 总时间”决定，两者数值量级未必接近

如果后续需要让 packet-level completion 严格对齐“成功解码完成”而不是“MAC 出队完成”，则需要把 packet 生命周期完成事件从当前的 MAC queue tracker 扣减路径，进一步绑定到 `tbErr == 0` 的 ACK / goodput 记账路径上。

### 5.5 `packet_effective_service_rate_mbps` 与 `goodput_mbps` 的区别

这两个指标虽然单位都写成 `Mbps`，但不要把它们当成同一种“吞吐量”去直接比较数值大小。

- `traffic.goodput_mbps`：
  - 定义为 `goodput_bytes * 8 / total_time_s / 1e6`
  - 分子是 PHY 侧成功解码的有效字节
  - 分母是整场仿真的 wall-clock 时间
  - 它是系统级 aggregate goodput，表示整个 cluster 在单位时间内真正成功送达了多少有效 bit
- `traffic.packet_effective_service_rate_mbps`：
  - 定义为 `R_pkt = total_delivered_bits / total_packet_system_time`
  - 当前实现里，统计对象是 MAC packet queue tracker 中“整包出队完成”的 packet
  - 分母不是 wall-clock 时间，而是所有 packet system time 的求和：
    - `sum_p (t_finish_p - t_arr_p)`
    - 且当前口径下每个成功 packet 至少贡献 `1 slot`
  - 它本质上是 packet-centric rate，更接近“单个已完成 packet 的平均有效服务速度”

在包长固定为 `L_pkt_bits` 时，这个关系可以写成：

- `R_pkt = sum(L_pkt_bits) / sum(W_p) = L_pkt_bits / E[W_p]`

也就是说，固定包长下：

- `packet_effective_service_rate_mbps` 近似等于“包长 / 平均 packet system time”
- 它更像时延指标的另一种写法
- 它不是 cluster aggregate throughput

如果忽略当前“MAC queue completion”和“PHY success completion”之间的事件边界差异，可以把两者的关系粗略理解成：

- `served_mbps_est ≈ R_pkt * N_bar`
- `goodput_mbps ≈ served_mbps_est * success_fraction`
- 因而 `goodput_mbps ≈ R_pkt * N_bar * success_fraction`

其中：

- `N_bar` 表示系统中平均同时处于 packet system time 统计口径内的 packet 并发度
- `success_fraction` 近似反映 `tbErr == 0` 的成功比例

所以 `goodput_mbps` 比 `R_pkt` 大很多通常并不奇怪，常见原因是：

- 多 cell、多 UE、多个 packet 在并发服务，aggregate goodput 会随并发度叠加
- `R_pkt` 是按每个 packet 自己的 system time 做平均，不会因为并发 packet 多就线性变大
- `goodput_mbps` 只看总成功 bit / 总时间，而 `R_pkt` 会把排队、等待调度、重传时间都压进分母

以当前这组 `1000 bytes`、`4000 TTI`、`slot=0.5 ms` 的实验为例：

- `traffic.goodput_mbps ≈ 46.49 Mbps`
- `traffic.packet_effective_service_rate_mbps = 16.0 Mbps`
- `traffic.served_mbps_est ≈ 57.65 Mbps`
- `global_kpi.global_tx_success_rate ≈ 0.807`

这组数可以这样读：

- `R_pkt = 16 Mbps`，因为固定 `1000 bytes = 8000 bits` 的包在当前口径下最小 system time 是 `0.5 ms`，因此理论上限就是 `8000 / 0.0005 / 1e6 = 16 Mbps`
- `served_mbps_est / R_pkt ≈ 57.65 / 16 ≈ 3.60`，说明系统 aggregate served 速率大约相当于 `3.6` 个 packet-level effective rate 在并行叠加
- `goodput_mbps / served_mbps_est ≈ 46.49 / 57.65 ≈ 0.806`，这又与当前 `global_tx_success_rate ≈ 0.807` 基本一致

什么时候两者会出现较大的差异：

- 并发度高时：更多 packet 同时在系统里流动，`goodput_mbps` 会明显高于 `R_pkt`
- packet system time 长时：排队、等待调度、重传会拉低 `R_pkt`，但不一定等比例拉低 goodput
- `BLER / tbErr` 高时：`served_mbps_est` 与 `goodput_mbps` 会进一步拉开，goodput 更低
- 包长固定且系统时延接近 `1 slot` 下界时：`R_pkt` 容易触顶，继续提升并发/成功率时，goodput 还可以继续上升，但 `R_pkt` 已经不再敏感
- 包长变化较大时：`R_pkt` 与 goodput 的关系会更复杂，因为 `R_pkt` 受 packet-size 分布和 packet-time 分布共同影响

因此：

- `goodput_mbps` 更适合看系统级吞吐能力
- `packet_effective_service_rate_mbps` 更适合和 `packet_delay_*` 一起看 packet 体验与 timeliness
- 两者都重要，但不应该按“哪个 Mbps 更大”这种方式直接判断调度器优劣

### 5.6 `packet_effective_service_rate_mbps` 与 `packet_effective_service_rate_per_packet_mean_mbps` 的区别

当前文档里这两个 packet-level `Mbps` 指标分别对应两种平均方式：

- `traffic.packet_effective_service_rate_mbps`：
  - `R_pkt = sum(L_p) / sum(W_p)`
  - 这是“先汇总 bits，再汇总 packet system time”的 ratio-of-sums 版本
  - 固定包长时可写成 `R_pkt = L / mean(W_p)`
  - 它更接近调和平均语义，对长 system-time packet 更敏感
- `traffic.packet_effective_service_rate_per_packet_mean_mbps`：
  - `R_pkt_mean = (1/N) * sum(L_p / W_p)`
  - 这是“每包先单独算速率，再做算术平均”的版本
  - 固定包长时可写成 `R_pkt_mean = L * mean(1 / W_p)`
  - 它会更偏向奖励那些很快完成的 packet

两者一般不相等。对固定包长场景，有：

- `mean(1 / W_p) >= 1 / mean(W_p)`
- 因而通常 `R_pkt_mean >= R_pkt`

只有在所有 packet 的 `W_p` 都完全相同时，这两个指标才会相同。

对当前 StageB 来说，可以这样理解：

- `R_pkt` 更保守，更能反映长尾等待和重传代价
- `R_pkt_mean` 更乐观，更像“典型 packet 的单包完成速率”
- 当系统里存在一批很快完成的 packet 和一批明显更慢的 packet 时，`R_pkt_mean` 往往会明显高于 `R_pkt`

因此：

- 如果你想看 packet timeliness / tail penalty，优先看 `R_pkt`
- 如果你想额外观察“每包单独折算速率后的平均体验”，可以同时看 `R_pkt_mean`
- 不建议只看 `R_pkt_mean`，因为它更容易被短时延 packet 拉高，从而掩盖慢 packet 的影响

### 5.7 调度优化建议

如果目标是评价或训练调度算法，当前更推荐：

- 主目标优先看 `goodput_mbps`
- 副目标或约束同时看 `packet_delay_p95_ms`、`packet_effective_service_rate_mbps`、`packet_effective_service_rate_per_packet_mean_mbps`、`expiry_drop_rate`、`global_tb_bler`

原因：

- `goodput_mbps` 最接近“系统最终真正送达了多少有效 bit/s”，是更硬的系统级收益指标
- 它天然同时约束了资源利用、UE 选择、链路质量匹配、BLER 和重传代价
- `packet_effective_service_rate_mbps` 虽然也用 `Mbps` 表示，但本质上更接近 packet delay 的倒数乘包长，是 packet-centric KPI
- `packet_effective_service_rate_per_packet_mean_mbps` 比 `R_pkt` 更容易被短时延 packet 拉高，因此更适合作为辅助观察，不适合替代系统级主目标

从“更容易出效果”看：

- 更容易被调出明显变化的，通常是 `packet_effective_service_rate_per_packet_mean_mbps`，其次是 `packet_effective_service_rate_mbps`
  - 尤其在中高负载下，只要 packet delay 有明显变化，`R_pkt` 就会立刻响应
  - `R_pkt_mean` 则会更快地被一批快速完成的 packet 拉高
  - 但这两个 packet-centric 指标都更容易受到统计口径、slot 下界、包长分布和 completion semantics 的影响
- 更难但也更有意义的，通常是 `goodput_mbps`
  - 想把它真正做高，必须同时兼顾调度效率和成功解码率
  - 这个指标更不容易“虚高”，因此更适合当主优化方向

对当前 StageB 来说，更稳妥的建议是：

- 若只能选一个主 KPI：优先优化 `goodput_mbps`
- 若业务确实强时延敏感，再把 `packet_effective_service_rate_mbps` 和 `packet_effective_service_rate_per_packet_mean_mbps` 作为 packet-centric 辅助 KPI
- 不建议把任一 packet-centric `Mbps` 指标单独当成唯一目标，否则调度器可能更偏向“缩短部分 packet 的系统时延”，但未必带来更高的真实成功吞吐

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
- `--baseline-scheduler pfq`
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

### 6.4 当前 PFQ 最终保留的改造

当前 `--baseline-scheduler pfq` 的最终 baseline，已经固定为**单阶段 queue-aware PFQ**，不再保留之前试验过的二阶段空闲 PRG 填充逻辑。

当前 PFQ 的核心改造只有三层：

- 基础分数仍然是原始 PF 形式：`rate^beta / avgRate`
- 在基础 PF 上叠加 queue-aware buffer 权重：
  - `1 + coeff * log2(1 + buffer_bytes / scale_bytes)`
- 在 `Type-0` 逐 PRG 分配时，再叠加两条单 TTI 内的约束：
  - `demand-aware cap`：按瞬时速率和 `slotDuration` 估算单个 PRG 可服务字节数；若 UE 在当前 TTI 已分到的字节已经接近 `buffer + slack`，则继续分配会被压低甚至截断
  - `intra-TTI decay`：同一 UE 在同一 TTI 内连续拿到多个 PRG 时，分数按 `1 / (1 + decay * assigned_prg_count)` 递减

当前 final baseline **明确不包含**：

- 二阶段 relaxed PF idle-PRG fill
- threshold-gated relaxed fill

这些都只是在 2026-04-20 的调参过程中做过试验，最终没有保留到当前 baseline。

这样定稿的设计目的很直接：

- 避免好信道 UE 在 `Type-0` 贪心分配里拿走远超当前需求的 PRG
- 让 PFQ 至少先做到“按需求收敛”，而不是继续无上限追逐瞬时速率
- 因此当前 PFQ 的 `global_kpi.prg_utilization_ratio` 可能低于 RR；如果 `goodput`、`delay`、`residual buffer` 没有同步恶化，这种“未把空闲 PRG 再补满”的行为是设计使然，不应单独视为 bug

当前运行期配置项如下：

- `CUMAC_PFQ_BUFFER_COEFF`，默认 `0.25`
- `CUMAC_PFQ_BUFFER_SCALE_BYTES`，默认 `packetSizeBytes`
- `CUMAC_PFQ_DEMAND_CAP`，默认 `1`
- `CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES`，默认 `packetSizeBytes`
- `CUMAC_PFQ_INTRA_TTI_DECAY_COEFF`，默认 `0.5`

对应代码位置：

- 参数注入与日志输出：[`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/multiCellSchedulerUeSelection/main.cpp)
- CPU Type-0 PFQ：[`cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/multiCellSchedulerCpu.cpp)
- GPU Type-0 PFQ 与 host 后处理：[`cuMAC/src/4T4R/multiCellScheduler.cu`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/src/4T4R/multiCellScheduler.cu)

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

## 8. RRQ/PFQ 比较脚本当前产物

当前推荐的 RRQ/PFQ 多 topology-seed baseline 命令为：

```bash
RUN_TS="$(date +%Y%m%d_%H%M%S)"; \
./cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh \
  --seed-list 41,42,43,44,45,46,47,48,49,50 \
  --reference-baseline rrq \
  --pf-baseline pfq \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --build-method cmake \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --tti 4000 \
  --prbs-per-group 16 \
  --precoding svd \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1 \
  --packet-ttl-ms 200 \
  --progress-tti 1000 \
  --kpi-tti-log 0 \
  --compare-tti 0 \
  --compact-output 1 \
  --exec-mode gpu \
  --tag rrq_pfq_3cell_s41_s50_ue36_ttl200_rbg16_svd \
  --compare-output-dir "output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s50_ue36_ttl200_rbg16_svd_${RUN_TS}"
```

当前保留的输出目录：

- [`output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s50_ue36_ttl200_rbg16_svd_20260506_091443`](/home/oai2/aerial-cuda-accelerated-ran/output/stageB_rrq_pfq_multiseed_compare_rrq_pfq_3cell_s41_s50_ue36_ttl200_rbg16_svd_20260506_091443)

核心产物命名会跟第二个 baseline 联动。
例如 `--pf-baseline pf` 时是 `rr_vs_pf_compare.*`，`--pf-baseline pfq` 时是 `rr_vs_pfq_compare.*`。

通用产物：

- `rr_wrapper_run.log`
- `<baseline>_wrapper_run.log`
- `compare_manifest.csv`
- `compare_summary.txt`
- `<SCENARIO>/rr_vs_<baseline>_compare.json`
- `<SCENARIO>/rr_vs_<baseline>_compare.csv`
- `<SCENARIO>/rr_vs_<baseline>_compare.txt`

其中 `rr_vs_<baseline>_compare.*` 会给出：

- 统一指标表
- per-cell delta
- Top-N UE throughput delta
- Top-N UE queue-delay delta

注意：

- 这些 `compare.csv/json/txt` 是最终 KPI 对比结果，不是 RL replay transition，不能直接拿去做 `bc_train.py` / `ppo_train.py`。
- 若要做离线训练，必须重新跑 `run_stageB_main_experiment.sh --replay-dump 1` 生成 `rl_replay_meta.json`、`rl_replay_schema.json`、`rl_replay_records.bin`。

## 9. 历史参考结果处理

2026-03 至 2026-04 的早期 RR/PF、RR/PFQ、no-SVD 或 3-seed 结果只保留为历史背景，不再作为当前 baseline 引用。相关本地输出目录已经清理；如果需要复现，应按第 8 节命令重新跑同口径结果。

当前报告不要再引用 2026-03 的旧 RR/PF 单 seed 输出或早期 `rr_vs_pfq_compare_mean.csv` 作为主结论来源。

## 10. 对后续智能算法开发的直接启示

当前 baseline 已经不是旧的 Type-1 连续块版本，而是：

- `Type-0 bitmap`
- 默认 `7-cell`
- `4T4R`
- native PF/RR 都可在 GPU 下运行

因此后续 GNN+RL 智能算法的第一阶段，应优先和这个 baseline 保持一致：

1. 先对齐同一场景、同一 traffic 模型、同一 KPI 口径
2. 明确“当前 baseline 对比的是 bitmap 分配策略”这一点
3. 先完成与 RRQ/PFQ 的同口径 compare automation，再做更复杂的联合动作空间扩展

## 11. 当前已知限制

- 纯 coordinated-cluster 模式下，`CDL` 仍不是当前主推荐回归模式，稳态对比建议继续优先用 `Rayleigh/TDL`
- `Type-0 + RR + HARQ` 当前不支持
- `gpu-only` 模式下不要把 `cpu_gpu_compare` 当成独立 baseline 校验
- online bridge 与 custom UE+PRG 还没有跟 `gpu-only` 运行模式打通
