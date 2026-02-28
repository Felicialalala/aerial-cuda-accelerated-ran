# 当前 Stage-B 场景实际生效配置梳理

## 1. 适用范围

本文档对应当前 Stage-B 主实验脚本标准化后的实际生效配置，入口脚本为：

```bash
./cuMAC/scripts/run_stageB_main_experiment.sh --build-method cmake --tti 400 --custom-ue-prg 0
```

本文档描述的是代码层面的“当前脚本注入后生效配置”，重点覆盖：

- 标准化簇几何：`1+6` 六边形 `7-site`
- 一圈簇外 interferers：总 `19-site`
- 同频簇内调度
- 频域分配从 `Type-1` 改为 `Type-0` 风格
- `nPrbsPerGrp=4` 作为最小频域粒度

说明：

- 当前 Stage-B 默认切回原生 PF 调度链路，即 `--custom-ue-prg 0`。
- `CustomUePrgScheduler` 当前仍只支持 `Type-1`，因此不再作为这个标准化 Stage-B 的默认路径。
- 脚本会在运行前修改 [`cuMAC/examples/parameters.h`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/parameters.h) 中的编译期宏，然后再执行编译。
- 当前脚本默认 `--restore-params 0`，也就是运行结束后不会自动回滚 `parameters.h`。

---

## 2. 标准化后的一眼结论

### 2.1 簇几何

当前场景已标准化为：

- `7-site` 协调簇：`1 + 6`
- `12-site` 外环干扰层
- 总站点数：`19`
- 当前按“每 site 1 sector / 1 cell”建模

因此：

- `numCoorCellConst = 7`
- `numCellConst = 19`
- `nInterfCell = 12`

### 2.2 簇内/簇外关系

- `7` 个内环站点参与 coordinated scheduling
- 外圈 `12` 个站点只作为同频干扰源，不承载本簇调度 UE
- 所有站点同频复用，属于 `reuse-1`

### 2.3 频域资源形态

当前已从原先的 `Type-1` 连续分配切换为：

- `Type-0` 风格的 bitmap / RBG 映射
- 频域最小粒度仍保持 `4 PRB`

这样做的直接意义是：

- 簇内各小区的频域占用不再被限制为单连续块
- 可以研究“频域 bitmap 形状”
- 可以研究跨小区错位/打孔/避让等频域协同策略

### 2.4 仍然未改动的关键点

以下参数仍沿用现有实现：

- 载频仍是 `2.5 GHz`，不是 `3.5 GHz`
- SCS 仍是 `30 kHz`
- `nPrbGrps = 68`
- `nPrbsPerGrp = 4`
- 有效总 PRB 数仍是 `272`
- 有效带宽仍是 `97.92 MHz`
- 当前仍是纯下行仿真入口，不是显式 TDD 时隙图样仿真

---

## 3. 运行控制参数

| 类别 | 参数 | 当前值 | 说明 |
|---|---|---:|---|
| 脚本 | `--tti` | `400` | 当前一次运行 400 个 slot |
| 脚本 | `--mode` | `dl` | 默认下行 |
| 脚本 | `--custom-ue-prg` | `0` | 原生 PF UE selection + scheduler |
| 脚本 | `--fading-mode` | `3` | GPU CDL，PRG 粒度 CFR |
| 脚本 | `--traffic-percent` | `100` | 全部配置 UE 都有业务流 |
| 脚本 | `--traffic-rate` | `5000` | 传给业务流模型的 `data_rate` 参数；在当前实现中等价为每包均值 `5000 bytes` |
| 脚本 | `--compact-output` | `1` | 常规输出 |
| 脚本 | `--compact-tti-log` | `1` | 压缩 TTI 细日志 |
| 脚本 | `--compare-tti` | `0` | 默认关闭逐 TTI CPU/GPU 对比 |
| 脚本 | `--restore-params` | `0` | 默认不恢复 `parameters.h` |

仿真时长：

- `slotDuration = 0.5 ms`
- `400 slot = 200 ms`

脚本实际会注入的关键编译期宏为：

- `numSimChnRlz = TTI_COUNT`
- `numCellConst = 19`
- `numCoorCellConst = 7`
- `numUePerCellConst = UE_PER_CELL`
- `numUeForGrpConst = UE_PER_CELL`
- `numActiveUePerCellConst = UE_PER_CELL`
- `totNumUesConst = numCoorCellConst * numUePerCellConst`
- `totNumActiveUesConst = numCoorCellConst * numActiveUePerCellConst`
- `nBsAntConst = 4`
- `nUeAntConst = 4`
- `nPrbsPerGrpConst = 4`
- `nPrbGrpsConst = 68`
- `gpuAllocTypeConst = 0`
- `cpuAllocTypeConst = 0`

---

## 4. 簇几何与站点拓扑

### 4.1 站点层级

当前采用两层六边形站点布局：

- 第 0 圈：`1` 个中心站点
- 第 1 圈：`6` 个协调站点
- 第 2 圈：`12` 个外环干扰站点

总计：

- `1 + 6 + 12 = 19 site`

### 4.2 协调簇与干扰层

| 层级 | 站点数 | 作用 |
|---|---:|---|
| 内层协调簇 | `7` | 有 UE、有调度、有 KPI |
| 外环干扰层 | `12` | 无本簇 UE，仅产生同频干扰 |
| 总计 | `19` | 全部进入信道与干扰计算 |

### 4.3 站点间距

当前采用：

- `cellRadiusConst = 500 m`
- `siteSpacing = 2 x cellRadius = 1000 m`

因此第一圈 6 个协调邻站相对中心站的典型坐标为：

- `(1000, 0)`
- `(500, 866.025)`
- `(-500, 866.025)`
- `(-1000, 0)`
- `(-500, -866.025)`
- `(500, -866.025)`

第二圈外环干扰站点由标准 2-ring hex layout 生成，共 12 个。

### 4.4 每站点 1 sector 的含义

当前简化为：

- 每个 site 只建一个 cell
- 每个 cell 只对应一个服务扇区
- 不再是旧版本里“每站 3 扇区平铺凑 7 cell”的方式

这样做的目的：

- 先把簇几何标准化
- 先把频域 bitmap 调度形态拉通
- 降低多扇区方向图和站内耦合对结果解释的干扰

### 4.5 站点朝向

当前每个站点都有单独的 `cellOrien`：

- 中心站点朝向取 `0`
- 外圈站点朝向取“指向簇中心”的方向

这意味着：

- 第一圈协调站点的单扇区是朝内指向中心区的
- 第二圈 interferer 站点也是朝内指向协调簇的

这样设置的作用是：

- 强化簇内与外环对协调簇的同频干扰耦合
- 避免外环 interferer 天线主瓣背离簇中心，导致干扰过弱

---

## 5. 小区、UE 与关联配置

### 5.1 小区数量

| 参数 | 当前值 |
|---|---:|
| `numCellConst` | `19` |
| `numCoorCellConst` | `7` |
| `totNumCell` | `19` |
| `nCell` | `7` |
| `nInterfCell` | `12` |

### 5.2 UE 数量

当前 UE 只配置在 `7` 个协调站点上：

| 参数 | 当前值 |
|---|---:|
| `numUePerCellConst` | `8` |
| `numActiveUePerCellConst` | `8` |
| `numUeForGrpConst` | `8` |
| `totNumUesConst` | `7 x 8 = 56` |
| `totNumActiveUesConst` | `7 x 8 = 56` |

外环 `12` 个 interferer 站点：

- 不配置本簇 UE
- 不进入本簇业务流与 KPI 统计
- 只作为发射端干扰源参与信道/干扰计算

### 5.3 UE 空间分布

当前每个协调 cell 的 UE 位置仍为随机生成，但生成范围已经基于“单 sector / 单 site”重解释：

- 极角范围：以该 cell 朝向为中心的 `±60°`
- 半径范围：`30 m ~ 500 m`
- 高度：`1.5 m`

更具体地说：

- `siteSpacing = 2 * cellRadius = 1000 m`
- 站点坐标由硬编码的 2-ring hex axial layout 生成
- 非中心站点的单扇区朝向设置为“指向簇中心”
- UE 极角通过 `uniform[-pi/3, +pi/3] + orientation` 采样
- UE 距离通过 `uniform[minD2Bs, cellRadius]` 采样

因此当前 UE 分布特征是：

- 仍是随机分布
- 不是手工分层的 center/mid/edge 采样
- 但已经和标准 1+6 site 几何对齐，不再混入“伪 3-sector 站点展开”的拓扑偏差

### 5.4 UE 与小区绑定

当前仍采用固定关联：

- 只在 `7` 个协调 cell 内部分配 UE
- `UE 0~7 -> cell 0`
- `UE 8~15 -> cell 1`
- ...
- `UE 48~55 -> cell 6`

当前没有：

- strongest-cell 动态关联
- handover
- 簇内动态负载均衡切换

第一个 TTI 后：

- `cellIdRenew = false`
- `cellAssocRenew = false`

因此当前研究重点仍然是：

- 同频干扰耦合
- 频域 bitmap 形状
- 调度与干扰的联合影响

而不是接入控制/切换控制。

---

## 6. 基站与 UE 无线参数

### 6.1 基站参数

| 参数 | 当前值 |
|---|---:|
| 场景 | `UMa` |
| `carrierFreq` | `2.5 GHz` |
| `bsHeight` | `25 m` |
| `bsTxPower` | `49 dBm` |
| `GEmax` | `9 dBi` |
| `bsAntDownTilt` | `102°` |

### 6.2 UE 参数

| 参数 | 当前值 |
|---|---:|
| `ueHeight` | `1.5 m` |
| `ueTxPower` | `23 dBm` |

### 6.3 4T4R 天线配置

BS：

| 参数 | 当前值 |
|---|---|
| `nBsAntConst` | `4` |
| `bsAntSizeConst` | `{1,2,2,1,1}` |
| `bsAntSpacingConst` | `{0.5,0.5,1.0,1.0}` |
| `bsAntPolarAnglesConst` | `{45,-45}` |
| `bsAntPatternConst` | `1`，38.901 |

UE：

| 参数 | 当前值 |
|---|---|
| `nUeAntConst` | `4` |
| `ueAntSizeConst` | `{2,2,1,1,1}` |
| `ueAntSpacingConst` | `{0.5,0.5,1.0,1.0}` |
| `ueAntPolarAnglesConst` | `{0,90}` |
| `ueAntPatternConst` | `0`，isotropic |

### 6.4 每天线每 RBG 功率

当前仍按 `nPrbGrp = 68`、`nBsAnt = 4` 计算：

- `bsTxPower_perAntPrg = 49 - 10log10(4 x 68) ≈ 24.65 dBm`
- `ueTxPower_perAntPrg = 23 - 10log10(4 x 68) ≈ -1.35 dBm`

---

## 7. 时频资源与分配方式

### 7.1 Numerology

| 参数 | 当前值 |
|---|---:|
| `slotDurationConst` | `0.5 ms` |
| `scsConst` | `30 kHz` |

### 7.2 PRB / RBG 粒度

| 参数 | 当前值 |
|---|---:|
| `nPrbsPerGrpConst` | `4` |
| `nPrbGrpsConst` | `68` |
| 总 PRB 数 | `272` |
| 单 PRB 带宽 | `360 kHz` |
| 单 group 带宽 | `1.44 MHz` |
| 总有效带宽 | `97.92 MHz` |

### 7.3 Type-0 风格频域分配

当前已切换为：

- `gpuAllocTypeConst = 0`
- `cpuAllocTypeConst = 0`

语义上是：

- 频域资源以 `RBG` 为单位做离散分配
- 每个 `RBG` 可映射给不同 UE
- 每个 cell 的频域占用形态可表现为一个 bitmap

注意：

- 当前最小频域粒度仍是 `4 PRB`
- 也就是“1 bit 对应 1 个 4-PRB 资源块”的研究风格
- 这正是你后续做簇内频域协同所需要的最小离散单元

### 7.4 为什么要从 Type-1 改到 Type-0

Type-1 的问题是：

- 更偏向单连续块
- 频域占用形状自由度有限
- 很难研究跨小区 bitmap 级的频域避让

Type-0 的好处是：

- 可显式形成离散 bitmap
- 可构造跨小区错位复用
- 可研究 protected RBG / muted RBG / edge RBG 打孔
- 更适合图调度 / RL 输出“频域 mask”

---

## 8. 信道与传播模型

### 8.1 大尺度传播

当前仍为：

- `UMa`
- `sfStd = 3 dB`
- `noiseFloor = -98.9 dBm`
- `minD2Bs = 30 m`

路径损耗公式：

```text
PL = 32.4 + 20*log10(fc_GHz) + 30*log10(d_3D)
```

其中：

- `fc_GHz = 2.5`
- `d_3D` 为 BS-UE 三维距离

### 8.2 小尺度衰落

当前 Stage-B 默认场景：

- `fading-mode = 3`
- `GPU CDL`
- `CFR on PRG only`

典型当前 profile：

- `CDL-C`
- `delay spread = 300 ns`
- `maxDopplerShift = 10 Hz`
- `CFO = 200 Hz`
- `numRay = 20`

### 8.3 簇外 interferer 的信道角色

外环 12 个 interferer 站点：

- 进入所有协调 UE 的 channel tensor
- 参与同频干扰叠加
- 不产生本簇业务 KPI

因此当前信道维度本质上是：

- 发射端：`19 cell`
- 被服务 UE：`56 active UE`
- 其中仅 `7 cell` 对应本簇服务小区

---

## 9. 调度与簇管理配置

### 9.1 当前调度器

当前标准化 Stage-B 默认：

- GPU：原生 `multiCellUeSelection` + `multiCellScheduler`
- CPU：原生 PF 参考路径
- `baseline = 0`
- `custom-ue-prg = 0`

### 9.2 为什么默认禁用 CustomUePrgScheduler

因为当前 `CustomUePrgScheduler` 仍然只支持：

- `allocType = 1`

而当前标准化 Stage-B 已切换到：

- `allocType = 0`

所以脚本默认已改为：

- `CUSTOM_UE_PRG=0`

并在命令行层面阻止 `Type-0 + custom=1` 这种不兼容组合。

### 9.3 簇管理的当前体现

当前簇管理体现在：

- 7 个协调站点进入统一调度簇
- 12 个外环站点进入统一同频干扰层
- 所有 19 个站点共用同一频谱
- 簇内 7 个协调站点的频域占用可按 bitmap 形成互相耦合

当前还不包含：

- 动态 cell association
- 异频层协同
- 小区开关机
- TDD pattern 协同
- 功率 mask per-RBG 控制

但对“同频簇级频域协同调度”来说，当前已经比旧版更接近可研究状态。

---

## 10. 业务流配置

当前每个协调 UE 对应一个 flow：

- flow 数：`56`
- 到达类型：`Poisson`
- 包到达率：`1 packet / TTI`
- 平均包长：`5000 bytes`
- 包长方差：`0`

这里需要明确：

- Stage-B 脚本的 `--traffic-rate` 名称沿用了原示例二进制的 CLI 文案
- 但在当前实现里，`main.cpp` 实际使用 `TrafficType basic_traffic(data_rate, 0, 1);`
- 因此当前 `-r 5000` 的实际效果是：
  - `packet_size = 5000 bytes`
  - `packet_stddev = 0`
  - `arrival_rate = 1 packet / TTI`
- 它不是“物理层已送达吞吐量目标值”，而是业务流生成侧的包大小参数

因此：

- 所有 56 个协调 UE 都有业务流
- 外环 interferer 没有本簇业务流

---

## 11. 当前配置相对旧版的核心改动清单

| 项目 | 旧版 | 当前标准化后 |
|---|---|---|
| 协调簇几何 | 7 个扇区 cell，非标准拓扑 | 标准 `1+6` 六边形 `7-site` |
| 簇外干扰层 | 无 | 增加外环 `12-site interferers` |
| 总站点数 | `7` | `19` |
| 每 site 扇区数 | 混合成伪 3-sector | 当前简化为 `1 sector/site` |
| 频域分配 | `Type-1` 连续块 | `Type-0` bitmap 风格 |
| 最小频域粒度 | `4 PRB` | 仍为 `4 PRB` |
| Stage-B 默认调度器 | custom=1 | native PF，`custom=0` |
| 参数文件回滚 | 运行后恢复 | 默认不恢复，`restore-params=0` |

---

## 12. 当前仍建议后续继续修正的点

1. 把 `carrierFreq` 从 `2.5 GHz` 改到 `3.5 GHz`
2. 把总带宽从 `272 PRB / 97.92 MHz` 进一步对齐到目标 `273 PRB / 100 MHz`
3. 增加 UE 分层布站，而不是纯随机扇区内采样
4. 加入最强小区关联或可控簇级关联
5. 在 Type-0 基础上继续扩展：
   - 时域门控
   - 功率域 per-RBG mask
   - 空域 beam / layer / precoder 协同

---

## 13. 本次核对结论

截至当前脚本实现，本文档已与以下事实对齐：

- Stage-B 入口脚本会先注入编译期参数，再按 `phase4/cmake/skip` 流程执行
- 当前标准 Stage-B 为 `19 total cells / 7 coordinated cells / Type-0 / 4T4R`
- 当前默认业务流模型是 `Poisson + 1 packet/TTI + 5000 bytes/packet`
- 当前默认不回滚 `parameters.h`
- 当前无 `rg` 依赖，脚本在无 `rg` 的容器内也可正常收尾
