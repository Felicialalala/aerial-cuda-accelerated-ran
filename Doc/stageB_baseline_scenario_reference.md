# Stage-B 基线场景参考（离线 RL，2026-03-10）

## 1. 适用范围

- 本文档描述当前离线 RL 回归使用的 Stage-B 基线场景（`run_stageB_main_experiment.sh`）。
- 用于回答“当前 baseline 的基站拓扑、UE 布局、无线参数到底是什么”。

## 2. 当前基线运行画像

- 建议命令见：`Doc/stageB_gnn_rl_offline_training_implementation.md`
- 当前常用 baseline 回归参数：
- `--fading-mode 0`（Rayleigh）
- `--tti 600`
- `--custom-ue-prg 0`（native PF）
- `--topology-seed 42`

## 3. 基站拓扑与小区布局

### 3.1 编译期注入（由 Stage-B 脚本覆盖）

脚本会在运行前注入：

- `cellRadiusConst=500`
- `numCellConst=7`
- `numCoorCellConst=7`
- `numUePerCellConst=8`
- `numActiveUePerCellConst=8`
- `nBsAntConst=4`
- `nUeAntConst=4`
- `nPrbsPerGrpConst=4`
- `nPrbGrpsConst=68`
- `gpuAllocTypeConst=0`
- `cpuAllocTypeConst=0`

对应代码位置：

- `cuMAC/scripts/run_stageB_main_experiment.sh`（参数注入段）
- `cuMAC/examples/parameters.h`（生效后的编译期宏）

### 3.2 站点几何

- `network::genNetTopology()` 使用硬编码 2-ring hex layout 生成站点坐标。
- 因为 `numCellConst=7`，实际只取前 7 个站点，形成 `1+6` 六边形簇。
- `siteSpacing = 2 * cellRadius = 1000 m`。
- 非中心小区朝向设置为“指向簇中心”。

对应代码：

- `cuMAC/examples/network.cu`（`genNetTopology()`）

## 4. UE 布局与关联

### 4.1 UE 数量与布局模式

- 默认每小区 `8` 个活动 UE（总计 `56`）。
- UE 布局模式默认 `uniform`，支持 `stratified`（由环境变量控制）：
- `CUMAC_UE_PLACEMENT_MODE`
- `CUMAC_UE_RADIUS_SPLITS`
- `CUMAC_UE_STRATA_COUNTS`
- `CUMAC_UE_VORONOI_CLIP`

对应代码：

- `cuMAC/examples/network.cu`（`getUePlacementConfig()`）
- `cuMAC/scripts/run_stageB_main_experiment.sh`（导出上述环境变量）

### 4.2 采样形态

- 极角范围：相对小区朝向 `±60°`
- 半径范围：`minD2Bs ~ cellRadius`（默认约 `30m~500m`）
- TTI 0 后关闭小区重绑定开关（固定关联模式）

对应代码：

- `cuMAC/examples/network.cu`（UE 极角/半径采样）
- `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`（`cellIdRenew/cellAssocRenew`）

## 5. 关键无线参数（当前默认）

- 载频：`2.5 GHz`
- SCS：`30 kHz`
- PRG：`68` 组，每组 `4 PRB`
- 天线：`4T4R`
- BS 功率：`49 dBm`（`PtConst=79.4328 W`）
- 仿真时隙：`slotDurationConst=0.5 ms`

对应代码：

- `cuMAC/examples/parameters.h`

## 6. 快速自检（跑完后）

- `run.log` 中应看到：
- `Topology configuration: seed=... ue_placement=...`
- `Cluster configuration: coordinated cells=7, interferer cells=0, total cells=7`
- `Using CPU multi-cell PF UE selection`（baseline）

## 7. 关联文档

- 离线训练主文档：`Doc/stageB_gnn_rl_offline_training_implementation.md`
- 在线训练设计：`Doc/stageB_gnn_rl_online_training_design.md`
- 历史配置梳理（含 19-site 版本）：`Doc/current_stageB_effective_configuration.md`
