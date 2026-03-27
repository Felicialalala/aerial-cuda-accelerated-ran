# Stage-B GNNRL 后续计划（与当前 RR/PF 基线对齐）

## 1. 当前状态快照（2026-03-27）

- Stage-B baseline 已固定到 `7-cell + 4T4R + Type-0 bitmap`
- native PF 和极限 RR 都已支持 GPU 下的 Type-0 运行
- 已有一键 RR/PF compare：
  - [`cuMAC/scripts/run_stageB_rr_pf_compare.sh`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/run_stageB_rr_pf_compare.sh)
- 当前 GNNRL 已具备：
  - replay v2
  - offline BC / offline PPO
  - ONNX 导出
  - C++ `gnnrl_model` 推理
  - online bridge + online PPO

但还没有完全和“当前 RR/PF baseline 语义”对齐。

## 2. 第一优先级：先补齐 baseline 一致性

### 2.1 统一 launcher，避免训练/评测配置漂移

问题：

- RR/PF baseline 统一通过 `run_stageB_main_experiment.sh`
- online PPO 目前直接启动 binary
- 这会导致 `packet_size_bytes`、`traffic_arrival_rate`、`topology_seed`、`exec_mode` 等参数容易漂移

建议修改：

- 新增一个专门的 Stage-B ML launcher，或者让 `ppo_online_train.py` 可直接复用 Stage-B 脚本参数

建议改动文件：

- [`training/gnnrl/ppo_online_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/ppo_online_train.py)
- [`training/gnnrl/aerial_env.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/aerial_env.py)
- [`cuMAC/scripts/run_stageB_main_experiment.sh`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/run_stageB_main_experiment.sh)
- 新增建议：`cuMAC/scripts/run_stageB_online_train.sh`

验收标准：

- 同一份 YAML/CLI 参数可同时驱动 baseline、replay、online 训练、导出评估
- 训练日志里能明确打印当前场景冻结参数

### 2.2 增加“固定 all-UE slot，只学习 PRG bitmap”的严格同口径模式

问题：

- 当前 native Type-0 baseline 先固定所有 active UE 进入 slot
- 当前 GNNRL 动作空间仍包含 UE selection
- 这会让智能算法与 RR/PF 的对比不完全同口径

建议修改：

- 增加 `policy_action_mode=prg_only_type0`
- 在该模式下：
  - slot 列表直接复用 baseline 的 all-UE 填法
  - 模型只输出 PRG bitmap / slot assignment
- 将“联合 UE + PRG”保留为后续扩展模式

建议改动文件：

- [`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/multiCellSchedulerUeSelection/main.cpp)
- [`cuMAC/examples/rlReplay/ReplayWriter.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/rlReplay/ReplayWriter.cpp)
- [`training/gnnrl/dataset.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/dataset.py)
- [`training/gnnrl/model.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/model.py)
- [`training/gnnrl/masks.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/masks.py)
- [`training/gnnrl/ppo_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/ppo_train.py)
- [`training/gnnrl/ppo_online_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/ppo_online_train.py)

验收标准：

- 可以生成与 baseline 完全同口径的 `policy vs RR/PF` 结果
- 同时保留联合动作空间实验分支

### 2.3 把 RR/PF compare 扩展成 RR/PF/GNNRL compare

问题：

- 现在只有 RR vs PF
- 还缺一个“导出模型 vs RR vs PF”的统一表格入口

建议修改：

- 把 `rr_vs_pf_compare.py` 泛化成多策略 compare
- 新增 `run_stageB_policy_compare.sh`

建议改动文件：

- [`cuMAC/scripts/rr_vs_pf_compare.py`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/rr_vs_pf_compare.py)
- [`cuMAC/scripts/run_stageB_rr_pf_compare.sh`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/run_stageB_rr_pf_compare.sh)
- 新增建议：`cuMAC/scripts/run_stageB_policy_compare.sh`

验收标准：

- 一次命令输出 `RR / PF / GNNRL` 的统一 CSV、JSON、TXT
- 能同时给出 per-cell 和 per-UE delta

## 3. 第二优先级：把训练目标改成与当前 KPI 更一致

### 3.1 将 packet-level delay 纳入 replay / online 奖励与 info

问题：

- 当前奖励仍以 `total_buffer_mb` 作为时延代理
- 但当前 traffic model 已经能导出 packet-level delay

建议修改：

- 在 replay 和 online bridge 中增加 packet delay 相关字段
- 奖励从“纯 backlog proxy”升级为“throughput + residual + packet delay + fairness”

建议改动文件：

- [`cuMAC/examples/trafficModel/trafficGenerator.hpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/trafficModel/trafficGenerator.hpp)
- [`cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp)
- [`cuMAC/examples/rlReplay/ReplayWriter.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/rlReplay/ReplayWriter.cpp)
- [`training/gnnrl/online_protocol.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/online_protocol.py)
- [`training/gnnrl/aerial_env.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/aerial_env.py)
- [`training/gnnrl/dataset.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/dataset.py)
- [`training/gnnrl/ppo_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/ppo_train.py)
- [`training/gnnrl/ppo_online_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/ppo_online_train.py)

验收标准：

- replay / online summary 可直接看到 packet delay
- checkpoint 选择不再只依赖 throughput + buffer

### 3.2 在 metadata 中显式记录 baseline 语义与运行模式

问题：

- 当前 replay / 训练产物里缺少一些“是否与当前 baseline 等价”的关键信息

建议补充 metadata：

- `baseline_scheduler`
- `exec_mode`
- `packet_size_bytes`
- `traffic_arrival_rate`
- `topology_seed`
- `slot_action_mode`

建议改动文件：

- [`cuMAC/examples/rlReplay/ReplayWriter.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/rlReplay/ReplayWriter.cpp)
- [`training/gnnrl/dataset.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/dataset.py)
- [`training/gnnrl/ppo_online_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/ppo_online_train.py)

## 4. 第三优先级：在线学习框架补齐到当前 baseline 工程能力

### 4.1 支持“训练 both / 评测 gpu-only”的标准流程

问题：

- 现在 online bridge 不能在 `gpu-only` 下工作
- 但当前 baseline compare 主要用 `exec-mode=gpu`

建议策略：

- 训练阶段继续 `exec-mode=both`
- 导出评测阶段强制支持 `exec-mode=gpu`
- 明确区分“训练模式”和“最终评测模式”

建议改动文件：

- [`training/gnnrl/ppo_online_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/ppo_online_train.py)
- [`cuMAC/scripts/run_stageB_main_experiment.sh`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/run_stageB_main_experiment.sh)
- [`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/examples/multiCellSchedulerUeSelection/main.cpp)

### 4.2 增加自动回归与 checkpoint 选择闭环

问题：

- 当前 online PPO 结束后仍需手动导出 ONNX、手动回归、手动比较

建议修改：

- 训练过程中按固定间隔：
  - 导出 best candidate
  - 跑 baseline / gnnrl_model 回归
  - 生成统一 compare 表
- checkpoint 选择规则从“objective 最大”改成“门禁加权最优”

建议改动文件：

- [`training/gnnrl/ppo_online_train.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/ppo_online_train.py)
- [`training/gnnrl/export_onnx.py`](/home/oai2/aerial-cuda-accelerated-ran/training/gnnrl/export_onnx.py)
- [`cuMAC/scripts/run_stageB_main_experiment.sh`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/run_stageB_main_experiment.sh)
- [`cuMAC/scripts/rr_vs_pf_compare.py`](/home/oai2/aerial-cuda-accelerated-ran/cuMAC/scripts/rr_vs_pf_compare.py)

验收标准：

- 训练输出目录直接包含“最佳回归结果”
- 不再需要手工拼接多份日志比较

## 5. 第四优先级：进入下一轮 GNN+RL 算法扩展

在前面三步都稳定后，再进入更高阶动作空间和更复杂特征。

建议顺序：

1. `PRG-only Type-0` 同口径版本先做稳
2. 在此基础上加入 `UE + PRG` 联合动作
3. 再加入更强邻区特征、跨小区 bitmap overlap、packet-delay-aware reward
4. 最后再考虑更复杂信道和更广场景泛化

## 6. 建议门禁

后续每轮模型至少统一检查：

- `cluster_sum_throughput_mbps`
- `ue_throughput_jain`
- `ue_throughput_p5_mbps`
- `residual_buffer_ratio`
- `packet_delay_p95_ms`
- `scheduled_ratio_p5`

并固定比较对象：

- RR
- PF
- GNNRL current best

## 7. 推荐执行顺序

### P0

- 统一 launcher
- 固定 all-UE slot / PRG-only mode
- RR/PF/GNNRL compare automation

### P1

- packet delay 纳入 replay / online reward
- 自动回归与 checkpoint 选择

### P2

- 联合动作空间
- 更复杂 GNN 特征与多场景泛化
