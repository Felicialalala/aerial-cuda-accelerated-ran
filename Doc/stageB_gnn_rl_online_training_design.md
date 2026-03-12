# Stage-B GNNRL 在线学习实现设计（PPO + Aerial 实时交互）

## 1. 目标

- 目标：实现 PPO 与 Aerial 仿真环境逐步交互训练，直接优化在线闭环 KPI。
- 重点：在不破坏现有 Stage-B 推理链路的前提下，新增可控、可回退的在线训练模式。

## 2. 非目标（首版）

- 不在首版引入复杂分布式训练框架（如 Ray/RLlib）。
- 不在首版追求跨硬件大规模并行。
- 不在首版引入复杂模型结构升级，优先复用现有 `StageBGnnPolicy`。

## 3. 总体架构

- 组件 1：`AerialSimWorker`（C++，单实例环境进程）。
- 组件 2：`AerialEnvClient`（Python，Gym-like 封装）。
- 组件 3：`ppo_online_train.py`（Python，采样 + 更新）。
- 组件 4：`OnlineEvalRunner`（训练中定期导出并回归评估）。

数据流：

- trainer 发送 action（含 mask 合法性约束后的决策）。
- sim 执行 1 TTI，返回 `obs, reward, done, info`。
- trainer 累积 `n_steps x n_env` rollout，执行 PPO update。
- 周期性导出 checkpoint/onnx，并触发 Stage-B 回归脚本。

## 4. 交互协议设计

建议采用 Unix Domain Socket + 长度前缀帧（单机低延迟）。

消息类型：

- `ResetReq`：seed、scenario 配置、episode 长度上限。
- `ResetRsp`：首帧 `obs` + 初始 mask + meta。
- `StepReq`：`ue_select`、`prg_alloc`、policy_id、step_idx。
- `StepRsp`：`next_obs`、`reward`、`done`、`mask`、`info`。
- `CloseReq/Rsp`：进程回收。

协议字段要求：

- obs 与离线 replay schema 保持一致（字段名、shape、dtype）。
- info 至少包含：`tti`、`sum_throughput`、`residual_buffer`、`fairness_proxy`。
- done 语义固定：episode 结束或仿真异常终止。

## 5. 训练循环（首版）

- 参数建议：
- `n_env=4`（先单机 4 并行）。
- `n_steps=256`。
- `update_epochs=4~6`。
- `minibatch_size=256`。
- `gamma=0.99`，`gae_lambda=0.95`。

流程：

- 从 BC best 加载 actor 初始参数。
- 在线采样 rollout，按当前实现继续使用 state/reward/adv 归一化。
- actor/critic 分离优化器，维持 huber value loss。
- 每 `K` 次更新做一次离线回归（固定 seed=42）并记录 KPI。

## 6. 工程落地拆分

### 阶段 A：环境接口打通（1 周）

- 新增 `training/gnnrl/aerial_env.py`。
- 新增 `cuMAC/examples/onlineTrainBridge/`（或在现有 main 扩展在线模式）。
- 支持 `reset/step/close` 最小闭环。

验收：

- Python 可调用 `env.reset()` 和连续 `env.step()` 到 `done=True`。
- 每 step 返回字段完整且与 replay schema 对齐。

### 阶段 B：在线 PPO 首版（1 周）

- 新增 `training/gnnrl/ppo_online_train.py`。
- 接入多 env rollout buffer。
- 输出 `online_ppo_summary.json`、`ppo_actor_best.pt`。

验收：

- 训练可稳定跑满 10k+ steps 不崩溃。
- KL/clipfrac/value_loss 在可控范围，无持续发散。

### 阶段 C：评估与门禁（3-5 天）

- 接入 `run_stageB_main_experiment.sh` 自动回归。
- 训练中周期触发 baseline/online-ppo 对比。

验收：

- 达到离线文档 M4 的最小门禁阈值，且多 seed 不出现大面积 `scheduled_ratio=0`。

## 7. 风险与缓解

- 风险：仿真步进吞吐低，训练时间过长。
- 缓解：多 env 并行、仅保留必要日志、异步评估。

- 风险：在线训练输出非法动作导致仿真异常。
- 缓解：动作 mask + C++ 二次校验 + fallback 到 legacy。

- 风险：在线接口与离线 schema 漂移。
- 缓解：共享 schema 定义文件，启动时做 shape/dtype handshake 校验。

## 8. 首批实施清单

- [ ] 定义并固化 `reset/step` 协议（字段与 dtype）。
- [ ] 实现 `aerial_env.py` 最小可运行客户端。
- [ ] 实现 C++ 侧在线桥接服务。
- [ ] 实现 `ppo_online_train.py`（单机多 env）。
- [ ] 打通导出与回归门禁。
