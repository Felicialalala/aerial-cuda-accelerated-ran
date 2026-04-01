# Stage-B 强化学习 + GNN 智能调度设计（历史归档）

本文档保留为历史设计归档，不再代表当前仓库实现状态。

原因：

- 文中将 replay 导出、online bridge、offline PPO、online PPO 等能力描述为“待实现”
- 这些能力目前已经在仓库主线中落地
- 继续把它当作当前状态文档会误导训练 readiness 判断

请改看以下当前文档：

- [`Doc/current_stageB_effective_configuration.md`](/home/oai2/aerial-cuda-accelerated-ran/Doc/current_stageB_effective_configuration.md)
- [`Doc/stageB_gnn_rl_online_training_design.md`](/home/oai2/aerial-cuda-accelerated-ran/Doc/stageB_gnn_rl_online_training_design.md)
- [`Doc/stageB_gnn_rl_training_implementation_tasklist.md`](/home/oai2/aerial-cuda-accelerated-ran/Doc/stageB_gnn_rl_training_implementation_tasklist.md)

如果需要继续做更高阶算法扩展，可以把旧文档里的研究方向当作“长期算法路线备忘”，但不要把其中的工程状态判断当作当前事实。
