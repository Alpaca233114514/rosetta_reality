# Coverage40 执行修订 003

002 已通过 79 项部署回归、Ruff、真实 batch-1/batch-4 无 optimizer forward 和
完整 sampler 检查，在 sample-contract 的错误形状断言停止。optimizer 为 0 步，
没有 smoke 或 B checkpoint。原始工作区与全部报告保留。

原生 forward 的合同是 state `[batch, n_obs_steps, 14]`，当前 n_obs_steps=1；
检查器误写 `[batch, 14]`。003 按真实合同修正检查器，并在异常中输出预期与
实际 shape；新增历史维缺失、历史长度错误、错误维数拒绝和正确形状通过的回归。
不 reshape 输入，不修改历史 trainer/processor，不放宽有效动作或状态维度。

新的 job 与 stage run name 使用 003。沿用 002 的开始时间与截止时间：
Unix 1789024448.3725536 至 1789026248.3725536，不追加 30 分钟。新守护确认
存活后，才依据旧 closure/exit/cwd/cmdline 核验并退役仅处于失败等待的旧
supervisor，避免相互干扰；不终止训练或其他实例任务。

本次学习假设及 8→40 主试验、所有噪声、评估与停止阈值均不变。002 失败属于
检查器工程缺陷，不是视觉泛化负结果。目标仍待实际 smoke、B 训练及评估验证。
