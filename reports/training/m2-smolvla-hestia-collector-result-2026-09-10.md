# Hestia 采集与封存模块：CPU 验收完成

本次完成新 B/C collector、实际保存配置校验、候选完整性封存及计划生成入口的 CPU 验收。**视觉泛化修复仍未证明，Hestia 主训练和真实 B/C 比较尚未运行，M2 未完成。**

执行源码为 `e289f45bcb75fea7c5830d708808103846d484f2`，独立 checkout 为 `dev/hestia-collector-cpu-20260910-e289f45`；注册计划是 `m2-smolvla-hestia-collector-plan-2026-09-10.md`，执行证据在 `reports/training/hestia-collector-20260910/001-*`。旧 A/B collector 与历史负结果保持不变。

## 实测结果

| 检查 | 结果与边界 |
|---|---|
| Ruff | 所有本轮源码及相关测试文件通过 |
| CPU 回归 | 184 passed，0 failed/error/skipped，15.36 秒；包含旧 A/B 数值评分、B/C 合同、观察器、v2 plan/launcher |
| 原生 sampler | 5120 个索引与独立生成顺序逐项一致；40 个场景各 128 次，仅合成索引，无真实样本 |
| 新计划 | YAML 经真实 `_resolve_plan` 回读一致；1280 updates、1280 decay、warmup 16、batch 4；合同仍为 draft 且在模型 import 前被拒绝 |
| B 已保存配置 | 实际完整 pretrained 文件清单及 SHA 与历史一致；256 步、resume=false、梯度累计 1、冻结范围、optimizer/scheduler 均通过 |
| B 恢复状态 | 完整恢复文件存在，training step=256、scheduler last_epoch=256、step_count=257、末 LR=2.5e-6；不读取或续跑旧 optimizer tensor |
| 候选配置回读 | 从真实两步 smoke 保存配置读取，以原生 TrainPipelineConfig 解码/编码合成候选；只证明配置可解析，不是 C 运行证据 |
| 资源 | 无 GPU 实例；总 worker 57.72 秒，exit=0，预算 600 秒；模型权重加载 0、真实样本加载 0、新 optimizer step 0 |

配置解码在无卡环境打印原生提示 `Device 'cuda' is not available. Switching to 'cpu'.`，因此本轮没有验证 CUDA 执行配置的完整运行行为；之前 GPU 两步预检仍以其独立报告为准，不扩张为本轮 collector 的真实模型验证。

原始 pytest XML 含 worker hostname，保留在私有运行证据，不发布；仓库保存逐测试模块计数、原始 XML SHA 和原始无敏感日志。其他已发布小证据逐文件 SHA 保留在 `001-raw-evidence-sha256.json`；未改写原始证据。

## 磁盘与下一步

实测可用 10,977,865,728 bytes。完整 B 恢复 checkpoint 为 1,614,046,930 bytes；按四个新 quarter checkpoint 加一个写入临时副本、另留 2 GiB，估算新增需求 10,217,718,298 bytes，余量 760,147,430 bytes。既有两步 smoke 已占盘，不再重复作为新增副本计入。该估算通过本轮预检，开炉前仍须按候选实际文件大小和现场余量重验，不删历史 checkpoint。

下一模块在无 GPU 占用时完成主 supervisor 串联：启动身份/资源核验 → 新基座和新 optimizer 的 1280 步观察训练 → 四个 checkpoint 封存 → B/C 各两次独立原生采集 → 七数组 reload → 历史 B 原数组复现 → 固定最终候选评分。补齐这条执行链和静态检查后，再需要带卡实例运行主实验。

本轮 CPU 通过只表明工程验收前置项完成。候选封存工具尚无真实 C checkpoint 输入，七数组真实采集与独立 reload、训练拟合、开发比较及新 Gate 3/4 均为 **not measured**。仍保留正确图优于全非自身错配平均误差、低于视觉无关均值下限、关节/夹爪/aggregate 同时至少 3/4 条件通过、标准动作非回退等门槛；hidden split 保持封存。

关机由同一独立 supervisor 在 worker 退出后保留 120 秒回收窗口，再检查平台 shutdown helper、无活动任务及 Trash 为空后执行；不释放实例。窗口后于 13:03:34 UTC 核验，SSH 返回连接关闭。最终 shutdown-request 文件未取回，平台电源与计费状态未独立核实，不能仅凭 SSH 关闭宣称平台账单停止。
