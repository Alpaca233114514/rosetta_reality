# SmolVLA 权威核验模块收线 — 2026-09-10

**本模块完成，已执行用户要求的关机；没有启动新训练，视觉泛化未修复，M2 未完成。**

本轮找回原 8 场景 step-256 checkpoint，核对权重与最终审计哈希，归档九份最终
源码及两份原始报告，补齐 40 场景审阅计划中的实际 smoke 入口、seed、optimizer、
scheduler 和资源边界。源码是选择性归档，尚未安装到当前本地代码树。
详情见 `reports/training/m2-smolvla-native-visual-authority-2026-09-10.md`。
主交付已推送当前功能分支，提交 `63221de6b318fa1797eb9901d332f18cdb11dd70`；
没有修改 main 或覆盖远端 dirty 工作树。

关机时间：**2026-09-10 11:21:35（Asia/Shanghai）**。关机前未发现项目计算
进程或 tmux 会话，最终审计文件仍一致、checkpoint 存在，已同步 durable 存储。
通过 Bash 调用已核验的平台关机脚本后，远端主动关闭 SSH；11:21:56 前完成的
一次连接复查也被关闭。平台控制台电源与计费状态没有独立核验。
实例未释放，未删除已有文件；关机脚本的回收站清理目标经两次检查均不存在。

第一次直接执行该脚本返回 `OSError: [Errno 8] Exec format error`，原因是脚本
没有 shebang。该失败和初次收线记录保留；随后用 Bash 解释同一哈希的脚本，
没有修改脚本或保护配置。远端收线记录位于 durable root 下
`control/visual-authority-closeout-20260910-001.json` 与 `-002.json`。

文档/JSON/hash/相对路径/split/预算静态检查及 staged diff 检查通过。
旧 47 项回归未重跑，新增 optimizer steps 为 0。全非自身错配评估、40 场景
训练、完整 chunk reload 和任务成功均为 `not measured`。

下一步先在新的服务器版本目录核对恢复源码，完成全非自身错配 evaluator、
完整预测保存和分阶段执行登记。随后需新计算授权、可见 GPU、缓存/磁盘核验、
真实 preflight 与独立两步 smoke，才运行一次 fresh-base 40 场景候选。
旧 A 只评估，不重训、不续 optimizer；共同协议检查 3/4 噪声、关节、夹爪、
训练拟合和完整 reload。hidden 封存，失败保留结果并停止，不自动开下一炉。
