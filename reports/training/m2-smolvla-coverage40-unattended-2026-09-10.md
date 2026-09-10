# SmolVLA 覆盖 8→40：无人值守执行登记

用户已授权按实际任务安排计算、部署服务器任务、配置匹配时长的看门狗及结束关机。
本登记延续既有审阅计划；不改评估阈值、学习合同、模型/数据身份或原始失败证据。

运行入口为 `scripts/run_visual_coverage_job.py`，真实输入及完整 smoke 预测检查为
`scripts/visual_coverage_job_checks.py`。输出新建在
`runs/visual-coverage40-unattended-001/`；同名目录已存在即拒绝重新准备。

执行链按顺序完成环境/回归/真实缓存检查、benchmark、batch 1/4 无 optimizer
预检、真实 40/5 frame-0 合同与完整采样顺序、fresh-base 两步 smoke/独立完整
chunk reload、A 历史结果重现和新 train8 拟合门禁、B fresh-base 256 步、冻结范围/
optimizer/LR/实际 1024 次采样审计、两臂统一评估与各自独立完整 reload、比较报告。
前置失败便停止，未运行阶段记 `not measured`。不重训 A，不续旧 optimizer，
不续 smoke，不自动修补后重开，不启动下一实验或闭环。

总计算上限 30 分钟，另留最多 2 分钟保存及终止本任务进程组并关机。历史 B4
256 步约 132 秒只用于时间估计；本次包含四组 45 场景完整评估/reload以及前置
验证。剩余时间不足时拒绝进入 smoke 或 B，不能挤掉 reload。守护进程独立于
SSH；按子进程退出事件推进阶段，无持续训练轮询。单进程内资源线程是停止条件
执行器：allocated 8 GiB、reserved 10 GiB、host RSS 10 GiB，超限终止并保存报告。

磁盘预检要求新 smoke、B、一次临时保存合计三个完整 checkpoint 加 2 GiB 余量。
旧 A checkpoint 和历史结果已完成服务器直传；源端与目标端全部文件大小、SHA
相等。源无卡实例随后执行官方关机并断开 SSH；未释放实例，计费状态未独立核验。
模型、数据、预测及完整 checkpoint 留在服务器；本机只接收脚本和小型清单。

`registration.json` 封存源码 commit/哈希、阶段 plan 哈希、控制 checkpoint
清单、磁盘预算及授权。`watchdog.json` 记录真实开始/截止时间和 supervisor PID；
`events.jsonl`、各阶段日志/资源、`closure.json`、`supervisor-exit.json` 和
`shutdown-request.json` 保留执行及收尾证据。关机使用校验过的官方 wrapper，
若发现其他任务或 wrapper 会删除已有 Trash，则记录 `shutdown-blocked.json`。

当前文档是启动登记，不能代替启动及运行结果。新增守护的进程组隔离、证据不覆盖、
未知关机 wrapper 拒绝三项测试已在 AutoDL 通过；完整任务仍待启动后实测。
视觉泛化未宣告修复，M2 未完成，hidden split 保持封存。

预部署检查补记：完整源码工作区的 Ruff 对本地包导入分类与独立 QA 目录不同，
发现 `I001`，在 prepare、守护和模型执行前退出。保留该工作区，仅补导入空行，
以新 commit 和新目录重新部署；没有 optimizer 步或实验结果被覆盖。
