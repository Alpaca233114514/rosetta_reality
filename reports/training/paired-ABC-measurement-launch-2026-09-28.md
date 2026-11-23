# 三臂固定终点测量挂载记录（2026-09-28）

新运行身份：`paired-ABC-measurement-20260928-003`，远端目录为 `runs/paired-ABC-measurement-20260928-003/`。源码 workspace 身份为 `20260928T053233Z-66a5bbcdbdcd-9297dde2ac3b`，composite SHA256 为 `9297dde2ac3b199c1724050a026a3c82c0eac27fd2b37e5d9cff94211ccd1c33`；注册计划是 `configs/diagnostics/paired-ABC-measurement-job-20260928-003.json`，SHA256 为 `5df393d0f40fee434fd5f640e68ba1ed720b2d37a015255253ee01a3fa2ca6a4`。

用户已表示在平台设置三小时定时关机；本轮没有独立读取平台控制面来确认该设置。队列只管理工作，`shutdown_authorized=false`，总工作截止为启动后 7800 秒。它依次执行：标准 AutoDL doctor、固定三终点输入准备、A/B/C 各自两进程完整动作数组重载及五场景和共同恢复 bank 测量。独立科学负结果或单臂失败按注册退出码留证后允许后续独立臂继续；身份、资源与物理安全错误仍受限。

实际启动收据报告 `launched=true`、`guard_handshake_verified=true`。只读复查显示 `init`、`doctor`、`prepare` 退出码均为 0，`reload-a` 退出码为 0 并写出 `reload-proof-A.json`。截至 `2026-09-28T05:35:25Z`，worker 尚在运行，活动阶段为 `measure-a`。这证明队列与 A 臂测量已实际接续，**不是** A/B/C 最终结果或整轮完成证明。

前一个 `paired-ABC-measurement-20260928-002` 身份因 doctor 所需新 job 目录变量未登记而在模型测量前退出；失败日志和守护清理收据保留在其独立远端目录。没有重用或覆盖该身份。

后续应从远端 `runs/paired-ABC-measurement-20260928-003/worker-exited.json`、各 `reload-proof-{A,B,C}.json`、`measurement-{A,B,C}/summary.json`、`measurement-{A,B,C}/recovery-bank/bank-summary.json` 及守护退出收据判断完成度，先回传并逐文件校验再解释效果。此独立测量不等同于正式 Gate 4 或 M2 验收。
