# Athena：历史重复权重清理

用户允许清理多余内容，要求 20 分钟内完成一个模块并自行关机。本模块仅做存储去重、容量复核、证据保存与关机，没有启动 Athena 训练。

实测释放 **5,988,974,592 bytes（约 5.99 GB）**，数据盘可用空间从 6,661,230,592 增至 **12,650,205,184 bytes**。已超过候选六份完整 checkpoint 加 2 GiB 余量的 11,831,765,228-byte 预算，剩余额外余量 818,439,956 bytes。下一次运行前必须重新核验，不能据此跳过真实预检。

## 核验与操作

没有把占用 25.37 GB 的历史 runtime 归档当作缓存删除。只对以下七个大文件做流式完整 SHA-256，发现两组字节完全一致的内容；每个文件 1,197,789,256 bytes，原本均为同文件系统上的独立 inode：

- Zen-uniform step 316：保留完整 checkpoint 中的权重；当前 deploy 和三个 `superseded` deploy 的四份权重与其完全相同，SHA 为 `da0e68cea003775ba64466ecf9017ace1a0b94cbe3ff55c000ebaee6472d71fb`。
- Zen-firstaction step 316：保留完整 checkpoint 中的权重；当前 deploy 的一份权重与其完全相同，SHA 为 `310a91fe9b36dc55e2312ec41e079fa52e95613658049edecb1d8003e249ccea`。

将五份重复导出权重原路径原子替换为同内容硬链接；两个保留 inode 设为 owner read-only（0600→0400，没有扩大访问权限）。操作前再次计算全部 SHA，检查原 inode、大小、link count、路径、同一文件系统及无活动 ML/tmux worker；先写 create-only 登记和逐项 fsync 日志，再创建临时硬链接并原子替换。操作后重算两份独立内容的 SHA，逐个确认原路径指向对应 inode。

逻辑重复字节为 5,988,946,280；实际可用空间增加包含文件系统块计量差异，以实测为准。没有移除唯一权重内容、原路径、配置、日志、失败记录或任何 optimizer/RNG/scheduler 状态。A/B checkpoint、基座、数据、源码 workspace 和既有独立 backup archive 未动。

同组路径现在共享 inode，**不能把这些链接当作独立备份**。历史权重只读；以后若要写入其中一个路径，须先通过新文件复制和原子替换解除共享，不能原地修改。读取、hash 校验和独立进程 reload 的路径仍保留；本次只验证字节和路径，没有加载模型。

具体 source/target 路径、原 inode、hash、每项操作和容量见同名 JSON，以及 `reports/training/athena-cpu-20260910/zen-duplicate-inspection.json`。数据盘位置采用相对路径，报告不含主机地址或凭证。

## 后续边界

本次完成存储模块；GPU、Athena 1280 步训练、B/C 评估和 Gate 3/4 均未运行。CPU 两批 79 / 21 项测试的独立证据见 CPU 预检报告。下一步先完成 B/C 评分合同及观察器接线，再做 batch 1/4 forward、两步 smoke/reload；前置条件全部通过后才执行新的固定预算诊断。M2 未完成，视觉忽略尚未修复。

关机结果追加记录于 `reports/training/m2-smolvla-athena-cpu-shutdown-2026-09-10.json`；历史关机不代表以后实例状态。
