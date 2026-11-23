# 后续数据用途审查、授权清理与正式训练启动

用户要求先分析后续实验用途再执行已批准删除，并将备份移出正式训练启动前置条件。未扩容、未重启、未恢复子代理或半小时自动监控。

## 用途与保留决策

`three-experiment-design-2026-09-27.md` 第3节、后续修订002及本轮探索授权表明：当前三臂使用ALOHA原D与新同域S/R。REBOOT WidowX/30Hz与ALOHA/50Hz合同分立；Sirius、MimicGen保留未来独立任务和标签审计用途，因此没有因当前未使用而一并删除它们。

实际Parquet footer检查确认USB-A `data` 的大体积主要来自四路 `*_depth.bytes`。后续RGB策略、state/action时序及恢复标签研究仍需要非深度列和RGB视频，不能只删整个data目录。此次先生成独立、明确标记的非深度列保留件，再清理经用户授权的部分原始深度载荷。

- 60个来源Parquet全部按原下载manifest SHA核验。
- 全部53,878帧非深度列保留，包括action、observation.state、timestamp、frame_index、episode_index、index和task_index；从原文件重新选择列，与独立重新读取的保留Parquet逐值/顺序精确比较。
- 60个保留文件共2,909,573字节，已回传本地并逐文件大小/SHA核验60/60。
- 原始完整深度样本 `data/chunk-000/file-000.parquet` 和 `file-001.parquet` 保留。这是存储示例，不是经过科学分层选择的深度训练子集。
- 全部RGB视频、原meta及原下载manifest保留；RJ45、Sirius、MimicGen、ALOHA、Q1/Q2/bank、所有checkpoint和失败证据均未删除。
- 今后完整深度实验仍须恢复58个原文件。保留件是新派生数据身份，不冒充原完整snapshot或已验证可直接训练的LeRobot view。

## 实际删除

UTC 2026-09-27 15:44:43，删除已重新逐文件SHA核验、没有符号链接/多重硬链接/可见打开句柄的58个原始文件：

`datasets_external/targeted-20260927-001/REBOOT26/USB-A_recovery_install/cfa3498a3982eb24554e88e77140247871eee3eb/data/chunk-000/file-002.parquet` 至 `file-059.parquet`。

原文件字节共33,302,251,300。数据盘可用空间从2,238,337,024增至35,540,840,448字节（约33.10GiB）。旧manifest不改写；独立删除收据明确原snapshot现在不完整。

远端保留目录：`datasets_external/usb-a-rgb-state-action-preserved-20260927-001`。
本地核验副本：`.work/usb-preserved-20260927-001/verified`。
保留manifest SHA：`5e0c801ef7d2e67c046d7770d6cfe7d564d7e5c594e9f2f9cd467e3fc906ae14`。
远端授权删除收据：上述保留目录下的 `authorized-deletion-receipt.json`，逐文件操作记录为 `authorized-deletion-progress.jsonl`。

## 正式训练启动

B/C独立冷重载已在 `paired-native-BC-smoke-reload-20260927-002` 全部rc0退出、worker/guard所属组清空。先前001仅因阶段名含大写被配置检查拒绝，未执行重载；保留失败身份。

新 `paired-native-ABC-formal-20260927-001` 已实际启动并完成guard握手；A/B/C分别绑定真实benchmark、两步优化及独立重载证据。每臂fresh base、batch4、5000成功更新，保存1250/2500/3750/5000四点。单GPU按A→B→C自动接续，SSH断开不影响已守护进程。总工作上限23000秒、每训练阶段7200秒、RSS8GiB、输出23GiB、最低磁盘余量2GiB。

启动边界实际观察：A已到140步；第100步loss0.569、gradient norm5.621、LR1e-4均有限，step耗时约0.205秒；GPU显存3219MiB、利用率53%。这只是早期运行稳定证据，不等同5000步完成或闭环任务成功。

当前队列只包含三臂训练，既有守护仍为work-only，`shutdown_authorized=false`。评价和整轮自动关机不能据此声称已部署完成。
