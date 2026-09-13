# Hestia 场景K/V对照：部署与启动模块

实现提交 `364bd067ed19dc134cb53998cf8257c163e3c5bc`，本地161项相关测试、远端30项新模块合成测试、Ruff及格式检查通过。14个上传文件均与Git提交逐字节一致，96项科学源/输入身份在本地与远端均匹配。代码或测试通过不代表真实推理与泛化成功。

增量包为133,120 bytes，SHA `e677497a358ce65610185c663061da0e9bdde7aa8d8024e671f859c6f4d4f10e`。新工作区identity为 `20260913T095208Z-364bd067ed19-a311a6791e8c`，composite SHA `a311a6791e8c31900953b94220cbb453a4188082e766f7d28075a3f9a71b2683`。既有基础工作区保留，未覆盖旧代码、删除文件、传输权重或下载数据。

模板SHA `ce9ab25293c3dfd56c08622f35c5c7f65f3069cb19020cb58a284561ca1c3c41`，登记run `hestia-scene-kv-ablation-20260913-001`。接收器已先就绪；supervisor PID2349在受保护窗口启动，doctor、workspace-local normalization和CPU门禁退出码均0，随后进入640原生控制。此模块只确认成功部署/启动，不声称1800次推理已完成；科学结论以之后的result报告为准。

两次Codex Auto Review拒绝均保留并通过明确授权解决，没有绕过：第一次要求针对内部源码/测试/登记增量包的具体上传授权，用户回复“授权”后，原命令通过审核并完成上传；第二次要求区分GPU计费运行与上传授权，用户明确选择“授权本次推理、回传与受保护关机”后，原启动命令通过。范围为最多1800次、0optimizer、40分钟worker/50分钟保护窗口、校验回传及受保护关机，不授权新训练或释放实例。

此前WSL内置SSH客户端返回No route to host，Bash调用本机既有SSH客户端随后连接成功；只读核验为原RTX4090D、无CUDA计算进程、持久盘约5.4GiB可用、基础workspace身份匹配。该网络错误不是关机或计费状态证明。上一轮image-value-component关机请求在同一窗口只读观察到，时间1789227216.8670177、wrapper SHA `0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027`、未请求释放；它也不证明当前平台状态。

本地过程证据在 `runs/hestia-scene-kv-preparation-20260913-001/`，包含精确payload清单、pack/stage/preflight/receiver/launch/status脚本及回执。首次上传阻断清单另存 `m2-smolvla-hestia-scene-kv-dispatch-blocked-2026-09-13.json`，是历史阶段而非当前队列。没有新Gate或M2验收。
