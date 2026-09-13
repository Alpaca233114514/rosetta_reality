# Hestia 参数互逆入口修复登记

`hestia-parameter-crossover-20260912-001` 的首次 GPU 调度在模型加载前失败：
运行 profile 为环境变量字符串，源码哈希函数要求 `Path`，产生
`AttributeError: 'str' object has no attribute 'open'`。同一问题也存在于
模板文件名迭代。worker 27.966 秒退出，未完成任何 condition；doctor、
normalization、CPU 预检 exit code 均为 0。实际四组推理均未开始。

新登记 `hestia-parameter-crossover-20260912-002`，只修复上述两处路径类型。
原 001 源码、模板、已上传包和远端工作区保留；不热修改旧 identity 或
把失败结果改成成功。新文件为 collector/supervisor/stream 的 `_v2` 入口。
两组原生端点、两组互逆 K/V、40/5 split、四噪声、50 槽、640/1280 两端
权重、Action Contract、processor/normalization、CUDA 运行时与全部精确
复现门槛完全沿用 001。仍是 720 forwards、0 optimizer、每项 180 秒、
共享工作 900 秒、最迟 1200 秒受保护结束。不是新训练炉或新实验轴。

补充测试调用真实 CLI `main()` 的完整模型前置链，使用合成文件、真实
Path/文件 SHA、PID 记录和参数解析，仅将最终 collect 与 CUDA 可用性替代：
旧版应复现原异常；修正版四种 condition 全部到达正确 collect 分派；
profile/input/upstream 篡改仍拒绝；无卡仍拒绝。没有把模拟 CUDA 当作
实际 GPU 执行证据。43 项本地相关测试通过，包括 34 项原有检查与 9 项
新增入口检查，固定离线 Linux Docker、2 CPU/2 GiB，未构造真实模型。

新工作区通过 staging helper 从已经核验的 001 组合工作区派生，原件只读。
仅增加六个新代码/测试/计划文件，复用远端已有且 SHA 相同的原历史数组与
参数审计摘要；排除旧 runs、normalization 绑定和旧 provenance marker，
新 workspace 记录基线组合身份及增量 SHA。没有新的权重或原始样本上传。
旧运行的 normalization 绑定不会被复制进新任务，避免二次绑定冲突。

恢复 SSH 后首先取回 001 的闭合失败证据。先前接收器在关机窗口后连接失败，
本地 archive 为空，不能声称完整失败黑匣子已经回传；只读状态快照已看到
明确的 worker 退出和异常。保留失败接收目录，使用新接收目录恢复旧结果。
再创建新内容寻址 workspace 并核验 002 的身份，按当前用户提供的卡模式
决定只做 CPU 准备还是继续已登记 CUDA 推理，不在无卡模式执行模型。

根因目标仍未完成。本次错误是诊断入口实现缺陷，不是视觉泛化的根因。
不得把路径修复、CPU 测试或旧 worker 的正常停止称为策略改善。新 Gate 3/4
仍为 not measured，M2 未完成。
