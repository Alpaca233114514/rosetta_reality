# Hestia 参数互逆 002：旧失败恢复与修复工作区无卡预检

本次 SSH 恢复后实测无 GPU。001 的完整闭合失败包已回传，7 个登记文件的
长度与 SHA 全部通过固定离线 Linux Docker 校验，匹配收据已送回。
证据位于 `runs/hestia-parameter-crossover-v1-recovered-20260912-001/verified/`。
异常与先前快照一致：`file_hash(profile)` 收到字符串，模型加载前退出，
completed_conditions 为空、optimizer_steps 为 0。旧空接收包与旧工作区保留。

另外取回并校验了旧 `shutdown-request.json`：请求时间 1789210803.8392704，
距 worker 退出约 120.124 秒，wrapper SHA 为
`0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027`。
这确认守护器记录了关机请求，不是计费或平台断电的独立验证。
没有请求释放实例。本轮无卡准备没有启动关机流程。

六文件修复增量已从旧 001 工作区只读派生到新的工作区身份
`20260912T113534Z-bd007f5402bf-1cd3c7fc0c28`，组合 SHA 为
`1cd3c7fc0c2886de2ac858b089994109d4bcd6fd3600c7e4d17f70a8b1b8c1fc`。
增量仍为 71,680 bytes，SHA
`a7bafef3e5a30f4c41b1fe41719c3c72e64633bfc39d42c73bcfc97af6aadd5e`；
没有再次上传原 14 文件包，没有更改旧实验 identity。
002 模板 SHA 为
`2615126e707692bac338217e7d81f7bb6d55a9cb097dd55c4e62ac8371cdafcf`。

通过 `scripts/run_autodl.sh shell` 在登记环境执行无卡预检：59 项源码/输入、
20 项 checkpoint 文件、2 项上游实现、2 项 normalization 文件和 1 项运行
profile，共 84 项 SHA 一致。checkpoint 仅流式校验，未物化权重。
运行时仍是 Python 3.12.3、Torch 2.8.0+cu128、NumPy 2.2.6、LeRobot 0.6.2。
`cuda_available=false`，`device_count=0`。

43 项 CPU 检查通过，测试 3.72 秒，全部预检 12.103 秒；父子进程峰值 RSS
保守相加为 1,183,858,688 bytes。未构造模型、未执行 forward 或 optimizer，
未绑定新工作区 normalization；CUDA doctor 未在无卡模式运行。
预检结果、测试日志、工作区身份及旧关机请求共四份证据已逐文件校验后保存于
`runs/hestia-parameter-crossover-v2-nocard-recovered-20260912-001/`。

带卡接续复用上述新工作区，不重新 staging。先确认原登记 GPU 与输入身份，
然后使用接收器先连接的 `runs/hestia-parameter-crossover-v2-dispatch-20260912-001/run.sh`
启动原已登记的 002 检验。四组共 720 forwards、0 optimizer，两个原生端点
精确复现后才允许两组互逆 K/V。执行仍须满足登记的生命周期与资源边界。
四组 CUDA 结果均为 not measured；本轮只解除诊断入口与传输问题，视觉泛化
根因未确定，没有新 Gate 3/4，M2 未完成。
