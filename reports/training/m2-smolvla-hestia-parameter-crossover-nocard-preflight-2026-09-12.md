# Hestia 参数互逆实验：无卡准备完成，等待带卡推理

用户明确允许 14 文件上传后提供了无卡 SSH。本轮已完成工作区准备、远端
身份检查和 CPU 预检；**没有构造真实模型或启动四组 CUDA 推理，根因尚未
确定。** 无卡模式没有被替换成其他后端来执行原 CUDA 实验。

当前唯一工作区身份是 `20260912T102718Z-bd007f5402bf-c3091a190b62`，组合 SHA：
`c3091a190b62611fd786ad2366a4e1d1c891e8a9681cc6f679a83b2e8c805abd`。
其组成仍是既有远端基线归档 `3a455a4748d8...` 与用户已批准的
14 文件包 `fd34c5972424...`。未再次外传整仓，未提交或推送。

## 传输实际情况

首次增量传输连接重置。检查发现新目录、组合身份和 12 个完整新增文件已经
到达；54 项身份中 52 项匹配。640 数组停在 720,896 bytes，1280 数组尚缺。
接收 tar 进程仍在等待，不能把本地连接退出当作远端进程已退出。

核对本次 tar 的 PID、start_ticks 和精确目标目录后，仅结束这个自有接收
进程。640 断片及其大小/SHA 保留在新工作区的
`runs/staging-interrupted-transfer-20260912-001/`。两份数组从同一远端既有
`hestia-cuda-checkpoint-curve-20260911-002` 的原件复制补齐；源/目标 SHA
均与冻结模板相同。历史原件没有修改，没有删除断片或覆盖其他工作区。
随后 54 项全部匹配，最终科学输入与批准清单逐项相同。

本机直接使用 Windows OpenSSH 的 Git Bash 调用完成补齐及核验，关闭了
Git Bash 对远端 Linux 路径的自动转换。此前系统 `python3` 缺少
`hashlib.file_digest`，只读核验改用已有登记 Python 3.12，没有安装依赖。
这些是传输和检查端的修复，不改变实验方法或后端。

## 无卡预检实测

通过 `scripts/run_autodl.sh shell` 进入原登记环境，离线核验以下身份：

- 54 个源码/输入文件。
- 两个 checkpoint 的全部 20 个登记文件。
- 两个上游实现文件、两个既有 normalization 摘要/manifest、一个运行 profile。

合计 **79 项 SHA 匹配**。运行时为 Python 3.12.3、Torch 2.8.0+cu128、
NumPy 2.2.6、LeRobot 0.6.2；`cuda_available=false`、`device_count=0`。
两份 checkpoint 仅按字节流校验，没有物化其权重或构造 policy。

34 项 CPU 合成/门禁/传输检查通过，测试耗时 3.58 秒；整个预检 11.675 秒。
父子进程峰值 RSS 保守相加为 1,188,278,272 bytes。另一次只读检查确认
容器内存上限 2 GiB，memory.events 的 max/oom/oom_kill 均为 0。
CPU 预检期间 SSH 又关闭，但之后读取到完整 passed 结果和测试日志，且没有
该测试进程；因此按终态证据认定已完成，没有重复运行。

预检结果、测试日志、断片恢复记录和工作区身份共四份证据已回传，本地核验
文件集合、长度和 SHA 全部通过，保存在
`runs/hestia-parameter-crossover-nocard-recovered-20260912-001/`。
远端没有绑定新工作区 normalization 路径，以保留正式守护器的首次绑定流程。
CUDA doctor 未在无卡模式运行；实际四项 endpoint/hybrid 结果仍为 not measured。

## 接续边界

恢复原 4090 D 带卡模式后，**复用上述已完成的工作区，不再次 staging**。
本机入口 `runs/hestia-parameter-crossover-dispatch-20260912-001/launch-prepared-git.sh`
保留当前 SSH 参数输入、显式仅推理/受保护结束开关，启动同一冻结 supervisor。
先核实当前设备和 source/缓存，再由 supervisor 进行 CUDA doctor、原生
normalization 绑定和 CPU 门禁；两个原生端点必须精确复现后才运行 hybrid。
四组 720 forwards，0 optimizer，工作上限 900 秒，受保护结束上限 1200 秒。

本轮未调用关机或释放实例；用户当前选择的是无卡准备。没有新增 Gate 3/4，
没有修复策略或选择 checkpoint，M2 未完成。
