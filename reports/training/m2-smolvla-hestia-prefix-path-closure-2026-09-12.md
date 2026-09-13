# Hestia 原生前缀采集已完成，输出精确复现

用户提供当前 SSH 后，`hestia-prefix-path-20260912-001` 在原 RTX 4090 D、
Torch 2.8.0+cu128 上完成。180 次 forward 的 normalized 和 standard 完整
输出均与历史 C 逐元素相同，最大绝对差为 0；500 个参数张量前后 hash 不变，
optimizer steps 为 0。原图、非视觉输入、噪声和目标身份检查通过。

第 1、15 个零基 cross-attention 层的 K/V 投影前后，实际摄像头各 64 个
token 已保存为 45 个 NPZ。每文件八个 `[1,64,320]` float32 数组，原始
数组共 29,491,200 bytes。激活在 10 次 denoising 和四个固定噪声之间逐值
相同；只读 hooks 没有改变原生输出。检查的是这两层，不外推其他层。

worker 无错误退出，耗时 155.672 秒，其中采集 130.851 秒；峰值 CUDA
allocated 1,437,593,600 bytes。远端 CPU 预检实际为 23 passed，不能混用
此前本地准备阶段的测试数。PyAV 是原环境已有的解码 fallback，未安装依赖。

创建并使用唯一内容寻址 workspace，归档 SHA 为
`3a455a4748d8ff588cdc74c7bce452862fb602d70a88b0610e068fcbeafa8c7d`。
传输层补充记录在 `runs/hestia-prefix-dispatch-20260912-001/dispatch-registration.json`：
WSL 原生网络不可达，正式权限审核后使用已有 Windows OpenSSH；新建私有
接收适配脚本接受 staging helper 实际产生的 `workspaces` 路径布局。
冻结采集代码和科学协议未改变，原接收器未覆盖，未推送或覆盖远端旧代码。

56 个结果文件全部通过路径、类型、文件集合、大小与 SHA 校验；归档共
15,697,920 bytes。handoff manifest SHA 为
`4db3c3737800313bdc1b3417068cb5f50054e6c100fb34df0cb8bb24e0e29773`。
本地证据在 `runs/hestia-prefix-recovered-20260912-001/verified/`，接收回执
在 `runs/hestia-prefix-receiver-20260912-001/receipt.json`。

匹配回执已成功送达受保护关机流程，随后 SSH 连接关闭。没有独立核实平台
电源或计费状态，未释放实例。历史 preparation/plan 中的 pending 是当时状态，
由本次完成证据接续，不能据此重复启动。

本次是观测采集，尚未实施参数或激活干预。后续本地八臂读出见
`reports/training/m2-smolvla-hestia-prefix-readout-result-2026-09-12.{md,json}`。
唯一根因仍未确定，没有修复策略、选择 checkpoint、新 Gate 3/4 或完成 M2。
