# Hestia 互逆 K/V 参数检验：本地准备完成，尚未执行 CUDA

根因目标尚未完成。上轮 `prefix-path` 已真实执行并回传；本文件登记的是
后续独立实验 `hestia-parameter-crossover-20260912-001`，没有复用旧执行许可。

前置真实字节审计已完成：两端各 500 个 tensor，345 个冻结 VLM tensor
逐字节相同；122 个参数 tensor 变化，其中 cross-attention K/V 共 16 个，
其余动作路径 106 个。两份 checkpoint 整文件 SHA 在前后匹配原 CUDA 身份，
按不超过 1 MiB 分块读取，未构造 policy。耗时 55.266 秒，峰值 RSS
20,045,824 bytes。证据在 `runs/hestia-parameter-crossover-audit-20260912-001/`。

这排除的是 640→1280 冻结 backbone 权重发生变化的解释，不证明该 backbone
已具备所有任务信息，也不证明剩余 122 个参数中哪一组导致泛化失败。

实现已冻结为四项：原 1280、原 640、1280 仅换入 640 的 K/V、640 仅换入
1280 的 K/V。16 个精确键和 320×320 shape 全部核验后才写内存参数；
其余参数前后 hash 必须相同，donor 复制逐值相同，推理期间全部参数不变。
两个端点完整 normalized/standard 输出须分别逐值复现历史 CUDA，之后才
准许运行 hybrid。四噪声 × 45 场景 × 四项，共 720 次 forward、0 optimizer。

最终本地 34 项检查通过，包含：精确文件 donor 复制、禁止非登记参数和
部分写入、schema/nonfinite 拒绝、两个完整端点前置、端点后续篡改拒绝、
许可/期限/顺序、失效 PID 不发信号、路径逃逸及非白名单权重传输拒绝。
Ruff/format 通过；Bash 接收与私有调度脚本语法检查通过。这些不是实际
CUDA 替换的成功证据；实际端点重现、hybrid 结果和本轮关机均为 not measured。

运行入口和固定身份：

- 同日期 `parameter-crossover-plan` MD 和 `parameter-crossover-template` JSON，
  后者绑定 54 个源文件/输入身份及两个 checkpoint 文件集合。
- `scripts/diagnose_hestia_parameter_crossover.py` 采集；
  `scripts/hestia_parameter_crossover.py` 实施精确互换。
- `scripts/run_hestia_parameter_crossover.py` 守护、独立 watchdog、受保护结束；
  每项 180 秒，共享工作上限 900 秒，最迟 1200 秒走受保护关机。
- `scripts/receive_hestia_parameter_crossover_results.sh` 和对应 stream/verify
  进行白名单、文件集合、类型、大小及 SHA 校验，匹配回执后结束 GPU 窗口。
- `runs/hestia-parameter-crossover-dispatch-20260912-001/` 为本机预备调度脚本，
  使用显式传入的当前 SSH 与已有 key 文件路径；未写入连接信息或 key 内容。
  stage 复用仓库的内容寻址 helper，launch 只补传既有历史数组和审计 JSON，
  不安装环境、复制权重、删除旧文件或覆盖旧 workspace。

待用户提供当前原 RTX 4090 D 带卡 SSH 后，先核实空闲 GPU、原环境、已存在
checkpoint/cache、受保护关机 wrapper 与 Trash 条件，再创建唯一 workspace，
启动带独立 watchdog 的仅推理任务并核实启动，接收闭合结果。所有数据均
使用既有缓存；不因缺缓存自行下载。退出/停止条件和解释边界见 plan。

这一试验首先检验后半程退化的参数来源，不能单独解释两个端点原有的全部
失败。未选 checkpoint、未重训、未改 Gate 阈值、未解封 hidden。M2 未完成。
