# Q0 单候选有限筛查执行登记

执行身份：`q0-coupled-translation-20260927-001`。
用户本轮提供原实例 SSH 并明确要求按最新修订部署、执行及安装看门狗。
平台定时关机由用户自行处理；规划截止为北京时间 2026-09-27 20:00，
本代理未核验平台设置。旧修订中的无执行授权字段保留为历史快照。

本次执行最新 revision-002 的 Q0 自动来源路径中的必要初筛，
候选设计见 `q0-coupled-translation-candidate-design-2026-09-27.md`。
仅一个新候选，不重跑旧失败身份。完整 Q0 的 G3/G4 尚无可执行实现，
所以本次最多 G0/G1/G2，任何结果都不自动启动采集或三臂训练。
Exp1 现有 base Gate3 失败，base Gate4 仍为 not_measured。

## 实测准备与环境

只读现场检查为 RTX 4090 D / 24564 MiB，GPU 无计算进程、无 tmux；
内存 cgroup 上限 64,424,509,440 bytes，数据盘剩余 11,882,274,816 bytes。
Python 3.12.3、PyTorch 2.8.0+cu128、MuJoCo 3.8.1、gym-aloha 0.1.4 已安装。
这些是准备时观测；启动和 worker 继续检查资源，不据此声称 runtime 全部准入。

现有环境缺 Mink 1.2.0、qpsolvers 4.13.0、DAQP 0.8.7。
已审阅 PyPI 官方固定版本的项目源、Requires-Dist、兼容 wheel 大小和 SHA。
仅取三份固定 wheel（合计 1,051,123 bytes）并逐个校验，拒绝路径穿越、
启动 .pth 和 scripts；通过 `--no-index --no-deps --no-compile --target`
安装到本次输出的 `dependencies/`。不修改既有 Python 环境、系统 PyTorch、
CUDA 或 MuJoCo。原 NumPy、SciPy 和 typing-extensions 满足已审查的依赖范围。
实际安装收据由 `scripts/prepare_q0_dependencies.py` create-only 保存。

AutoDL 本身为登记的 Linux 平台容器，不嵌套 Docker。
使用 `configs/runtime/autodl_rtx4090.yaml` 记录参考身份，并经
`scripts/run_autodl.sh shell` 启动。`autodl-platform-container-observed`
是观察标签而非已验证 image digest；保存版本、mount namespace 和 cgroup。
本次不加载模型权重、训练数据或 optimizer，不是 CUDA 模型 doctor/smoke 准入。

## 固定命令链与预算

配套机器登记：`configs/diagnostics/q0_coupled_translation_20260927_001.json`，
包含源文件完整 SHA、命令 argv、候选、观察器和固定截止。
源码通过 `scripts/stage_autodl_from_wsl.sh --verified-source-delta`
生成新 content-addressed workspace，并逐文件核验。旧 workspace 和已有修改保留。

| 阶段 | 工作 | 秒数上限 |
| --- | --- | ---: |
| check_env | 独立依赖目录、版本/环境检查 | 180 |
| synthetic | 看门狗、supervisor、teacher contract 和候选回归 | 120 |
| benchmark | seed10 最多3个零增量保持步，固定54次 IK | 180 |
| register | 新候选和原协议绑定 | 30 |
| G0 | 原输入、确定性、显式拒绝检查 | 60 |
| G1 | 原 seed10，最多500步，原成功与安全判据 | 420 |
| G2 | 原1900–1904，各最多500步，全部成功并另核安全 | 1800 |

整个 worker 上限2700秒，实际 work deadline 取启动加2700秒与平台规划截止减300秒的较早值。
独立 guard 在 work deadline 停止本任务进程组，在其后最多300秒请求受保护关机。
正常完成或失败后保留最多300秒的回传窗口。平台截止独立由用户管理。
3个 benchmark 步与最多3000个 gate 步单独计数；benchmark 不作为 G1。
模型 forward、optimizer update、collection episode 均为零。

benchmark 用6次冷启动成本，加上实测54次 IK 和最大热 step/render/observer/safety
耗时乘3000、再乘1.5作为保守估计；超过1800秒则停止，不能改变门槛凑通过。
该估计不保证 loaded-contact 尾部耗时，独立工作截止仍生效。
进程组 RSS 上限8GiB，任务目录上限256MiB，磁盘至少保留1GiB。
无自动 retry、resume、扩大预算或下一炉。

## 看门狗及证据边界

`scripts/q0_bounded_guard.py` 为本次专用版本，复用原 guard 的时间和平台关机保护。
补充组长退出后的子进程清理，避免只因组长消失便漏掉原组内进程。
旧 `scripts/autodl_bounded_guard.py` 保留。
worker 在 guard armed 并绑定 PID/截止后才启动，各阶段另有限时与资源检查。
关机仍核对原 wrapper SHA、Trash、无关 GPU/tmux/项目进程；不释放实例。

首个阶段失败即保存证据并停止。G2 即使成功也仅为 partial screening，
`provider_qualified=false`；G3–G5、恢复标签、正式 cohort 和训练保持未执行。
输出为相对远端数据根 `runs/q0-coupled-translation-20260927-001`，
封存逐文件大小/SHA、phase、失败或结果。部署完成与实际成功分别记录，M2不变。
