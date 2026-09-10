# Athena：无卡 CPU 预检与存储核验

CPU 合同与观察器模块通过；**存储预检失败，Athena 未启动，视觉忽略仍未修复，M2 未完成**。本报告追加证据，不改写 coverage40 负结果及原始草案。

后续同日更新：用户授权清理多余内容后，存储缺口已通过保留全部路径与唯一权重的去重解决，见 `reports/training/m2-smolvla-athena-storage-cleanup-2026-09-10.{md,json}`。本报告保留清理前的实测字节与失败状态。

## 实测结果

| 模块 | 不可变源码 | 结果 |
| --- | --- | --- |
| 原 A 实例 schema / launcher / sampler 回归 | `c42d8354c4ee1a178ab3d1b167c43c6ef30c27fe` | 79 passed，3.76 秒，Ruff 通过 |
| 原生 CPU scheduler / DataLoaderShard 合成探针 | 同上 | 通过，11.496 秒，峰值 RSS 845,336,576 bytes |
| B 实例成功更新观察器 | `016eb497e6a168db9c7edbf55190604d08aeef4b` | 21 passed，12.28 秒，零跳过，Ruff 通过，作业退出 0 |
| B 实例 A/B 权重流式 SHA 与文件清单 | 同上，独立只读阶段 | 两个权重 SHA 均匹配；未反序列化或修改 checkpoint |

两组测试在不同 workspace 运行，不能写成一次联合回归。原 A 实例的首次 CPU 作业因 Ruff I001 退出，后续只修正空行后另建 workspace 通过，首次日志保留。

原生 trainer SHA 为 `4d15d283ea54583f552b32088db0b6c195250905ca6daf06d4670383790e2059`，与此前 worker / 官方固定 revision 的证据相同。B 实例为 AutoDL 平台容器，不嵌套 Docker；Python 3.12.3、torch 2.8.0+cu128、lerobot 0.6.2、transformers 5.5.4、accelerate 1.14.0、numpy 2.2.6。现场 GPU 查询返回 `No devices were found`，所有更新验证只使用 CPU 合成参数。

## 本模块解决的问题与边界

真实 schema 确认旧候选 1280 training / 256 decay 被拒绝，1280 / 1280 可通过结构和 CLI 检查；formal phase 仍拒绝固定视觉样本。没有放宽入口或 optimizer 校验。实际原生 LR 序列仅前 16 个 update 相同，候选改变训练步数和退火路径，不能解释成纯重复次数效应。

原生 sampler + DataLoaderShard 在 2、10、256、320、1280 步的实测 consumed / emitted 为 8/12、40/40、1024/1028、1280/1280、5120/5120。1280 步刚好落在完整 epoch 边界，40 场景各暴露 128 次；这里是合成索引的预期轨迹验证，不是完成了 1280 步 SmolVLA 训练。

新增 `src/rosetta_reality/vla/training/observation.py` 区分已交付批次、成功 optimizer step 和完整 native update。预处理失败、更新前/后失败、跳过 step、重复 step、缺批、多批、错序、错误 episode/frame、非同步梯度累积均不能冒充完整成功。真实 CPU AdamW/LambdaLR 以及固定原生 `update_policy` + CPU Accelerator 的对照中，权重、RNG、loss 和 LR 一致；观察器交付原始 batch 对象，结束后恢复 wrapper。

观察器还未接入 Athena launcher，也未在真实 SmolVLA processor / CUDA 运行中验证。单参数合成 policy 不代表 SmolVLA forward、数据合同、视觉泛化或闭环成功。旧探针 JSON 中“production observer 未实现”的字段描述其运行时状态，原件保留；新观察器证据在单独文件。

## checkpoint 和容量

- A 权重 SHA：`fd95a19ca8d8f9e1c6a5c477807aaf69fc3e19bb15e626bd53e4224da8ad3a5c`；完整恢复目录 1,614,046,416 bytes。
- B 权重 SHA：`4e91624b34c3a16e22e0cbba19d8bcf3352a0d3835522ea6973ee22f0397accf`；完整恢复目录 1,614,046,930 bytes。
- 可用空间：6,661,238,784 bytes；六份 B 尺寸完整 checkpoint 加 2 GiB 余量：11,831,765,228 bytes；缺口：5,170,526,444 bytes。
- 六份分别为四个 quarter、一个 smoke、一个原子保存临时副本。完整 optimizer/RNG/scheduler 状态计入，没有删除、迁移历史文件或减少保留份数。容量是候选预检预算，不是新 checkpoint 已保存的实测尺寸。
- 两台无卡实例只能顺序开启。原 A 实例小型结果已取回；当前 B 实例同时保留 A、B，不再要求两台同时在线。未传输模型、数据或 optimizer state 到本机。

## 执行与审计限制

观察器测试由 tmux 和 600 秒 timeout 守护，已退出 0；随后没有训练进程。源码提交和部分远端操作曾因自动审核超时未执行；权重校验还经历审核服务网络断流。重试前检查了执行状态和脚本仅只读 checkpoint、仅向本次新目录 create-only 写摘要，之后获准执行，未绕过审核。

由于审核延迟和会话中断，checkpoint 核验被拆为测试退出后的独立 180 秒 timeout 阶段，其采样时间晚于观察器登记 600 秒。**整段墙钟不能宣称在一个连续 10 分钟窗口内完成**；没有复活已退出的观察器 worker、延长其 watchdog 或启动 GPU。CPU 时间和两个阶段的证据分别记录。

## 下一步

先补足已登记持久存储；可将 B 实例数据盘增加至少 6 GB，建议 10 GB 留出额外日志和配置余量，或核验用户指定的其他可靠持久位置。扩容后重新量测可用字节，不把本次结果当作以后状态。

随后完成独立 B/C 评分合同与观察器接线，再冻结代码/config/plan 身份、运行真实预检及独立两步 smoke/reload。各前置门禁通过才运行候选 1280 步诊断。保持冻结范围、原生损失、seed 和既定 split；不续跑 B optimizer，不重训 A/B，不打开 hidden，不叠加新 loss 或解冻。训练拟合和开发验收失败即保留负结果并停止该候选。

当前 Athena 训练、GPU 资源、B/C 视觉分数、CUDA 观察器验证和 Gate 3/4 均为 `not measured`。历史 B 开发视觉验收仍为 0/4。

机器摘要及可发布的小型证据见同名 JSON 与 `reports/training/athena-cpu-20260910/`。实例结束状态另以追加的关机记录为准。
