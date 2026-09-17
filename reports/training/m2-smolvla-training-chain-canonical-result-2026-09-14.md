# 训练链规范化修正：真实 CUDA 验收结果

run `training-chain-gpu-audit-20260914-007` 已完成九个阶段，全部退出码为 0，
worker error 为 null。修正后的实际 CUDA loss、梯度、动作、两步 optimizer
执行和独立进程重载通过本轮有界验收。这里的通过不代表全数据训练、模型
泛化修复或 M2 完成；既有 Iris control 的 Gate 4 仍为 0/5。

本报告接续 `m2-smolvla-training-chain-gpu-continuation-result-2026-09-14.md`，
更新其中“规范化 GPU 待验收”的状态。旧报告、旧计划与失败证据保留。

## 身份与证据

- template 013 SHA-256：`da335114f316c96ff411562179eb98b4ef7a25cbaa50d89f3c6f445a1c440969`。
- 内容寻址工作区：`20260914T091557Z-95cf9cf9483b-8f4430ffcb0a`。
- composite SHA-256：`8f4430ffcb0a951fd05ff83c66f8b09874fde720763bf7b215e28ec3991e1757`。
- 证据根：`runs/training-chain-gpu-received-20260914-012`，handoff manifest 的
  40 个成员逐文件 SHA-256 校验通过；QA 日志也随包取回。
- GPU 上相关回归为 105 passed、0 skipped、4 warnings，包含 CUDA 全部
  256 个像素值的精确一致性测试；Ruff 通过。警告为多线程进程 fork 提醒，
  本轮测试未死锁，不据此推断任意多进程配置都可靠。

## 修复了什么，怎样重测

旧路径在 CUDA 上对 uint8 除以 255，离线入口在 CPU 上做同一操作，存在
float32 舍入差异。run 006 在非零帧测得完整归一化动作最大差 0.0399322509765625，
frame-0 单独检查会漏掉它。它不是已证明的历史 Gate 失败唯一原因。

`src/rosetta_reality/vla/image_scaling.py` 使用 CPU 定义的 256 值 float32 查表，
`canonical_image_scaling` opt-in 特性在原生 cycle 入口应用它，原生 uint8-only
缩放分支不再重复除法。模型入口再以独立 CPU `/255` 参考精确检查图像，
同时核对 state、完整动作 chunk、padding 和元数据身份。

相同 pinned base、RNG、BF16 和完整推理噪声下重新执行实际模型：

| episode 49 输入帧 | CPU/CUDA loss 绝对差 | 完整归一化动作最大差 | 精确相同梯度张量 |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0 | 155/155 |
| 249 | 0 | 0 | 155/155 |
| 499 | 0 | 0 | 155/155 |

相同规范化下，native 与 masked-camera skip 的 loss、梯度、完整动作也精确
一致。此阶段未更新参数。结论来自真实 forward/backward 和完整输出比较，
不是仅比较像素或假模型结果。

## 实际训练帧和更新

新 run 从既定 base 开始，仅 batch 4、两次成功 optimizer 更新，不是 run 006
的继续训练。两个诊断 run 合计四次更新，不能称作一次四步训练。

- 实际消费顺序为 `(49,0),(4,0),(49,249),(4,249),(49,450),(4,450),(49,499),(4,499)`。
  delivered、completed、unique input 均为 8；每个 episode 四个输入帧。
- 400 个动作槽位中 302 有效、98 padding；300 个唯一目标帧、2 次重复覆盖。
  相对 20000 个训练输入帧，输入覆盖 0.04%，唯一目标覆盖 1.5%。这不是全量遍历。
- 真正送入 policy 的每样本 state 为 `[1,14]`，RGB 为 `[3,480,640]`，
  action 为 `[50,14]`；原生 observation delta 为 `[0]`，action delta 为 0..49。
- 两次实际更新 LR 分别为 `1e-4`、`5.125e-5`；不能把更新后 scheduler 的值
  当作当前 update 的 LR。345 个冻结参数张量保持不变。
- 从 CLI、resolved config、FeatureStack、sampler/cycle、processor、policy
  输入到成功 optimizer 调用均有绑定记录。processor 会移除 frame_index，
  观察器在其前面取身份，再在模型入口核对真实张量，没有向模型伪造身份字段。

## 保存与重载

两个独立进程 PID 18436、30773 实际加载权重及保存的 processor，各做
16 次推理（8 帧 × 两个噪声条件），参数不变、无 optimizer 更新。
完整 normalized action、standard action、32 维 noise、sample identity 和
valid mask 精确一致；14 维动作不意味着截掉其余推理噪声。

`reload-proof.json` 本身只证明已存数组的比较，实际模型执行由两个独立
execution 文件补足。checkpoint 的 image preprocessing sidecar 记录新 recipe。
完整 checkpoint 共 17 文件、1614046590 字节已取回并逐文件 SHA-256 校验，
包括模型、processor、optimizer、scheduler、RNG 和 recipe。备份位于
`runs/training-chain-checkpoint-recovered-20260914-007/verified`。传输瞬断的失败
分片保留，补齐缺失文件后才判定完整。无卡模式只取文件，完成后已在刷新后的
平台页面确认原实例“已关机”；详细收据见同名 JSON 的 recovery 字段。

## Gate 与局限

run 006 对既有 **1280-step Iris control** 重测：桥接完整 chunk 精确一致，
Gate 3 通过，Gate 4 五个 seed 均执行 500 步，成功 **0/5**。非法动作、关节
越界及意外碰撞均为 0，最低成功率条件仍失败。run 007 未重复未受此训练特性
影响的固定 Gate；不能把既有端点的 Gate 结果记到新两步 checkpoint 名下。

修复以显式 opt-in 方式验收，不追溯改变历史训练的预处理或历史结果；尚未
证明所有其他入口都自动采用该 recipe。正式 resume、长训练稳定性、全数据
拟合与开发泛化、隐藏集、视频物理时间对齐以及真机表现均未由本轮证明。
新两步模型未做任务成功验收。工程一致性改善不能代替模型能力验证。

## 清理范围

已按逐路径授权删除本地备份且校验一致的 Iris 传输归档约 2.25 GiB，以及
旧 run 006 诊断 checkpoint 约 1.50 GiB，共约 3.75 GiB。清理收据为
`training-chain-storage-cleanup-20260914-001.json`、`002.json`；本地完整备份保留。
本轮新 run 007 checkpoint 保留远端和本地副本，未扩大删除范围。
最终数据盘可用 5626368000 字节（约 5.24 GiB），系统盘可用 3093442560
字节（约 2.88 GiB）；系统盘没有清理。未提交或推送代码。
