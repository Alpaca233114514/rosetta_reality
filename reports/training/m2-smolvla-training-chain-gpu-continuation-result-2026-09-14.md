# 训练链 GPU 续测、执行差异与修正验收边界

本轮已完成旧执行路径的有界 GPU 验证和固定 Iris control Gate 重测；发现并
定位了会改变数值结果的 CPU/CUDA 图像缩放差异。规范化修正已实现、82 项
CPU 回归通过，但原实例空闲 GPU 已不足，修正后的 CUDA/真实模型验收未完成。
不能据本报告宣称整条代码链无影响、模型泛化修复或 M2 完成。

## 实际执行与训练帧

权威运行 `training-chain-gpu-audit-20260914-006`，template 010，新内容寻址
工作区 `20260914T082946Z-95cf9cf9483b-dce774ba55fa`。12 个阶段退出码均为 0，
worker error 为 null；Gate 4 引擎返回 1 被正确保留为模型负结果，未改判通过。
运行前的代码检查与 63 项相关回归通过，完整黑匣子 68 个成员回传并逐个 SHA
校验。原始证据根：
`runs/training-chain-gpu-received-20260914-009/runs/training-chain-gpu-audit-20260914-006`。

证据链包含实际 CLI 启动清单、runtime experiment、native train config、
FeatureStack 安装记录、原生 sampler/DataLoader 身份、policy forward 输入、
optimizer 完成记录、checkpoint 配置及独立进程重载，不能只看计划中的行数。

- 原生 batch 4、两次实际 optimizer 更新；不是恢复训练或正式训练。
- 按顺序消费 `(49,0),(4,0),(49,249),(4,249),(49,450),(4,450),(49,499),(4,499)`。
  delivered/completed/unique input 均为 8，两个 episode 各四个输入帧。
- 共 400 个动作槽位，302 有效、98 padding；唯一目标 `(episode,frame)` 为
  300，尾帧被重复覆盖两次。输入覆盖为原 20000 个训练帧的 0.04%，唯一目标
  覆盖 1.5%，不能称为完整训练集遍历。
- 实际 `n_obs_steps=1`；state 每样本为 `[1,14]`，图像为 `[3,480,640]`，
  action 为 `[50,14]`，原生 observation delta 为 `[0]`，action delta 为 0..49。
  原生 processor 移除了 `frame_index`，所以先在 cycle 输出观察身份，再将
  身份绑定到逐样本图像/state/action/padding 校验；不向模型输入注入假元数据。
- 345 个冻结参数张量保持不变。实际 optimizer LR 为 `1e-4`、`5.125e-5`；
  日志和 step-2 checkpoint 的 `2.5e-6` 是 scheduler 更新后的值，不是第二次
  update 使用的值。native scheduler 的两步缩放为 warmup 0、decay 2。
- step-2 checkpoint loss 为 0.7720232605934143，仅是这两批的训练诊断，不能
  与不同帧分布、mask 或正式调度的历史 loss 作学习效果比较。

## 查到的真实代码执行差异

四个独立探针的七个文件均回传校验，证据见
`runs/training-chain-video-probes-received-20260914-001`，失败探针也保留。
8 帧的原始缓存与 native train-only view 的图像、state 和 action 精确一致；
CPU 上 native processor 也未改变图像，帧和像素源没有在这组样本中损坏。

实际执行顺序是：`make_dataset(return_uint8=True)` → Accelerate 将 batch
放到 CUDA → native training loop 转 float32 并 `/255` → processor。
旧离线入口是在 CPU 上 `/255` 后才送 CUDA。全部 256 个像素值中 126 个
float32 结果不同，8 帧共 516766 个标量出现差异，最大绝对差为
`5.960464477539063e-8`，但所有 uint8 像素都可精确还原。

在相同 pinned base、RNG、BF16、固定推理噪声和未变参数下，重测得到：

| episode 49 输入帧 | CPU/CUDA 缩放的 loss 绝对差 | 完整归一化动作最大差 | 逐字节相同的梯度张量 |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0 | 155/155 |
| 249 | 0.001466989517211914 | 0.0399322509765625 | 0/155 |
| 499 | 0.0033729076385498047 | 0.019439697265625 | 0/155 |

动作差是完整 50×14 **归一化输出**的差，不是弧度或成功率。梯度哈希不同不
表示梯度方向或范数出现同样比例变化；本轮未量化这些梯度差的幅度。
结果说明仅检查 frame-0 或忽略最低有效位会漏掉执行路径影响；不证明该差异
是历史任务失败的唯一原因。相同缩放路径下，原生实现与 masked-camera skip
在三个时刻的 loss、全部 155 个梯度张量和完整动作仍精确一致。

## 重载与真实 Gate

两步 checkpoint 在 PID 33793、46132 两个独立进程中分别加载权重和已保存
processor，各运行 16 次推理（8 个帧 × 两个噪声条件）。完整 normalized action、
standard action、32 维 noise、sample identity、valid mask 全部精确一致。
这有实际模型执行日志支持，不仅是数组文件比较；未验证正式 resume 或本地
再次加载 450M 模型。新增 `noise_action_dim` 合同保留全部噪声，没有截断成14维。

固定 Gate 使用**已有 1280-step Iris control**，不是本次两步 checkpoint。
原生到 adapter 的原始/投影后完整 chunk 桥接精确一致；Gate 3 通过。Gate 4：

- seeds 1000..1004，五次均完成 500 步，成功 **0/5**；仅最低成功率条件未通过。
- 非法动作率、执行/原始/投影后越界率为 0，关节越界和意外碰撞为 0。
- 平均动作平滑度 L2 为 0.07026611045002937。
- 平均 policy 推理 0.20897001872360707 秒，仿真步 0.009012427625060082 秒。
  这是当前单步闭环仿真协议下的实测，不等于真机实时控制能力。

Gate 文件位于上述接收目录同级 `training-chain-gpu-audit-20260914-006-gate/results/`
下；判定保留 failed，没有用安全项通过替代任务成功。M2 仍未完成。

## 已实现修正与未完成验收

`src/rosetta_reality/vla/image_scaling.py` 定义同一张 float32 像素查表；新增
opt-in FeatureStack 特性 `canonical_image_scaling`。它在原生循环入口把 uint8
转换为规范值，使 native uint8-only 缩放分支跳过第二次除法。原始字节、帧
元数据和历史计划不改动。新 checkpoint 会记录输入 recipe sidecar。

82 项 CPU 回归通过，包含 256 值穷举、非连续 RGB、输入保持、错误 dtype/shape
拒绝、无重复缩放、元数据身份、安装/恢复及原生 observer/optimizer 计数。
证据：`runs/training-chain-canonical-qa-received-20260914-001`，五个文件已校验。
其中 CUDA 用例被有意留待 GPU 空闲；**尚未跑修正后的真实模型和两步更新**。

待续 run 007 / template 012 已密封并校验源文件；其 plan 要求 canonical CPU/CUDA
在真实 loss、全部梯度及动作上精确一致，再进行新的两步 smoke 与独立重载。
不能把旧路径 run 006 的通过结果移作新修正的验收。新 GPU 分配须重新核对 UUID，
不能关闭准入断言。正式 resume、全数据训练、隐藏泛化与真机验证也不在本轮成果中。

## 清理、备份与停机

两项均有用户逐路径授权：删除与本地 SHA 完全一致的 Iris 传输 tar（约2.25 GiB），
以及旧像素路径两步诊断的远端 step-2 目录（约1.50 GiB）。后者模型、processor、
optimizer、scheduler、RNG 共16文件完整备份并校验后，再次核对远端 SHA 才删除。
没有删除 Gate 端点、唯一模型/数据或历史失败日志。清理记录为
`training-chain-storage-cleanup-20260914-001.json` 与 `002.json`。
完整 checkpoint 保留在 `runs/training-chain-checkpoint-recovered-20260914-006/verified`。
最终数据盘剩余约6.8 GiB，系统盘约2.9 GiB，系统盘未清理。

GPU 不足后，无卡启动曾因计费授权不明确被自动审核拒绝。取得用户明确批准后，
仅以无卡模式恢复结果、执行已授权清理并关机；未绕过审核，未在无卡环境训练。
