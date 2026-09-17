# 修复后训练链 GPU 验证：部分完成，保留阻断

已在原登记 RTX 4090 D 上验证：episode 49 的 frame 0、249、499，原生实现与
masked-camera encoder skip 在匹配 RNG/BF16 条件下，loss、每次全部 155 个梯度
张量和完整预测动作块精确相同；参数哈希未变。三项 loss 分别为
5.5850138664245605、2.7107512950897217、2.2632803916931152。
这只支持所测实现路径及样本的数值等价，不证明整套代码没有隐藏问题。

## 实际训练帧与失败边界

所有运行采用新内容寻址工作区，旧失败和计划未覆盖。每个已启动 GPU 工作区
的 Ruff 与 63 项相关回归通过；这些检查有重复，不相加作为独立测试总数。
前一阶段 SSH CPU 审计见 `m2-smolvla-training-chain-ssh-result-2026-09-14.md`。

| 运行 | 已完成 | 阻断 | 成功优化器更新 |
| --- | --- | --- | --- |
| 001 | doctor、benchmark、真实 batch-1 preflight | 新采集器漏加 action batch 维 | 0 |
| 002 | 上述及三个真实时刻完整数值 parity | 新计划复用 preflight/smoke run name，被禁止覆盖保护拒绝 | 0 |
| 003 | 上述及 native DataLoader 交付首批四帧 | 新 forward 钩子读取 processor 已移除的 frame_index | 0 |

003 的 observer 实测交付 `(49,0),(4,0),(49,249),(4,249)`；completed samples、
successful optimizer steps 均为 0。不能将交付、推理 parity 或日志中的 dataset
1000 frames 当作实际完成训练覆盖。预注册全八帧计划还包括两 episode 的
450、499；本轮未完成这些训练更新。

以上三项是新诊断 harness 的问题，不是历史训练漏帧的证据。001/002/003
分别有 15/20/20 个 manifest 文件回传并逐文件 SHA 校验；空接收目录和部分
传输工作区保留。002/003 的 parity 输出一致。003 shutdown request 已取回，
另由 Chrome 实测关机；申请恢复 GPU 时平台明确返回空闲卡数 0。

最新本地 004 修正使用原生 cycle 输出记录身份，与单个实际 policy forward
绑定，在模型入口继续独立核对图像、state、全动作标签与 padding；不向模型
增添元数据，不更改张量。该修正通过远端无卡 Ruff 静态检查，尚未通过 GPU
执行。各阶段使用独立名称。不得重用已存在输出目录。

## 外延检查与限制

- 保存数组验证器原先错误要求 noise 与 action 末维相同。真实保留数组是
  `[1,50,32]` noise 和 `[1,50,14]` action。现以显式 `noise_action_dim` 验证
  完整 noise，保留旧等宽合同和 finite/hash 检查；相关六个新回归通过。
- 实际 native scheduler 在两步 smoke 下自动缩放 warmup 16→0、decay 1280→2。
  这不是正式 1280-step 学习曲线比较，本轮也没有产生有效训练指标。
- 原生视频后端因既有 torchcodec ABI 不兼容回退到 PyAV；没有改动依赖。
  前一 CPU 阶段已经用直接 PyAV 解码核对时间帧与像素。
- 未完成：两步 optimizer smoke、训练后 checkpoint/独立模型重载、固定 Iris
  control Gate 3/4 重测、正式 resume、全生产数据训练覆盖。没有新 Gate 结论。
- 旧 5120 exposures 来自 40 个 frame-0 输入，是已登记小样本设计；不能由此
  推定意外丢帧，亦不能定位历史 0/5 的唯一原因。M2 仍未完成。

## 接续

仅原实例有卡时，按 amendment-004 与新 template 的密封身份建立全新工作区，
先检查环境/缓存/磁盘，再执行同一有界阶段。尚未发生任何成功更新，计划仍
最多两步。已通过的 parity 证据不依赖未来结果。不要重复旧任务、扩大训练、
放宽比较或覆盖失败。待重新取得 GPU 后继续其余已授权测试。

无卡证据恢复结束后，已通过 Chrome 确认原实例处于已关机状态，未释放实例。
待续模板为 `training-chain-gpu-template-20260914-006.json`，SHA-256
`d17291690a90b75f8fce1d6935ba20b281d1cb699c2d8e611072af432b49b8dc`；
当前仅已在本地密封，未启动 GPU run 004。
