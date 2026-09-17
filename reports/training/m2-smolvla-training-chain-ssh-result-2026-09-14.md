# 修复后训练链 SSH 核验 · 2026-09-14

**CPU、训练帧和接口合同核验通过；CUDA 与真实模型/Gate 等价性仍未测。**
本轮不能保证代码绝无隐藏问题，也不能将测试通过当作模型真实性能或 Gate 通过。

用户授权通过 Chrome 打开 SSH 并测试修复代码。原实例
`44db45aec7-cd880eb6` 的带卡开机被平台拒绝（空闲 GPU 为 0），因此在同一
实例的无卡模式运行已登记 CPU 阶段，没有换实例、下载、安装或正式训练。
新源码工作区为 `20260914T025746Z-95cf9cf9483b-f40f5df56020`，
归档 SHA 为 `f40f5df56020bffae6d9232e04bbe2df827cad53206359ac858b3adc34a67bf7`。
执行前后 1,332 个源码/配置/文档文件的校验身份一致，本地复核也无漂移。

## 实测结果

| 检查 | 结果 | 证据含义 |
|---|---|---|
| 修复相关回归 | 569 passed、0 failed、1 skipped、2 deselected | CUDA normalization 因无卡跳过；两项 data 测试未执行，另行数据审计不能冒充这两项测试 |
| 额外 Gate 合同回归 | 30 passed | 评价/报告合同测试，不是 policy rollout 或新 Gate |
| 全非隐藏数值帧 | 20,000 train + 2,500 dev | 连续索引、时间戳、动作/状态维度、finite 和原注册越界容忍通过 |
| 真实 temporal feature → native sampler → DataLoader | 2 个训练 episode，各取 0/249/450/499，共 8 个唯一输入 | 实际交付的样本顺序、state、完整 50×14 action、padding 与独立原始行一致；没有模型更新 |
| 独立视频对照 | 上述 8 个输入逐像素一致；PTS 最大误差约 7.11e-15 秒 | 视频全局时间偏移和数字帧对应通过；不是物理图像/标签语义证明 |
| 动作尾部 | frame 450 有 50 个有效槽，frame 499 有 1 个有效槽 | 尾部复制与 padding 未跨轨迹；合成原生 update 回归另覆盖 batch 1/4、workers 0/2 |
| 两臂保存 processor | 每臂 2 episodes × 3 时段，归一化独立算式误差 0 | 动作往返最大误差 1.1920928955078125e-7，图像和 padding 保持 |
| 生产默认视频后端 | 现场选择 PyAV | torchcodec ABI 加载失败后上游回退 PyAV；保留警告，没有升级依赖或认定训练失败 |
| 资源与源码 | 工作进程树峰值 1,092,997,120 bytes；两个监督阶段均未超限 | 原平台容器 2 GiB；无嵌套 Docker；零生产模型优化器更新，合成测试有更新 |

训练覆盖必须继续分开报告：旧首帧设计的 **5,120 次暴露是 40 个唯一输入帧**，
占 20,000 训练帧的 0.2%；完整动作块覆盖 2,000 个唯一目标帧，占 10%。
新增跨时段草稿的 5,120 个唯一输入只是草稿覆盖，未运行正式模型训练。
本次全量数值审计和 8 帧 loader 核验也不是历史 optimizer 覆盖记录。
首帧小样本是已登记实验设计，不能据此断言意外丢帧或历史 Gate 4 的唯一原因。

## 本轮补出的部署边界

新工作区第一次回归为 553 passed、8 failed、8 errors，根因是历史 protocol 测试
需要的 normalization 文件没有随源码归档传输。使用已有
`scripts/iris_protocol.py:bind_inputs` 按原 SHA 绑定实例上的不可变先决证据后，
受影响批次从 299 passed 变为 315 passed；其他批次的 254 passed 保持有效。
未改历史计划、生产 hash 校验或历史结果。失败记录保留。

这说明新工作区必须执行先决证据绑定，不能仅凭源码上传成功便宣称可训练。
显式 PyAV 审计与默认解码器选择也需要分别记录；本轮已实测默认选择与审计后端相同，
未来环境变化仍需重新检查。

## 限制与下一阶段

- 未执行真实 450M forward/backward、完整模型/processor 独立进程 reload、CUDA
  数值等价或新 Gate 3/4。保存 processor 执行不能替代独立模型加载和推理。
- 新 temporal feature 只支持单独登记的 bounded smoke；不是正式全轨迹训练方案。
  本轮没有启动新炉、恢复历史 optimizer 或选择新 checkpoint。
- 数字帧、像素、PTS 和动作数值一致，仍不足以证明物理画面与示范动作的全部语义关系。
- 原 GPU 可用后，需要密封新小规模计划，先 doctor/benchmark，再真实原生前后向与
  当前实现的受控等价比较，记录实际消费帧、梯度/更新、完整动作张量、独立 reload，
  最后按原协议验证 Gate。不得用本轮 CPU 结果替代这些步骤。
- Iris 已有 Gate 3 两臂通过、Gate 4 两臂各 0/5；本轮没有改变该结果，M2 未完成。

## 证据与收尾

- 第一轮：`runs/training-chain-ssh-received-20260914-001/runs/training-chain-ssh-audit-20260914-001/`
- 第二轮：`runs/training-chain-ssh-received-20260914-003/runs/training-chain-ssh-audit-20260914-002/`
- 两轮 manifest 共 35 个成员逐文件大小/SHA 核验通过；同名 JSON 记录具体证据哈希。
- 两次长连接传输失败目录、第一轮测试失败和第二次空接收器均保留。最终经
  46 个压缩分块逐块校验完成源码传输；未覆盖旧工作区。
- 未提交或推送。实例最终电源状态以同名 JSON 的 Chrome 现场观测为准；不以 SSH
  断开推定关机，不释放实例。
